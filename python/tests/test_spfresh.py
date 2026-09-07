"""SPFresh correctness tests. From `python/`: uv run pytest -q tests/test_spfresh.py

Invariants checked:
- ids are contiguous and stable across insert / delete / gc / compact / save+open
- probing every posting on an unquantized index reproduces brute-force k-NN exactly
- deletes are hidden immediately and stay hidden after gc; live counts track
- no posting ever reaches max_posting_size after any mutation
- quantized variants with reranking return exact squared-L2 distances
- batch search == single search; persistence round-trips
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import pydiskann as d

DIM = 32
MAX_POSTING = 64


CENTERS = [[random.Random(c * 1000 + j).uniform(-1.0, 1.0) for j in range(DIM)] for c in range(25)]


def clustered(n: int, seed: int, spread: float = 0.12) -> list[list[float]]:
    """Points around the shared CENTERS; `seed` only changes the sample."""
    rng = random.Random(seed)
    return [[c + rng.gauss(0.0, spread) for c in CENTERS[i % len(CENTERS)]] for i in range(n)]


def uniform(n: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(n)]


def l2sq(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def brute(vectors: list[list[float]], q: list[float], k: int, live: set[int] | None = None) -> list[int]:
    ids = range(len(vectors)) if live is None else live
    return sorted(ids, key=lambda i: l2sq(q, vectors[i]))[:k]


def recall(idx: d.SPFresh, vectors, queries, k: int, n_probe: int, live: set[int] | None = None) -> float:
    hits = 0
    for q in queries:
        hits += len(set(idx.search(q, k=k, n_probe=n_probe)) & set(brute(vectors, q, k, live)))
    return hits / (k * len(queries))


def all_probes(idx: d.SPFresh) -> int:
    return idx.stats()["postings"]


def assert_bounds(idx: d.SPFresh) -> None:
    st = idx.stats()
    assert st["max_posting"] < MAX_POSTING, st
    assert st["postings"] >= 1


@pytest.fixture
def built(tmp_path: Path):
    vs = clustered(2000, seed=1)
    idx = d.SPFresh.build(vs, str(tmp_path / "spf"), max_posting_size=MAX_POSTING, min_posting_size=8)
    return idx, vs


def test_build_stats_and_bounds(built) -> None:
    idx, vs = built
    st = idx.stats()
    assert st["live"] == len(vs)
    assert st["deleted"] == 0
    assert st["postings"] >= len(vs) // MAX_POSTING
    assert idx.dim == DIM
    assert_bounds(idx)
    for i in (0, 7, 1999):
        assert idx.get_vector(i) == pytest.approx(vs[i], abs=1e-6)


def test_exhaustive_probe_matches_brute_force(built) -> None:
    idx, vs = built
    for q in uniform(20, seed=2):
        got = idx.search_with_dists(q, k=10, n_probe=all_probes(idx))
        assert [i for i, _ in got] == brute(vs, q, 10)
        for i, dist in got:
            assert dist == pytest.approx(l2sq(q, vs[i]), rel=1e-4, abs=1e-5)


def test_recall_improves_with_probes(built) -> None:
    idx, vs = built
    qs = clustered(50, seed=3)
    r1, r4, r16 = (recall(idx, vs, qs, 10, p) for p in (1, 4, 16))
    assert r16 >= r4 >= r1 - 0.02
    assert r4 >= 0.9
    assert r16 >= 0.98


def test_insert_ids_contiguous_and_searchable(built) -> None:
    idx, vs = built
    new = clustered(300, seed=4)
    ids = idx.insert(new)
    assert ids == list(range(len(vs), len(vs) + len(new)))
    vs = vs + new
    assert idx.stats()["live"] == len(vs)
    assert_bounds(idx)
    for i in ids[::30]:
        top = idx.search_with_dists(vs[i], k=1, n_probe=all_probes(idx))
        assert top[0][0] == i
        assert top[0][1] == pytest.approx(0.0, abs=1e-5)
        assert idx.get_vector(i) == pytest.approx(vs[i], abs=1e-6)
    self_hits = sum(idx.search(vs[i], k=1, n_probe=4) == [i] for i in ids)
    assert self_hits / len(ids) >= 0.9
    assert recall(idx, vs, clustered(30, seed=5), 10, all_probes(idx)) == 1.0


def test_delete_hidden_then_gc(built) -> None:
    idx, vs = built
    dead = list(range(0, 500, 10))
    idx.delete(dead)
    live = set(range(len(vs))) - set(dead)
    st = idx.stats()
    assert st["deleted"] == len(dead)
    assert st["live"] == len(live)
    for i in dead:
        assert idx.is_deleted(i)
        assert idx.get_vector(i) is None
    qs = [vs[i] for i in dead[:10]]
    for q in qs:
        assert not set(idx.search(q, k=10, n_probe=all_probes(idx))) & set(dead)
    assert recall(idx, vs, qs, 10, all_probes(idx), live) == 1.0

    idx.delete(dead)
    idx.delete([10**9])
    assert idx.stats()["live"] == len(live)

    idx.gc()
    assert idx.stats()["live"] == len(live)
    assert_bounds(idx)
    assert recall(idx, vs, qs, 10, all_probes(idx), live) == 1.0
    for q in qs:
        assert not set(idx.search(q, k=10, n_probe=all_probes(idx))) & set(dead)


def test_many_inserts_keep_bounds_and_recall(tmp_path: Path) -> None:
    vs = uniform(1000, seed=6)
    idx = d.SPFresh.build(vs, str(tmp_path / "spf"), max_posting_size=MAX_POSTING, min_posting_size=8)
    for b in range(5):
        new = uniform(600, seed=10 + b)
        ids = idx.insert(new)
        assert ids[0] == len(vs)
        vs += new
        assert_bounds(idx)
    assert idx.stats()["live"] == len(vs)
    qs = uniform(30, seed=7)
    assert recall(idx, vs, qs, 10, all_probes(idx)) == 1.0
    assert recall(idx, vs, qs, 10, 16) >= 0.6


def test_save_open_roundtrip_and_insert_after_open(tmp_path: Path) -> None:
    path = str(tmp_path / "spf")
    vs = clustered(1500, seed=8)
    idx = d.SPFresh.build(vs, path, max_posting_size=MAX_POSTING, min_posting_size=8)
    vs += clustered(200, seed=9)
    idx.insert(vs[1500:])
    dead = [3, 50, 1600]
    idx.delete(dead)
    idx.save()
    qs = clustered(20, seed=10)
    before = [idx.search_with_dists(q, k=10, n_probe=all_probes(idx)) for q in qs]
    st = idx.stats()
    del idx

    idx2 = d.SPFresh.open(path)
    assert idx2.stats() == st
    assert idx2.dim == DIM
    for i in dead:
        assert idx2.is_deleted(i)
    after = [idx2.search_with_dists(q, k=10, n_probe=all_probes(idx2)) for q in qs]
    for a, b in zip(after, before):
        assert [i for i, _ in a] == [i for i, _ in b]
        assert [x for _, x in a] == pytest.approx([x for _, x in b], rel=1e-6)

    ids = idx2.insert(clustered(50, seed=11))
    assert ids == list(range(len(vs), len(vs) + 50))
    assert idx2.get_vector(0) == pytest.approx(vs[0], abs=1e-6)


def test_compact_preserves_results(built) -> None:
    idx, vs = built
    idx.insert(clustered(400, seed=12))
    vs = vs + clustered(400, seed=12)
    qs = clustered(20, seed=13)
    before = [idx.search(q, k=10, n_probe=all_probes(idx)) for q in qs]
    st = idx.stats()
    idx.compact()
    assert idx.stats() == st
    assert [idx.search(q, k=10, n_probe=all_probes(idx)) for q in qs] == before
    assert recall(idx, vs, qs, 10, 8) >= 0.9


@pytest.mark.parametrize(
    "quantizer,min_recall",
    [("f16", 0.98), ("int8", 0.9), ("rabitq", 0.95), ("pq", 0.5)],
)
def test_quantizers_rerank_exact(tmp_path: Path, quantizer: str, min_recall: float) -> None:
    vs = clustered(2000, seed=14)
    idx = d.SPFresh.build(
        vs,
        str(tmp_path / quantizer),
        quantizer=quantizer,
        rerank_size=64,
        pq_subspaces=8,
        max_posting_size=MAX_POSTING,
        min_posting_size=8,
    )
    assert_bounds(idx)
    qs = clustered(30, seed=15)
    assert recall(idx, vs, qs, 10, 16) >= min_recall
    for q in qs[:10]:
        for i, dist in idx.search_with_dists(q, k=10, n_probe=16):
            assert dist == pytest.approx(l2sq(q, vs[i]), rel=1e-4, abs=1e-5)
    ids = idx.insert(clustered(100, seed=16))
    assert ids[0] == len(vs)
    assert_bounds(idx)


def test_bad_quantizer_and_metric_raise(tmp_path: Path) -> None:
    vs = clustered(100, seed=17)
    with pytest.raises(ValueError):
        d.SPFresh.build(vs, str(tmp_path / "q"), quantizer="nope")
    with pytest.raises(ValueError):
        d.SPFresh.build(vs, str(tmp_path / "m"), metric="nope")


def test_dim_mismatch_raises(built) -> None:
    idx, _ = built
    with pytest.raises(RuntimeError):
        idx.insert([[0.0] * (DIM + 1)])
    assert idx.stats()["live"] == 2000


def test_batch_matches_single(built) -> None:
    idx, _ = built
    qs = clustered(25, seed=18)
    assert idx.search_batch(qs, k=10, n_probe=8) == [idx.search(q, k=10, n_probe=8) for q in qs]
