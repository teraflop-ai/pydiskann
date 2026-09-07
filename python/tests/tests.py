"""Python correctness tests for RaBitQ + filtered DiskANN.

Place at:
    python/tests/test_rabitq_filtered.py

Run from the repository's `python/` directory:
    uv run --with pytest pytest -q tests/test_rabitq_filtered.py

Notes:
- The Python binding's `metric="l2"` returns *squared* L2 distances.
- Standalone `QuantizedDiskANN` currently does not expose filtered search in Python.
  The public Python path for filtered + quantized + incremental composition is
  `IncrementalDiskANN.build(..., labels=..., quantizer="rabitq")`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

import pydiskann as d


DIM = 64


def random_vectors(n: int, dim: int = DIM, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n)]


def l2_sq(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def assert_same_results(
    actual: list[tuple[int, float]],
    expected: list[tuple[int, float]],
    *,
    abs_tol: float = 1e-6,
) -> None:
    assert [idx for idx, _ in actual] == [idx for idx, _ in expected]
    assert len(actual) == len(expected)
    for (_, actual_dist), (_, expected_dist) in zip(actual, expected):
        assert actual_dist == pytest.approx(expected_dist, abs=abs_tol)


def assert_ranked_results(
    results: list[tuple[int, float]],
    *,
    max_len: int,
    exact_len: int | None = None,
) -> None:
    if exact_len is not None:
        assert len(results) == exact_len
    else:
        assert len(results) <= max_len

    ids = [idx for idx, _ in results]
    assert len(ids) == len(set(ids)), f"duplicate ids in results: {results}"

    for idx, dist in results:
        assert isinstance(idx, int)
        assert math.isfinite(dist), f"non-finite distance for {idx}: {dist}"
        assert dist >= 0.0, f"negative distance for {idx}: {dist}"

    distances = [dist for _, dist in results]
    assert distances == sorted(distances), f"results not distance-sorted: {results}"


def test_filter_python_api_composes_predicates() -> None:
    f = d.Filter.and_(
        [
            d.Filter.eq(0, 3),
            d.Filter.range(1, 10, 20),
            d.Filter.in_(2, [7, 8, 9]),
        ]
    )

    assert f.matches([3, 10, 7])
    assert f.matches([3, 20, 9])
    assert not f.matches([2, 10, 7])
    assert not f.matches([3, 21, 7])
    assert not f.matches([3, 15, 99])

    either = d.Filter.or_([d.Filter.lt(0, 2), d.Filter.gt(0, 8)])
    assert either.matches([1])
    assert either.matches([9])
    assert not either.matches([5])

    assert d.Filter.none().matches([])


def test_filtered_diskann_respects_filter_and_roundtrips(tmp_path: Path) -> None:
    vectors = random_vectors(180, seed=10_001)
    labels = [[i % 5, i % 100] for i in range(len(vectors))]
    path = tmp_path / "filtered"

    index = d.FilteredDiskANN.build(
        vectors,
        labels,
        str(path),
        metric="l2",
        max_degree=32,
        build_beam_width=64,
    )

    filt = d.Filter.and_(
        [
            d.Filter.eq(0, 3),
            d.Filter.range(1, 10, 90),
        ]
    )

    assert index.count_matching(filt) == sum(filt.matches(x) for x in labels)

    results = index.search_filtered_with_dists(
        vectors[33],
        filt,
        k=10,
        beam_width=128,
    )
    assert results
    assert_ranked_results(results, max_len=10)

    for idx, dist in results:
        assert filt.matches(labels[idx])
        assert index.get_labels(idx) == labels[idx]
        assert dist == pytest.approx(l2_sq(vectors[33], vectors[idx]), abs=1e-5)

    restored = d.FilteredDiskANN.from_bytes(bytes(index.to_bytes()), metric="l2")
    restored_results = restored.search_filtered_with_dists(
        vectors[33],
        filt,
        k=10,
        beam_width=128,
    )
    assert_same_results(restored_results, results, abs_tol=1e-5)


def test_quantized_rabitq_rerank_returns_exact_squared_l2(tmp_path: Path) -> None:
    vectors = random_vectors(200, seed=10_002)
    path = tmp_path / "rabitq.db"

    index = d.QuantizedDiskANN.build(
        vectors,
        str(path),
        quantizer="rabitq",
        metric="l2",
        rerank_size=48,
        max_degree=32,
        build_beam_width=64,
    )

    assert index.num_vectors == len(vectors)
    assert index.dim == DIM

    query = vectors[0]
    results = index.search_with_dists(query, k=10, beam_width=128)
    assert_ranked_results(results, max_len=10, exact_len=10)

    # Re-ranking uses full vectors, so returned distances must be exact squared L2.
    for idx, dist in results:
        assert dist == pytest.approx(l2_sq(query, vectors[idx]), abs=1e-5)


def test_quantized_rabitq_bytes_roundtrip_preserves_search(tmp_path: Path) -> None:
    vectors = random_vectors(160, seed=10_003)
    path = tmp_path / "rabitq_roundtrip.db"

    index = d.QuantizedDiskANN.build(
        vectors,
        str(path),
        quantizer="rabitq",
        metric="l2",
        rerank_size=32,
        max_degree=32,
        build_beam_width=64,
    )

    query = vectors[17]
    before = index.search_with_dists(query, k=12, beam_width=128)

    restored = d.QuantizedDiskANN.from_bytes(
        bytes(index.to_bytes()),
        metric="l2",
        rerank_size=32,
    )
    after = restored.search_with_dists(query, k=12, beam_width=128)

    assert_same_results(after, before, abs_tol=1e-5)


def test_incremental_rabitq_filtered_search_respects_base_labels(tmp_path: Path) -> None:
    vectors = random_vectors(192, seed=10_004)
    labels = [[i % 4, i % 96] for i in range(len(vectors))]
    path = tmp_path / "incremental_rabitq.db"

    index = d.IncrementalDiskANN.build(
        vectors,
        str(path),
        metric="l2",
        labels=labels,
        quantizer="rabitq",
        rerank_size=48,
    )

    filt = d.Filter.and_(
        [
            d.Filter.eq(0, 2),
            d.Filter.range(1, 8, 90),
        ]
    )

    results = index.search_filtered(
        vectors[42],
        filt,
        k=12,
        beam_width=160,
    )

    assert results, "expected filtered RaBitQ search to return base matches"
    assert len(results) <= 12
    assert len(results) == len(set(results))

    # Before delta inserts, all IDs are base IDs, so they index directly into labels.
    for idx in results:
        assert 0 <= idx < len(vectors)
        assert filt.matches(labels[idx])


def test_incremental_rabitq_delta_filter_delete_and_bytes_roundtrip(
    tmp_path: Path,
) -> None:
    base = random_vectors(144, seed=10_005)
    base_labels = [[i % 3] for i in range(len(base))]
    path = tmp_path / "incremental_rabitq_delta.db"

    index = d.IncrementalDiskANN.build(
        base,
        str(path),
        metric="l2",
        labels=base_labels,
        quantizer="rabitq",
        rerank_size=64,
    )

    # Label 99 exists only in the delta layer. This is the important composition
    # case: incremental + labels + RaBitQ quantized traversal.
    delta = random_vectors(16, seed=10_006)
    delta_labels = [[99] for _ in delta]
    delta_ids = index.add_vectors_with_labels(delta, delta_labels)
    delta_id_set = set(delta_ids)

    assert len(delta_ids) == len(delta)
    assert index.stats()["delta_vectors"] == len(delta)

    only_delta = d.Filter.eq(0, 99)

    before_delete = index.search_filtered(
        delta[0],
        only_delta,
        k=8,
        beam_width=192,
    )

    assert before_delete, "delta-only filtered search returned no results"
    assert set(before_delete).issubset(delta_id_set)
    assert delta_ids[0] in before_delete, (
        "querying an inserted vector should retrieve its own delta id "
        "when exact reranking is enabled"
    )

    deleted = delta_ids[0]
    index.delete_vectors([deleted])
    assert index.is_deleted(deleted)

    after_delete = index.search_filtered(
        delta[0],
        only_delta,
        k=8,
        beam_width=192,
    )
    assert deleted not in after_delete
    assert set(after_delete).issubset(delta_id_set - {deleted})

    # A filter with no matching label should cleanly return no results.
    assert index.search_filtered(
        delta[0],
        d.Filter.eq(0, 123_456),
        k=8,
        beam_width=192,
    ) == []

    # Verify the whole composed state survives serialization:
    # base graph + RaBitQ state + labels + delta vectors + tombstones.
    restored = d.IncrementalDiskANN.from_bytes(
        bytes(index.to_bytes()),
        metric="l2",
    )

    assert restored.is_deleted(deleted)
    assert restored.stats()["base_vectors"] == len(base)
    assert restored.stats()["delta_vectors"] == len(delta)
    assert restored.stats()["tombstones"] >= 1

    restored_results = restored.search_filtered(
        delta[0],
        only_delta,
        k=8,
        beam_width=192,
    )
    assert deleted not in restored_results
    assert set(restored_results).issubset(delta_id_set - {deleted})


def test_incremental_rabitq_plain_delta_search(tmp_path: Path) -> None:
    base = random_vectors(128, seed=10_007)
    path = tmp_path / "incremental_rabitq_plain.db"

    index = d.IncrementalDiskANN.build(
        base,
        str(path),
        metric="l2",
        quantizer="rabitq",
        rerank_size=48,
    )

    delta = random_vectors(12, seed=10_008)
    delta_ids = index.add_vectors(delta)

    results = index.search(delta[0], k=10, beam_width=160)
    assert len(results) == 10
    assert delta_ids[0] in results
    assert any(idx in set(delta_ids) for idx in results)
