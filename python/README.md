# pydiskann

Python bindings for [diskann-rs](https://github.com/lukaesch/diskann-rs): memory-mapped
Vamana/DiskANN vector search with filtered search, incremental updates and quantization
(F16, Int8, PQ, RaBitQ).

```bash
uv add pydiskann
```

```python
import pydiskann as d

idx = d.DiskANN.build(vectors, "index.db", metric="l2")
idx.search(query, k=10, beam_width=128)
```

Vectors are `list[list[float]]` (`arr.tolist()` for numpy). `l2` returns squared distances.

```py
docker run --rm -v "$PWD":/io -w /io/python quay.io/pypa/manylinux_2_28_x86_64 bash -c \
  'curl -sSf https://sh.rustup.rs | sh -s -- -y -q && . ~/.cargo/env && pipx run maturin build --release --sdist --out dist'
uv publish python/dist/*
```