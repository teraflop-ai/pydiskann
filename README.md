# SPFresh Implementation in Rust
Supports incremental builds, spfresh, quantization with fp16, int8, rabitq, batch search, and s3 cache. Numerous additional improvements will be made.

# Installation
```python
uv add pydiskann
```

# Testing
Tests:
```
cargo clean &&
CC=gcc-12 cargo build --release &&
CC=gcc-12 cargo test --release

CC=gcc-12 cargo bench --bench spfresh
```
```
CC=gcc-12 uv run pytest tests/tests.py
```

# Acknowledgement
Originally began as a fork of https://docs.rs/diskann_rs/latest/diskann_rs/