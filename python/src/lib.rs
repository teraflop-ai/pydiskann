use diskann::pq::PQConfig;
use diskann::{
    formats, DiskANN, DiskAnnError, DiskAnnParams, DistCosine, DistDot, DistL2Sq, Filter,
    FilteredDiskANN, IncrementalConfig, IncrementalDiskANN, IncrementalQuantizedConfig,
    QuantizedConfig, QuantizedDiskANN, QuantizerKind, SPFresh, SPFreshConfig,
};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::HashMap;

// Metric: "l2" = squared L2, "cosine" = 1 - cos, "dot" = 1 - a·b.
enum Any<A, B, C> {
    L2(A),
    Cos(B),
    Dot(C),
}
macro_rules! any { ($t:ident) => { Any<$t<DistL2Sq>, $t<DistCosine>, $t<DistDot>> } }
macro_rules! d {
    ($s:expr, $i:ident => $e:expr) => {
        match $s { Any::L2($i) => $e, Any::Cos($i) => $e, Any::Dot($i) => $e }
    };
}
macro_rules! mk {
    ($m:expr, $e:expr) => {
        match $m {
            "l2" => { type T = DistL2Sq; Any::L2($e) }
            "cosine" => { type T = DistCosine; Any::Cos($e) }
            "dot" => { type T = DistDot; Any::Dot($e) }
            m => return Err(PyValueError::new_err(format!("unknown metric: {m}"))),
        }
    };
}

fn err(e: DiskAnnError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
fn params(max_degree: usize, build_beam_width: usize, alpha: f32) -> DiskAnnParams {
    DiskAnnParams { max_degree, build_beam_width, alpha }
}
fn qkind(quantizer: Option<&str>, pq_subspaces: usize) -> PyResult<Option<QuantizerKind>> {
    Ok(match quantizer {
        None => None,
        Some("f16") => Some(QuantizerKind::F16),
        Some("int8") => Some(QuantizerKind::Int8),
        Some("pq") => Some(QuantizerKind::PQ(PQConfig { num_subspaces: pq_subspaces, ..Default::default() })),
        Some("rabitq") => Some(QuantizerKind::RaBitQ),
        Some(q) => return Err(PyValueError::new_err(format!("unknown quantizer: {q}"))),
    })
}

#[pyclass(name = "Filter", from_py_object)]
#[derive(Clone)]
struct PyFilter(Filter);

#[pymethods]
impl PyFilter {
    #[staticmethod]
    fn none() -> Self { Self(Filter::None) }
    #[staticmethod]
    fn eq(field: usize, value: u64) -> Self { Self(Filter::label_eq(field, value)) }
    #[staticmethod]
    fn in_(field: usize, values: Vec<u64>) -> Self { Self(Filter::label_in(field, values)) }
    #[staticmethod]
    fn lt(field: usize, value: u64) -> Self { Self(Filter::label_lt(field, value)) }
    #[staticmethod]
    fn gt(field: usize, value: u64) -> Self { Self(Filter::label_gt(field, value)) }
    #[staticmethod]
    fn range(field: usize, min: u64, max: u64) -> Self { Self(Filter::label_range(field, min, max)) }
    #[staticmethod]
    fn and_(filters: Vec<PyFilter>) -> Self { Self(Filter::and(filters.into_iter().map(|f| f.0).collect())) }
    #[staticmethod]
    fn or_(filters: Vec<PyFilter>) -> Self { Self(Filter::or(filters.into_iter().map(|f| f.0).collect())) }
    fn matches(&self, labels: Vec<u64>) -> bool { self.0.matches(&labels) }
}

#[pyclass(name = "DiskANN")]
struct PyDiskANN(any!(DiskANN));

#[pymethods]
impl PyDiskANN {
    #[staticmethod]
    #[pyo3(signature = (vectors, path, metric="l2", max_degree=64, build_beam_width=128, alpha=1.2))]
    fn build(py: Python<'_>, vectors: Vec<Vec<f32>>, path: &str, metric: &str, max_degree: usize, build_beam_width: usize, alpha: f32) -> PyResult<Self> {
        let p = params(max_degree, build_beam_width, alpha);
        py.detach(|| Ok(Self(mk!(metric, DiskANN::<T>::build_index_with_params(&vectors, T::default(), path, p).map_err(err)?))))
    }
    #[staticmethod]
    #[pyo3(signature = (path, metric="l2"))]
    fn open(path: &str, metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, DiskANN::<T>::open_index_with(path, T::default()).map_err(err)?)))
    }
    #[staticmethod]
    #[pyo3(signature = (data, metric="l2"))]
    fn from_bytes(data: &[u8], metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, DiskANN::<T>::from_bytes(data.to_vec(), T::default()).map_err(err)?)))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &d!(&self.0, i => i.to_bytes()))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<u32> {
        d!(&self.0, i => i.search(&query, k, beam_width))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search_with_dists(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        d!(&self.0, i => i.search_with_dists(&query, k, beam_width))
    }
    #[pyo3(signature = (queries, k=10, beam_width=128))]
    fn search_batch(&self, py: Python<'_>, queries: Vec<Vec<f32>>, k: usize, beam_width: usize) -> Vec<Vec<u32>> {
        py.detach(|| queries.par_iter().map(|q| d!(&self.0, i => i.search(q, k, beam_width))).collect())
    }
    fn get_vector(&self, id: usize) -> PyResult<Vec<f32>> {
        if id >= self.num_vectors() { return Err(PyIndexError::new_err(id)); }
        Ok(d!(&self.0, i => i.get_vector(id)))
    }
    #[getter]
    fn num_vectors(&self) -> usize { d!(&self.0, i => i.num_vectors) }
    #[getter]
    fn dim(&self) -> usize { d!(&self.0, i => i.dim) }
    #[getter]
    fn max_degree(&self) -> usize { d!(&self.0, i => i.max_degree) }
}

#[pyclass(name = "IncrementalDiskANN")]
struct PyIncremental(any!(IncrementalDiskANN));

#[pymethods]
impl PyIncremental {
    #[staticmethod]
    #[pyo3(signature = (vectors, path, metric="l2", labels=None, quantizer=None, rerank_size=0, pq_subspaces=8, delta_threshold=10_000, tombstone_ratio=0.1))]
    fn build(py: Python<'_>, vectors: Vec<Vec<f32>>, path: &str, metric: &str, labels: Option<Vec<Vec<u64>>>, quantizer: Option<&str>, rerank_size: usize, pq_subspaces: usize, delta_threshold: usize, tombstone_ratio: f32) -> PyResult<Self> {
        let cfg = IncrementalConfig { delta_threshold, tombstone_ratio_threshold: tombstone_ratio, ..Default::default() };
        let qc = IncrementalQuantizedConfig { rerank_size };
        let qk = qkind(quantizer, pq_subspaces)?;
        py.detach(|| Ok(Self(mk!(metric, match (labels, qk) {
            (None, None) => IncrementalDiskANN::<T>::build_with_config(&vectors, path, cfg),
            (Some(l), None) => IncrementalDiskANN::<T>::build_with_labels(&vectors, &l, path, cfg),
            (None, Some(QuantizerKind::F16)) => IncrementalDiskANN::<T>::build_quantized_f16(&vectors, path, cfg, qc),
            (None, Some(QuantizerKind::Int8)) => IncrementalDiskANN::<T>::build_quantized_int8(&vectors, path, cfg, qc),
            (None, Some(QuantizerKind::PQ(p))) => IncrementalDiskANN::<T>::build_quantized_pq(&vectors, path, cfg, p, qc),
            (None, Some(QuantizerKind::RaBitQ)) => IncrementalDiskANN::<T>::build_quantized_rabitq(&vectors, path, cfg, qc),
            (Some(l), Some(q)) => IncrementalDiskANN::<T>::build_full(&vectors, &l, path, cfg, q, qc),
        }.map_err(err)?))))
    }
    #[staticmethod]
    #[pyo3(signature = (path, metric="l2"))]
    fn open(path: &str, metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, IncrementalDiskANN::<T>::open(path).map_err(err)?)))
    }
    #[staticmethod]
    #[pyo3(signature = (data, metric="l2"))]
    fn from_bytes(data: &[u8], metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, IncrementalDiskANN::<T>::from_bytes(data, T::default(), Default::default()).map_err(err)?)))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &d!(&self.0, i => i.to_bytes()))
    }
    fn add_vectors(&self, py: Python<'_>, vectors: Vec<Vec<f32>>) -> PyResult<Vec<u64>> {
        py.detach(|| d!(&self.0, i => i.add_vectors(&vectors)).map_err(err))
    }
    fn add_vectors_with_labels(&self, py: Python<'_>, vectors: Vec<Vec<f32>>, labels: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
        py.detach(|| d!(&self.0, i => i.add_vectors_with_labels(&vectors, &labels)).map_err(err))
    }
    fn delete_vectors(&self, ids: Vec<u64>) -> PyResult<()> {
        d!(&self.0, i => i.delete_vectors(&ids)).map_err(err)
    }
    fn is_deleted(&self, id: u64) -> bool { d!(&self.0, i => i.is_deleted(id)) }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<u64> {
        d!(&self.0, i => i.search(&query, k, beam_width))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search_with_dists(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<(u64, f32)> {
        d!(&self.0, i => i.search_with_dists(&query, k, beam_width))
    }
    #[pyo3(signature = (query, filter, k=10, beam_width=128))]
    fn search_filtered(&self, query: Vec<f32>, filter: &PyFilter, k: usize, beam_width: usize) -> Vec<u64> {
        d!(&self.0, i => i.search_filtered(&query, k, beam_width, &filter.0))
    }
    #[pyo3(signature = (queries, k=10, beam_width=128))]
    fn search_batch(&self, py: Python<'_>, queries: Vec<Vec<f32>>, k: usize, beam_width: usize) -> Vec<Vec<u64>> {
        py.detach(|| d!(&self.0, i => i.search_batch(&queries, k, beam_width)))
    }
    fn get_vector(&self, id: u64) -> Option<Vec<f32>> { d!(&self.0, i => i.get_vector(id)) }
    fn should_compact(&self) -> bool { d!(&self.0, i => i.should_compact()) }
    fn compact(&mut self, py: Python<'_>, new_path: &str) -> PyResult<()> {
        py.detach(|| d!(&mut self.0, i => i.compact(new_path)).map_err(err))
    }
    fn stats(&self) -> HashMap<&'static str, usize> {
        let s = d!(&self.0, i => i.stats());
        HashMap::from([
            ("base_vectors", s.base_vectors),
            ("delta_vectors", s.delta_vectors),
            ("tombstones", s.tombstones),
            ("total_live", s.total_live),
            ("dim", s.dim),
        ])
    }
    #[getter]
    fn dim(&self) -> usize { d!(&self.0, i => i.dim()) }
}

#[pyclass(name = "FilteredDiskANN")]
struct PyFiltered(any!(FilteredDiskANN));

#[pymethods]
impl PyFiltered {
    #[staticmethod]
    #[pyo3(signature = (vectors, labels, path, metric="l2", max_degree=64, build_beam_width=128, alpha=1.2))]
    fn build(py: Python<'_>, vectors: Vec<Vec<f32>>, labels: Vec<Vec<u64>>, path: &str, metric: &str, max_degree: usize, build_beam_width: usize, alpha: f32) -> PyResult<Self> {
        let p = params(max_degree, build_beam_width, alpha);
        py.detach(|| Ok(Self(mk!(metric, FilteredDiskANN::<T>::build_with_params(&vectors, &labels, path, p).map_err(err)?))))
    }
    #[staticmethod]
    #[pyo3(signature = (path, metric="l2"))]
    fn open(path: &str, metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, FilteredDiskANN::<T>::open(path).map_err(err)?)))
    }
    #[staticmethod]
    #[pyo3(signature = (data, metric="l2"))]
    fn from_bytes(data: &[u8], metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, FilteredDiskANN::<T>::from_bytes(data.to_vec(), T::default()).map_err(err)?)))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &d!(&self.0, i => i.to_bytes()))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<u32> {
        d!(&self.0, i => i.search(&query, k, beam_width))
    }
    #[pyo3(signature = (query, filter, k=10, beam_width=128))]
    fn search_filtered(&self, query: Vec<f32>, filter: &PyFilter, k: usize, beam_width: usize) -> Vec<u32> {
        d!(&self.0, i => i.search_filtered(&query, k, beam_width, &filter.0))
    }
    #[pyo3(signature = (query, filter, k=10, beam_width=128))]
    fn search_filtered_with_dists(&self, query: Vec<f32>, filter: &PyFilter, k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        d!(&self.0, i => i.search_filtered_with_dists(&query, k, beam_width, &filter.0))
    }
    #[pyo3(signature = (queries, filter, k=10, beam_width=128))]
    fn search_filtered_batch(&self, py: Python<'_>, queries: Vec<Vec<f32>>, filter: &PyFilter, k: usize, beam_width: usize) -> Vec<Vec<u32>> {
        py.detach(|| d!(&self.0, i => i.search_filtered_batch(&queries, k, beam_width, &filter.0)))
    }
    fn count_matching(&self, filter: &PyFilter) -> usize { d!(&self.0, i => i.count_matching(&filter.0)) }
    fn get_labels(&self, id: usize) -> Option<Vec<u64>> { d!(&self.0, i => i.get_labels(id).map(|l| l.to_vec())) }
    #[getter]
    fn num_vectors(&self) -> usize { d!(&self.0, i => i.num_vectors()) }
    #[getter]
    fn num_fields(&self) -> usize { d!(&self.0, i => i.num_fields()) }
}

#[pyclass(name = "QuantizedDiskANN")]
struct PyQuantized(any!(QuantizedDiskANN));

#[pymethods]
impl PyQuantized {
    #[staticmethod]
    #[pyo3(signature = (vectors, path, quantizer="f16", metric="l2", rerank_size=0, pq_subspaces=8, max_degree=64, build_beam_width=128, alpha=1.2))]
    fn build(py: Python<'_>, vectors: Vec<Vec<f32>>, path: &str, quantizer: &str, metric: &str, rerank_size: usize, pq_subspaces: usize, max_degree: usize, build_beam_width: usize, alpha: f32) -> PyResult<Self> {
        let (p, c) = (params(max_degree, build_beam_width, alpha), QuantizedConfig { rerank_size });
        let pq = PQConfig { num_subspaces: pq_subspaces, ..Default::default() };
        py.detach(|| Ok(Self(mk!(metric, match quantizer {
            "f16" => QuantizedDiskANN::<T>::build_f16(&vectors, T::default(), path, p, c),
            "int8" => QuantizedDiskANN::<T>::build_int8(&vectors, T::default(), path, p, c),
            "pq" => QuantizedDiskANN::<T>::build_pq(&vectors, T::default(), path, p, pq, c),
            "rabitq" => QuantizedDiskANN::<T>::build_rabitq(&vectors, T::default(), path, p, c),
            q => return Err(PyValueError::new_err(format!("unknown quantizer: {q}"))),
        }.map_err(err)?))))
    }
    #[staticmethod]
    #[pyo3(signature = (base_path, quantized_path, metric="l2", rerank_size=0))]
    fn open(base_path: &str, quantized_path: &str, metric: &str, rerank_size: usize) -> PyResult<Self> {
        Ok(Self(mk!(metric, QuantizedDiskANN::<T>::open(base_path, quantized_path, T::default(), QuantizedConfig { rerank_size }).map_err(err)?)))
    }
    fn save_quantized(&self, path: &str) -> PyResult<()> { d!(&self.0, i => i.save_quantized(path)).map_err(err) }
    #[staticmethod]
    #[pyo3(signature = (data, metric="l2", rerank_size=0))]
    fn from_bytes(data: &[u8], metric: &str, rerank_size: usize) -> PyResult<Self> {
        Ok(Self(mk!(metric, QuantizedDiskANN::<T>::from_bytes(data, T::default(), QuantizedConfig { rerank_size }).map_err(err)?)))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &d!(&self.0, i => i.to_bytes()))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<u32> {
        d!(&self.0, i => i.search(&query, k, beam_width))
    }
    #[pyo3(signature = (query, k=10, beam_width=128))]
    fn search_with_dists(&self, query: Vec<f32>, k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        d!(&self.0, i => i.search_with_dists(&query, k, beam_width))
    }
    #[pyo3(signature = (queries, k=10, beam_width=128))]
    fn search_batch(&self, py: Python<'_>, queries: Vec<Vec<f32>>, k: usize, beam_width: usize) -> Vec<Vec<u32>> {
        py.detach(|| d!(&self.0, i => i.search_batch(&queries, k, beam_width)))
    }
    #[getter]
    fn num_vectors(&self) -> usize { d!(&self.0, i => i.num_vectors()) }
    #[getter]
    fn dim(&self) -> usize { d!(&self.0, i => i.dim()) }
}

#[pyclass(name = "SPFresh")]
struct PySPFresh(any!(SPFresh));

#[pymethods]
impl PySPFresh {
    #[staticmethod]
    #[pyo3(signature = (vectors, path, metric="l2", quantizer=None, rerank_size=0, pq_subspaces=8, max_posting_size=128, min_posting_size=16, reassign_neighbors=16, probe_ratio=f32::INFINITY))]
    fn build(py: Python<'_>, vectors: Vec<Vec<f32>>, path: &str, metric: &str, quantizer: Option<&str>, rerank_size: usize, pq_subspaces: usize, max_posting_size: usize, min_posting_size: usize, reassign_neighbors: usize, probe_ratio: f32) -> PyResult<Self> {
        let cfg = SPFreshConfig { max_posting_size, min_posting_size, probe_ratio, rerank_size, reassign_neighbors, ..Default::default() };
        let qk = qkind(quantizer, pq_subspaces)?;
        py.detach(|| Ok(Self(mk!(metric, SPFresh::<T>::build(&vectors, path, cfg, qk).map_err(err)?))))
    }
    #[staticmethod]
    #[pyo3(signature = (path, metric="l2"))]
    fn open(path: &str, metric: &str) -> PyResult<Self> {
        Ok(Self(mk!(metric, SPFresh::<T>::open(path).map_err(err)?)))
    }
    fn save(&self) -> PyResult<()> { d!(&self.0, i => i.save()).map_err(err) }
    fn insert(&self, py: Python<'_>, vectors: Vec<Vec<f32>>) -> PyResult<Vec<u64>> {
        py.detach(|| d!(&self.0, i => i.insert(&vectors)).map_err(err))
    }
    fn delete(&self, ids: Vec<u64>) { d!(&self.0, i => i.delete(&ids)) }
    fn is_deleted(&self, id: u64) -> bool { d!(&self.0, i => i.is_deleted(id)) }
    #[pyo3(signature = (query, k=10, n_probe=8))]
    fn search(&self, query: Vec<f32>, k: usize, n_probe: usize) -> Vec<u64> {
        d!(&self.0, i => i.search(&query, k, n_probe))
    }
    #[pyo3(signature = (query, k=10, n_probe=8))]
    fn search_with_dists(&self, query: Vec<f32>, k: usize, n_probe: usize) -> Vec<(u64, f32)> {
        d!(&self.0, i => i.search_with_dists(&query, k, n_probe))
    }
    #[pyo3(signature = (queries, k=10, n_probe=8))]
    fn search_batch(&self, py: Python<'_>, queries: Vec<Vec<f32>>, k: usize, n_probe: usize) -> Vec<Vec<u64>> {
        py.detach(|| d!(&self.0, i => i.search_batch(&queries, k, n_probe)))
    }
    fn get_vector(&self, id: u64) -> Option<Vec<f32>> { d!(&self.0, i => i.get_vector(id)) }
    fn gc(&self, py: Python<'_>) -> PyResult<()> { py.detach(|| d!(&self.0, i => i.gc()).map_err(err)) }
    fn compact(&self, py: Python<'_>) -> PyResult<()> { py.detach(|| d!(&self.0, i => i.compact()).map_err(err)) }
    fn stats(&self) -> HashMap<&'static str, usize> {
        let s = d!(&self.0, i => i.stats());
        HashMap::from([("live", s.live), ("deleted", s.deleted), ("postings", s.postings), ("max_posting", s.max_posting)])
    }
    #[getter]
    fn dim(&self) -> usize { d!(&self.0, i => i.dim()) }
    /// Incremental snapshot to `s3://bucket/prefix` (AWS_* env credentials) or `file:///dir/prefix`; returns the version.
    #[cfg(feature = "object-store")]
    fn snapshot(&self, py: Python<'_>, url: &str) -> PyResult<u64> {
        let (s, prefix) = remote::store(url)?;
        py.detach(|| d!(&self.0, i => remote::rt().block_on(i.snapshot(s.as_ref(), &prefix))).map(|m| m.version).map_err(err))
    }
    /// Materialise the latest snapshot at `url` into `{local}.*` and open it.
    #[cfg(feature = "object-store")]
    #[staticmethod]
    #[pyo3(signature = (url, local, metric="l2"))]
    fn restore(py: Python<'_>, url: &str, local: &str, metric: &str) -> PyResult<Self> {
        let (s, prefix) = remote::store(url)?;
        py.detach(|| Ok(Self(mk!(metric, remote::rt().block_on(SPFresh::<T>::restore(s.as_ref(), &prefix, local)).map_err(err)?))))
    }
}

#[cfg(feature = "object-store")]
mod remote {
    use super::*;
    use object_store::ObjectStore;
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    pub fn rt() -> &'static tokio::runtime::Runtime { RT.get_or_init(|| tokio::runtime::Runtime::new().unwrap()) }
    pub fn store(url: &str) -> PyResult<(Box<dyn ObjectStore>, String)> {
        let e = |e: object_store::Error| PyValueError::new_err(e.to_string());
        if let Some(p) = url.strip_prefix("file://") {
            let (dir, prefix) = p.rsplit_once('/').unwrap_or((p, ""));
            return Ok((Box::new(object_store::local::LocalFileSystem::new_with_prefix(dir).map_err(e)?), prefix.to_string()));
        }
        let (bucket, prefix) = url.trim_start_matches("s3://").split_once('/').unwrap_or((url, ""));
        Ok((Box::new(object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket).build().map_err(e)?), prefix.to_string()))
    }
}

#[pyfunction]
fn read_fvecs(path: &str) -> PyResult<Vec<Vec<f32>>> { formats::read_fvecs(path).map_err(err) }
#[pyfunction]
fn read_ivecs(path: &str) -> PyResult<Vec<Vec<i32>>> { formats::read_ivecs(path).map_err(err) }
#[pyfunction]
fn read_bvecs_as_f32(path: &str) -> PyResult<Vec<Vec<f32>>> { formats::read_bvecs_as_f32(path).map_err(err) }
#[pyfunction]
fn write_fvecs(path: &str, vectors: Vec<Vec<f32>>) -> PyResult<()> { formats::write_fvecs(path, &vectors).map_err(err) }
#[pyfunction]
fn simd_info() -> String { diskann::simd_info() }

#[pymodule]
fn pydiskann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiskANN>()?;
    m.add_class::<PyIncremental>()?;
    m.add_class::<PyFiltered>()?;
    m.add_class::<PyQuantized>()?;
    m.add_class::<PySPFresh>()?;
    m.add_class::<PyFilter>()?;
    m.add_function(wrap_pyfunction!(read_fvecs, m)?)?;
    m.add_function(wrap_pyfunction!(read_ivecs, m)?)?;
    m.add_function(wrap_pyfunction!(read_bvecs_as_f32, m)?)?;
    m.add_function(wrap_pyfunction!(write_fvecs, m)?)?;
    m.add_function(wrap_pyfunction!(simd_info, m)?)
}