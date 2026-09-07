use crate::pq::{PQConfig, ProductQuantizer};
use crate::rabitq::{RaBitQ, RaBitQQuery};
use crate::sq::{F16Quantizer, Int8Quantizer, VectorQuantizer};
use crate::Distance;
use crate::{beam_search, BeamSearchConfig, DiskANN, DiskAnnError, DiskAnnParams, GraphIndex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

/// Compute quantized distance for a single candidate from flat code buffer.
#[inline]
pub(crate) fn quantized_distance_from_codes<D: Distance<f32> + 'static>(
    dist: &D,
    query: &[f32],
    idx: usize,
    codes: &[u8],
    code_size: usize,
    quantizer: &QuantizerState,
    prep: &Prepared,
) -> f32 {
    use std::any::TypeId;

    let start = idx * code_size;
    let code = &codes[start..start + code_size];
    let l2 = TypeId::of::<D>() == TypeId::of::<crate::DistL2>();

    if !l2 && TypeId::of::<D>() != TypeId::of::<crate::DistL2Sq>() {
        let decoded = match quantizer {
            QuantizerState::PQ(q) => q.decode(code),
            QuantizerState::F16(q) => q.decode(code),
            QuantizerState::Int8(q) => q.decode(code),
            QuantizerState::RaBitQ(q) => q.decode(code),
        };
        return dist.eval(query, &decoded);
    }

    let d = match (quantizer, prep) {
        (QuantizerState::PQ(q), Prepared::Table(t)) => q.distance_with_table(t, code),
        (QuantizerState::PQ(q), _) => q.asymmetric_distance(query, code),
        (QuantizerState::RaBitQ(q), Prepared::RaBitQ(p)) => q.distance(p, code),
        (QuantizerState::RaBitQ(q), _) => q.asymmetric_distance(query, code),
        (QuantizerState::F16(q), Prepared::F16(p)) => q.distance_f16(p, code),
        (QuantizerState::Int8(q), Prepared::U8(p)) => q.distance_u8(p, code),
        (QuantizerState::F16(q), _) => q.asymmetric_distance(query, code),
        (QuantizerState::Int8(q), _) => q.asymmetric_distance(query, code),
    };

    if l2 {
        d.sqrt()
    } else {
        d
    }
}

/// Per-query precomputed state (PQ distance table or RaBitQ query).
pub(crate) enum Prepared {
    Table(Vec<f32>),
    RaBitQ(RaBitQQuery),
    F16(Vec<numkong::f16>),
    U8(Vec<u8>),
}

/// Shared quantized search implementation usable with any `GraphIndex`.
///
/// Performs beam search using quantized distances, with optional re-ranking
/// using exact distances from the graph and optional label-based filtering.
fn quantized_search<D: Distance<f32> + 'static>(
    graph: &dyn GraphIndex,
    dist: &D,
    codes: &[u8],
    code_size: usize,
    quantizer: &QuantizerState,
    start_ids: &[u32],
    query: &[f32],
    k: usize,
    beam_width: usize,
    rerank_size: usize,
    filter_fn: impl Fn(u32) -> bool,
    config: BeamSearchConfig,
) -> Vec<(u32, f32)> {
    let prep = quantizer.prepare(query);

    let search_k = if rerank_size > 0 {
        rerank_size.max(k)
    } else {
        k
    };

    let mut results = beam_search(
        start_ids,
        beam_width,
        search_k,
        |id| {
            quantized_distance_from_codes(
                dist,
                query,
                id as usize,
                codes,
                code_size,
                quantizer,
                &prep,
            )
        },
        |id| graph.get_neighbors(id),
        &filter_fn,
        config,
    );

    // Re-ranking phase
    if rerank_size > 0 {
        results = results
            .iter()
            .map(|&(id, _)| {
                let exact_dist = graph.distance_to(query, id);
                (id, exact_dist)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
    }

    results
}

/// Magic number for quantized sidecar files: "QANN"
const MAGIC: u32 = 0x51414E4E;
/// Current sidecar file format version
const VERSION: u32 = 2;

/// Configuration for quantized search.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedConfig {
    /// Number of candidates to re-rank with exact vectors after quantized search.
    /// Set to 0 to disable re-ranking (faster, lower recall).
    pub rerank_size: usize,
}

impl Default for QuantizedConfig {
    fn default() -> Self {
        Self { rerank_size: 0 }
    }
}

/// Monomorphic quantizer state — avoids dynamic dispatch on the hot path.
///
/// PQ uses `create_distance_table()` once per query then `distance_with_table()`
/// per candidate (O(M) lookups). F16/Int8 use `asymmetric_distance()` directly.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum QuantizerState {
    PQ(ProductQuantizer),
    F16(F16Quantizer),
    Int8(Int8Quantizer),
    RaBitQ(RaBitQ),
}

impl QuantizerState {
    fn quantizer_type_id(&self) -> u8 {
        match self {
            QuantizerState::PQ(_) => 0,
            QuantizerState::F16(_) => 1,
            QuantizerState::Int8(_) => 2,
            QuantizerState::RaBitQ(_) => 3,
        }
    }

    pub(crate) fn prepare(&self, query: &[f32]) -> Prepared {
        match self {
            QuantizerState::PQ(pq) => Prepared::Table(pq.create_distance_table(query)),
            QuantizerState::RaBitQ(rq) => Prepared::RaBitQ(rq.query(query)),
            QuantizerState::F16(f16q) => Prepared::F16(f16q.prepare(query)),
            QuantizerState::Int8(int8q) => Prepared::U8(int8q.encode(query)),
        }
    }
}

/// Quantized DiskANN index — wraps a `DiskANN<D>` with compressed in-memory codes.
///
/// During beam search, distance computations use the compressed codes in RAM
/// instead of reading full f32 vectors from the backing store. After search,
/// an optional re-ranking phase re-scores top candidates with exact vectors.
pub struct QuantizedDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// The underlying DiskANN index (graph + full vectors on disk/mmap)
    base: DiskANN<D>,
    /// Compressed codes in RAM — contiguous flat buffer.
    /// `codes[i * code_size .. (i+1) * code_size]` is the code for vector i.
    codes: Vec<u8>,
    /// Bytes per code
    code_size: usize,
    /// Quantizer state (PQ, F16, or Int8)
    quantizer: QuantizerState,
    /// Number of candidates to re-rank with exact vectors (0 = no re-ranking)
    rerank_size: usize,
}

// ---------------------------------------------------------------------------
// Construction from existing components
// ---------------------------------------------------------------------------

impl<D> QuantizedDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Create from an existing index and a pre-trained ProductQuantizer.
    ///
    /// Encodes all vectors from `base` into the in-memory codes buffer.
    pub fn from_pq(base: DiskANN<D>, pq: ProductQuantizer, config: QuantizedConfig) -> Self {
        let n = base.num_vectors;
        let code_size = pq.stats().code_size_bytes;
        let codes = encode_all_pq(&base, &pq, n);
        Self {
            base,
            codes,
            code_size,
            quantizer: QuantizerState::PQ(pq),
            rerank_size: config.rerank_size,
        }
    }

    /// Create from an existing index using F16 quantization.
    pub fn from_f16(base: DiskANN<D>, config: QuantizedConfig) -> Self {
        let dim = base.dim;
        let n = base.num_vectors;
        let f16q = F16Quantizer::new(dim);
        let code_size = dim * 2;
        let codes = encode_all_generic(&base, &f16q, n, code_size);
        Self {
            base,
            codes,
            code_size,
            quantizer: QuantizerState::F16(f16q),
            rerank_size: config.rerank_size,
        }
    }

    /// Create from an existing index and a trained RaBitQ quantizer.
    pub fn from_rabitq(base: DiskANN<D>, rq: RaBitQ, config: QuantizedConfig) -> Self {
        let n = base.num_vectors;
        let code_size = rq.code_size();
        let codes = encode_all_generic(&base, &rq, n, code_size);
        Self {
            base,
            codes,
            code_size,
            quantizer: QuantizerState::RaBitQ(rq),
            rerank_size: config.rerank_size,
        }
    }

    /// Create from an existing index and a pre-trained Int8Quantizer.
    pub fn from_int8(base: DiskANN<D>, int8q: Int8Quantizer, config: QuantizedConfig) -> Self {
        let n = base.num_vectors;
        let code_size = int8q.dim();
        let codes = encode_all_generic(&base, &int8q, n, code_size);
        Self {
            base,
            codes,
            code_size,
            quantizer: QuantizerState::Int8(int8q),
            rerank_size: config.rerank_size,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of vectors in the index.
    pub fn num_vectors(&self) -> usize {
        self.base.num_vectors
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.base.dim
    }

    /// Reference to the underlying base index.
    pub fn base(&self) -> &DiskANN<D> {
        &self.base
    }

    /// Get a full-precision vector from the base index.
    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        self.base.get_vector(idx)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search with quantized beam search, returning `(id, distance)` pairs.
    ///
    /// Distances in the returned results are:
    /// - If `rerank_size > 0`: exact distances from the base index
    /// - Otherwise: approximate distances from the quantizer
    pub fn search_with_dists(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.base.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.base.dim
        );

        quantized_search(
            &self.base,
            &self.base.dist,
            &self.codes,
            self.code_size,
            &self.quantizer,
            &[self.base.medoid_id],
            query,
            k,
            beam_width,
            self.rerank_size,
            |_| true,
            BeamSearchConfig::default(),
        )
    }

    /// Search returning only neighbor IDs.
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Batch search (parallel over queries).
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, beam_width: usize) -> Vec<Vec<u32>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k, beam_width))
            .collect()
    }

    /// Search with quantized beam search and a metadata filter.
    ///
    /// Composes quantized distance computation with label-based filtering.
    /// `labels` must have one entry per vector in the index, with the same
    /// ordering as the base index vectors.
    ///
    /// Distances in the returned results are:
    /// - If `rerank_size > 0`: exact distances from the base index
    /// - Otherwise: approximate distances from the quantizer
    pub fn search_filtered_with_dists(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        labels: &[Vec<u64>],
        filter: &crate::Filter,
    ) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.base.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.base.dim
        );
        assert_eq!(
            labels.len(),
            self.base.num_vectors,
            "Labels count {} != index vectors {}",
            labels.len(),
            self.base.num_vectors
        );

        // For unfiltered search, use the fast path
        if matches!(filter, crate::Filter::None) {
            return self.search_with_dists(query, k, beam_width);
        }

        let expanded_beam = (beam_width * 4).max(k * 10);

        quantized_search(
            &self.base,
            &self.base.dist,
            &self.codes,
            self.code_size,
            &self.quantizer,
            &[self.base.medoid_id],
            query,
            k,
            beam_width,
            self.rerank_size,
            |id| filter.matches(&labels[id as usize]),
            BeamSearchConfig {
                expanded_beam: Some(expanded_beam),
                max_iterations: Some(expanded_beam * 2),
                early_term_factor: Some(1.5),
            },
        )
    }

    /// Search with filter, returning only neighbor IDs.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        labels: &[Vec<u64>],
        filter: &crate::Filter,
    ) -> Vec<u32> {
        self.search_filtered_with_dists(query, k, beam_width, labels, filter)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Persistence — sidecar file
    // -----------------------------------------------------------------------

    /// Save the quantizer state and codes to a sidecar file.
    ///
    /// The base index should be saved separately via `DiskANN::build_index` or
    /// already persisted on disk.
    ///
    /// ## Sidecar Format
    /// ```text
    /// [magic: u32 = 0x51414E4E]
    /// [version: u32 = 1]
    /// [quantizer_type: u8]        // 0=PQ, 1=F16, 2=Int8
    /// [num_vectors: u64]
    /// [code_size: u64]
    /// [quantizer_data_len: u64]
    /// [quantizer_data: bincode]
    /// [codes: num_vectors * code_size bytes]
    /// ```
    pub fn save_quantized(&self, path: &str) -> Result<(), DiskAnnError> {
        let bytes = self.quantized_to_bytes();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    }

    /// Open a QuantizedDiskANN from a base index path and a sidecar path.
    pub fn open(
        base_path: &str,
        quantized_path: &str,
        dist: D,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let base = DiskANN::open_index_with(base_path, dist)?;
        let bytes = std::fs::read(quantized_path)?;
        Self::from_quantized_bytes(base, &bytes, config)
    }

    /// Serialize the quantizer + codes to bytes (without the base index).
    pub fn to_bytes(&self) -> Vec<u8> {
        // Format: [base_len:u64][base_bytes][quantized_bytes]
        let base_bytes = self.base.to_bytes();
        let quantized_bytes = self.quantized_to_bytes();
        let mut out = Vec::with_capacity(8 + base_bytes.len() + quantized_bytes.len());
        out.extend_from_slice(&(base_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&base_bytes);
        out.extend_from_slice(&quantized_bytes);
        out
    }

    /// Deserialize from bytes (base index + quantized data).
    pub fn from_bytes(
        bytes: &[u8],
        dist: D,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        if bytes.len() < 8 {
            return Err(DiskAnnError::IndexError("Buffer too small".into()));
        }
        let base_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + base_len {
            return Err(DiskAnnError::IndexError(
                "Buffer too small for base index".into(),
            ));
        }
        let base = DiskANN::from_bytes(bytes[8..8 + base_len].to_vec(), dist)?;
        Self::from_quantized_bytes(base, &bytes[8 + base_len..], config)
    }

    // -----------------------------------------------------------------------
    // Internal serialization helpers
    // -----------------------------------------------------------------------

    fn quantized_to_bytes(&self) -> Vec<u8> {
        let quantizer_data = bincode::serialize(&self.quantizer).unwrap();
        let num_vectors = self.base.num_vectors as u64;
        let code_size = self.code_size as u64;
        let quantizer_data_len = quantizer_data.len() as u64;

        let header_size = 4 + 4 + 1 + 8 + 8 + 8; // magic + version + type + num_vectors + code_size + qdata_len
        let total = header_size + quantizer_data.len() + self.codes.len();
        let mut out = Vec::with_capacity(total);

        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.push(self.quantizer.quantizer_type_id());
        out.extend_from_slice(&num_vectors.to_le_bytes());
        out.extend_from_slice(&code_size.to_le_bytes());
        out.extend_from_slice(&quantizer_data_len.to_le_bytes());
        out.extend_from_slice(&quantizer_data);
        out.extend_from_slice(&self.codes);
        out
    }

    fn from_quantized_bytes(
        base: DiskANN<D>,
        bytes: &[u8],
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let err = |m: &str| DiskAnnError::IndexError(m.into());
        if bytes.len() < 33 {
            return Err(err("Quantized data too small"));
        }
        let u32_at = |p: usize| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        let u64_at = |p: usize| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;

        let magic = u32_at(0);
        if magic != MAGIC {
            return Err(DiskAnnError::IndexError(format!(
                "Invalid magic: expected 0x{:08X}, got 0x{:08X}",
                MAGIC, magic
            )));
        }
        let version = u32_at(4);
        if version != VERSION {
            return Err(DiskAnnError::IndexError(format!(
                "Unsupported version: {}",
                version
            )));
        }
        let quantizer_type = bytes[8];
        let num_vectors = u64_at(9);
        let code_size = u64_at(17);
        let quantizer_data_len = u64_at(25);
        let mut pos = 33;

        if num_vectors != base.num_vectors {
            return Err(DiskAnnError::IndexError(format!(
                "Sidecar has {} vectors, base index has {}",
                num_vectors, base.num_vectors
            )));
        }
        if bytes.len() < pos + quantizer_data_len {
            return Err(err("Truncated quantizer data"));
        }
        let quantizer: QuantizerState =
            bincode::deserialize(&bytes[pos..pos + quantizer_data_len])?;
        if quantizer.quantizer_type_id() != quantizer_type {
            return Err(err("Quantizer type mismatch"));
        }
        pos += quantizer_data_len;

        let codes_len = num_vectors * code_size;
        if bytes.len() < pos + codes_len {
            return Err(err("Truncated codes data"));
        }
        let codes = bytes[pos..pos + codes_len].to_vec();

        Ok(Self {
            base,
            codes,
            code_size,
            quantizer,
            rerank_size: config.rerank_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Combined builders (vectors -> graph + quantizer + codes)
// ---------------------------------------------------------------------------

impl<D> QuantizedDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build a PQ-quantized index from scratch.
    pub fn build_pq(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        ann_params: DiskAnnParams,
        pq_config: PQConfig,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let base = DiskANN::build_index_with_params(vectors, dist, file_path, ann_params)?;
        let pq = ProductQuantizer::train(vectors, pq_config)?;
        Ok(Self::from_pq(base, pq, config))
    }

    /// Build an F16-quantized index from scratch.
    pub fn build_f16(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        ann_params: DiskAnnParams,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let base = DiskANN::build_index_with_params(vectors, dist, file_path, ann_params)?;
        Ok(Self::from_f16(base, config))
    }

    /// Build a RaBitQ-quantized (1 bit/dim) index from scratch.
    pub fn build_rabitq(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        ann_params: DiskAnnParams,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let base = DiskANN::build_index_with_params(vectors, dist, file_path, ann_params)?;
        let rq = RaBitQ::train(vectors)?;
        Ok(Self::from_rabitq(base, rq, config))
    }

    /// Build an Int8-quantized index from scratch.
    pub fn build_int8(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        ann_params: DiskAnnParams,
        config: QuantizedConfig,
    ) -> Result<Self, DiskAnnError> {
        let base = DiskANN::build_index_with_params(vectors, dist, file_path, ann_params)?;
        let int8q = Int8Quantizer::train(vectors)?;
        Ok(Self::from_int8(base, int8q, config))
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers — parallel encoding of all vectors
// ---------------------------------------------------------------------------

fn encode_all_pq<D>(base: &DiskANN<D>, pq: &ProductQuantizer, n: usize) -> Vec<u8>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    let code_size = pq.stats().code_size_bytes;
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| base.get_vector(i)).collect();
    let encoded: Vec<Vec<u8>> = vectors.par_iter().map(|v| pq.encode(v)).collect();
    let mut flat = Vec::with_capacity(n * code_size);
    for code in &encoded {
        flat.extend_from_slice(code);
    }
    flat
}

fn encode_all_generic<D, Q>(base: &DiskANN<D>, quantizer: &Q, n: usize, code_size: usize) -> Vec<u8>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
    Q: VectorQuantizer,
{
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| base.get_vector(i)).collect();
    let encoded: Vec<Vec<u8>> = vectors.par_iter().map(|v| quantizer.encode(v)).collect();
    let mut flat = Vec::with_capacity(n * code_size);
    for code in &encoded {
        flat.extend_from_slice(code);
    }
    flat
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistL2;
    use rand::prelude::*;
    use rand::SeedableRng;
    use std::collections::HashSet;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
            .collect()
    }

    fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
        let mut dists: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (i as u32, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.iter().take(k).map(|(i, _)| *i).collect()
    }

    fn recall_at_k(retrieved: &[u32], ground_truth: &[u32]) -> f32 {
        let gt_set: HashSet<u32> = ground_truth.iter().copied().collect();
        let hits = retrieved.iter().filter(|id| gt_set.contains(id)).count();
        hits as f32 / ground_truth.len() as f32
    }

    // Test 1: PQ basic search
    #[test]
    fn test_quantized_pq_basic() {
        let path = "test_quantized_pq_basic.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(200, dim, 42);
        let pq_config = PQConfig {
            num_subspaces: 4,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };
        let config = QuantizedConfig { rerank_size: 0 };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        };

        let index = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            pq_config,
            config,
        )
        .unwrap();

        let query = &vectors[0];
        let results = index.search(query, 10, 64);
        assert_eq!(results.len(), 10);

        let gt = brute_force_knn(&vectors, query, 10);
        let recall = recall_at_k(&results, &gt);
        assert!(
            recall >= 0.3,
            "PQ recall@10 too low: {recall} (expected >= 0.3)"
        );

        let _ = std::fs::remove_file(path);
    }

    // Test 2: F16 basic search
    #[test]
    fn test_quantized_f16_basic() {
        let path = "test_quantized_f16_basic.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(200, dim, 43);
        let config = QuantizedConfig { rerank_size: 0 };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        };

        let index =
            QuantizedDiskANN::<DistL2>::build_f16(&vectors, DistL2 {}, path, ann_params, config)
                .unwrap();

        let query = &vectors[0];
        let results = index.search(query, 10, 64);
        assert_eq!(results.len(), 10);

        let gt = brute_force_knn(&vectors, query, 10);
        let recall = recall_at_k(&results, &gt);
        assert!(
            recall >= 0.7,
            "F16 recall@10 too low: {recall} (expected >= 0.7)"
        );

        let _ = std::fs::remove_file(path);
    }

    // Test 3: Int8 basic search
    #[test]
    fn test_quantized_int8_basic() {
        let path = "test_quantized_int8_basic.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(200, dim, 44);
        let config = QuantizedConfig { rerank_size: 0 };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        };

        let index =
            QuantizedDiskANN::<DistL2>::build_int8(&vectors, DistL2 {}, path, ann_params, config)
                .unwrap();

        let query = &vectors[0];
        let results = index.search(query, 10, 64);
        assert_eq!(results.len(), 10);

        let gt = brute_force_knn(&vectors, query, 10);
        let recall = recall_at_k(&results, &gt);
        assert!(
            recall >= 0.7,
            "Int8 recall@10 too low: {recall} (expected >= 0.7)"
        );

        let _ = std::fs::remove_file(path);
    }

    // Test 4: Re-ranking improves recall
    #[test]
    fn test_reranking_improves_recall() {
        let path_no_rr = "test_quantized_no_rerank.db";
        let path_rr = "test_quantized_rerank.db";
        let _ = std::fs::remove_file(path_no_rr);
        let _ = std::fs::remove_file(path_rr);

        let dim = 32;
        let vectors = random_vectors(200, dim, 45);
        let pq_config = PQConfig {
            num_subspaces: 4,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        };

        let index_no_rr = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path_no_rr,
            ann_params,
            pq_config,
            QuantizedConfig { rerank_size: 0 },
        )
        .unwrap();

        let index_rr = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path_rr,
            ann_params,
            pq_config,
            QuantizedConfig { rerank_size: 50 },
        )
        .unwrap();

        // Average recall over several queries
        let num_queries = 20;
        let mut total_no_rr = 0.0f32;
        let mut total_rr = 0.0f32;

        for i in 0..num_queries {
            let query = &vectors[i];
            let gt = brute_force_knn(&vectors, query, 10);
            let res_no_rr = index_no_rr.search(query, 10, 64);
            let res_rr = index_rr.search(query, 10, 64);
            total_no_rr += recall_at_k(&res_no_rr, &gt);
            total_rr += recall_at_k(&res_rr, &gt);
        }

        let avg_no_rr = total_no_rr / num_queries as f32;
        let avg_rr = total_rr / num_queries as f32;

        // Re-ranking should generally improve or at least match recall
        assert!(
            avg_rr >= avg_no_rr - 0.05,
            "Re-ranking should not significantly degrade recall: no_rr={avg_no_rr}, rr={avg_rr}"
        );

        let _ = std::fs::remove_file(path_no_rr);
        let _ = std::fs::remove_file(path_rr);
    }

    // Test 5: Save/load sidecar round-trip
    #[test]
    fn test_save_load_roundtrip() {
        let base_path = "test_quantized_save_base.db";
        let sidecar_path = "test_quantized_save_sidecar.qann";
        let _ = std::fs::remove_file(base_path);
        let _ = std::fs::remove_file(sidecar_path);

        let dim = 32;
        let vectors = random_vectors(100, dim, 46);
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 64,
            alpha: 1.2,
        };
        let config = QuantizedConfig { rerank_size: 10 };

        let index = QuantizedDiskANN::<DistL2>::build_f16(
            &vectors,
            DistL2 {},
            base_path,
            ann_params,
            config,
        )
        .unwrap();

        // Search before save
        let query = &vectors[0];
        let res_before = index.search(query, 5, 32);

        // Save sidecar
        index.save_quantized(sidecar_path).unwrap();

        // Reload
        let loaded =
            QuantizedDiskANN::<DistL2>::open(base_path, sidecar_path, DistL2 {}, config).unwrap();

        assert_eq!(loaded.num_vectors(), index.num_vectors());
        assert_eq!(loaded.dim(), index.dim());

        let res_after = loaded.search(query, 5, 32);
        assert_eq!(res_before, res_after);

        let _ = std::fs::remove_file(base_path);
        let _ = std::fs::remove_file(sidecar_path);
    }

    // Test 6: to_bytes / from_bytes round-trip
    #[test]
    fn test_to_bytes_from_bytes() {
        let path = "test_quantized_bytes_rt.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(100, dim, 47);
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 64,
            alpha: 1.2,
        };
        let config = QuantizedConfig { rerank_size: 0 };

        let index =
            QuantizedDiskANN::<DistL2>::build_int8(&vectors, DistL2 {}, path, ann_params, config)
                .unwrap();

        let query = &vectors[0];
        let res_before = index.search(query, 5, 32);

        let bytes = index.to_bytes();
        let loaded = QuantizedDiskANN::<DistL2>::from_bytes(&bytes, DistL2 {}, config).unwrap();

        assert_eq!(loaded.num_vectors(), index.num_vectors());
        let res_after = loaded.search(query, 5, 32);
        assert_eq!(res_before, res_after);

        let _ = std::fs::remove_file(path);
    }

    // Test 7: PQ distance table produces valid results
    #[test]
    fn test_pq_distance_table_used() {
        let path = "test_quantized_pq_table.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(100, dim, 48);
        let pq_config = PQConfig {
            num_subspaces: 4,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };
        let config = QuantizedConfig { rerank_size: 0 };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 64,
            alpha: 1.2,
        };

        let index = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            pq_config,
            config,
        )
        .unwrap();

        let query = &vectors[0];
        let results = index.search_with_dists(query, 10, 32);

        // All returned distances should be non-negative
        for (id, dist) in &results {
            assert!(*dist >= 0.0, "Negative distance for id {id}: {dist}");
            assert!(*dist < f32::MAX, "MAX distance for id {id}");
        }

        // Distances should be non-decreasing
        for pair in results.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1 + 1e-6,
                "Distances not sorted: {} > {}",
                pair[0].1,
                pair[1].1
            );
        }

        let _ = std::fs::remove_file(path);
    }

    // Test 8: search_with_dists returns correct distances
    #[test]
    fn test_quantized_search_with_dists() {
        let path = "test_quantized_with_dists.db";
        let _ = std::fs::remove_file(path);

        let dim = 32;
        let vectors = random_vectors(100, dim, 49);
        let config = QuantizedConfig { rerank_size: 20 };
        let ann_params = DiskAnnParams {
            max_degree: 32,
            build_beam_width: 64,
            alpha: 1.2,
        };

        let index =
            QuantizedDiskANN::<DistL2>::build_f16(&vectors, DistL2 {}, path, ann_params, config)
                .unwrap();

        let query = &vectors[0];
        let results = index.search_with_dists(query, 5, 32);
        assert_eq!(results.len(), 5);

        // With reranking, distances should be exact L2 (DistL2 returns sqrt of sum-of-squares)
        for (id, dist) in &results {
            let v = index.get_vector(*id as usize);
            let exact: f32 = query
                .iter()
                .zip(&v)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            assert!(
                (dist - exact).abs() < 1e-4,
                "Distance mismatch for id {id}: returned {dist}, exact {exact}"
            );
        }

        // Distances should be non-decreasing
        for pair in results.windows(2) {
            assert!(pair[0].1 <= pair[1].1 + 1e-6);
        }

        let _ = std::fs::remove_file(path);
    }
}
