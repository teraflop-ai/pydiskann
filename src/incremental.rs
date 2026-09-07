use crate::filtered::Filter;
use crate::pq::{PQConfig, ProductQuantizer};
use crate::quantized::{quantized_distance_from_codes, QuantizerState};
use crate::rabitq::RaBitQ;
use crate::sq::{F16Quantizer, Int8Quantizer, VectorQuantizer};
use crate::Distance;
use crate::{
    beam_search, BeamSearchConfig, DiskANN, DiskAnnError, DiskAnnParams, GraphIndex, PAD_U32,
};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::sync::RwLock;

/// Magic number for incremental index format: "INCR"
const INCR_MAGIC: u32 = 0x494E4352;
/// Current incremental format version
const INCR_FORMAT_VERSION: u32 = 2;

/// Configuration for the incremental index behavior
#[derive(Clone, Copy, Debug)]
pub struct IncrementalConfig {
    /// Maximum vectors in delta before suggesting compaction
    pub delta_threshold: usize,
    /// Maximum tombstone ratio before suggesting compaction (0.0-1.0)
    pub tombstone_ratio_threshold: f32,
    /// Parameters for delta graph construction
    pub delta_params: DiskAnnParams,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 10_000,
            tombstone_ratio_threshold: 0.1,
            delta_params: DiskAnnParams {
                max_degree: 32, // Smaller for delta
                build_beam_width: 64,
                alpha: 1.2,
            },
        }
    }
}

/// Configuration for quantized incremental index construction.
#[derive(Clone, Copy, Debug)]
pub struct IncrementalQuantizedConfig {
    /// Number of candidates to re-rank with exact vectors after quantized search.
    pub rerank_size: usize,
}

impl Default for IncrementalQuantizedConfig {
    fn default() -> Self {
        Self { rerank_size: 0 }
    }
}

/// Internal candidate for search merging
#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u64, // Global ID (base or delta)
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Delta layer: in-memory vectors with a small navigation graph
pub(crate) struct DeltaLayer {
    /// Vectors added after base index was built
    pub(crate) vectors: Vec<Vec<f32>>,
    /// Small Vamana-style adjacency graph for delta vectors
    /// graph[i] contains neighbor indices (local to delta)
    pub(crate) graph: Vec<Vec<u32>>,
    /// Entry point for delta searches (local index)
    pub(crate) entry_point: Option<u32>,
    /// Max degree for delta graph
    pub(crate) max_degree: usize,
}

#[allow(dead_code)]
impl DeltaLayer {
    fn new(max_degree: usize) -> Self {
        Self {
            vectors: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            max_degree,
        }
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Add vectors to the delta layer and update the graph
    fn add_vectors<D: Distance<f32> + Copy + Sync>(
        &mut self,
        vectors: &[Vec<f32>],
        dist: D,
    ) -> Vec<u64> {
        let start_idx = self.vectors.len();
        let mut new_ids = Vec::with_capacity(vectors.len());

        for (i, v) in vectors.iter().enumerate() {
            let local_idx = start_idx + i;
            new_ids.push(DELTA_ID_OFFSET + local_idx as u64);

            self.vectors.push(v.clone());
            self.graph.push(Vec::new());

            if local_idx > 0 {
                let neighbors = self.find_and_prune_neighbors(local_idx, dist);
                self.graph[local_idx] = neighbors.clone();

                for &nb in &neighbors {
                    let nb_idx = nb as usize;
                    if self.graph[nb_idx].contains(&(local_idx as u32)) {
                        continue;
                    }
                    if self.graph[nb_idx].len() < self.max_degree {
                        self.graph[nb_idx].push(local_idx as u32);
                    } else {
                        let pool: Vec<(u32, f32)> = self.graph[nb_idx]
                            .iter()
                            .copied()
                            .chain(std::iter::once(local_idx as u32))
                            .map(|c| {
                                (
                                    c,
                                    dist.eval(&self.vectors[nb_idx], &self.vectors[c as usize]),
                                )
                            })
                            .collect();
                        self.graph[nb_idx] = self.prune_neighbors(nb_idx, &pool, dist);
                    }
                }
            }

            if self.entry_point.is_none() {
                self.entry_point = Some(0);
            }
        }

        if self.vectors.len() > 1 {
            self.entry_point = Some(self.compute_medoid(dist));
        }

        new_ids
    }

    fn compute_medoid<D: Distance<f32> + Copy + Sync>(&self, dist: D) -> u32 {
        if self.vectors.is_empty() {
            return 0;
        }

        // Compute centroid
        let dim = self.vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for v in &self.vectors {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in &mut centroid {
            *val /= self.vectors.len() as f32;
        }

        // Find closest to centroid
        let (best_idx, _) = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, dist.eval(&centroid, v)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, f32::MAX));

        best_idx as u32
    }

    fn find_and_prune_neighbors<D: Distance<f32> + Copy>(
        &self,
        node_idx: usize,
        dist: D,
    ) -> Vec<u32> {
        let query = &self.vectors[node_idx];
        let beam_width = (self.max_degree * 2).max(16);

        // Greedy search from entry point
        let candidates = if let Some(entry) = self.entry_point {
            self.greedy_search_internal(query, entry as usize, beam_width, dist)
        } else {
            // No entry point yet, just compute distances to all
            self.vectors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != node_idx)
                .map(|(i, v)| (i as u32, dist.eval(query, v)))
                .collect()
        };

        // Alpha-prune
        self.prune_neighbors(node_idx, &candidates, dist)
    }

    fn greedy_search_internal<D: Distance<f32> + Copy>(
        &self,
        query: &[f32],
        start: usize,
        beam_width: usize,
        dist: D,
    ) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() || start >= self.vectors.len() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        let start_dist = dist.eval(query, &self.vectors[start]);
        let start_cand = Candidate {
            dist: start_dist,
            id: start as u64,
        };
        frontier.push(Reverse(start_cand));
        results.push(start_cand);
        visited.insert(start);

        while let Some(Reverse(best)) = frontier.peek().copied() {
            if results.len() >= beam_width {
                if let Some(worst) = results.peek() {
                    if best.dist >= worst.dist {
                        break;
                    }
                }
            }
            let Reverse(current) = frontier.pop().unwrap();
            let cur_idx = current.id as usize;

            if cur_idx < self.graph.len() {
                for &nb in &self.graph[cur_idx] {
                    let nb_idx = nb as usize;
                    if !visited.insert(nb_idx) {
                        continue;
                    }
                    if nb_idx >= self.vectors.len() {
                        continue;
                    }

                    let d = dist.eval(query, &self.vectors[nb_idx]);
                    let cand = Candidate {
                        dist: d,
                        id: nb as u64,
                    };

                    if results.len() < beam_width {
                        results.push(cand);
                        frontier.push(Reverse(cand));
                    } else if d < results.peek().unwrap().dist {
                        results.pop();
                        results.push(cand);
                        frontier.push(Reverse(cand));
                    }
                }
            }
        }

        results
            .into_vec()
            .into_iter()
            .map(|c| (c.id as u32, c.dist))
            .collect()
    }

    fn prune_neighbors<D: Distance<f32> + Copy>(
        &self,
        node_idx: usize,
        candidates: &[(u32, f32)],
        dist: D,
    ) -> Vec<u32> {
        crate::robust_prune(node_idx as u32, candidates, self.max_degree, 1.2, |a, b| {
            dist.eval(&self.vectors[a as usize], &self.vectors[b as usize])
        })
    }

    fn search<D: Distance<f32> + Copy>(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        dist: D,
    ) -> Vec<(u64, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap_or(0) as usize;
        let mut results = self.greedy_search_internal(query, entry, beam_width, dist);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        // Convert local IDs to global delta IDs
        results
            .into_iter()
            .map(|(local_id, d)| (DELTA_ID_OFFSET + local_id as u64, d))
            .collect()
    }

    fn get_vector(&self, local_idx: usize) -> Option<&Vec<f32>> {
        self.vectors.get(local_idx)
    }
}

/// Offset for delta vector IDs to distinguish from base IDs
const DELTA_ID_OFFSET: u64 = 1u64 << 48;

/// Check if an ID is from the delta layer
#[inline]
pub fn is_delta_id(id: u64) -> bool {
    id >= DELTA_ID_OFFSET
}

/// Convert a delta global ID to local delta index
#[inline]
pub fn delta_local_idx(id: u64) -> usize {
    (id - DELTA_ID_OFFSET) as usize
}

// =========================================================================
// UnifiedView — adapts base + delta into a single GraphIndex
// =========================================================================

/// Adapter that presents base + delta as a single `GraphIndex`.
///
/// ID mapping:
/// - `0..base_count` → base vector IDs (unchanged)
/// - `base_count..base_count+delta_count` → delta local IDs (offset by base_count)
pub(crate) struct UnifiedView<'a, D: Distance<f32> + Copy + Send + Sync + 'static> {
    base: Option<&'a DiskANN<D>>,
    delta: &'a DeltaLayer,
    tombstones: &'a HashSet<u64>,
    dist: D,
    base_count: usize,
}

impl<'a, D: Distance<f32> + Copy + Send + Sync + 'static> UnifiedView<'a, D> {
    fn new(
        base: Option<&'a DiskANN<D>>,
        delta: &'a DeltaLayer,
        tombstones: &'a HashSet<u64>,
        dist: D,
    ) -> Self {
        let base_count = base.map(|b| b.num_vectors).unwrap_or(0);
        Self {
            base,
            delta,
            tombstones,
            dist,
            base_count,
        }
    }

    /// Return entry points for multi-seed search: one from base (if any) and one from delta (if any).
    fn entry_points(&self) -> Vec<u32> {
        let mut seeds = Vec::with_capacity(2);
        if let Some(base) = self.base {
            seeds.push(base.medoid_id);
        }
        if let Some(ep) = self.delta.entry_point {
            seeds.push(self.base_count as u32 + ep);
        }
        seeds
    }

    /// Convert internal u32 ID to the external u64 ID space used by IncrementalDiskANN.
    fn to_global_u64(&self, id: u32) -> u64 {
        let id_usize = id as usize;
        if id_usize < self.base_count {
            id_usize as u64
        } else {
            DELTA_ID_OFFSET + (id_usize - self.base_count) as u64
        }
    }
}

impl<'a, D: Distance<f32> + Copy + Send + Sync + 'static> GraphIndex for UnifiedView<'a, D> {
    fn num_vectors(&self) -> usize {
        self.base_count + self.delta.len()
    }

    fn dim(&self) -> usize {
        if let Some(base) = self.base {
            base.dim
        } else if !self.delta.vectors.is_empty() {
            self.delta.vectors[0].len()
        } else {
            0
        }
    }

    fn entry_point(&self) -> u32 {
        // Prefer base medoid, fall back to delta entry
        if let Some(base) = self.base {
            base.medoid_id
        } else if let Some(ep) = self.delta.entry_point {
            self.base_count as u32 + ep
        } else {
            0
        }
    }

    fn distance_to(&self, query: &[f32], id: u32) -> f32 {
        let id_usize = id as usize;
        if id_usize < self.base_count {
            self.base.unwrap().distance_to(query, id_usize)
        } else {
            let delta_idx = id_usize - self.base_count;
            self.dist.eval(query, &self.delta.vectors[delta_idx])
        }
    }

    fn get_neighbors(&self, id: u32) -> Vec<u32> {
        let id_usize = id as usize;
        if id_usize < self.base_count {
            self.base
                .unwrap()
                .get_neighbors(id)
                .iter()
                .copied()
                .filter(|&nb| nb != PAD_U32)
                .collect()
        } else {
            let delta_idx = id_usize - self.base_count;
            if delta_idx < self.delta.graph.len() {
                // Remap delta-local IDs to unified space
                self.delta.graph[delta_idx]
                    .iter()
                    .map(|&nb| nb + self.base_count as u32)
                    .collect()
            } else {
                Vec::new()
            }
        }
    }

    fn get_vector(&self, id: u32) -> Vec<f32> {
        let id_usize = id as usize;
        if id_usize < self.base_count {
            self.base.unwrap().get_vector(id_usize)
        } else {
            let delta_idx = id_usize - self.base_count;
            self.delta.vectors[delta_idx].clone()
        }
    }

    fn is_live(&self, id: u32) -> bool {
        let global = if (id as usize) < self.base_count {
            id as u64
        } else {
            DELTA_ID_OFFSET + (id as usize - self.base_count) as u64
        };
        !self.tombstones.contains(&global)
    }
}

// =========================================================================
// Quantizer kind enum for builder API
// =========================================================================

fn tombstone_config(beam_width: usize, k: usize, tombstones: usize) -> BeamSearchConfig {
    if tombstones == 0 {
        return BeamSearchConfig::default();
    }
    let expanded = (beam_width * 2).max(k + tombstones.min(beam_width));
    BeamSearchConfig {
        expanded_beam: Some(expanded),
        max_iterations: Some(expanded * 2),
        early_term_factor: Some(1.5),
    }
}

/// Specifies which quantizer to use for incremental quantized builds.
pub enum QuantizerKind {
    F16,
    Int8,
    PQ(PQConfig),
    RaBitQ,
}

// =========================================================================
// IncrementalDiskANN
// =========================================================================

/// An incremental DiskANN index supporting add/delete without full rebuild
pub struct IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// The base immutable index (memory-mapped)
    base: Option<DiskANN<D>>,
    /// Delta layer for newly added vectors
    delta: RwLock<DeltaLayer>,
    /// Set of deleted vector IDs (tombstones)
    tombstones: RwLock<HashSet<u64>>,
    /// Distance metric
    dist: D,
    /// Configuration
    config: IncrementalConfig,
    /// Path to the base index file
    base_path: Option<String>,
    /// Dimensionality
    dim: usize,
    // --- Labels (optional, for filtered search) ---
    base_labels: Option<Vec<Vec<u64>>>,
    delta_labels: RwLock<Vec<Vec<u64>>>,
    num_label_fields: usize,
    // --- Quantizer (optional, for quantized search) ---
    quantizer: Option<QuantizerState>,
    base_codes: Option<Vec<u8>>,
    code_size: usize,
    rerank_size: usize,
}

impl<D> IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    /// Build a new incremental index with default parameters
    pub fn build_default(vectors: &[Vec<f32>], file_path: &str) -> Result<Self, DiskAnnError> {
        Self::build_with_config(vectors, file_path, IncrementalConfig::default())
    }

    /// Open an existing index for incremental updates
    pub fn open(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_with_config(path, IncrementalConfig::default())
    }
}

impl<D> IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build a new incremental index with custom configuration
    pub fn build_with_config(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let dist = D::default();
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);

        let base = DiskANN::<D>::build_index_default(vectors, dist, file_path)?;

        Ok(Self {
            base: Some(base),
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: Some(file_path.to_string()),
            dim,
            base_labels: None,
            delta_labels: RwLock::new(Vec::new()),
            num_label_fields: 0,
            quantizer: None,
            base_codes: None,
            code_size: 0,
            rerank_size: 0,
        })
    }

    /// Open an existing index with custom configuration
    pub fn open_with_config(path: &str, config: IncrementalConfig) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let dist = D::default();
        let base = DiskANN::<D>::open_index_default_metric(path)?;
        let dim = base.dim;

        Ok(Self {
            base: Some(base),
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: Some(path.to_string()),
            dim,
            base_labels: None,
            delta_labels: RwLock::new(Vec::new()),
            num_label_fields: 0,
            quantizer: None,
            base_codes: None,
            code_size: 0,
            rerank_size: 0,
        })
    }

    /// Create an empty incremental index (no base, delta-only)
    pub fn new_empty(dim: usize, dist: D, config: IncrementalConfig) -> Self {
        Self {
            base: None,
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: None,
            dim,
            base_labels: None,
            delta_labels: RwLock::new(Vec::new()),
            num_label_fields: 0,
            quantizer: None,
            base_codes: None,
            code_size: 0,
            rerank_size: 0,
        }
    }

    // =====================================================================
    // Labeled constructors (incremental + filtered)
    // =====================================================================

    /// Build an incremental index with per-vector labels for filtered search.
    pub fn build_with_labels(
        vectors: &[Vec<f32>],
        labels: &[Vec<u64>],
        file_path: &str,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        if vectors.len() != labels.len() {
            return Err(DiskAnnError::IndexError(format!(
                "vectors.len() ({}) != labels.len() ({})",
                vectors.len(),
                labels.len()
            )));
        }
        let num_fields = labels.first().map(|l| l.len()).unwrap_or(0);
        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        idx.base_labels = Some(labels.to_vec());
        idx.num_label_fields = num_fields;
        Ok(idx)
    }

    // =====================================================================
    // Quantized constructors (incremental + quantized)
    // =====================================================================

    /// Build an incremental index with F16 quantization.
    pub fn build_quantized_f16(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
        quant_config: IncrementalQuantizedConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        let f16q = F16Quantizer::new(dim);
        let code_size = dim * 2;
        let codes = encode_all_vecs(vectors, &f16q, code_size);
        idx.quantizer = Some(QuantizerState::F16(f16q));
        idx.base_codes = Some(codes);
        idx.code_size = code_size;
        idx.rerank_size = quant_config.rerank_size;
        Ok(idx)
    }

    /// Build an incremental index with RaBitQ (1 bit/dim) quantization.
    pub fn build_quantized_rabitq(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
        quant_config: IncrementalQuantizedConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        let rq = RaBitQ::train(vectors)?;
        let code_size = rq.code_size();
        let codes = encode_all_vecs(vectors, &rq, code_size);
        idx.quantizer = Some(QuantizerState::RaBitQ(rq));
        idx.base_codes = Some(codes);
        idx.code_size = code_size;
        idx.rerank_size = quant_config.rerank_size;
        Ok(idx)
    }

    /// Build an incremental index with Int8 quantization.
    pub fn build_quantized_int8(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
        quant_config: IncrementalQuantizedConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        let int8q = Int8Quantizer::train(vectors)?;
        let code_size = int8q.dim();
        let codes = encode_all_vecs(vectors, &int8q, code_size);
        idx.quantizer = Some(QuantizerState::Int8(int8q));
        idx.base_codes = Some(codes);
        idx.code_size = code_size;
        idx.rerank_size = quant_config.rerank_size;
        Ok(idx)
    }

    /// Build an incremental index with PQ quantization.
    pub fn build_quantized_pq(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
        pq_config: PQConfig,
        quant_config: IncrementalQuantizedConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        let pq = ProductQuantizer::train(vectors, pq_config)?;
        let code_size = pq.stats().code_size_bytes;
        let codes = encode_all_pq_vecs(vectors, &pq, code_size);
        idx.quantizer = Some(QuantizerState::PQ(pq));
        idx.base_codes = Some(codes);
        idx.code_size = code_size;
        idx.rerank_size = quant_config.rerank_size;
        Ok(idx)
    }

    /// Build an incremental index with labels AND quantization.
    pub fn build_full(
        vectors: &[Vec<f32>],
        labels: &[Vec<u64>],
        file_path: &str,
        config: IncrementalConfig,
        quantizer_kind: QuantizerKind,
        quant_config: IncrementalQuantizedConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        if vectors.len() != labels.len() {
            return Err(DiskAnnError::IndexError(format!(
                "vectors.len() ({}) != labels.len() ({})",
                vectors.len(),
                labels.len()
            )));
        }
        let num_fields = labels.first().map(|l| l.len()).unwrap_or(0);
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);

        let mut idx = Self::build_with_config(vectors, file_path, config)?;
        idx.base_labels = Some(labels.to_vec());
        idx.num_label_fields = num_fields;
        idx.rerank_size = quant_config.rerank_size;

        match quantizer_kind {
            QuantizerKind::F16 => {
                let f16q = F16Quantizer::new(dim);
                let code_size = dim * 2;
                let codes = encode_all_vecs(vectors, &f16q, code_size);
                idx.quantizer = Some(QuantizerState::F16(f16q));
                idx.base_codes = Some(codes);
                idx.code_size = code_size;
            }
            QuantizerKind::Int8 => {
                let int8q = Int8Quantizer::train(vectors)?;
                let code_size = int8q.dim();
                let codes = encode_all_vecs(vectors, &int8q, code_size);
                idx.quantizer = Some(QuantizerState::Int8(int8q));
                idx.base_codes = Some(codes);
                idx.code_size = code_size;
            }
            QuantizerKind::PQ(pq_config) => {
                let pq = ProductQuantizer::train(vectors, pq_config)?;
                let code_size = pq.stats().code_size_bytes;
                let codes = encode_all_pq_vecs(vectors, &pq, code_size);
                idx.quantizer = Some(QuantizerState::PQ(pq));
                idx.base_codes = Some(codes);
                idx.code_size = code_size;
            }
            QuantizerKind::RaBitQ => {
                let rq = RaBitQ::train(vectors)?;
                let code_size = rq.code_size();
                let codes = encode_all_vecs(vectors, &rq, code_size);
                idx.quantizer = Some(QuantizerState::RaBitQ(rq));
                idx.base_codes = Some(codes);
                idx.code_size = code_size;
            }
        }

        Ok(idx)
    }

    // =====================================================================
    // Mutation
    // =====================================================================

    /// Add new vectors to the index. Returns their assigned IDs.
    pub fn add_vectors(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>, DiskAnnError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but index expects {}",
                    i,
                    v.len(),
                    self.dim
                )));
            }
        }

        // Hold both locks simultaneously to keep labels and vectors aligned
        let mut delta = self.delta.write().unwrap();
        if self.num_label_fields > 0 {
            let mut delta_labels = self.delta_labels.write().unwrap();
            for _ in 0..vectors.len() {
                delta_labels.push(vec![0u64; self.num_label_fields]);
            }
        }
        let ids = delta.add_vectors(vectors, self.dist);
        Ok(ids)
    }

    /// Add new vectors with labels to the index. Returns their assigned IDs.
    pub fn add_vectors_with_labels(
        &self,
        vectors: &[Vec<f32>],
        labels: &[Vec<u64>],
    ) -> Result<Vec<u64>, DiskAnnError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        if vectors.len() != labels.len() {
            return Err(DiskAnnError::IndexError(format!(
                "vectors.len() ({}) != labels.len() ({})",
                vectors.len(),
                labels.len()
            )));
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but index expects {}",
                    i,
                    v.len(),
                    self.dim
                )));
            }
        }

        // Validate label fields
        for (i, l) in labels.iter().enumerate() {
            if self.num_label_fields > 0 && l.len() != self.num_label_fields {
                return Err(DiskAnnError::IndexError(format!(
                    "Label {} has {} fields, expected {}",
                    i,
                    l.len(),
                    self.num_label_fields
                )));
            }
        }

        // Hold both locks simultaneously to keep labels and vectors aligned
        let mut delta = self.delta.write().unwrap();
        let mut delta_labels = self.delta_labels.write().unwrap();
        delta_labels.extend_from_slice(labels);
        let ids = delta.add_vectors(vectors, self.dist);
        Ok(ids)
    }

    /// Delete vectors by their IDs (lazy deletion via tombstones)
    pub fn delete_vectors(&self, ids: &[u64]) -> Result<(), DiskAnnError> {
        let mut tombstones = self.tombstones.write().unwrap();
        for &id in ids {
            tombstones.insert(id);
        }
        Ok(())
    }

    /// Check if a vector ID has been deleted
    pub fn is_deleted(&self, id: u64) -> bool {
        self.tombstones.read().unwrap().contains(&id)
    }

    // =====================================================================
    // Search
    // =====================================================================

    /// Search the index, merging results from base and delta, excluding tombstones
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u64> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Search returning (id, distance) pairs
    pub fn search_with_dists(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<(u64, f32)> {
        let tombstones = self.tombstones.read().unwrap();
        let delta = self.delta.read().unwrap();
        let view = UnifiedView::new(self.base.as_ref(), &delta, &tombstones, self.dist);
        let start_ids = view.entry_points();

        if start_ids.is_empty() {
            return Vec::new();
        }

        if let (Some(ref quantizer), Some(ref base_codes)) = (&self.quantizer, &self.base_codes) {
            let base_count = view.base_count;
            let code_size = self.code_size;
            let rerank_size = self.rerank_size;
            let prep = quantizer.prepare(query);
            let search_k = if rerank_size > 0 {
                rerank_size.max(k)
            } else {
                k
            };

            let mut results = beam_search(
                &start_ids,
                beam_width,
                search_k,
                |id| {
                    let id_usize = id as usize;
                    if id_usize < base_count {
                        quantized_distance_from_codes(
                            &self.dist, query, id_usize, base_codes, code_size, quantizer, &prep,
                        )
                    } else {
                        view.distance_to(query, id)
                    }
                },
                |id| view.get_neighbors(id),
                |id| view.is_live(id),
                tombstone_config(beam_width, search_k, tombstones.len()),
            );

            if rerank_size > 0 {
                results = results
                    .iter()
                    .map(|&(id, _)| (id, view.distance_to(query, id)))
                    .collect();
                results.sort_by(|a, b| a.1.total_cmp(&b.1));
                results.truncate(k);
            }

            return results
                .into_iter()
                .map(|(id, d)| (view.to_global_u64(id), d))
                .collect();
        }

        let results = beam_search(
            &start_ids,
            beam_width,
            k,
            |id| view.distance_to(query, id),
            |id| view.get_neighbors(id),
            |id| view.is_live(id),
            tombstone_config(beam_width, k, tombstones.len()),
        );

        results
            .into_iter()
            .map(|(id, d)| (view.to_global_u64(id), d))
            .collect()
    }

    /// Search with a filter predicate (requires labels).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        filter: &Filter,
    ) -> Vec<u64> {
        self.search_filtered_with_dists(query, k, beam_width, filter)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Search with filter, returning (id, distance) pairs.
    pub fn search_filtered_with_dists(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        filter: &Filter,
    ) -> Vec<(u64, f32)> {
        if matches!(filter, Filter::None) || self.base_labels.is_none() {
            return self.search_with_dists(query, k, beam_width);
        }

        let tombstones = self.tombstones.read().unwrap();
        let delta = self.delta.read().unwrap();
        let delta_labels = self.delta_labels.read().unwrap();
        let view = UnifiedView::new(self.base.as_ref(), &delta, &tombstones, self.dist);
        let start_ids = view.entry_points();

        if start_ids.is_empty() {
            return Vec::new();
        }

        let base_labels = self.base_labels.as_ref().unwrap();
        let base_count = view.base_count;
        let label_of = |i: usize| -> Option<&[u64]> {
            if i < base_count {
                base_labels.get(i)
            } else {
                delta_labels.get(i - base_count)
            }
            .map(Vec::as_slice)
        };

        let expanded_beam = (beam_width * 4).max(k * 10);

        let results = beam_search(
            &start_ids,
            beam_width,
            k,
            |id| view.distance_to(query, id),
            |id| view.get_neighbors(id),
            |id| view.is_live(id) && label_of(id as usize).map_or(false, |l| filter.matches(l)),
            BeamSearchConfig {
                expanded_beam: Some(expanded_beam),
                max_iterations: Some(expanded_beam * 2),
                early_term_factor: Some(1.5),
            },
        );

        results
            .into_iter()
            .map(|(id, d)| (view.to_global_u64(id), d))
            .collect()
    }

    /// Parallel batch search
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, beam_width: usize) -> Vec<Vec<u64>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k, beam_width))
            .collect()
    }

    /// Get a vector by its ID (works for both base and delta)
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        if is_delta_id(id) {
            let delta = self.delta.read().unwrap();
            delta.get_vector(delta_local_idx(id)).cloned()
        } else if let Some(ref base) = self.base {
            let idx = id as usize;
            if idx < base.num_vectors {
                Some(base.get_vector(idx))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if compaction is recommended
    pub fn should_compact(&self) -> bool {
        let delta = self.delta.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();

        let base_size = self.base.as_ref().map(|b| b.num_vectors).unwrap_or(0);
        let total_size = base_size + delta.len();

        // Check delta threshold
        if delta.len() >= self.config.delta_threshold {
            return true;
        }

        // Check tombstone ratio
        if total_size > 0 {
            let tombstone_ratio = tombstones.len() as f32 / total_size as f32;
            if tombstone_ratio >= self.config.tombstone_ratio_threshold {
                return true;
            }
        }

        false
    }

    /// Compact the index: merge base + delta, remove tombstones, write new file
    pub fn compact(&mut self, new_path: &str) -> Result<(), DiskAnnError>
    where
        D: Default,
    {
        let tombstones = self.tombstones.read().unwrap().clone();
        let delta = self.delta.read().unwrap();
        let delta_labels = self.delta_labels.read().unwrap();

        // Collect all live vectors (and labels if present)
        let mut all_vectors: Vec<Vec<f32>> = Vec::new();
        let mut all_labels: Option<Vec<Vec<u64>>> = if self.base_labels.is_some() {
            Some(Vec::new())
        } else {
            None
        };

        // Add base vectors (excluding tombstones)
        if let Some(ref base) = self.base {
            for i in 0..base.num_vectors {
                if !tombstones.contains(&(i as u64)) {
                    all_vectors.push(base.get_vector(i));
                    if let (Some(ref mut al), Some(ref bl)) = (&mut all_labels, &self.base_labels) {
                        al.push(bl[i].clone());
                    }
                }
            }
        }

        // Add delta vectors (excluding tombstones)
        for (i, v) in delta.vectors.iter().enumerate() {
            let global_id = DELTA_ID_OFFSET + i as u64;
            if !tombstones.contains(&global_id) {
                all_vectors.push(v.clone());
                if let Some(ref mut al) = all_labels {
                    if i < delta_labels.len() {
                        al.push(delta_labels[i].clone());
                    } else {
                        al.push(vec![0u64; self.num_label_fields]);
                    }
                }
            }
        }

        drop(delta);
        drop(delta_labels);
        drop(tombstones);

        if all_vectors.is_empty() {
            return Err(DiskAnnError::IndexError(
                "Cannot compact: no vectors remaining after removing tombstones".to_string(),
            ));
        }

        // Build new index
        let new_base = DiskANN::<D>::build_index_default(&all_vectors, self.dist, new_path)?;

        // Re-encode quantized codes if quantizer is present
        let new_codes = if let Some(ref quantizer) = self.quantizer {
            let codes = match quantizer {
                QuantizerState::PQ(pq) => encode_all_pq_vecs(&all_vectors, pq, self.code_size),
                QuantizerState::F16(f16q) => encode_all_vecs(&all_vectors, f16q, self.code_size),
                QuantizerState::Int8(int8q) => encode_all_vecs(&all_vectors, int8q, self.code_size),
                QuantizerState::RaBitQ(rq) => encode_all_vecs(&all_vectors, rq, self.code_size),
            };
            Some(codes)
        } else {
            None
        };

        // Replace state
        self.base = Some(new_base);
        self.delta = RwLock::new(DeltaLayer::new(self.config.delta_params.max_degree));
        self.tombstones = RwLock::new(HashSet::new());
        self.base_path = Some(new_path.to_string());
        self.base_labels = all_labels;
        self.delta_labels = RwLock::new(Vec::new());
        self.base_codes = new_codes;

        Ok(())
    }

    // =====================================================================
    // Serialization
    // =====================================================================

    /// Serialize the entire incremental index to bytes.
    ///
    /// New format (v1):
    /// ```text
    /// [magic:u32 = 0x494E4352]["INCR"]
    /// [version:u32 = 1]
    /// [has_base:u8][base_len:u64][base_bytes...]
    /// [dim:u64]
    /// [num_delta:u64][delta_vectors flat]
    /// [num_delta_graph:u64][graph data]
    /// [entry_point:i64]
    /// [max_degree:u64]
    /// [num_tombstones:u64][tombstone_ids]
    /// [has_labels:u8]
    ///   if has_labels:
    ///     [num_label_fields:u64]
    ///     [num_base_labels:u64][base_labels flat]
    ///     [num_delta_labels:u64][delta_labels flat]
    /// [has_quantizer:u8]
    ///   if has_quantizer:
    ///     [code_size:u64]
    ///     [rerank_size:u64]
    ///     [quantizer_data_len:u64][quantizer_data]
    ///     [base_codes_len:u64][base_codes]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let delta = self.delta.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();
        let delta_labels = self.delta_labels.read().unwrap();

        let mut out = Vec::new();

        // Magic + version
        out.extend_from_slice(&INCR_MAGIC.to_le_bytes());
        out.extend_from_slice(&INCR_FORMAT_VERSION.to_le_bytes());

        // Base index
        if let Some(ref base) = self.base {
            out.push(1u8);
            let base_bytes = base.to_bytes();
            out.extend_from_slice(&(base_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&base_bytes);
        } else {
            out.push(0u8);
        }

        // Dimension
        out.extend_from_slice(&(self.dim as u64).to_le_bytes());

        // Delta vectors
        out.extend_from_slice(&(delta.vectors.len() as u64).to_le_bytes());
        for v in &delta.vectors {
            let bytes: &[u8] = bytemuck::cast_slice(v);
            out.extend_from_slice(bytes);
        }

        // Delta graph
        out.extend_from_slice(&(delta.graph.len() as u64).to_le_bytes());
        for neighbors in &delta.graph {
            out.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            let bytes: &[u8] = bytemuck::cast_slice(neighbors);
            out.extend_from_slice(bytes);
        }

        // Entry point
        let ep = delta.entry_point.map(|e| e as i64).unwrap_or(-1);
        out.extend_from_slice(&ep.to_le_bytes());

        // Max degree
        out.extend_from_slice(&(delta.max_degree as u64).to_le_bytes());

        // Tombstones
        out.extend_from_slice(&(tombstones.len() as u64).to_le_bytes());
        for &id in tombstones.iter() {
            out.extend_from_slice(&id.to_le_bytes());
        }

        // Labels section
        if let Some(ref base_labels) = self.base_labels {
            out.push(1u8); // has_labels
            out.extend_from_slice(&(self.num_label_fields as u64).to_le_bytes());
            // Base labels
            out.extend_from_slice(&(base_labels.len() as u64).to_le_bytes());
            for lv in base_labels {
                for &val in lv {
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
            // Delta labels
            out.extend_from_slice(&(delta_labels.len() as u64).to_le_bytes());
            for lv in delta_labels.iter() {
                for &val in lv {
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
        } else {
            out.push(0u8); // no labels
        }

        // Quantizer section
        if let Some(ref quantizer) = self.quantizer {
            out.push(1u8); // has_quantizer
            out.extend_from_slice(&(self.code_size as u64).to_le_bytes());
            out.extend_from_slice(&(self.rerank_size as u64).to_le_bytes());
            let qdata = bincode::serialize(quantizer).unwrap();
            out.extend_from_slice(&(qdata.len() as u64).to_le_bytes());
            out.extend_from_slice(&qdata);
            if let Some(ref base_codes) = self.base_codes {
                out.extend_from_slice(&(base_codes.len() as u64).to_le_bytes());
                out.extend_from_slice(base_codes);
            } else {
                out.extend_from_slice(&0u64.to_le_bytes());
            }
        } else {
            out.push(0u8); // no quantizer
        }

        out
    }

    /// Load an incremental index from bytes.
    ///
    /// Supports both old format (has_base byte first) and new format (INCR magic).
    pub fn from_bytes(
        bytes: &[u8],
        dist: D,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError> {
        if bytes.len() < 4 {
            return Err(DiskAnnError::IndexError(
                "Incremental buffer too small".into(),
            ));
        }

        // Detect format: check for INCR magic
        let first_u32 = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if first_u32 == INCR_MAGIC {
            Self::from_bytes_v1(bytes, dist, config)
        } else {
            // Old format: first byte is 0 or 1 (has_base flag)
            Self::from_bytes_legacy(bytes, dist, config)
        }
    }

    /// Parse old format (backward compatible)
    fn from_bytes_legacy(
        bytes: &[u8],
        dist: D,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError> {
        let mut pos = 0;

        macro_rules! read_bytes {
            ($n:expr) => {{
                if pos + $n > bytes.len() {
                    return Err(DiskAnnError::IndexError(
                        "Incremental buffer truncated".into(),
                    ));
                }
                let slice = &bytes[pos..pos + $n];
                pos += $n;
                slice
            }};
        }

        // Base index
        let has_base = read_bytes!(1)[0];
        let base = if has_base == 1 {
            let base_len = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let base_data = read_bytes!(base_len).to_vec();
            Some(DiskANN::<D>::from_bytes(base_data, dist)?)
        } else {
            None
        };

        // Dimension
        let dim = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;

        // Delta vectors
        let num_delta = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut delta_vectors = Vec::with_capacity(num_delta);
        for _ in 0..num_delta {
            let vbytes = read_bytes!(dim * 4);
            let floats: Vec<f32> = vbytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            delta_vectors.push(floats);
        }

        // Delta graph
        let num_graph = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut delta_graph = Vec::with_capacity(num_graph);
        for _ in 0..num_graph {
            let deg = u32::from_le_bytes(read_bytes!(4).try_into().unwrap()) as usize;
            let nbytes = read_bytes!(deg * 4);
            let neighbors: Vec<u32> = nbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            delta_graph.push(neighbors);
        }

        // Entry point
        let ep = i64::from_le_bytes(read_bytes!(8).try_into().unwrap());
        let entry_point = if ep >= 0 { Some(ep as u32) } else { None };

        // Max degree
        let max_degree = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;

        // Tombstones
        let num_tombstones = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut tombstones = HashSet::with_capacity(num_tombstones);
        for _ in 0..num_tombstones {
            let id = u64::from_le_bytes(read_bytes!(8).try_into().unwrap());
            tombstones.insert(id);
        }

        Ok(Self {
            base,
            delta: RwLock::new(DeltaLayer {
                vectors: delta_vectors,
                graph: delta_graph,
                entry_point,
                max_degree,
            }),
            tombstones: RwLock::new(tombstones),
            dist,
            config,
            base_path: None,
            dim,
            base_labels: None,
            delta_labels: RwLock::new(Vec::new()),
            num_label_fields: 0,
            quantizer: None,
            base_codes: None,
            code_size: 0,
            rerank_size: 0,
        })
    }

    /// Parse new format (v1 with magic/version)
    #[allow(unused_assignments)]
    fn from_bytes_v1(
        bytes: &[u8],
        dist: D,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError> {
        let mut pos = 0;

        macro_rules! read_bytes {
            ($n:expr) => {{
                if pos + $n > bytes.len() {
                    return Err(DiskAnnError::IndexError(
                        "Incremental buffer truncated".into(),
                    ));
                }
                let slice = &bytes[pos..pos + $n];
                pos += $n;
                slice
            }};
        }

        // Magic
        let magic = u32::from_le_bytes(read_bytes!(4).try_into().unwrap());
        if magic != INCR_MAGIC {
            return Err(DiskAnnError::IndexError(format!(
                "Invalid incremental magic: 0x{:08X}",
                magic
            )));
        }

        // Version
        let version = u32::from_le_bytes(read_bytes!(4).try_into().unwrap());
        if version != INCR_FORMAT_VERSION {
            return Err(DiskAnnError::IndexError(format!(
                "Unsupported incremental format version: {}",
                version
            )));
        }

        // Base index
        let has_base = read_bytes!(1)[0];
        let base = if has_base == 1 {
            let base_len = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let base_data = read_bytes!(base_len).to_vec();
            Some(DiskANN::<D>::from_bytes(base_data, dist)?)
        } else {
            None
        };

        // Dimension
        let dim = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;

        // Delta vectors
        let num_delta = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut delta_vectors = Vec::with_capacity(num_delta);
        for _ in 0..num_delta {
            let vbytes = read_bytes!(dim * 4);
            let floats: Vec<f32> = vbytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            delta_vectors.push(floats);
        }

        // Delta graph
        let num_graph = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut delta_graph = Vec::with_capacity(num_graph);
        for _ in 0..num_graph {
            let deg = u32::from_le_bytes(read_bytes!(4).try_into().unwrap()) as usize;
            let nbytes = read_bytes!(deg * 4);
            let neighbors: Vec<u32> = nbytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            delta_graph.push(neighbors);
        }

        // Entry point
        let ep = i64::from_le_bytes(read_bytes!(8).try_into().unwrap());
        let entry_point = if ep >= 0 { Some(ep as u32) } else { None };

        // Max degree
        let max_degree = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;

        // Tombstones
        let num_tombstones = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
        let mut tombstones = HashSet::with_capacity(num_tombstones);
        for _ in 0..num_tombstones {
            let id = u64::from_le_bytes(read_bytes!(8).try_into().unwrap());
            tombstones.insert(id);
        }

        // Labels section
        let has_labels = read_bytes!(1)[0];
        let (base_labels, delta_labels_vec, num_label_fields) = if has_labels == 1 {
            let num_fields = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let num_base = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let mut bl = Vec::with_capacity(num_base);
            for _ in 0..num_base {
                let mut lv = Vec::with_capacity(num_fields);
                for _ in 0..num_fields {
                    lv.push(u64::from_le_bytes(read_bytes!(8).try_into().unwrap()));
                }
                bl.push(lv);
            }
            let num_dl = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let mut dl = Vec::with_capacity(num_dl);
            for _ in 0..num_dl {
                let mut lv = Vec::with_capacity(num_fields);
                for _ in 0..num_fields {
                    lv.push(u64::from_le_bytes(read_bytes!(8).try_into().unwrap()));
                }
                dl.push(lv);
            }
            (Some(bl), dl, num_fields)
        } else {
            (None, Vec::new(), 0)
        };

        // Quantizer section
        let has_quantizer = read_bytes!(1)[0];
        let (quantizer, base_codes, code_size, rerank_size) = if has_quantizer == 1 {
            let cs = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let rs = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let qdata_len = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let qdata = read_bytes!(qdata_len);
            let q: QuantizerState = bincode::deserialize(qdata)?;
            let codes_len = u64::from_le_bytes(read_bytes!(8).try_into().unwrap()) as usize;
            let codes = if codes_len > 0 {
                Some(read_bytes!(codes_len).to_vec())
            } else {
                None
            };
            (Some(q), codes, cs, rs)
        } else {
            (None, None, 0, 0)
        };

        Ok(Self {
            base,
            delta: RwLock::new(DeltaLayer {
                vectors: delta_vectors,
                graph: delta_graph,
                entry_point,
                max_degree,
            }),
            tombstones: RwLock::new(tombstones),
            dist,
            config,
            base_path: None,
            dim,
            base_labels,
            delta_labels: RwLock::new(delta_labels_vec),
            num_label_fields,
            quantizer,
            base_codes,
            code_size,
            rerank_size,
        })
    }

    /// Get statistics about the index
    pub fn stats(&self) -> IncrementalStats {
        let delta = self.delta.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();
        let base_count = self.base.as_ref().map(|b| b.num_vectors).unwrap_or(0);

        IncrementalStats {
            base_vectors: base_count,
            delta_vectors: delta.len(),
            tombstones: tombstones.len(),
            total_live: (base_count + delta.len()).saturating_sub(tombstones.len()),
            dim: self.dim,
        }
    }

    /// Get the dimensionality of vectors in this index
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether this index has labels configured.
    pub fn has_labels(&self) -> bool {
        self.base_labels.is_some()
    }

    /// Whether this index has quantization configured.
    pub fn has_quantizer(&self) -> bool {
        self.quantizer.is_some()
    }
}

/// Statistics about an incremental index
#[derive(Debug, Clone)]
pub struct IncrementalStats {
    pub base_vectors: usize,
    pub delta_vectors: usize,
    pub tombstones: usize,
    pub total_live: usize,
    pub dim: usize,
}

// =========================================================================
// Encoding helpers
// =========================================================================

fn encode_all_vecs<Q: VectorQuantizer>(
    vectors: &[Vec<f32>],
    quantizer: &Q,
    code_size: usize,
) -> Vec<u8> {
    let encoded: Vec<Vec<u8>> = vectors.par_iter().map(|v| quantizer.encode(v)).collect();
    let mut flat = Vec::with_capacity(vectors.len() * code_size);
    for code in &encoded {
        flat.extend_from_slice(code);
    }
    flat
}

fn encode_all_pq_vecs(vectors: &[Vec<f32>], pq: &ProductQuantizer, code_size: usize) -> Vec<u8> {
    let encoded: Vec<Vec<u8>> = vectors.par_iter().map(|v| pq.encode(v)).collect();
    let mut flat = Vec::with_capacity(vectors.len() * code_size);
    for code in &encoded {
        flat.extend_from_slice(code);
    }
    flat
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistL2;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_incremental_basic() {
        let path = "test_incremental_basic.db";
        let _ = fs::remove_file(path);

        // Build initial index
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Search should work
        let results = index.search(&[0.1, 0.1], 2, 8);
        assert_eq!(results.len(), 2);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_add() {
        let path = "test_incremental_add.db";
        let _ = fs::remove_file(path);

        let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Add new vectors
        let new_vecs = vec![vec![0.5, 0.5], vec![2.0, 2.0]];
        let new_ids = index.add_vectors(&new_vecs).unwrap();
        assert_eq!(new_ids.len(), 2);
        assert!(is_delta_id(new_ids[0]));

        // Search should find the new vector
        let results = index.search_with_dists(&[0.5, 0.5], 1, 8);
        assert!(!results.is_empty());

        // The closest should be the one we just added at [0.5, 0.5]
        let (_best_id, best_dist) = results[0];
        assert!(
            best_dist < 0.01,
            "Expected to find [0.5, 0.5], got dist {}",
            best_dist
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_delete() {
        let path = "test_incremental_delete.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1
            vec![0.0, 1.0], // id 2
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Delete vector 0
        index.delete_vectors(&[0]).unwrap();
        assert!(index.is_deleted(0));

        // Search near [0,0] should not return id 0
        let results = index.search(&[0.0, 0.0], 3, 8);
        assert!(
            !results.contains(&0),
            "Deleted vector should not appear in results"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_compact() {
        let path1 = "test_compact_v1.db";
        let path2 = "test_compact_v2.db";
        let _ = fs::remove_file(path1);
        let _ = fs::remove_file(path2);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let mut index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path1).unwrap();

        // Add some vectors
        index
            .add_vectors(&[vec![2.0, 2.0], vec![3.0, 3.0]])
            .unwrap();

        // Delete some
        index.delete_vectors(&[0, 1]).unwrap();

        let stats_before = index.stats();
        assert_eq!(stats_before.base_vectors, 4);
        assert_eq!(stats_before.delta_vectors, 2);
        assert_eq!(stats_before.tombstones, 2);

        // Compact
        index.compact(path2).unwrap();

        let stats_after = index.stats();
        assert_eq!(stats_after.base_vectors, 4); // 4 base - 2 deleted + 2 delta = 4
        assert_eq!(stats_after.delta_vectors, 0);
        assert_eq!(stats_after.tombstones, 0);

        // Search should still work
        let results = index.search(&[2.0, 2.0], 1, 8);
        assert!(!results.is_empty());

        let _ = fs::remove_file(path1);
        let _ = fs::remove_file(path2);
    }

    #[test]
    fn test_delta_only() {
        // Test index with no base, only delta
        let config = IncrementalConfig::default();
        let index = IncrementalDiskANN::<DistL2>::new_empty(2, DistL2 {}, config);

        // Add vectors
        let vecs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        index.add_vectors(&vecs).unwrap();

        // Search
        let results = index.search_with_dists(&[0.5, 0.5], 3, 8);
        assert_eq!(results.len(), 3);

        // Closest should be [0.5, 0.5]
        let best_vec = index.get_vector(results[0].0).unwrap();
        let dist = euclid(&best_vec, &[0.5, 0.5]);
        assert!(dist < 0.01);
    }

    #[test]
    fn test_incremental_to_bytes_from_bytes() {
        let path = "test_incr_bytes_rt.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Add delta vectors
        index
            .add_vectors(&[vec![0.5, 0.5], vec![2.0, 2.0]])
            .unwrap();

        // Delete one
        index.delete_vectors(&[0]).unwrap();

        let bytes = index.to_bytes();

        let index2 = IncrementalDiskANN::<DistL2>::from_bytes(
            &bytes,
            DistL2 {},
            IncrementalConfig::default(),
        )
        .unwrap();

        let stats = index2.stats();
        assert_eq!(stats.base_vectors, 4);
        assert_eq!(stats.delta_vectors, 2);
        assert_eq!(stats.tombstones, 1);

        // Search should exclude tombstone and include delta
        let results = index2.search(&[0.5, 0.5], 3, 8);
        assert!(!results.contains(&0), "Deleted vector should not appear");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_backward_compat_bytes() {
        let path = "test_incr_compat.db";
        let _ = fs::remove_file(path);

        let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        // Build old-format bytes manually: [has_base:u8][base_len:u64][base_bytes][dim:u64]...
        let base = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        let base_bytes = base.to_bytes();
        let mut old_bytes = Vec::new();
        old_bytes.push(1u8); // has_base
        old_bytes.extend_from_slice(&(base_bytes.len() as u64).to_le_bytes());
        old_bytes.extend_from_slice(&base_bytes);
        old_bytes.extend_from_slice(&(2u64).to_le_bytes()); // dim
        old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_delta = 0
        old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_graph = 0
        old_bytes.extend_from_slice(&(-1i64).to_le_bytes()); // entry_point = None
        old_bytes.extend_from_slice(&(32u64).to_le_bytes()); // max_degree
        old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_tombstones = 0

        let loaded = IncrementalDiskANN::<DistL2>::from_bytes(
            &old_bytes,
            DistL2 {},
            IncrementalConfig::default(),
        )
        .unwrap();

        assert_eq!(loaded.stats().base_vectors, 3);
        assert_eq!(loaded.stats().delta_vectors, 0);
        assert!(!loaded.has_labels());
        assert!(!loaded.has_quantizer());

        let results = loaded.search(&[0.0, 0.0], 2, 8);
        assert_eq!(results.len(), 2);

        let _ = fs::remove_file(path);
    }
}
