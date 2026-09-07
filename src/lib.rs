mod filtered;
pub mod formats;
mod incremental;
mod metric;
pub mod pq;
mod quantized;
pub mod rabitq;
mod spfresh;
pub mod sq;
pub mod storage;

pub use quantized::{QuantizedConfig, QuantizedDiskANN};
pub use spfresh::{Manifest, SPFresh, SPFreshConfig, SPFreshStats, RAW_CHUNK};

pub use incremental::{
    delta_local_idx, is_delta_id, IncrementalConfig, IncrementalDiskANN,
    IncrementalQuantizedConfig, IncrementalStats, QuantizerKind,
};

pub use filtered::{Filter, FilteredDiskANN};

pub use pq::{PQConfig, PQStats, ProductQuantizer};

pub use storage::Storage;

pub use sq::{F16Quantizer, Int8Quantizer, VectorQuantizer};

pub use rabitq::{RaBitQ, RaBitQQuery};

use bytemuck;
pub use metric::{simd_info, DistCosine, DistDot, DistL2, DistL2Sq, Distance};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use thiserror::Error;

/// Padding sentinel for adjacency slots (avoid colliding with node 0).
pub(crate) const PAD_U32: u32 = u32::MAX;

/// Magic number for the core index format: "DANN"
const CORE_MAGIC: u32 = 0x44414E4E;
/// Current core index format version
const CORE_FORMAT_VERSION: u32 = 1;

/// Defaults for in-memory DiskANN builds
pub const DISKANN_DEFAULT_MAX_DEGREE: usize = 64;
pub const DISKANN_DEFAULT_BUILD_BEAM: usize = 128;
pub const DISKANN_DEFAULT_ALPHA: f32 = 1.2;

/// Optional bag of knobs if you want to override just a few.
#[derive(Clone, Copy, Debug)]
pub struct DiskAnnParams {
    pub max_degree: usize,
    pub build_beam_width: usize,
    pub alpha: f32,
}
impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            max_degree: DISKANN_DEFAULT_MAX_DEGREE,
            build_beam_width: DISKANN_DEFAULT_BUILD_BEAM,
            alpha: DISKANN_DEFAULT_ALPHA,
        }
    }
}

/// Custom error type for DiskAnnRS operations
#[derive(Debug, Error)]
pub enum DiskAnnError {
    /// Represents I/O errors during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Represents serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Represents index-specific errors
    #[error("Index error: {0}")]
    IndexError(String),
}

/// Internal metadata structure stored in the index file
#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    distance_name: String,
}

/// Candidate for search/frontier queues
#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    pub dist: f32,
    pub id: u32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Natural order by distance: smaller is "less".
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Internal abstraction for a searchable graph index with u32 IDs.
#[allow(dead_code)]
pub(crate) trait GraphIndex: Send + Sync {
    fn num_vectors(&self) -> usize;
    fn dim(&self) -> usize;
    fn entry_point(&self) -> u32;
    fn distance_to(&self, query: &[f32], id: u32) -> f32;
    fn get_neighbors(&self, id: u32) -> Vec<u32>; // PAD_U32 already filtered
    fn get_vector(&self, id: u32) -> Vec<f32>;
    fn is_live(&self, _id: u32) -> bool {
        true
    }
}

impl<D> GraphIndex for DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn entry_point(&self) -> u32 {
        self.medoid_id
    }
    fn distance_to(&self, query: &[f32], id: u32) -> f32 {
        DiskANN::distance_to(self, query, id as usize)
    }
    fn get_neighbors(&self, id: u32) -> Vec<u32> {
        DiskANN::get_neighbors(self, id)
            .iter()
            .copied()
            .filter(|&nb| nb != PAD_U32)
            .collect()
    }
    fn get_vector(&self, id: u32) -> Vec<f32> {
        DiskANN::get_vector(self, id as usize)
    }
}

/// Configuration for the unified beam search.
pub(crate) struct BeamSearchConfig {
    /// If set, use an expanded working set of this size (for filtered search).
    /// Candidates in the expanded set participate in graph exploration but
    /// only candidates passing the filter are added to the results.
    pub expanded_beam: Option<usize>,
    /// Maximum iterations before forced termination (for filtered search).
    pub max_iterations: Option<usize>,
    /// Early termination factor: stop when best frontier > worst_result * factor
    /// (for filtered search).
    pub early_term_factor: Option<f32>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            expanded_beam: None,
            max_iterations: None,
            early_term_factor: None,
        }
    }
}

/// Unified beam search used by all search variants (base, quantized, filtered).
///
/// - `start_ids`: entry point nodes (typically medoid, or multiple seeds)
/// - `beam_width`: working set size (number of closest candidates maintained)
/// - `k`: number of results to return
/// - `distance_fn`: computes distance from query to node id
/// - `neighbors_fn`: returns neighbor ids for a node (filtered, no PAD_U32)
/// - `filter_fn`: returns true if a candidate should be included in results
/// - `config`: optional expanded beam / iteration limits for filtered search
pub(crate) fn beam_search(
    start_ids: &[u32],
    beam_width: usize,
    k: usize,
    distance_fn: impl Fn(u32) -> f32,
    neighbors_fn: impl Fn(u32) -> Vec<u32>,
    filter_fn: impl Fn(u32) -> bool,
    config: BeamSearchConfig,
) -> Vec<(u32, f32)> {
    let working_beam = config.expanded_beam.unwrap_or(beam_width);
    let is_filtered = config.expanded_beam.is_some();

    let mut visited = HashSet::new();
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    let mut w: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut results: Vec<(u32, f32)> = Vec::with_capacity(if is_filtered { k } else { 0 });

    for &sid in start_ids {
        if !visited.insert(sid) {
            continue;
        }
        let d = distance_fn(sid);
        let cand = Candidate { dist: d, id: sid };
        frontier.push(Reverse(cand));
        w.push(cand);
        if is_filtered && filter_fn(sid) {
            results.push((sid, d));
        }
    }
    if is_filtered {
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
    }

    let mut iterations = 0;
    let max_iterations = config.max_iterations.unwrap_or(usize::MAX);
    let early_term_factor = config.early_term_factor.unwrap_or(f32::INFINITY);

    while let Some(Reverse(best)) = frontier.peek().copied() {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        if is_filtered && results.len() >= k {
            if let Some((_, worst_dist)) = results.last() {
                if early_term_factor.is_finite()
                    && *worst_dist >= 0.0
                    && best.dist > *worst_dist * early_term_factor.max(1.0)
                {
                    break;
                }
            }
        }

        if w.len() >= working_beam {
            if let Some(worst) = w.peek() {
                if best.dist >= worst.dist {
                    break;
                }
            }
        }

        let Reverse(current) = frontier.pop().unwrap();

        for nb in neighbors_fn(current.id) {
            if !visited.insert(nb) {
                continue;
            }

            let d = distance_fn(nb);
            let cand = Candidate { dist: d, id: nb };

            if w.len() < working_beam {
                w.push(cand);
                frontier.push(Reverse(cand));
            } else if d < w.peek().unwrap().dist {
                w.pop();
                w.push(cand);
                frontier.push(Reverse(cand));
            }

            if is_filtered && filter_fn(nb) {
                let pos = results
                    .iter()
                    .position(|(_, dist)| d < *dist)
                    .unwrap_or(results.len());
                if pos < k {
                    results.insert(pos, (nb, d));
                    if results.len() > k {
                        results.pop();
                    }
                }
            }
        }
    }

    if is_filtered {
        results
    } else {
        let mut candidates: Vec<_> = w.into_vec();
        candidates.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        candidates.truncate(k);
        candidates.into_iter().map(|c| (c.id, c.dist)).collect()
    }
}

/// Main struct representing a DiskANN index (generic over distance)
pub struct DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Dimensionality of vectors in the index
    pub dim: usize,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Maximum number of edges per node
    pub max_degree: usize,
    /// Informational: type name of the distance (from metadata)
    pub distance_name: String,

    /// ID of the medoid (used as entry point)
    pub(crate) medoid_id: u32,
    // Offsets
    pub(crate) vectors_offset: u64,
    pub(crate) adjacency_offset: u64,

    /// Backing storage (mmap, owned bytes, or shared bytes)
    pub(crate) storage: Storage,

    /// The distance strategy
    pub(crate) dist: D,
}

// constructors

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default parameters: (M=32, L=256, alpha=1.2).
    pub fn build_index_default(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            DISKANN_DEFAULT_MAX_DEGREE,
            DISKANN_DEFAULT_BUILD_BEAM,
            DISKANN_DEFAULT_ALPHA,
            dist,
            file_path,
        )
    }

    /// Build with a `DiskAnnParams` bundle.
    pub fn build_index_with_params(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        p: DiskAnnParams,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            p.max_degree,
            p.build_beam_width,
            p.alpha,
            dist,
            file_path,
        )
    }
}

/// Extra sugar when your distance type implements `Default` (most unit-struct metrics do).
impl<D> DiskANN<D>
where
    D: Distance<f32> + Default + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default params **and** `D::default()` metric.
    pub fn build_index_default_metric(
        vectors: &[Vec<f32>],
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index_default(vectors, D::default(), file_path)
    }

    /// Open an index using `D::default()` as the distance (matches what you built with).
    pub fn open_index_default_metric(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_index_with(path, D::default())
    }
}

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Builds a new index from provided vectors
    ///
    /// # Arguments
    /// * `vectors` - The vectors to index (slice of `Vec<f32>`)
    /// * `max_degree` - Maximum edges per node (M ~ 24-64)
    /// * `build_beam_width` - Construction L (e.g., 128-400)
    /// * `alpha` - Pruning parameter (1.2–2.0)
    /// * `dist` - Any `anndists::Distance<f32>` (e.g., `DistL2`)
    /// * `file_path` - Path of index file
    pub fn build_index(
        vectors: &[Vec<f32>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors provided".to_string()));
        }

        let num_vectors = vectors.len();
        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        // Reserve space for metadata (we'll write it after data)
        let vectors_offset = 1024 * 1024;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;

        // Write vectors contiguous (sequential I/O is fastest)
        file.seek(SeekFrom::Start(vectors_offset))?;
        for vector in vectors {
            let bytes = bytemuck::cast_slice(vector);
            file.write_all(bytes)?;
        }

        // Compute medoid using provided distance (parallelized distance eval)
        let medoid_id = calculate_medoid(vectors, dist);

        // Build Vamana-like graph (stronger refinement, parallel inner loops)
        let adjacency_offset = vectors_offset as u64 + total_vector_bytes;
        let graph = build_vamana_graph(
            vectors,
            max_degree,
            build_beam_width,
            alpha,
            dist,
            medoid_id as u32,
        );

        // Write adjacency lists (fixed max_degree, pad with PAD_U32)
        file.seek(SeekFrom::Start(adjacency_offset))?;
        for neighbors in &graph {
            let mut padded = neighbors.clone();
            padded.resize(max_degree, PAD_U32);
            let bytes = bytemuck::cast_slice(&padded);
            file.write_all(bytes)?;
        }

        // Write metadata
        let metadata = Metadata {
            dim,
            num_vectors,
            max_degree,
            medoid_id: medoid_id as u32,
            vectors_offset: vectors_offset as u64,
            adjacency_offset,
            distance_name: std::any::type_name::<D>().to_string(),
        };

        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&CORE_MAGIC.to_le_bytes())?;
        file.write_all(&CORE_FORMAT_VERSION.to_le_bytes())?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        // Memory map the file
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            storage: Storage::Mmap(mmap),
            dist,
        })
    }

    /// Opens an existing index file, supplying the distance strategy explicitly.
    pub fn open_index_with(path: &str, dist: D) -> Result<Self, DiskAnnError> {
        let mut file = OpenOptions::new().read(true).write(false).open(path)?;

        // Read first 4 bytes to detect format (magic or old-style md_len)
        let mut buf4 = [0u8; 4];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut buf4)?;
        let first_u32 = u32::from_le_bytes(buf4);

        let md_offset = if first_u32 == CORE_MAGIC {
            // New format: [magic:u32][version:u32][md_len:u64][metadata...]
            let mut ver_buf = [0u8; 4];
            file.read_exact(&mut ver_buf)?;
            let version = u32::from_le_bytes(ver_buf);
            if version != CORE_FORMAT_VERSION {
                return Err(DiskAnnError::IndexError(format!(
                    "Unsupported core format version: {}",
                    version
                )));
            }
            8u64 // magic + version = 8 bytes, then md_len starts
        } else {
            // Old format: [md_len:u64][metadata...]
            file.seek(SeekFrom::Start(0))?;
            0u64
        };

        // Read metadata length
        let mut buf8 = [0u8; 8];
        file.seek(SeekFrom::Start(md_offset))?;
        file.read_exact(&mut buf8)?;
        let md_len = u64::from_le_bytes(buf8);

        // Sanity check: metadata length must be reasonable (< 1 MiB and < file size)
        let file_size = file.seek(SeekFrom::End(0))?;
        if md_len > 1024 * 1024 || md_offset + 8 + md_len > file_size {
            return Err(DiskAnnError::IndexError(format!(
                "Invalid metadata length {} (file size {})",
                md_len, file_size
            )));
        }
        file.seek(SeekFrom::Start(md_offset + 8))?;

        // Read metadata
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact(&mut md_bytes)?;
        let metadata: Metadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Optional sanity/logging: warn if type differs from recorded name
        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            storage: Storage::Mmap(mmap),
            dist,
        })
    }

    /// Load an index from an owned byte buffer (no file needed).
    pub fn from_bytes(bytes: Vec<u8>, dist: D) -> Result<Self, DiskAnnError> {
        let metadata = Self::parse_metadata(&bytes)?;

        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            storage: Storage::Owned(bytes),
            dist,
        })
    }

    /// Load an index from a shared byte buffer (cheap clone, multi-reader).
    pub fn from_shared_bytes(bytes: Arc<[u8]>, dist: D) -> Result<Self, DiskAnnError> {
        let metadata = Self::parse_metadata(&bytes)?;

        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            storage: Storage::Shared(bytes),
            dist,
        })
    }

    /// Serialize the index to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.storage.to_vec()
    }

    /// Parse metadata from raw bytes (shared helper for from_bytes / from_shared_bytes).
    /// Handles both new format (with magic/version) and old format (raw md_len).
    fn parse_metadata(bytes: &[u8]) -> Result<Metadata, DiskAnnError> {
        if bytes.len() < 8 {
            return Err(DiskAnnError::IndexError(
                "Buffer too small for metadata length".into(),
            ));
        }

        // Detect format: check first 4 bytes for magic
        let first_u32 = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let md_offset = if first_u32 == CORE_MAGIC {
            // New format: skip magic(4) + version(4)
            if bytes.len() < 16 {
                return Err(DiskAnnError::IndexError(
                    "Buffer too small for header".into(),
                ));
            }
            let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            if version != CORE_FORMAT_VERSION {
                return Err(DiskAnnError::IndexError(format!(
                    "Unsupported core format version: {}",
                    version
                )));
            }
            8
        } else {
            0
        };

        if bytes.len() < md_offset + 8 {
            return Err(DiskAnnError::IndexError(
                "Buffer too small for metadata length".into(),
            ));
        }
        let md_len =
            u64::from_le_bytes(bytes[md_offset..md_offset + 8].try_into().unwrap()) as usize;
        if bytes.len() < md_offset + 8 + md_len {
            return Err(DiskAnnError::IndexError(
                "Buffer too small for metadata".into(),
            ));
        }
        let metadata: Metadata =
            bincode::deserialize(&bytes[md_offset + 8..md_offset + 8 + md_len])?;
        Ok(metadata)
    }

    /// Searches the index for nearest neighbors using a best-first beam search.
    /// Like `search` but also returns the distance for each neighbor.
    pub fn search_with_dists(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        beam_search(
            &[self.medoid_id],
            beam_width,
            k,
            |id| self.distance_to(query, id as usize),
            |id| {
                self.get_neighbors(id)
                    .iter()
                    .copied()
                    .filter(|&nb| nb != PAD_U32)
                    .collect()
            },
            |_| true,
            BeamSearchConfig::default(),
        )
    }
    /// search but only return neighbor ids
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _dist)| id)
            .collect()
    }

    /// Gets the neighbors of a node from the (fixed-degree) adjacency region
    pub(crate) fn get_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.adjacency_offset + (node_id as u64 * self.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.max_degree * 4);
        let bytes = &self.storage[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between `query` and vector `idx`
    pub(crate) fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.storage[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        self.dist.eval(query, vector)
    }

    /// Gets a vector from the index (useful for tests)
    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.storage[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        vector.to_vec()
    }
}

/// Calculates the medoid (vector closest to the centroid) using distance `D`
/// Parallelizes the per-vector distance evaluations.
fn calculate_medoid<D: Distance<f32> + Copy + Sync>(vectors: &[Vec<f32>], dist: D) -> usize {
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];

    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in &mut centroid {
        *val /= vectors.len() as f32;
    }

    let (best_idx, _best_dist) = vectors
        .par_iter()
        .enumerate()
        .map(|(idx, v)| (idx, dist.eval(&centroid, v)))
        .reduce(|| (0usize, f32::MAX), |a, b| if a.1 <= b.1 { a } else { b });

    best_idx
}

/// Builds a strengthened Vamana-like graph using multi-pass refinement.
/// - Multi-seed candidate gathering (medoid + random seeds)
/// - Union with current adjacency before α-prune
/// - 2 refinement passes with symmetrization after each pass
fn build_vamana_graph<D: Distance<f32> + Copy + Sync>(
    vectors: &[Vec<f32>],
    max_degree: usize,
    build_beam_width: usize,
    alpha: f32,
    dist: D,
    medoid_id: u32,
) -> Vec<Vec<u32>> {
    let n = vectors.len();
    let mut graph = vec![Vec::<u32>::new(); n];

    // Light random bootstrap to avoid disconnected starts
    {
        let mut rng = thread_rng();
        for i in 0..n {
            let mut s = HashSet::new();
            let target = (max_degree / 2).max(2).min(n.saturating_sub(1));
            while s.len() < target {
                let nb = rng.gen_range(0..n);
                if nb != i {
                    s.insert(nb as u32);
                }
            }
            graph[i] = s.into_iter().collect();
        }
    }

    // Refinement passes
    const PASSES: usize = 2;
    const EXTRA_SEEDS: usize = 2;

    let mut rng = thread_rng();
    for _pass in 0..PASSES {
        // Shuffle visit order each pass
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        // Snapshot read of graph for parallel candidate building
        let snapshot = &graph;

        // Build new neighbor proposals in parallel
        let new_graph: Vec<Vec<u32>> = order
            .par_iter()
            .map(|&u| {
                let mut candidates: Vec<(u32, f32)> =
                    Vec::with_capacity(build_beam_width * (2 + EXTRA_SEEDS));

                // Include current adjacency with distances
                for &nb in &snapshot[u] {
                    let d = dist.eval(&vectors[u], &vectors[nb as usize]);
                    candidates.push((nb, d));
                }

                // Seeds: always medoid + some random starts
                let mut seeds = Vec::with_capacity(1 + EXTRA_SEEDS);
                seeds.push(medoid_id as usize);
                let mut trng = thread_rng();
                for _ in 0..EXTRA_SEEDS {
                    seeds.push(trng.gen_range(0..n));
                }

                // Gather candidates from greedy searches
                for start in seeds {
                    let mut part = greedy_search(
                        &vectors[u],
                        vectors,
                        snapshot,
                        start,
                        build_beam_width,
                        dist,
                    );
                    candidates.append(&mut part);
                }

                // Deduplicate by id keeping best distance
                candidates.sort_by(|a, b| a.0.cmp(&b.0));
                candidates.dedup_by(|a, b| {
                    if a.0 == b.0 {
                        if a.1 < b.1 {
                            *b = *a;
                        }
                        true
                    } else {
                        false
                    }
                });

                // α-prune around u
                prune_neighbors(u, &candidates, vectors, max_degree, alpha, dist)
            })
            .collect();

        // Symmetrize: union incoming + outgoing, then α-prune again (parallel)
        // Build inverse map: node-id -> position in `order`
        let mut pos_of = vec![0usize; n];
        for (pos, &u) in order.iter().enumerate() {
            pos_of[u] = pos;
        }

        // Build incoming as CSR
        let (incoming_flat, incoming_off) = build_incoming_csr(&order, &new_graph, n);

        // Union + prune in parallel
        graph = (0..n)
            .into_par_iter()
            .map(|u| {
                let ng = &new_graph[pos_of[u]]; // outgoing from this pass
                let inc = &incoming_flat[incoming_off[u]..incoming_off[u + 1]]; // incoming to u

                // pool = union(outgoing ∪ incoming) with tiny, cache-friendly ops
                let mut pool_ids: Vec<u32> = Vec::with_capacity(ng.len() + inc.len());
                pool_ids.extend_from_slice(ng);
                pool_ids.extend_from_slice(inc);
                pool_ids.sort_unstable();
                pool_ids.dedup();

                // compute distances once, then α-prune
                let pool: Vec<(u32, f32)> = pool_ids
                    .into_iter()
                    .filter(|&id| id as usize != u)
                    .map(|id| (id, dist.eval(&vectors[u], &vectors[id as usize])))
                    .collect();

                prune_neighbors(u, &pool, vectors, max_degree, alpha, dist)
            })
            .collect();
    }

    // Final cleanup (ensure <= max_degree everywhere)
    graph
        .into_par_iter()
        .enumerate()
        .map(|(u, neigh)| {
            if neigh.len() <= max_degree {
                return neigh;
            }
            let pool: Vec<(u32, f32)> = neigh
                .iter()
                .map(|&id| (id, dist.eval(&vectors[u], &vectors[id as usize])))
                .collect();
            prune_neighbors(u, &pool, vectors, max_degree, alpha, dist)
        })
        .collect()
}

/// Greedy search used during construction (read-only on `graph`)
/// Same termination rule as query-time search.
fn greedy_search<D: Distance<f32> + Copy>(
    query: &[f32],
    vectors: &[Vec<f32>],
    graph: &[Vec<u32>],
    start_id: usize,
    beam_width: usize,
    dist: D,
) -> Vec<(u32, f32)> {
    let mut visited = HashSet::new();
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new(); // min-heap by dist
    let mut w: BinaryHeap<Candidate> = BinaryHeap::new(); // max-heap by dist

    let start_dist = dist.eval(query, &vectors[start_id]);
    let start = Candidate {
        dist: start_dist,
        id: start_id as u32,
    };
    frontier.push(Reverse(start));
    w.push(start);
    visited.insert(start_id as u32);

    while let Some(Reverse(best)) = frontier.peek().copied() {
        if w.len() >= beam_width {
            if let Some(worst) = w.peek() {
                if best.dist >= worst.dist {
                    break;
                }
            }
        }
        let Reverse(cur) = frontier.pop().unwrap();

        for &nb in &graph[cur.id as usize] {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist.eval(query, &vectors[nb as usize]);
            let cand = Candidate { dist: d, id: nb };

            if w.len() < beam_width {
                w.push(cand);
                frontier.push(Reverse(cand));
            } else if d < w.peek().unwrap().dist {
                w.pop();
                w.push(cand);
                frontier.push(Reverse(cand));
            }
        }
    }

    let mut v = w.into_vec();
    v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    v.into_iter().map(|c| (c.id, c.dist)).collect()
}

/// RobustPrune (Vamana): strict RNG pass at α=1, then relaxed passes up to `alpha`
/// (as in Microsoft DiskANN's `occlude_list`), then backfill with the closest.
pub(crate) fn robust_prune(
    node_id: u32,
    candidates: &[(u32, f32)],
    max_degree: usize,
    alpha: f32,
    dist: impl Fn(u32, u32) -> f32,
) -> Vec<u32> {
    let mut sorted: Vec<(u32, f32)> = candidates
        .iter()
        .copied()
        .filter(|c| c.0 != node_id)
        .collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    sorted.dedup_by_key(|c| c.0);
    let n = sorted.len();
    let alpha = alpha.max(1.0);

    let mut occlude = vec![0.0f32; n];
    let mut selected = vec![false; n];
    let mut pruned = Vec::with_capacity(max_degree.min(n));
    let mut cur_alpha = 1.0f32;
    while cur_alpha <= alpha && pruned.len() < max_degree {
        for i in 0..n {
            if pruned.len() >= max_degree {
                break;
            }
            if selected[i] || occlude[i] > cur_alpha {
                continue;
            }
            selected[i] = true;
            pruned.push(sorted[i].0);
            for j in i + 1..n {
                if selected[j] || occlude[j] > alpha {
                    continue;
                }
                let djk = dist(sorted[j].0, sorted[i].0);
                occlude[j] = if djk == 0.0 {
                    f32::INFINITY
                } else {
                    occlude[j].max(sorted[j].1 / djk)
                };
            }
        }
        cur_alpha *= 1.2;
    }

    for i in 0..n {
        if pruned.len() >= max_degree {
            break;
        }
        if !selected[i] {
            selected[i] = true;
            pruned.push(sorted[i].0);
        }
    }

    pruned
}

fn prune_neighbors<D: Distance<f32> + Copy>(
    node_id: usize,
    candidates: &[(u32, f32)],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    dist: D,
) -> Vec<u32> {
    robust_prune(node_id as u32, candidates, max_degree, alpha, |a, b| {
        dist.eval(&vectors[a as usize], &vectors[b as usize])
    })
}

fn build_incoming_csr(order: &[usize], new_graph: &[Vec<u32>], n: usize) -> (Vec<u32>, Vec<usize>) {
    // 1) count in-degree per node
    let mut indeg = vec![0usize; n];
    for (pos, _u) in order.iter().enumerate() {
        for &v in &new_graph[pos] {
            indeg[v as usize] += 1;
        }
    }
    // 2) prefix sums → offsets
    let mut off = vec![0usize; n + 1];
    for i in 0..n {
        off[i + 1] = off[i] + indeg[i];
    }
    // 3) fill flat incoming list
    let mut cur = off.clone();
    let mut incoming_flat = vec![0u32; off[n]];
    for (pos, &u) in order.iter().enumerate() {
        for &v in &new_graph[pos] {
            let idx = cur[v as usize];
            incoming_flat[idx] = u as u32;
            cur[v as usize] += 1;
        }
    }
    (incoming_flat, off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistCosine, DistL2};
    use rand::Rng;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_small_index_l2() {
        let path = "test_small_l2.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();

        let q = vec![0.1, 0.1];
        let nns = index.search(&q, 3, 8);
        assert_eq!(nns.len(), 3);

        // Verify the first neighbor is quite close in L2
        let v = index.get_vector(nns[0] as usize);
        assert!(euclid(&q, &v) < 1.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cosine() {
        let path = "test_cosine.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let index =
            DiskANN::<DistCosine>::build_index_default(&vectors, DistCosine {}, path).unwrap();

        let q = vec![2.0, 0.0, 0.0]; // parallel to [1,0,0]
        let nns = index.search(&q, 2, 8);
        assert_eq!(nns.len(), 2);

        // Top neighbor should have high cosine similarity (close direction)
        let v = index.get_vector(nns[0] as usize);
        let dot = v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>();
        let n1 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (n1 * n2);
        assert!(cos > 0.7);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_persistence_and_open() {
        let path = "test_persist.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        {
            let _idx = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        }

        let idx2 = DiskANN::<DistL2>::open_index_default_metric(path).unwrap();
        assert_eq!(idx2.num_vectors, 4);
        assert_eq!(idx2.dim, 2);

        let q = vec![0.9, 0.9];
        let res = idx2.search(&q, 2, 8);
        // [1,1] should be best
        assert_eq!(res[0], 3);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_grid_connectivity() {
        let path = "test_grid.db";
        let _ = fs::remove_file(path);

        // 5x5 grid
        let mut vectors = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                vectors.push(vec![i as f32, j as f32]);
            }
        }

        let index = DiskANN::<DistL2>::build_index_with_params(
            &vectors,
            DistL2 {},
            path,
            DiskAnnParams {
                max_degree: 4,
                build_beam_width: 64,
                alpha: 1.5,
            },
        )
        .unwrap();

        for target in 0..vectors.len() {
            let q = &vectors[target];
            let nns = index.search(q, 10, 32);
            if !nns.contains(&(target as u32)) {
                let v = index.get_vector(nns[0] as usize);
                assert!(euclid(q, &v) < 2.0);
            }
            for &nb in nns.iter().take(5) {
                let v = index.get_vector(nb as usize);
                assert!(euclid(q, &v) < 5.0);
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_medium_random() {
        let path = "test_medium.db";
        let _ = fs::remove_file(path);

        let n = 200usize;
        let d = 32usize;
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        let index = DiskANN::<DistL2>::build_index_with_params(
            &vectors,
            DistL2 {},
            path,
            DiskAnnParams {
                max_degree: 32,
                build_beam_width: 128,
                alpha: 1.2,
            },
        )
        .unwrap();

        let q: Vec<f32> = (0..d).map(|_| rng.r#gen::<f32>()).collect();
        let res = index.search(&q, 10, 64);
        assert_eq!(res.len(), 10);

        // Ensure distances are nondecreasing
        let dists: Vec<f32> = res
            .iter()
            .map(|&id| {
                let v = index.get_vector(id as usize);
                euclid(&q, &v)
            })
            .collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, sorted);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_to_bytes_from_bytes_round_trip() {
        let path = "test_bytes_rt.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        let bytes = index.to_bytes();

        let index2 = DiskANN::<DistL2>::from_bytes(bytes, DistL2 {}).unwrap();
        assert_eq!(index2.num_vectors, 5);
        assert_eq!(index2.dim, 2);

        let q = vec![0.9, 0.9];
        let res1 = index.search(&q, 3, 8);
        let res2 = index2.search(&q, 3, 8);
        assert_eq!(res1, res2);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_from_shared_bytes() {
        let path = "test_shared_bytes.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        let bytes = index.to_bytes();
        let shared: std::sync::Arc<[u8]> = bytes.into();

        let index2 = DiskANN::<DistL2>::from_shared_bytes(shared, DistL2 {}).unwrap();
        assert_eq!(index2.num_vectors, 4);
        assert_eq!(index2.dim, 2);

        let q = vec![0.9, 0.9];
        let res = index2.search(&q, 2, 8);
        assert_eq!(res[0], 3); // [1,1]

        let _ = fs::remove_file(path);
    }

    // ================================================================
    // Unit tests for graph algorithms (no file I/O)
    // ================================================================

    #[test]
    fn test_candidate_ordering() {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let a = Candidate { dist: 1.0, id: 0 };
        let b = Candidate { dist: 2.0, id: 1 };
        let c = Candidate { dist: 0.5, id: 2 };

        // Natural ordering: smaller dist is "less"
        assert!(a < b);
        assert!(c < a);

        // Min-heap via Reverse
        let mut min_heap: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        min_heap.push(Reverse(a));
        min_heap.push(Reverse(b));
        min_heap.push(Reverse(c));
        assert_eq!(min_heap.pop().unwrap().0.id, 2); // dist 0.5
        assert_eq!(min_heap.pop().unwrap().0.id, 0); // dist 1.0
        assert_eq!(min_heap.pop().unwrap().0.id, 1); // dist 2.0

        // Max-heap (natural order)
        let mut max_heap: BinaryHeap<Candidate> = BinaryHeap::new();
        max_heap.push(a);
        max_heap.push(b);
        max_heap.push(c);
        assert_eq!(max_heap.peek().unwrap().id, 1); // dist 2.0 at top
    }

    #[test]
    fn test_beam_search_small_graph() {
        // Hand-crafted 5-node graph:
        //   0 --1.0-- 1 --1.0-- 2
        //   |                   |
        //  2.0                 1.0
        //   |                   |
        //   3 ------1.5------- 4
        //
        // Node positions: 0=(0,0), 1=(1,0), 2=(2,0), 3=(0,2), 4=(2,1)
        let positions: Vec<[f32; 2]> = vec![
            [0.0, 0.0], // 0
            [1.0, 0.0], // 1
            [2.0, 0.0], // 2
            [0.0, 2.0], // 3
            [2.0, 1.0], // 4
        ];

        let neighbors: Vec<Vec<u32>> = vec![
            vec![1, 3], // 0 -> 1, 3
            vec![0, 2], // 1 -> 0, 2
            vec![1, 4], // 2 -> 1, 4
            vec![0, 4], // 3 -> 0, 4
            vec![2, 3], // 4 -> 2, 3
        ];

        // Query near node 4: (2.1, 0.9)
        let query = [2.1f32, 0.9];

        let results = beam_search(
            &[0], // start from node 0
            5,
            3,
            |id| {
                let p = &positions[id as usize];
                ((query[0] - p[0]).powi(2) + (query[1] - p[1]).powi(2)).sqrt()
            },
            |id| neighbors[id as usize].clone(),
            |_| true,
            BeamSearchConfig::default(),
        );

        assert_eq!(results.len(), 3);
        // Node 4 (2,1) should be closest to query (2.1, 0.9)
        assert_eq!(results[0].0, 4);
        // Node 2 (2,0) should be second closest
        assert_eq!(results[1].0, 2);
        // Distances should be sorted
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
    }

    #[test]
    fn test_beam_search_with_filter() {
        // Same 5-node graph as above
        let positions: Vec<[f32; 2]> =
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 1.0]];
        let neighbors: Vec<Vec<u32>> =
            vec![vec![1, 3], vec![0, 2], vec![1, 4], vec![0, 4], vec![2, 3]];

        // Query near node 4, but filter out nodes 4 and 2 (even IDs only allowed: 0, 2, 4... but let's filter for odd IDs)
        let query = [2.1f32, 0.9];

        let results = beam_search(
            &[0],
            5,
            3,
            |id| {
                let p = &positions[id as usize];
                ((query[0] - p[0]).powi(2) + (query[1] - p[1]).powi(2)).sqrt()
            },
            |id| neighbors[id as usize].clone(),
            |id| id % 2 == 1, // only odd IDs: 1 and 3
            BeamSearchConfig {
                expanded_beam: Some(10),
                max_iterations: Some(20),
                early_term_factor: Some(1.5),
            },
        );

        // Should only contain odd IDs
        for (id, _) in &results {
            assert!(id % 2 == 1, "Expected only odd IDs, got {}", id);
        }
        // Should find at least nodes 1 and 3
        let ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_prune_neighbors_alpha() {
        // 3 candidates around node 0:
        //   node 1 at distance 1.0
        //   node 2 at distance 1.5 but close to node 1 (should be pruned with high alpha)
        //   node 3 at distance 2.0 but far from both (should survive)
        let vectors = vec![
            vec![0.0, 0.0], // node 0 (center)
            vec![1.0, 0.0], // node 1
            vec![1.2, 0.0], // node 2 (close to node 1)
            vec![0.0, 2.0], // node 3 (far from node 1)
        ];

        let candidates: Vec<(u32, f32)> = vec![
            (1, DistL2 {}.eval(&vectors[0], &vectors[1])),
            (2, DistL2 {}.eval(&vectors[0], &vectors[2])),
            (3, DistL2 {}.eval(&vectors[0], &vectors[3])),
        ];

        // With alpha=1.0 (strict pruning), node 2 should be pruned because
        // dist(1,2) < alpha * dist(0,2), meaning node 1 is a better representative
        let pruned = prune_neighbors(0, &candidates, &vectors, 3, 1.0, DistL2 {});

        // Node 1 should always be included (closest)
        assert!(pruned.contains(&1));
        // Node 3 should be included (it's in a different direction)
        assert!(pruned.contains(&3));
        // With strict alpha=1.0 and max_degree=3, node 2 might still be added in the fill phase
        // but the alpha-pruning step itself should prefer diverse directions
    }

    #[test]
    fn test_prune_neighbors_max_degree() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];

        let candidates: Vec<(u32, f32)> = (1..6)
            .map(|i| (i as u32, DistL2 {}.eval(&vectors[0], &vectors[i])))
            .collect();

        // max_degree=2: should return at most 2 neighbors
        let pruned = prune_neighbors(0, &candidates, &vectors, 2, 1.2, DistL2 {});
        assert_eq!(pruned.len(), 2);
        assert!(!pruned.is_empty());

        // max_degree=5: should return all 5
        let pruned = prune_neighbors(0, &candidates, &vectors, 5, 1.2, DistL2 {});
        assert_eq!(pruned.len(), 5);

        // max_degree=1: should return exactly 1 (the closest)
        let pruned = prune_neighbors(0, &candidates, &vectors, 1, 1.2, DistL2 {});
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn test_core_magic_number_in_bytes() {
        let path = "test_magic.db";
        let _ = fs::remove_file(path);

        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        let bytes = index.to_bytes();

        // First 4 bytes should be CORE_MAGIC
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(
            magic, CORE_MAGIC,
            "Expected magic 0x{:08X}, got 0x{:08X}",
            CORE_MAGIC, magic
        );

        // Next 4 bytes should be version
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, CORE_FORMAT_VERSION);

        let _ = fs::remove_file(path);
    }
}
