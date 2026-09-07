use crate::quantized::{quantized_distance_from_codes, Prepared, QuantizerState};
use crate::sq::VectorQuantizer;
use crate::{
    DiskAnnError, Distance, F16Quantizer, IncrementalConfig, IncrementalDiskANN, Int8Quantizer,
    ProductQuantizer, QuantizerKind, RaBitQ,
};
use memmap2::MmapMut;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use numkong::Euclidean;
use std::any::TypeId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::sync::RwLock;

const FREE: u64 = u64::MAX;
/// Below this many live postings, `nearest` scans centroids instead of searching the graph.
const BRUTE_FORCE_CENTROIDS: usize = 4096;
const CHUNK: usize = 4096;
/// Raw vectors per object in a snapshot.
pub const RAW_CHUNK: usize = 4096;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SPFreshConfig {
    /// Split a posting when it reaches this many entries.
    pub max_posting_size: usize,
    /// `gc()` merges postings with fewer live entries than this.
    pub min_posting_size: usize,
    /// Skip probed postings whose centroid is farther than `probe_ratio × nearest`.
    pub probe_ratio: f32,
    /// Re-rank this many candidates with exact vectors (quantized only, 0 = off).
    pub rerank_size: usize,
    /// Neighbouring postings checked for reassignment after a split.
    pub reassign_neighbors: usize,
    /// Beam width for centroid-graph searches.
    pub centroid_beam: usize,
}

impl Default for SPFreshConfig {
    fn default() -> Self {
        Self {
            max_posting_size: 128,
            min_posting_size: 16,
            probe_ratio: f32::INFINITY,
            rerank_size: 0,
            reassign_neighbors: 16,
            centroid_beam: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SPFreshStats {
    pub live: usize,
    pub deleted: usize,
    pub postings: usize,
    pub max_posting: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct Meta {
    dim: usize,
    cfg: SPFreshConfig,
    quantizer: Option<QuantizerState>,
    block_cid: Vec<u64>,
    block_centroid: Vec<Vec<f32>>,
    next_id: u64,
    deleted: Vec<u64>,
    live: usize,
}

/// Snapshot descriptor stored at `{prefix}/manifest` (see [`SPFresh::snapshot`]).
#[derive(Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u64,
    meta: Meta,
    /// Snapshot version that last wrote each posting block (0 = never).
    pub block_ver: Vec<u64>,
    /// Snapshot version that last wrote each `RAW_CHUNK`-vector chunk of the raw store.
    pub raw_chunk_ver: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq)]
struct Cand(f32, u64);
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&o.0)
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap_or(Ordering::Equal)
    }
}

fn code_size(q: &Option<QuantizerState>, dim: usize) -> usize {
    match q {
        None => dim * 4,
        Some(QuantizerState::F16(_)) => dim * 2,
        Some(QuantizerState::Int8(_)) => dim,
        Some(QuantizerState::PQ(p)) => p.stats().code_size_bytes,
        Some(QuantizerState::RaBitQ(r)) => r.code_size(),
    }
}

fn block_bytes(cfg: &SPFreshConfig, q: &Option<QuantizerState>, dim: usize) -> usize {
    4 + 2 * cfg.max_posting_size * (8 + code_size(q, dim))
}

fn grow(f: &File, m: &mut MmapMut, need: usize) -> Result<(), DiskAnnError> {
    if m.len() < need {
        f.set_len((need as u64).max(2 * m.len() as u64))?;
        *m = unsafe { MmapMut::map_mut(f)? };
    }
    Ok(())
}

/// 2-means on raw vectors; falls back to an index split when degenerate.
fn kmeans2<D: Distance<f32> + Copy>(vs: &[&[f32]], dist: D) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let (n, dim) = (vs.len(), vs[0].len());
    let means = |side: &[bool]| {
        let (mut s, mut c) = ([vec![0.0f32; dim], vec![0.0f32; dim]], [0usize; 2]);
        for (v, &b) in vs.iter().zip(side) {
            c[b as usize] += 1;
            s[b as usize].iter_mut().zip(*v).for_each(|(a, x)| *a += x);
        }
        for k in 0..2 {
            s[k].iter_mut().for_each(|x| *x /= c[k].max(1) as f32);
        }
        s
    };
    let degenerate = |side: &[bool]| side.iter().all(|&s| s) || side.iter().all(|&s| !s);
    let far = (1..n)
        .max_by(|&i, &j| {
            dist.eval(vs[0], vs[i])
                .partial_cmp(&dist.eval(vs[0], vs[j]))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);
    let mut c = [vs[0].to_vec(), vs[far].to_vec()];
    let mut side = vec![false; n];
    for _ in 0..5 {
        for (v, s) in vs.iter().zip(side.iter_mut()) {
            *s = dist.eval(v, &c[1]) < dist.eval(v, &c[0]);
        }
        if degenerate(&side) {
            break;
        }
        c = means(&side);
    }
    if degenerate(&side) {
        side.iter_mut()
            .enumerate()
            .for_each(|(i, s)| *s = i >= n / 2);
        c = means(&side);
    }
    let [c1, c2] = c;
    (c1, c2, side)
}

#[cfg_attr(not(feature = "object-store"), allow(dead_code))]
struct Inner<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    dim: usize,
    cfg: SPFreshConfig,
    dist: D,
    quantizer: Option<QuantizerState>,
    code_size: usize,
    entry: usize,
    block: usize,
    graph: IncrementalDiskANN<D>,
    block_cid: Vec<u64>,
    block_centroid: Vec<Vec<f32>>,
    cid_block: HashMap<u64, u32>,
    free: Vec<(u32, u64)>,
    created: Vec<u32>,
    epoch: u64,
    dirty: HashMap<u32, u64>,
    write_seq: u64,
    synced_id: u64,
    manifest: Option<Manifest>,
    postings: MmapMut,
    postings_file: File,
    raw: MmapMut,
    raw_file: File,
    next_id: u64,
    deleted: HashSet<u64>,
    live: usize,
    path: String,
}

impl<D> Inner<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    fn new(
        path: &str,
        dim: usize,
        mut cfg: SPFreshConfig,
        quantizer: Option<QuantizerState>,
        graph: IncrementalDiskANN<D>,
        truncate: bool,
    ) -> Result<Self, DiskAnnError> {
        cfg.max_posting_size = cfg.max_posting_size.max(2);
        let code_size = code_size(&quantizer, dim);
        let entry = 8 + code_size;
        let block = block_bytes(&cfg, &quantizer, dim);
        let open = |suffix: &str, min: usize| -> Result<(File, MmapMut), DiskAnnError> {
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(truncate)
                .open(format!("{path}.{suffix}"))?;
            if (f.metadata()?.len() as usize) < min {
                f.set_len(min as u64)?;
            }
            let m = unsafe { MmapMut::map_mut(&f)? };
            Ok((f, m))
        };
        let (postings_file, postings) = open("postings", block)?;
        let (raw_file, raw) = open("vectors", 1024 * dim * 4)?;
        Ok(Self {
            dim,
            cfg,
            dist: D::default(),
            quantizer,
            code_size,
            entry,
            block,
            graph,
            block_cid: Vec::new(),
            block_centroid: Vec::new(),
            cid_block: HashMap::new(),
            free: Vec::new(),
            created: Vec::new(),
            epoch: 1,
            dirty: HashMap::new(),
            write_seq: 0,
            synced_id: 0,
            manifest: None,
            postings,
            postings_file,
            raw,
            raw_file,
            next_id: 0,
            deleted: HashSet::new(),
            live: 0,
            path: path.to_string(),
        })
    }

    fn blk(&self, b: u32) -> &[u8] {
        &self.postings[b as usize * self.block..][..self.block]
    }
    fn len(&self, b: u32) -> usize {
        u32::from_le_bytes(self.blk(b)[..4].try_into().unwrap()) as usize
    }
    fn is_live(&self, b: u32) -> bool {
        self.block_cid[b as usize] != FREE
    }
    fn entry(&self, b: u32, i: usize) -> (u64, &[u8]) {
        let e = &self.blk(b)[4 + i * self.entry..][..self.entry];
        (u64::from_le_bytes(e[..8].try_into().unwrap()), &e[8..])
    }
    fn raw(&self, id: u64) -> &[f32] {
        bytemuck::cast_slice(&self.raw[id as usize * self.dim * 4..][..self.dim * 4])
    }

    fn live_entries(&self, b: u32) -> Vec<(u64, Vec<u8>)> {
        (0..self.len(b))
            .map(|i| self.entry(b, i))
            .filter(|(id, _)| !self.deleted.contains(id))
            .map(|(id, c)| (id, c.to_vec()))
            .collect()
    }

    fn mark_dirty(&mut self, b: u32) {
        if self.dirty.is_empty() {
            let _ = std::fs::remove_file(format!("{}.cache", self.path));
        }
        self.write_seq += 1;
        self.dirty.insert(b, self.write_seq);
    }

    fn meta(&self) -> Meta {
        Meta {
            dim: self.dim,
            cfg: self.cfg,
            quantizer: self.quantizer.clone(),
            block_cid: self.block_cid.clone(),
            block_centroid: self.block_centroid.clone(),
            next_id: self.next_id,
            deleted: self.deleted.iter().copied().collect(),
            live: self.live,
        }
    }

    fn rewrite(&mut self, b: u32, entries: &[(u64, Vec<u8>)]) {
        self.mark_dirty(b);
        let (block, entry) = (self.block, self.entry);
        let blk = &mut self.postings[b as usize * block..][..block];
        blk[..4].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        for (i, (id, code)) in entries.iter().enumerate() {
            let e = &mut blk[4 + i * entry..][..entry];
            e[..8].copy_from_slice(&id.to_le_bytes());
            e[8..].copy_from_slice(code);
        }
    }

    fn push(&mut self, b: u32, id: u64, code: &[u8]) {
        self.mark_dirty(b);
        let (n, block, entry) = (self.len(b), self.block, self.entry);
        let blk = &mut self.postings[b as usize * block..][..block];
        let e = &mut blk[4 + n * entry..][..entry];
        e[..8].copy_from_slice(&id.to_le_bytes());
        e[8..].copy_from_slice(code);
        blk[..4].copy_from_slice(&((n + 1) as u32).to_le_bytes());
    }

    /// Code for `v` stored in posting `b`. RaBitQ codes are residuals against the
    /// posting centroid, so entries must be re-encoded whenever they change posting.
    fn encode(&self, v: &[f32], b: u32) -> Vec<u8> {
        match &self.quantizer {
            None => bytemuck::cast_slice(v).to_vec(),
            Some(QuantizerState::PQ(q)) => q.encode(v),
            Some(QuantizerState::F16(q)) => q.encode(v),
            Some(QuantizerState::Int8(q)) => q.encode(v),
            Some(QuantizerState::RaBitQ(q)) => q.encode_with(v, &self.block_centroid[b as usize]),
        }
    }
    fn encode_id(&self, id: u64, b: u32) -> Vec<u8> {
        self.encode(self.raw(id), b)
    }
    fn residual(&self) -> bool {
        matches!(self.quantizer, Some(QuantizerState::RaBitQ(_)))
    }

    fn code_dist(&self, query: &[f32], code: &[u8], prep: &Option<Prepared>, b: u32) -> f32 {
        match (&self.quantizer, prep) {
            (Some(QuantizerState::RaBitQ(q)), _) => self
                .dist
                .eval(query, &q.decode_with(code, &self.block_centroid[b as usize])),
            (Some(q), Some(p)) => {
                quantized_distance_from_codes(&self.dist, query, 0, code, self.code_size, q, p)
            }
            _ => self.dist.eval(query, bytemuck::cast_slice(code)),
        }
    }

    /// Nearest live postings as `(block, centroid distance)`, closest first.
    /// Small centroid sets are scanned exactly; larger ones go through the graph.
    fn nearest(&self, v: &[f32], n: usize) -> Vec<(u32, f32)> {
        if self.cid_block.len() <= BRUTE_FORCE_CENTROIDS {
            let mut all: Vec<(u32, f32)> = self
                .cid_block
                .values()
                .map(|&b| (b, self.dist.eval(v, &self.block_centroid[b as usize])))
                .collect();
            let n = n.min(all.len());
            if n < all.len() {
                all.select_nth_unstable_by(n, |a, b| a.1.total_cmp(&b.1));
                all.truncate(n);
            }
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            return all;
        }
        self.graph
            .search_with_dists(v, n, self.cfg.centroid_beam.max(n))
            .into_iter()
            .filter_map(|(cid, d)| self.cid_block.get(&cid).map(|&b| (b, d)))
            .collect()
    }

    fn alloc(&mut self) -> Result<u32, DiskAnnError> {
        if let Some(i) = self.free.iter().position(|&(_, e)| e < self.epoch) {
            return Ok(self.free.swap_remove(i).0);
        }
        let b = self.block_cid.len() as u32;
        grow(
            &self.postings_file,
            &mut self.postings,
            (b as usize + 1) * self.block,
        )?;
        self.block_cid.push(FREE);
        self.block_centroid.push(Vec::new());
        Ok(b)
    }

    fn new_posting(&mut self, centroid: Vec<f32>) -> Result<u32, DiskAnnError> {
        let b = self.alloc()?;
        let cid = self.graph.add_vectors(std::slice::from_ref(&centroid))?[0];
        self.block_cid[b as usize] = cid;
        self.cid_block.insert(cid, b);
        self.block_centroid[b as usize] = centroid;
        self.created.push(b);
        self.rewrite(b, &[]);
        Ok(b)
    }

    fn free_posting(&mut self, b: u32) -> Result<(), DiskAnnError> {
        let cid = std::mem::replace(&mut self.block_cid[b as usize], FREE);
        self.graph.delete_vectors(&[cid])?;
        self.cid_block.remove(&cid);
        self.block_centroid[b as usize] = Vec::new();
        self.rewrite(b, &[]);
        self.free.push((b, self.epoch));
        Ok(())
    }

    fn insert(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>, DiskAnnError> {
        if let Some(v) = vectors.iter().find(|v| v.len() != self.dim) {
            return Err(DiskAnnError::IndexError(format!(
                "Vector dim {} != index dim {}",
                v.len(),
                self.dim
            )));
        }
        let ids: Vec<u64> = (self.next_id..self.next_id + vectors.len() as u64).collect();
        grow(
            &self.raw_file,
            &mut self.raw,
            (self.next_id as usize + vectors.len()) * self.dim * 4,
        )?;
        for (&id, v) in ids.iter().zip(vectors) {
            let o = id as usize * self.dim * 4;
            self.raw[o..o + self.dim * 4].copy_from_slice(bytemuck::cast_slice(v));
        }
        self.next_id += vectors.len() as u64;
        self.live += vectors.len();
        let mut queue = Vec::new();
        for (chunk_ids, chunk) in ids.chunks(CHUNK).zip(vectors.chunks(CHUNK)) {
            self.epoch += 1;
            self.created.clear();
            let me = &*self;
            let targets: Vec<Option<u32>> = chunk
                .par_iter()
                .map(|v| me.nearest(v, 1).first().map(|t| t.0))
                .collect();
            for ((&id, v), t) in chunk_ids.iter().zip(chunk).zip(targets) {
                let mut best = t
                    .filter(|&b| self.is_live(b))
                    .map(|b| (b, self.dist.eval(v, &self.block_centroid[b as usize])))
                    .or_else(|| self.nearest(v, 1).first().copied());
                for i in 0..self.created.len() {
                    let c = self.created[i];
                    if self.is_live(c) {
                        let d = self.dist.eval(v, &self.block_centroid[c as usize]);
                        if best.map_or(true, |(_, bd)| d < bd) {
                            best = Some((c, d));
                        }
                    }
                }
                let b = match best {
                    Some((b, _)) => b,
                    None => self.new_posting(v.clone())?,
                };
                let code = self.encode(v, b);
                self.push(b, id, &code);
                if self.len(b) >= self.cfg.max_posting_size {
                    queue.push(b);
                    self.drain(&mut queue)?;
                }
            }
            if self.graph.should_compact() {
                self.compact()?;
            }
        }
        self.compact_if_dirty()?;
        Ok(ids)
    }

    /// Rebuild the centroid graph if any posting was created or freed since the
    /// last rebuild, so routing never depends on the delta layer between calls.
    fn compact_if_dirty(&mut self) -> Result<(), DiskAnnError> {
        let s = self.graph.stats();
        if s.delta_vectors + s.tombstones > 0 {
            self.compact()?;
        }
        Ok(())
    }

    /// Entries of `p` that are closer to one of `cands` than to `p`'s own centroid.
    fn local_moves(&self, p: u32, cands: &[u32]) -> Vec<(u64, u32)> {
        let cp = &self.block_centroid[p as usize];
        self.live_entries(p)
            .iter()
            .filter_map(|(id, _)| {
                let v = self.raw(*id);
                let dp = self.dist.eval(v, cp);
                cands
                    .iter()
                    .filter(|&&t| t != p && self.is_live(t))
                    .map(|&t| (t, self.dist.eval(v, &self.block_centroid[t as usize])))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .filter(|&(_, d)| d < dp)
                    .map(|(t, _)| (*id, t))
            })
            .collect()
    }

    /// Live postings around `p`'s centroid (including `p`).
    fn neighbourhood(&self, p: u32) -> Vec<u32> {
        let c = self.block_centroid[p as usize].clone();
        self.nearest(&c, self.cfg.reassign_neighbors + 2)
            .into_iter()
            .map(|(b, _)| b)
            .collect()
    }

    fn drain(&mut self, queue: &mut Vec<u32>) -> Result<(), DiskAnnError> {
        while let Some(b) = queue.pop() {
            if !self.is_live(b) {
                continue;
            }
            if self.len(b) >= self.cfg.max_posting_size {
                self.split(b, queue)?;
            } else {
                let moves = self.local_moves(b, &self.neighbourhood(b));
                self.relocate(b, &moves, queue)?;
            }
        }
        Ok(())
    }

    /// Move `(id, target)` entries out of `from`; targets at capacity keep the entry in place.
    fn relocate(
        &mut self,
        from: u32,
        moves: &[(u64, u32)],
        queue: &mut Vec<u32>,
    ) -> Result<(), DiskAnnError> {
        if moves.is_empty() {
            return Ok(());
        }
        let (moved, keep): (Vec<_>, Vec<_>) = self
            .live_entries(from)
            .into_iter()
            .partition(|(id, _)| moves.iter().any(|m| m.0 == *id));
        self.rewrite(from, &keep);
        for (id, code) in moved {
            let t = moves.iter().find(|m| m.0 == id).unwrap().1;
            let t = if self.is_live(t) && self.len(t) < 2 * self.cfg.max_posting_size {
                t
            } else {
                if !queue.contains(&from) {
                    queue.insert(0, from);
                }
                if self.is_live(t) {
                    queue.push(t);
                }
                from
            };
            let code = if self.residual() { self.encode_id(id, t) } else { code };
            self.push(t, id, &code);
            if self.len(t) >= self.cfg.max_posting_size {
                queue.push(t);
            }
        }
        Ok(())
    }

    fn split(&mut self, b: u32, queue: &mut Vec<u32>) -> Result<(), DiskAnnError> {
        let entries = self.live_entries(b);
        if entries.len() < 2 {
            self.rewrite(b, &entries);
            return Ok(());
        }
        let vs: Vec<&[f32]> = entries.iter().map(|(id, _)| self.raw(*id)).collect();
        let (c1, c2, side) = kmeans2(&vs, self.dist);
        self.free_posting(b)?;
        let nb = [self.new_posting(c1)?, self.new_posting(c2)?];
        for ((id, code), &s) in entries.iter().zip(&side) {
            let t = nb[s as usize];
            let owned;
            let code: &[u8] = if self.residual() {
                owned = self.encode_id(*id, t);
                &owned
            } else {
                code
            };
            self.push(t, *id, code);
        }
        // NPA (SPFresh): vectors of the two new postings may belong to a nearby
        // centroid, and vectors of nearby postings may now belong to a new one.
        let mut around: Vec<u32> = nb.iter().flat_map(|&p| self.neighbourhood(p)).collect();
        around.sort_unstable();
        around.dedup();
        for &p in &nb {
            let moves = self.local_moves(p, &around);
            self.relocate(p, &moves, queue)?;
        }
        for &q in &around {
            if nb.contains(&q) || !self.is_live(q) {
                continue;
            }
            let moves = self.local_moves(q, &nb);
            self.relocate(q, &moves, queue)?;
        }
        for &p in &nb {
            if self.len(p) >= self.cfg.max_posting_size {
                queue.push(p);
            }
        }
        Ok(())
    }

    fn gc(&mut self) -> Result<(), DiskAnnError> {
        self.epoch += 1;
        let mut queue = Vec::new();
        for b in 0..self.block_cid.len() as u32 {
            if !self.is_live(b) {
                continue;
            }
            let entries = self.live_entries(b);
            if entries.len() < self.cfg.min_posting_size && self.cid_block.len() > 1 {
                self.free_posting(b)?;
                for (id, code) in entries {
                    let v = self.raw(id).to_vec();
                    let t = self
                        .nearest(&v, 1)
                        .first()
                        .map(|t| t.0)
                        .unwrap_or_else(|| *self.cid_block.values().next().unwrap());
                    let code = if self.residual() { self.encode_id(id, t) } else { code };
                    self.push(t, id, &code);
                    if self.len(t) >= self.cfg.max_posting_size {
                        queue.push(t);
                        self.drain(&mut queue)?;
                    }
                }
            } else if entries.len() != self.len(b) {
                self.rewrite(b, &entries);
            }
        }
        self.drain(&mut queue)?;
        self.compact_if_dirty()
    }

    fn compact(&mut self) -> Result<(), DiskAnnError> {
        let live: Vec<u32> = (0..self.block_cid.len() as u32)
            .filter(|&b| self.is_live(b))
            .collect();
        let cents: Vec<Vec<f32>> = live
            .iter()
            .map(|&b| self.block_centroid[b as usize].clone())
            .collect();
        let cfg = IncrementalConfig::default();
        self.graph = if cents.is_empty() {
            IncrementalDiskANN::new_empty(self.dim, self.dist, cfg)
        } else {
            IncrementalDiskANN::build_with_config(
                &cents,
                &format!("{}.centroids.base", self.path),
                cfg,
            )?
        };
        self.cid_block.clear();
        for (i, &b) in live.iter().enumerate() {
            self.block_cid[b as usize] = i as u64;
            self.cid_block.insert(i as u64, b);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<(u64, f32)> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );
        let probes = self.nearest(query, n_probe.max(1));
        let Some(&(_, d0)) = probes.first() else {
            return Vec::new();
        };
        let ratio = self.cfg.probe_ratio;
        let probes = probes
            .into_iter()
            .take_while(move |&(_, d)| !(ratio.is_finite() && d0 >= 0.0 && d > d0 * ratio.max(1.0)))
            .map(|(b, _)| b);
        let l2 = TypeId::of::<D>() == TypeId::of::<crate::DistL2>();
        if let (Some(QuantizerState::RaBitQ(rq)), true) = (
            &self.quantizer,
            l2 || TypeId::of::<D>() == TypeId::of::<crate::DistL2Sq>(),
        ) {
            return self.search_rabitq(rq, query, k, probes, l2);
        }
        let prep = self.quantizer.as_ref().map(|q| q.prepare(query));
        let rerank = self.quantizer.is_some() && self.cfg.rerank_size > 0;
        let want = if rerank {
            k.max(self.cfg.rerank_size)
        } else {
            k
        };
        let mut heap = BinaryHeap::new();
        for b in probes {
            for i in 0..self.len(b) {
                let (id, code) = self.entry(b, i);
                if self.deleted.contains(&id) {
                    continue;
                }
                let dist = self.code_dist(query, code, &prep, b);
                if heap.len() < want {
                    heap.push(Cand(dist, id));
                } else if dist < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push(Cand(dist, id));
                }
            }
        }
        let mut out: Vec<(u64, f32)> = heap.into_iter().map(|c| (c.1, c.0)).collect();
        if rerank {
            for r in &mut out {
                r.1 = self.dist.eval(query, self.raw(r.0));
            }
        }
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out.truncate(k);
        out
    }

    /// RaBitQ scan with error-bound reranking (paper Alg. 2): a candidate gets an
    /// exact distance only if its lower bound beats the current k-th exact distance.
    fn search_rabitq(
        &self,
        rq: &RaBitQ,
        query: &[f32],
        k: usize,
        probes: impl Iterator<Item = u32>,
        sqrt: bool,
    ) -> Vec<(u64, f32)> {
        let mut heap: BinaryHeap<Cand> = BinaryHeap::new();
        for b in probes {
            let qb = rq.query_with(query, &self.block_centroid[b as usize]);
            for i in 0..self.len(b) {
                let (id, code) = self.entry(b, i);
                if self.deleted.contains(&id) {
                    continue;
                }
                let (est, err) = rq.distance_with_bound(&qb, code);
                let kth = if heap.len() < k {
                    f32::INFINITY
                } else {
                    heap.peek().unwrap().0
                };
                if est - err < kth {
                    let exact = f32::sqeuclidean(query, self.raw(id)).unwrap() as f32;
                    if heap.len() < k {
                        heap.push(Cand(exact, id));
                    } else if exact < kth {
                        heap.pop();
                        heap.push(Cand(exact, id));
                    }
                }
            }
        }
        let mut out: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|c| (c.1, if sqrt { c.0.sqrt() } else { c.0 }))
            .collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }
}

/// SPFresh index: SPANN partitions with in-place inserts/deletes and quantized postings.
pub struct SPFresh<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    inner: RwLock<Inner<D>>,
}

impl<D> SPFresh<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    /// Build at `path` (creates `{path}.spf`, `.postings`, `.vectors`, `.centroids`).
    pub fn build(
        vectors: &[Vec<f32>],
        path: &str,
        cfg: SPFreshConfig,
        quantizer: Option<QuantizerKind>,
    ) -> Result<Self, DiskAnnError> {
        let dim = vectors
            .first()
            .map(|v| v.len())
            .ok_or_else(|| DiskAnnError::IndexError("No vectors provided".into()))?;
        let quantizer = match quantizer {
            None => None,
            Some(QuantizerKind::F16) => Some(QuantizerState::F16(F16Quantizer::new(dim))),
            Some(QuantizerKind::Int8) => Some(QuantizerState::Int8(Int8Quantizer::train(vectors)?)),
            Some(QuantizerKind::PQ(c)) => {
                Some(QuantizerState::PQ(ProductQuantizer::train(vectors, c)?))
            }
            Some(QuantizerKind::RaBitQ) => Some(QuantizerState::RaBitQ(RaBitQ::train(vectors)?)),
        };
        let graph = IncrementalDiskANN::new_empty(dim, D::default(), IncrementalConfig::default());
        let mut inner = Inner::new(path, dim, cfg, quantizer, graph, true)?;
        let k = (vectors.len() / (inner.cfg.max_posting_size / 2).max(1)).max(1);
        for v in vectors.choose_multiple(&mut thread_rng(), k) {
            inner.new_posting(v.clone())?;
        }
        inner.insert(vectors)?;
        let s = Self {
            inner: RwLock::new(inner),
        };
        s.save()?;
        Ok(s)
    }

    pub fn open(path: &str) -> Result<Self, DiskAnnError> {
        let meta: Meta = bincode::deserialize(&std::fs::read(format!("{path}.spf"))?)?;
        let graph = IncrementalDiskANN::from_bytes(
            &std::fs::read(format!("{path}.centroids"))?,
            D::default(),
            IncrementalConfig::default(),
        )?;
        let mut inner = Inner::new(path, meta.dim, meta.cfg, meta.quantizer, graph, false)?;
        inner.cid_block = meta
            .block_cid
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != FREE)
            .map(|(b, &c)| (c, b as u32))
            .collect();
        inner.free = meta
            .block_cid
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == FREE)
            .map(|(b, _)| (b as u32, 0))
            .collect();
        inner.block_cid = meta.block_cid;
        inner.block_centroid = meta.block_centroid;
        inner.next_id = meta.next_id;
        inner.deleted = meta.deleted.into_iter().collect();
        inner.live = meta.live;
        Ok(Self {
            inner: RwLock::new(inner),
        })
    }

    /// Flush postings/vectors and write metadata + centroid graph.
    pub fn save(&self) -> Result<(), DiskAnnError> {
        let g = self.inner.read().unwrap();
        g.postings.flush()?;
        g.raw.flush()?;
        std::fs::write(format!("{}.centroids", g.path), g.graph.to_bytes())?;
        std::fs::write(format!("{}.spf", g.path), bincode::serialize(&g.meta())?)?;
        Ok(())
    }

    /// Insert vectors in place; returns their ids.
    pub fn insert(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>, DiskAnnError> {
        self.inner.write().unwrap().insert(vectors)
    }

    /// Tombstone ids; entries are purged on the next rewrite or `gc()`.
    pub fn delete(&self, ids: &[u64]) {
        let mut g = self.inner.write().unwrap();
        for &id in ids {
            if id < g.next_id && g.deleted.insert(id) {
                g.live -= 1;
            }
        }
    }

    pub fn is_deleted(&self, id: u64) -> bool {
        self.inner.read().unwrap().deleted.contains(&id)
    }

    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        let g = self.inner.read().unwrap();
        (id < g.next_id && !g.deleted.contains(&id)).then(|| g.raw(id).to_vec())
    }

    /// Scan the `n_probe` nearest postings; distances are quantized estimates unless re-ranked.
    pub fn search_with_dists(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<(u64, f32)> {
        self.inner.read().unwrap().search(query, k, n_probe)
    }

    pub fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<u64> {
        self.search_with_dists(query, k, n_probe)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, n_probe: usize) -> Vec<Vec<u64>> {
        let g = self.inner.read().unwrap();
        queries
            .par_iter()
            .map(|q| {
                g.search(q, k, n_probe)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect()
    }

    /// Purge tombstones and merge postings below `min_posting_size`.
    pub fn gc(&self) -> Result<(), DiskAnnError> {
        self.inner.write().unwrap().gc()
    }

    /// Rebuild the centroid graph from scratch (after many splits/merges).
    pub fn compact(&self) -> Result<(), DiskAnnError> {
        self.inner.write().unwrap().compact()
    }

    pub fn dim(&self) -> usize {
        self.inner.read().unwrap().dim
    }

    pub fn stats(&self) -> SPFreshStats {
        let g = self.inner.read().unwrap();
        let sizes = (0..g.block_cid.len() as u32)
            .filter(|&b| g.is_live(b))
            .map(|b| g.len(b));
        SPFreshStats {
            live: g.live,
            deleted: g.deleted.len(),
            postings: g.cid_block.len(),
            max_posting: sizes.max().unwrap_or(0),
        }
    }
}

#[cfg(feature = "object-store")]
mod remote {
    use super::*;
    use object_store::{path::Path, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
    use std::io::{Seek, SeekFrom, Write};

    fn oe(e: object_store::Error) -> DiskAnnError {
        DiskAnnError::IndexError(e.to_string())
    }
    async fn put(store: &dyn ObjectStore, key: String, bytes: Vec<u8>) -> Result<(), DiskAnnError> {
        store
            .put(&Path::from(key), PutPayload::from(bytes))
            .await
            .map(|_| ())
            .map_err(oe)
    }
    async fn get(store: &dyn ObjectStore, key: String) -> Result<Vec<u8>, DiskAnnError> {
        Ok(store
            .get(&Path::from(key))
            .await
            .map_err(oe)?
            .bytes()
            .await
            .map_err(oe)?
            .to_vec())
    }

    impl<D> SPFresh<D>
    where
        D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
    {
        /// Upload changed posting blocks, new raw-vector chunks, the centroid graph and a new
        /// manifest under `{prefix}/`. Layout: `v{n}/postings/{block}`, `v{n}/vectors/{chunk}`,
        /// `v{n}/centroids`, `manifest-v{n}` (create-only: concurrent publishers fail) and
        /// `manifest` (latest). Data is copied under a read lock, so search/insert keep running.
        pub async fn snapshot(
            &self,
            store: &dyn ObjectStore,
            prefix: &str,
        ) -> Result<Manifest, DiskAnnError> {
            let (mut m, blocks, raw, graph, next_id) = {
                let g = self.inner.read().unwrap();
                let base = g.manifest.as_ref();
                let n = g.block_cid.len();
                let dirty: Vec<(u32, u64)> = match base {
                    None => (0..n as u32)
                        .map(|b| (b, g.dirty.get(&b).copied().unwrap_or(0)))
                        .collect(),
                    Some(_) => g.dirty.iter().map(|(&b, &s)| (b, s)).collect(),
                };
                let blocks: Vec<(u32, u64, Vec<u8>)> = dirty
                    .into_iter()
                    .map(|(b, s)| (b, s, g.blk(b).to_vec()))
                    .collect();
                let synced = if base.is_some() {
                    g.synced_id as usize
                } else {
                    0
                };
                let stride = RAW_CHUNK * g.dim * 4;
                let last = (g.next_id as usize).div_ceil(RAW_CHUNK);
                let raw: Vec<(usize, Vec<u8>)> = (synced / RAW_CHUNK..last)
                    .map(|c| {
                        (
                            c,
                            g.raw[c * stride
                                ..((c + 1) * stride).min(g.next_id as usize * g.dim * 4)]
                                .to_vec(),
                        )
                    })
                    .collect();
                let mut block_ver = base.map_or(Vec::new(), |m| m.block_ver.clone());
                block_ver.resize(n, 0);
                let mut raw_chunk_ver = base.map_or(Vec::new(), |m| m.raw_chunk_ver.clone());
                raw_chunk_ver.resize(last, 0);
                let m = Manifest {
                    version: base.map_or(1, |m| m.version + 1),
                    meta: g.meta(),
                    block_ver,
                    raw_chunk_ver,
                };
                (m, blocks, raw, g.graph.to_bytes(), g.next_id)
            };
            let v = m.version;
            for (b, _, bytes) in &blocks {
                put(store, format!("{prefix}/v{v}/postings/{b}"), bytes.clone()).await?;
                m.block_ver[*b as usize] = v;
            }
            for (c, bytes) in &raw {
                put(store, format!("{prefix}/v{v}/vectors/{c}"), bytes.clone()).await?;
                m.raw_chunk_ver[*c] = v;
            }
            put(store, format!("{prefix}/v{v}/centroids"), graph).await?;
            let bytes = bincode::serialize(&m)?;
            store
                .put_opts(
                    &Path::from(format!("{prefix}/manifest-v{v}")),
                    PutPayload::from(bytes.clone()),
                    PutOptions {
                        mode: PutMode::Create,
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| {
                    DiskAnnError::IndexError(format!("snapshot v{v} already published: {e}"))
                })?;
            put(store, format!("{prefix}/manifest"), bytes.clone()).await?;
            let mut g = self.inner.write().unwrap();
            for (b, s, _) in &blocks {
                if g.dirty.get(b) == Some(s) {
                    g.dirty.remove(b);
                }
            }
            g.synced_id = g.synced_id.max(next_id);
            g.manifest = Some(m.clone());
            if g.dirty.is_empty() {
                std::fs::write(format!("{}.cache", g.path), &bytes)?;
            }
            Ok(m)
        }

        /// Materialise the latest snapshot under `{prefix}/` into `{local}.*`, fetching only the
        /// blocks/chunks whose version differs from `{local}.cache`, then open it.
        pub async fn restore(
            store: &dyn ObjectStore,
            prefix: &str,
            local: &str,
        ) -> Result<Self, DiskAnnError> {
            let bytes = get(store, format!("{prefix}/manifest")).await?;
            let m: Manifest = bincode::deserialize(&bytes)?;
            let cached: Option<Manifest> = std::fs::read(format!("{local}.cache"))
                .ok()
                .and_then(|b| bincode::deserialize(&b).ok());
            let (dim, meta) = (m.meta.dim, &m.meta);
            let block = block_bytes(&meta.cfg, &meta.quantizer, dim);
            let stride = RAW_CHUNK * dim * 4;
            let file = |suffix: &str, len: usize| -> Result<File, DiskAnnError> {
                let f = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(format!("{local}.{suffix}"))?;
                f.set_len(len as u64)?;
                Ok(f)
            };
            let mut postings = file("postings", (m.block_ver.len() * block).max(block))?;
            for (b, &v) in m.block_ver.iter().enumerate() {
                if cached.as_ref().and_then(|c| c.block_ver.get(b)) == Some(&v) {
                    continue;
                }
                let bytes = if v == 0 {
                    vec![0; block]
                } else {
                    get(store, format!("{prefix}/v{v}/postings/{b}")).await?
                };
                postings.seek(SeekFrom::Start((b * block) as u64))?;
                postings.write_all(&bytes)?;
            }
            let mut vectors = file(
                "vectors",
                (meta.next_id as usize * dim * 4).max(1024 * dim * 4),
            )?;
            for (c, &v) in m.raw_chunk_ver.iter().enumerate() {
                if cached.as_ref().and_then(|x| x.raw_chunk_ver.get(c)) == Some(&v) {
                    continue;
                }
                vectors.seek(SeekFrom::Start((c * stride) as u64))?;
                vectors.write_all(&get(store, format!("{prefix}/v{v}/vectors/{c}")).await?)?;
            }
            std::fs::write(
                format!("{local}.centroids"),
                get(store, format!("{prefix}/v{}/centroids", m.version)).await?,
            )?;
            std::fs::write(format!("{local}.spf"), bincode::serialize(&m.meta)?)?;
            std::fs::write(format!("{local}.cache"), &bytes)?;
            let s = Self::open(local)?;
            {
                let mut g = s.inner.write().unwrap();
                g.synced_id = g.next_id;
                g.manifest = Some(m);
            }
            Ok(s)
        }
    }
}

#[cfg(all(test, feature = "object-store"))]
mod remote_tests {
    use super::tests::{cleanup, rc};
    use super::*;
    use crate::DistL2;

    #[tokio::test]
    async fn snapshot_restore_roundtrip() {
        let dir = std::env::temp_dir().join(format!("spf_store_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = object_store::local::LocalFileSystem::new_with_prefix(&dir).unwrap();
        let (src, dst) = ("test_spf_snap_src", "test_spf_snap_dst");
        cleanup(src);
        cleanup(dst);
        let vs = rc(1500, 16, 7);
        let cfg = SPFreshConfig {
            max_posting_size: 32,
            rerank_size: 20,
            ..Default::default()
        };
        let idx = SPFresh::<DistL2>::build(&vs, src, cfg, Some(QuantizerKind::F16)).unwrap();
        assert_eq!(idx.snapshot(&store, "idx").await.unwrap().version, 1);
        idx.insert(&rc(5, 16, 8)).unwrap();
        idx.delete(&[1, 2]);
        let m2 = idx.snapshot(&store, "idx").await.unwrap();
        assert_eq!(m2.version, 2);
        let touched = m2.block_ver.iter().filter(|&&v| v == 2).count();
        assert!(
            touched <= 20 && touched < m2.block_ver.len(),
            "snapshot was not incremental: {touched}"
        );
        assert!(idx.inner.read().unwrap().dirty.is_empty());
        idx.insert(&rc(295, 16, 11)).unwrap();
        assert_eq!(idx.snapshot(&store, "idx").await.unwrap().version, 3);
        let q = vs[3].clone();
        let expect = idx.search_with_dists(&q, 5, 4);
        for _ in 0..2 {
            let r = SPFresh::<DistL2>::restore(&store, "idx", dst)
                .await
                .unwrap();
            assert_eq!(r.search_with_dists(&q, 5, 4), expect);
            assert!(
                r.is_deleted(1)
                    && r.stats().live == 1798
                    && r.stats().postings == idx.stats().postings
            );
        }
        let r = SPFresh::<DistL2>::restore(&store, "idx", dst)
            .await
            .unwrap();
        r.insert(&rc(10, 16, 9)).unwrap();
        assert_eq!(r.snapshot(&store, "idx").await.unwrap().version, 4);
        idx.insert(&rc(10, 16, 10)).unwrap();
        assert!(
            idx.snapshot(&store, "idx").await.is_err(),
            "concurrent publish must fail"
        );
        cleanup(src);
        cleanup(dst);
        let _ = std::fs::remove_file(format!("{dst}.cache"));
        let _ = std::fs::remove_file(format!("{src}.cache"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistL2, PQConfig};

    fn rv(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut r = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| r.r#gen::<f32>()).collect())
            .collect()
    }

    /// 40 Gaussian clusters in [0,1]^dim with sigma 0.05 (embedding-like data).
    pub(super) fn rc(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut r = StdRng::seed_from_u64(seed);
        let mut cr = StdRng::seed_from_u64(99);
        let centers: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..dim).map(|_| cr.r#gen::<f32>()).collect())
            .collect();
        (0..n)
            .map(|i| {
                centers[i % 40]
                    .iter()
                    .map(|c| c + 0.05 * (r.r#gen::<f32>() - 0.5))
                    .collect()
            })
            .collect()
    }

    pub(super) fn cleanup(path: &str) {
        for s in [
            "spf",
            "postings",
            "vectors",
            "centroids",
            "centroids.base",
            "cache",
        ] {
            let _ = std::fs::remove_file(format!("{path}.{s}"));
        }
    }

    fn recall(idx: &SPFresh<DistL2>, vs: &[Vec<f32>], queries: &[Vec<f32>], probe: usize) -> f32 {
        let mut hits = 0;
        for q in queries {
            let mut d: Vec<(u64, f32)> = vs
                .iter()
                .enumerate()
                .filter(|(i, _)| !idx.is_deleted(*i as u64))
                .map(|(i, v)| (i as u64, DistL2.eval(q, v)))
                .collect();
            d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: HashSet<u64> = d.iter().take(10).map(|x| x.0).collect();
            hits += idx
                .search(q, 10, probe)
                .iter()
                .filter(|id| gt.contains(id))
                .count();
        }
        hits as f32 / (10 * queries.len()) as f32
    }

    fn npa_ratio(idx: &SPFresh<DistL2>) -> f32 {
        let g = idx.inner.read().unwrap();
        let (mut ok, mut n) = (0, 0);
        for b in (0..g.block_cid.len() as u32).filter(|&b| g.is_live(b)) {
            for (id, _) in g.live_entries(b) {
                let v = g.raw(id);
                let best = (0..g.block_cid.len() as u32)
                    .filter(|&x| g.is_live(x))
                    .min_by(|&x, &y| {
                        DistL2
                            .eval(v, &g.block_centroid[x as usize])
                            .partial_cmp(&DistL2.eval(v, &g.block_centroid[y as usize]))
                            .unwrap()
                    })
                    .unwrap();
                ok += (best == b) as usize;
                n += 1;
            }
        }
        ok as f32 / n as f32
    }

    #[test]
    fn build_search_recall() {
        let path = "test_spf_build";
        cleanup(path);
        let vs = rc(3000, 32, 1);
        let cfg = SPFreshConfig {
            max_posting_size: 64,
            ..Default::default()
        };
        let idx = SPFresh::<DistL2>::build(&vs, path, cfg, None).unwrap();
        let st = idx.stats();
        assert_eq!(st.live, 3000);
        assert!(
            st.max_posting < 64,
            "postings not split: {}",
            st.max_posting
        );
        assert!(recall(&idx, &vs, &rc(50, 32, 2), 8) >= 0.95);
        assert!(npa_ratio(&idx) >= 0.95);
        cleanup(path);
    }

    #[test]
    fn uniform_data_splits_keep_npa() {
        let path = "test_spf_uniform";
        cleanup(path);
        let vs = rv(3000, 32, 1);
        let cfg = SPFreshConfig {
            max_posting_size: 64,
            reassign_neighbors: 128,
            ..Default::default()
        };
        let idx = SPFresh::<DistL2>::build(&vs, path, cfg, None).unwrap();
        assert!(idx.stats().max_posting < 64);
        assert!(npa_ratio(&idx) >= 0.95);
        cleanup(path);
    }

    #[test]
    fn insert_splits_delete_gc() {
        let path = "test_spf_insert";
        cleanup(path);
        let vs = rc(3000, 32, 3);
        let cfg = SPFreshConfig {
            max_posting_size: 64,
            min_posting_size: 8,
            ..Default::default()
        };
        let idx = SPFresh::<DistL2>::build(&vs[..500], path, cfg, None).unwrap();
        let p0 = idx.stats().postings;
        for chunk in vs[500..].chunks(700) {
            idx.insert(chunk).unwrap();
        }
        assert!(idx.stats().postings > p0);
        assert!(idx.stats().max_posting < 64);
        let qs = rc(50, 32, 4);
        assert!(recall(&idx, &vs, &qs, 8) >= 0.9);
        assert!(npa_ratio(&idx) >= 0.95);
        let del: Vec<u64> = (0..3000).step_by(3).collect();
        idx.delete(&del);
        assert!(idx.is_deleted(0) && !idx.is_deleted(1) && idx.get_vector(0).is_none());
        assert!(idx.search(&vs[0], 10, 8).iter().all(|id| id % 3 != 0));
        assert!(recall(&idx, &vs, &qs, 8) >= 0.9);
        let p1 = idx.stats().postings;
        idx.gc().unwrap();
        let st = idx.stats();
        assert!(st.postings <= p1 && st.live == 2000 && st.max_posting < 64);
        assert!(recall(&idx, &vs, &qs, 8) >= 0.9);
        cleanup(path);
    }

    #[test]
    fn quantized_variants() {
        let vs = rc(2000, 32, 5);
        let qs = rc(30, 32, 6);
        let pq = PQConfig {
            num_subspaces: 8,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };
        for (name, q, min) in [
            ("f16", QuantizerKind::F16, 0.9),
            ("int8", QuantizerKind::Int8, 0.9),
            ("pq", QuantizerKind::PQ(pq), 0.6),
            ("rabitq", QuantizerKind::RaBitQ, 0.6),
        ] {
            let path = format!("test_spf_{name}");
            cleanup(&path);
            let cfg = SPFreshConfig {
                max_posting_size: 64,
                rerank_size: 40,
                ..Default::default()
            };
            let idx = SPFresh::<DistL2>::build(&vs, &path, cfg, Some(q)).unwrap();
            let r = recall(&idx, &vs, &qs, 8);
            assert!(r >= min, "{name}: recall {r}");
            let (id, d) = idx.search_with_dists(&vs[7], 1, 8)[0];
            assert_eq!(id, 7);
            assert!(d.abs() < 1e-3);
            cleanup(&path);
        }
    }

    #[test]
    fn save_open_compact() {
        let path = "test_spf_persist";
        cleanup(path);
        let vs = rc(1500, 16, 7);
        let cfg = SPFreshConfig {
            max_posting_size: 32,
            rerank_size: 20,
            ..Default::default()
        };
        let idx = SPFresh::<DistL2>::build(&vs, path, cfg, Some(QuantizerKind::F16)).unwrap();
        idx.delete(&[1, 2]);
        let before = idx.search_with_dists(&vs[3], 5, 4);
        idx.save().unwrap();
        drop(idx);
        let idx = SPFresh::<DistL2>::open(path).unwrap();
        assert_eq!(idx.search_with_dists(&vs[3], 5, 4), before);
        assert!(idx.is_deleted(1) && idx.stats().live == 1498);
        idx.compact().unwrap();
        assert_eq!(idx.search_with_dists(&vs[3], 5, 4), before);
        let ids = idx.insert(&rc(300, 16, 8)).unwrap();
        assert_eq!(ids[0], 1500);
        assert_eq!(
            idx.search(&idx.get_vector(ids[5]).unwrap(), 1, 4),
            vec![ids[5]]
        );
        cleanup(path);
    }
}
