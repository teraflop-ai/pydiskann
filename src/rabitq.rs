use crate::sq::VectorQuantizer;
use crate::DiskAnnError;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

const ROUNDS: usize = 3;
const QBITS: u32 = 4;
/// Error-bound constant from the RaBitQ paper (ε₀).
const EPS0: f32 = 1.9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaBitQ {
    dim: usize,
    padded: usize,
    centroid: Vec<f32>,
    /// ROUNDS × padded, entries ±1/√padded (scale folded in)
    signs: Vec<f32>,
}

/// Rotated, 4-bit quantized query plus estimator constants.
pub struct RaBitQQuery {
    planes: Vec<u64>,
    words: usize,
    nq2: f32,
    s: f32,
    k1: f32,
    k2: f32,
    k3: f32,
}

fn fht(v: &mut [f32]) {
    let mut h = 1;
    while h < v.len() {
        for i in (0..v.len()).step_by(2 * h) {
            for j in i..i + h {
                let (x, y) = (v[j], v[j + h]);
                v[j] = x + y;
                v[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

#[inline(always)]
fn ip_bits(code: &[u8], planes: &[u64]) -> (u32, u32) {
    let w = planes.len() / 4;
    let (mut s, mut c) = (0u32, 0u32);
    for (i, ch) in code.chunks_exact(8).enumerate() {
        let b = u64::from_le_bytes(ch.try_into().unwrap());
        c += b.count_ones();
        s += (b & planes[i]).count_ones()
            + ((b & planes[w + i]).count_ones() << 1)
            + ((b & planes[2 * w + i]).count_ones() << 2)
            + ((b & planes[3 * w + i]).count_ones() << 3);
    }
    (s, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn ip_bits_popcnt(code: &[u8], planes: &[u64]) -> (u32, u32) {
    ip_bits(code, planes)
}

#[inline]
fn ip_bits_dispatch(code: &[u8], planes: &[u64]) -> (u32, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe { ip_bits_popcnt(code, planes) };
        }
    }
    ip_bits(code, planes)
}

impl RaBitQ {
    pub fn train(vectors: &[Vec<f32>]) -> Result<Self, DiskAnnError> {
        let dim = vectors
            .first()
            .map(|v| v.len())
            .ok_or_else(|| DiskAnnError::IndexError("No vectors to train on".into()))?;
        let padded = dim.max(64).next_power_of_two();
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    v.len()
                )));
            }
            centroid.iter_mut().zip(v).for_each(|(c, x)| *c += x);
        }
        centroid.iter_mut().for_each(|c| *c /= vectors.len() as f32);
        let scale = 1.0 / (padded as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(0x7ab1_7);
        let signs = (0..ROUNDS * padded)
            .map(|_| if rng.r#gen::<bool>() { scale } else { -scale })
            .collect();
        Ok(Self {
            dim,
            padded,
            centroid,
            signs,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded
    }

    pub fn code_size(&self) -> usize {
        self.padded / 8 + 8
    }

    fn rotate(&self, v: &mut [f32]) {
        for s in self.signs.chunks_exact(self.padded) {
            v.iter_mut().zip(s).for_each(|(x, s)| *x *= s);
            fht(v);
        }
    }

    fn unrotate(&self, v: &mut [f32]) {
        for s in self.signs.chunks_exact(self.padded).rev() {
            fht(v);
            v.iter_mut().zip(s).for_each(|(x, s)| *x *= s);
        }
    }

    /// Center on `centroid`, normalize, pad, rotate. Returns (o', ‖v - centroid‖).
    fn prep_with(&self, v: &[f32], centroid: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(v.len(), self.dim, "Vector dimension mismatch");
        let mut r = vec![0.0f32; self.padded];
        r.iter_mut()
            .zip(v)
            .zip(centroid)
            .for_each(|((r, x), c)| *r = x - c);
        let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            r.iter_mut().for_each(|x| *x /= norm);
        }
        self.rotate(&mut r);
        (r, norm)
    }

    /// Per-query precomputation (do once, then `distance` per candidate).
    pub fn query(&self, q: &[f32]) -> RaBitQQuery {
        self.query_with(q, &self.centroid)
    }

    /// Like [`query`](Self::query) but centered on `centroid` (residual codes, see [`encode_with`](Self::encode_with)).
    pub fn query_with(&self, q: &[f32], centroid: &[f32]) -> RaBitQQuery {
        let (r, nq) = self.prep_with(q, centroid);
        let (lo, hi) = r
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        let levels = ((1u32 << QBITS) - 1) as f32;
        let delta = (hi - lo) / levels;
        let inv = if delta > 0.0 { 1.0 / delta } else { 0.0 };
        let words = self.padded / 64;
        let mut planes = vec![0u64; 4 * words];
        let mut sum_u = 0u32;
        for (i, &x) in r.iter().enumerate() {
            let u = (((x - lo) * inv).round() as u32).min(levels as u32);
            sum_u += u;
            for j in 0..QBITS as usize {
                if (u >> j) & 1 == 1 {
                    planes[j * words + i / 64] |= 1 << (i % 64);
                }
            }
        }
        let sd = (self.padded as f32).sqrt();
        RaBitQQuery {
            planes,
            words,
            nq2: nq * nq,
            s: 2.0 * nq,
            k1: 2.0 * delta / sd,
            k2: 2.0 * lo / sd,
            k3: -(delta * sum_u as f32 + self.padded as f32 * lo) / sd,
        }
    }

    /// Estimated squared L2 distance between the prepared query and a code.
    #[inline]
    pub fn distance(&self, q: &RaBitQQuery, code: &[u8]) -> f32 {
        let w = 8 * q.words;
        let (s, c) = ip_bits_dispatch(&code[..w], &q.planes);
        let n2 = f32::from_le_bytes(code[w..w + 4].try_into().unwrap());
        let fac = f32::from_le_bytes(code[w + 4..w + 8].try_into().unwrap());
        (n2 + q.nq2 - q.s * fac * (q.k1 * s as f32 + q.k2 * c as f32 + q.k3)).max(0.0)
    }

    /// Estimated squared L2 distance plus its high-probability error radius
    /// (RaBitQ Thm 3.2 with ε₀ = 1.9): the true distance is ≥ `est - err` w.h.p.
    #[inline]
    pub fn distance_with_bound(&self, q: &RaBitQQuery, code: &[u8]) -> (f32, f32) {
        let w = 8 * q.words;
        let (s, c) = ip_bits_dispatch(&code[..w], &q.planes);
        let n2 = f32::from_le_bytes(code[w..w + 4].try_into().unwrap());
        let fac = f32::from_le_bytes(code[w + 4..w + 8].try_into().unwrap());
        let est = (n2 + q.nq2 - q.s * fac * (q.k1 * s as f32 + q.k2 * c as f32 + q.k3)).max(0.0);
        let ip = if fac > 0.0 { n2.sqrt() / fac } else { 0.0 };
        let err = if ip > 0.0 {
            q.s * fac * EPS0 * ((1.0 - ip * ip).max(0.0) / (self.padded as f32 - 1.0)).sqrt()
        } else {
            f32::INFINITY
        };
        (est, err)
    }

    /// Encode `v - centroid` (residual quantization for IVF/SPFresh postings).
    pub fn encode_with(&self, v: &[f32], centroid: &[f32]) -> Vec<u8> {
        let (r, norm) = self.prep_with(v, centroid);
        self.pack(&r, norm)
    }

    /// Decode a code produced by [`encode_with`](Self::encode_with) with the same `centroid`.
    pub fn decode_with(&self, codes: &[u8], centroid: &[f32]) -> Vec<f32> {
        assert_eq!(codes.len(), self.code_size(), "Code length mismatch");
        let w = self.padded / 8;
        let sd = (self.padded as f32).sqrt();
        let mut r: Vec<f32> = codes[..w]
            .chunks_exact(8)
            .flat_map(|ch| {
                let b = u64::from_le_bytes(ch.try_into().unwrap());
                (0..64).map(move |i| if (b >> i) & 1 == 1 { 1.0 / sd } else { -1.0 / sd })
            })
            .collect();
        self.unrotate(&mut r);
        let norm = f32::from_le_bytes(codes[w..w + 4].try_into().unwrap()).sqrt();
        r.iter().zip(centroid).map(|(x, c)| c + norm * x).collect()
    }

    fn pack(&self, r: &[f32], norm: f32) -> Vec<u8> {
        let mut code = Vec::with_capacity(self.code_size());
        let mut ip = 0.0f32;
        for chunk in r.chunks_exact(64) {
            let mut w = 0u64;
            for (i, &x) in chunk.iter().enumerate() {
                w |= ((x > 0.0) as u64) << i;
                ip += x.abs();
            }
            code.extend_from_slice(&w.to_le_bytes());
        }
        ip /= (self.padded as f32).sqrt();
        let fac = if ip > 0.0 { norm / ip } else { 0.0 };
        code.extend_from_slice(&(norm * norm).to_le_bytes());
        code.extend_from_slice(&fac.to_le_bytes());
        code
    }
}

impl VectorQuantizer for RaBitQ {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.encode_with(vector, &self.centroid)
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        self.decode_with(codes, &self.centroid)
    }

    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        self.distance(&self.query(query), codes)
    }

    fn compression_ratio(&self, dim: usize) -> f32 {
        (dim * 4) as f32 / self.code_size() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn test_rotation_is_orthogonal() {
        let vs = random_vectors(4, 100, 1);
        let q = RaBitQ::train(&vs).unwrap();
        assert_eq!(q.padded_dim(), 128);
        let mut v = vec![0.0f32; 128];
        v[..100].copy_from_slice(&vs[0]);
        let n0: f32 = v.iter().map(|x| x * x).sum();
        let orig = v.clone();
        q.rotate(&mut v);
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3 * n0);
        q.unrotate(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_self_distance_near_zero_and_table_matches() {
        let vs = random_vectors(200, 128, 2);
        let q = RaBitQ::train(&vs).unwrap();
        assert_eq!(q.code_size(), 24);
        let codes: Vec<Vec<u8>> = vs.iter().map(|v| q.encode(v)).collect();
        let pq = q.query(&vs[0]);
        let scale = l2(&vs[0], &vs[1]);
        assert!(q.distance(&pq, &codes[0]).abs() < 0.05 * scale);
        for (v, c) in vs.iter().zip(&codes).take(20) {
            assert_eq!(q.distance(&pq, c), q.asymmetric_distance(&vs[0], c));
            let est = q.distance(&pq, c);
            let exact = l2(&vs[0], v);
            assert!(
                (est - exact).abs() < 0.35 * exact + 0.02 * scale,
                "est={est} exact={exact}"
            );
        }
    }

    #[test]
    fn test_decode_and_ordering() {
        let vs = random_vectors(500, 64, 3);
        let q = RaBitQ::train(&vs).unwrap();
        let codes: Vec<Vec<u8>> = vs.iter().map(|v| q.encode(v)).collect();
        let d = q.decode(&codes[0]);
        assert_eq!(d.len(), 64);
        assert!(l2(&vs[0], &d) < l2(&vs[0], &vs[1]));

        let query = &vs[0];
        let mut exact: Vec<(usize, f32)> = vs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let pq = q.query(query);
        let mut est: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| (i, q.distance(&pq, c)))
            .collect();
        est.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top: std::collections::HashSet<usize> = exact.iter().take(10).map(|x| x.0).collect();
        let hits = est.iter().take(30).filter(|x| top.contains(&x.0)).count();
        assert!(hits >= 7, "recall@10 in top-30 too low: {hits}/10");
    }
}
