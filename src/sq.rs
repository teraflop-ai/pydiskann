use crate::DiskAnnError;
use numkong::{cast, f16, EachScale, Euclidean};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Shared interface for vector quantizers (PQ, F16, Int8).
pub trait VectorQuantizer: Send + Sync {
    /// Encode a float vector into compressed bytes.
    fn encode(&self, vector: &[f32]) -> Vec<u8>;

    /// Decode compressed bytes back to an approximate float vector.
    fn decode(&self, codes: &[u8]) -> Vec<f32>;

    /// Compute asymmetric distance: exact query vs quantized database vector.
    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32;

    /// Compression ratio for a given dimension (original bytes / compressed bytes).
    fn compression_ratio(&self, dim: usize) -> f32;
}

// =============================================================================
// F16 Quantizer
// =============================================================================

pub(crate) fn f16_codes(codes: &[u8]) -> &[f16] {
    let bits: &[u16] = bytemuck::cast_slice(codes);
    unsafe { std::slice::from_raw_parts(bits.as_ptr().cast(), bits.len()) } // f16 is repr(transparent) over u16
}

/// Half-precision (f16) quantizer.
///
/// Each f32 dimension is stored as f16 (2 bytes), giving 2x compression.
/// Uses SIMD-accelerated conversion and fused distance when available.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct F16Quantizer {
    dim: usize,
}

impl F16Quantizer {
    /// Create a new F16 quantizer for vectors of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn prepare(&self, query: &[f32]) -> Vec<f16> {
        let mut q = vec![f16::ZERO; self.dim];
        cast(query, &mut q).expect("Vector dimension mismatch");
        q
    }
    #[inline]
    pub fn distance_f16(&self, query: &[f16], codes: &[u8]) -> f32 {
        f16::sqeuclidean(query, f16_codes(codes)).expect("Code length mismatch")
    }

    /// Save to file.
    pub fn save(&self, path: &str) -> Result<(), DiskAnnError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load from file.
    pub fn load(path: &str) -> Result<Self, DiskAnnError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let q: Self = bincode::deserialize_from(reader)?;
        Ok(q)
    }

    /// Get stats.
    pub fn stats(&self) -> SQStats {
        SQStats {
            kind: "F16".to_string(),
            dim: self.dim,
            code_size_bytes: self.dim * 2,
            compression_ratio: 2.0,
            trained: true, // F16 needs no training
        }
    }
}

impl VectorQuantizer for F16Quantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.prepare(vector)
            .iter()
            .flat_map(|h| h.0.to_le_bytes())
            .collect()
    }
    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        cast(f16_codes(codes), &mut out).expect("Code length mismatch");
        out
    }
    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        self.distance_f16(&self.prepare(query), codes)
    }

    fn compression_ratio(&self, dim: usize) -> f32 {
        (dim * 4) as f32 / (dim * 2) as f32
    }
}

// =============================================================================
// Int8 Quantizer
// =============================================================================

/// Per-dimension affine Int8 quantizer.
///
/// Maps each dimension independently: `u8_val = round((f32_val - offset) / scale * 255)`
/// where `offset = min` and `scale = max - min` per dimension.
///
/// Trained from sample vectors to learn the per-dimension ranges.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Int8Quantizer {
    dim: usize,
    scale: f32,
    offset: f32,
}

impl Int8Quantizer {
    pub fn train(vectors: &[Vec<f32>]) -> Result<Self, DiskAnnError> {
        let dim = vectors
            .first()
            .map(|v| v.len())
            .ok_or_else(|| DiskAnnError::IndexError("No vectors to train on".into()))?;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in vectors {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    v.len()
                )));
            }
            for &x in v {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        let range = hi - lo;
        Ok(Self {
            dim,
            scale: if range.abs() < f32::EPSILON {
                1.0
            } else {
                range / 255.0
            },
            offset: lo,
        })
    }
    pub fn from_params(dim: usize, scale: f32, offset: f32) -> Self {
        Self { dim, scale, offset }
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn offset(&self) -> f32 {
        self.offset
    }
    #[inline]
    pub fn distance_u8(&self, query: &[u8], codes: &[u8]) -> f32 {
        u8::sqeuclidean(query, codes).expect("Code length mismatch") as f32
            * self.scale
            * self.scale
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Save to file.
    pub fn save(&self, path: &str) -> Result<(), DiskAnnError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load from file.
    pub fn load(path: &str) -> Result<Self, DiskAnnError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let q: Self = bincode::deserialize_from(reader)?;
        Ok(q)
    }

    /// Get stats.
    pub fn stats(&self) -> SQStats {
        SQStats {
            kind: "Int8".to_string(),
            dim: self.dim,
            code_size_bytes: self.dim,
            compression_ratio: 4.0,
            trained: true,
        }
    }
}

impl VectorQuantizer for Int8Quantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut scaled = vec![0.0f32; self.dim];
        f32::each_scale(
            vector,
            1.0 / self.scale,
            -self.offset / self.scale,
            &mut scaled,
        )
        .expect("Vector dimension mismatch");
        let mut codes = vec![0u8; self.dim];
        cast(&scaled, &mut codes).unwrap(); // rounds + saturates to [0,255] (verified)
        codes
    }
    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .map(|&c| c as f32 * self.scale + self.offset)
            .collect()
    }
    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        self.distance_u8(&self.encode(query), codes)
    }

    fn compression_ratio(&self, dim: usize) -> f32 {
        (dim * 4) as f32 / dim as f32
    }
}

// =============================================================================
// VectorQuantizer impl for ProductQuantizer
// =============================================================================

impl VectorQuantizer for crate::pq::ProductQuantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.encode(vector)
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        self.decode(codes)
    }

    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        self.asymmetric_distance(query, codes)
    }

    fn compression_ratio(&self, _dim: usize) -> f32 {
        self.stats().compression_ratio
    }
}

// =============================================================================
// Stats
// =============================================================================

/// Statistics for a scalar quantizer.
#[derive(Debug, Clone)]
pub struct SQStats {
    pub kind: String,
    pub dim: usize,
    pub code_size_bytes: usize,
    pub compression_ratio: f32,
    pub trained: bool,
}

impl std::fmt::Display for SQStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} Quantizer Stats:", self.kind)?;
        writeln!(f, "  Dimension: {}", self.dim)?;
        writeln!(f, "  Code size: {} bytes", self.code_size_bytes)?;
        writeln!(f, "  Compression ratio: {:.1}x", self.compression_ratio)?;
        writeln!(f, "  Trained: {}", self.trained)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 10.0 - 5.0).collect())
            .collect()
    }

    // ---- F16 tests ----

    #[test]
    fn test_f16_encode_decode_round_trip() {
        let q = F16Quantizer::new(4);
        let vec = vec![1.0f32, -2.5, 0.0, 3.14];
        let codes = q.encode(&vec);
        assert_eq!(codes.len(), 8); // 4 dims * 2 bytes
        let decoded = q.decode(&codes);
        assert_eq!(decoded.len(), 4);
        for (orig, dec) in vec.iter().zip(&decoded) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_f16_asymmetric_distance() {
        let q = F16Quantizer::new(4);
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let target = vec![5.0f32, 6.0, 7.0, 8.0];
        let codes = q.encode(&target);

        let dist = q.asymmetric_distance(&query, &codes);
        let decoded = q.decode(&codes);
        let expected: f32 = query
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        assert!(
            (dist - expected).abs() < 0.1,
            "dist={dist}, expected={expected}"
        );
    }

    #[test]
    fn test_f16_large_vectors() {
        let q = F16Quantizer::new(128);
        let vectors = random_vectors(100, 128, 42);
        for v in &vectors {
            let codes = q.encode(v);
            let decoded = q.decode(&codes);
            let max_err: f32 = v
                .iter()
                .zip(&decoded)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max_err < 0.05, "Max f16 error too high: {max_err}");
        }
    }

    #[test]
    fn test_f16_save_load() {
        let path = "test_f16q.bin";
        let q = F16Quantizer::new(64);
        q.save(path).unwrap();
        let loaded = F16Quantizer::load(path).unwrap();
        assert_eq!(q.dim(), loaded.dim());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_f16_compression_ratio() {
        let q = F16Quantizer::new(128);
        assert!((q.compression_ratio(128) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_stats() {
        let q = F16Quantizer::new(128);
        let stats = q.stats();
        assert_eq!(stats.dim, 128);
        assert_eq!(stats.code_size_bytes, 256);
        assert!((stats.compression_ratio - 2.0).abs() < 0.01);
    }

    // ---- Int8 tests ----

    #[test]
    fn test_int8_train_encode_decode() {
        let vectors = random_vectors(500, 32, 42);
        let q = Int8Quantizer::train(&vectors).unwrap();

        let original = &vectors[0];
        let codes = q.encode(original);
        assert_eq!(codes.len(), 32);

        let decoded = q.decode(&codes);
        assert_eq!(decoded.len(), 32);

        // Reconstruction error should be small relative to range
        let max_err: f32 = original
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // Each dimension has range ~10 (from -5 to 5), quantized to 256 levels
        // So max error should be ~10/255 ≈ 0.04
        assert!(max_err < 0.1, "Max int8 error too high: {max_err}");
    }

    #[test]
    fn test_int8_asymmetric_distance() {
        let vectors = random_vectors(500, 32, 123);
        let q = Int8Quantizer::train(&vectors).unwrap();

        let query = &vectors[0];
        let target = &vectors[100];
        let codes = q.encode(target);

        let asym_dist = q.asymmetric_distance(query, &codes);
        let decoded = q.decode(&codes);
        let expected: f32 = query
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        // Should be very close since both use same dequantization
        assert!(
            (asym_dist - expected).abs() < 0.01 * expected,
            "asym={asym_dist}, expected={expected}"
        );
    }

    #[test]
    fn test_int8_constant_dimension() {
        // Edge case: a dimension with all same values
        let vectors = vec![
            vec![1.0, 5.0, 5.0],
            vec![2.0, 5.0, 5.0],
            vec![3.0, 5.0, 5.0],
        ];
        let q = Int8Quantizer::train(&vectors).unwrap();
        let codes = q.encode(&vectors[0]);
        let decoded = q.decode(&codes);
        // Constant dim should decode back accurately
        assert!((decoded[1] - 5.0).abs() < 0.1);
        assert!((decoded[2] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_int8_save_load() {
        let path = "test_int8q.bin";
        let vectors = random_vectors(200, 16, 42);
        let q = Int8Quantizer::train(&vectors).unwrap();

        let codes_before = q.encode(&vectors[0]);
        q.save(path).unwrap();

        let loaded = Int8Quantizer::load(path).unwrap();
        let codes_after = loaded.encode(&vectors[0]);

        assert_eq!(codes_before, codes_after);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_int8_compression_ratio() {
        let vectors = random_vectors(100, 128, 42);
        let q = Int8Quantizer::train(&vectors).unwrap();
        assert!((q.compression_ratio(128) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_int8_stats() {
        let vectors = random_vectors(100, 64, 42);
        let q = Int8Quantizer::train(&vectors).unwrap();
        let stats = q.stats();
        assert_eq!(stats.dim, 64);
        assert_eq!(stats.code_size_bytes, 64);
        assert!((stats.compression_ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_int8_preserves_ordering() {
        let vectors = random_vectors(200, 32, 456);
        let q = Int8Quantizer::train(&vectors).unwrap();

        let query = &vectors[0];

        // True distances
        let mut true_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, v)| {
                let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, d)
            })
            .collect();
        true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Quantized distances
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let mut quant_dists: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| (i, q.asymmetric_distance(query, c)))
            .collect();
        quant_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check recall@10
        let true_top10: std::collections::HashSet<_> =
            true_dists.iter().take(10).map(|(i, _)| *i).collect();
        let quant_top10: std::collections::HashSet<_> =
            quant_dists.iter().take(10).map(|(i, _)| *i).collect();
        let recall = true_top10.intersection(&quant_top10).count() as f32 / 10.0;
        assert!(recall >= 0.6, "Int8 recall@10 too low: {recall}");
    }

    // ---- VectorQuantizer trait usage ----

    #[test]
    fn test_trait_object_dispatch() {
        let f16q: Box<dyn VectorQuantizer> = Box::new(F16Quantizer::new(4));
        let vec = vec![1.0f32, 2.0, 3.0, 4.0];
        let codes = f16q.encode(&vec);
        let decoded = f16q.decode(&codes);
        assert_eq!(decoded.len(), 4);

        let vectors = random_vectors(50, 4, 42);
        let int8q: Box<dyn VectorQuantizer> = Box::new(Int8Quantizer::train(&vectors).unwrap());
        let codes2 = int8q.encode(&vec);
        let decoded2 = int8q.decode(&codes2);
        assert_eq!(decoded2.len(), 4);
    }
}
