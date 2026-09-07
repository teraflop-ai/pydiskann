use numkong::{cap, Angular, Dot, Euclidean};

pub trait Distance<T> {
    fn eval(&self, a: &[T], b: &[T]) -> f32;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistL2;
#[derive(Clone, Copy, Debug, Default)]
pub struct DistL2Sq;
#[derive(Clone, Copy, Debug, Default)]
pub struct DistCosine;
#[derive(Clone, Copy, Debug, Default)]
pub struct DistDot;

impl Distance<f32> for DistL2 {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        f32::euclidean(a, b).expect("dimension mismatch") as f32
    }
}
impl Distance<f32> for DistL2Sq {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        f32::sqeuclidean(a, b).expect("dimension mismatch") as f32
    }
}
impl Distance<f32> for DistCosine {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        f32::angular(a, b).expect("dimension mismatch") as f32
    }
}
impl Distance<f32> for DistDot {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - f32::dot(a, b).expect("dimension mismatch") as f32
    }
}

pub fn simd_info() -> String {
    let caps = numkong::available();
    [
        (cap::HASWELL, "AVX2"),
        (cap::SKYLAKE, "AVX-512"),
        (cap::ICELAKE, "AVX-512 VNNI"),
        (cap::GENOA, "AVX-512 BF16"),
        (cap::SAPPHIRE, "AVX-512 FP16"),
        (cap::SAPPHIREAMX, "AMX"),
        (cap::NEON, "NEON"),
        (cap::NEONHALF, "NEON FP16"),
        (cap::NEONSDOT, "NEON SDOT"),
        (cap::NEONBFDOT, "NEON BFDOT"),
        (cap::SVE, "SVE"),
        (cap::SVE2, "SVE2"),
        (cap::SME, "SME"),
        (cap::RVV, "RVV"),
    ]
    .iter()
    .filter(|(bit, _)| caps & bit != 0)
    .map(|(_, n)| *n)
    .collect::<Vec<_>>()
    .join(", ")
}
