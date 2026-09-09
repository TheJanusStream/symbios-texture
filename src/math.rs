//! Small shared maths helpers.
//!
//! These existed as private per-module copies — `smoothstep` in eight
//! generator files at once — which is how two of them drift apart without
//! anything failing.  One definition, `pub(crate)` so the public API is
//! unchanged.

/// Hermite interpolation between two edges: `0` below `edge0`, `1` above
/// `edge1`, and a smooth `3t² − 2t³` ramp between them.
///
/// A degenerate or inverted pair (`edge1 <= edge0`) has no ramp to take, so it
/// falls back to a hard step at `edge1`.
#[inline]
pub(crate) fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Deterministic integer cell hash → `[0, 1]`: the same mix the paver and
/// stained-glass cells use, for callers that need a per-cell random value
/// on a torus.
#[inline]
pub(crate) fn cell_hash(a: i64, b: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (a as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (b as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
