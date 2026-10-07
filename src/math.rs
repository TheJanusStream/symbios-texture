//! Small shared maths helpers.
//!
//! These existed as private per-module copies (`smoothstep` in eight
//! generator files at once, `cell_hash` in nine), which is how two of them
//! drift apart without anything failing.  One definition each: `smoothstep`
//! is `pub(crate)`, and `cell_hash` is public through its old path,
//! [`crate::noise::cell_hash`], so the public API is unchanged.

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

/// Deterministic integer hash → \[0, 1\].  Drives Voronoi site jitter and
/// per-cell variance for the cell-decomposition generators.
///
/// Pair it with [`CellularSample`](crate::noise::CellularSample)'s
/// `cell_x`/`cell_y` to give every cell its own stable colour, height, or
/// orientation.  Vary `seed` to decorrelate several such attributes on the
/// same lattice.
#[inline]
pub fn cell_hash(a: i64, b: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (a as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (b as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
