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
