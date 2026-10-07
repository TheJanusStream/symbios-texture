//! Helpers the generators' unit tests share.

use crate::generator::TextureMap;

/// The luminance of each texel of a bake's base level, row-major.
pub(crate) fn luminance(map: &TextureMap) -> Vec<f64> {
    let texels = (map.width * map.height) as usize;
    map.albedo[..texels * 4]
        .chunks_exact(4)
        .map(|px| 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64)
        .collect()
}

/// How the step across a square bake's wrap compares with the steps across
/// its other structural boundaries, along U (`along_u`) or V: the mean
/// absolute luminance step from the last column (row) back to the first,
/// over the mean step across each interior boundary `period` texels apart.
///
/// A seamless tile scores about `1`: its wrap is one more boundary of the
/// same kind.  Pass the texels between the generator's own boundaries -
/// columns, courses - as `period`, or `1` for a field with none, so that a
/// tile edge is compared with tile edges and not with smooth interior.
pub(crate) fn wrap_step_ratio(l: &[f64], size: usize, period: usize, along_u: bool) -> f64 {
    let (wrap, inside) = wrap_and_inside_steps(l, size, period, along_u);
    wrap / inside.max(1e-9)
}

/// The two sums [`wrap_step_ratio`] divides, for a test that adds them up
/// over several bakes before it divides - a layout whose boundaries fall at
/// random places scores about `1` only on average.
pub(crate) fn wrap_and_inside_steps(
    l: &[f64],
    size: usize,
    period: usize,
    along_u: bool,
) -> (f64, f64) {
    let at = |a: usize, b: usize| {
        if along_u {
            l[b * size + a]
        } else {
            l[a * size + b]
        }
    };
    let step = |a: usize| {
        (0..size)
            .map(|b| (at(a, b) - at((a + 1) % size, b)).abs())
            .sum::<f64>()
    };
    let wrap = step(size - 1);
    let boundaries: Vec<usize> = (1..size / period).map(|k| k * period - 1).collect();
    let inside = boundaries.iter().map(|&a| step(a)).sum::<f64>() / boundaries.len() as f64;
    (wrap, inside)
}
