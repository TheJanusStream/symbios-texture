//! Bake-time de-repetition: histogram-preserving hex-tile blending.
//!
//! # What it does
//!
//! A tileable texture laid across hundreds of metres of terrain repeats at
//! its tile period, and the eye locks onto whatever large structure the tile
//! carries — a dark patch, a distinctive stone — marching across the ground
//! in step.  This operator rebuilds a tile from an *example* so that nothing
//! wider than a chosen cell survives.  The output is covered by a hexagonal
//! lattice; each vertex shows the example through its own random window;
//! every texel is a blend of the three vertices around it.  Features finer
//! than a cell come through at full contrast, in one window or another.
//! Anything wider than a cell is averaged across independent windows and
//! fades.  The tile still repeats — it is a tile — but with its landmarks
//! broken up there is far less for the eye to lock onto: on a coarse rock
//! laid over the same ground, the plain tile's repeat is obvious and the
//! same-size blend's is faint at best (the contrast of structure wider than
//! two cells drops by half, measured).  To *remove* the repeat within an
//! extent rather than soften it, ask for a larger output than the example: a
//! 1024² tile from a 768² example doubles the period in metres, at four
//! times the texture memory, because every cell draws its own window and the
//! output never repeats internally at any size.
//!
//! # Structured textures
//!
//! This is an operator for stochastic materials — rock, gravel, sand,
//! ground, anything a splat is made of.  A texture whose meaning is in its
//! structure, a cobble or a paver bond, ghosts at the default weights: the
//! three windows show three different stone layouts and the blend zones
//! overlap them translucently.  A [`blend_exponent`](HexBlendConfig::blend_exponent)
//! around 8 narrows the zones to seams and keeps the stones whole; whether
//! the seams read is the material's call.
//!
//! # Why blending does not go grey
//!
//! A plain weighted average of three samples halves the variance in the
//! blend zones and the result looks washed out.  Following Heitz & Neyret
//! (*High-Performance By-Example Noise using a Histogram-Preserving Blending
//! Operator*, 2018), each channel is first mapped to a Gaussian distribution
//! through its own histogram, the blend is taken there with the
//! variance-preserving weights `w / sqrt(Σ w²)` after mean subtraction, and
//! the inverse map takes it back — so every channel of the output has the
//! example's histogram, and the same contrast everywhere.  That guarantee is
//! as good as the Gaussian fit: a channel with a continuous histogram (rock,
//! gravel, sand — anything a splat is made of) comes back within a couple of
//! percent, while a flat-shaded tile with a dozen distinct values per channel
//! has its mass redistributed among those values, because a blend of three
//! atoms lands between them.  Measured in the tests.  The exponent on
//! the weights is Mikkelsen's (*Practical Real-Time Hex-Tiling*, 2022): above
//! `1` it narrows the blend zones so each cell reads more as a whole window.
//!
//! # Tiling
//!
//! The lattice is the hexagonal paver layout's even-column construction
//! (the crate-private `HexLattice`), so it tiles the unit square exactly;
//! each vertex's
//! window is hashed from its canonical id on the torus; and every tap is
//! taken relative to its *vertex*, so a texel one tile over sees the same
//! three windows at the same weights, bit for bit.  The example has to tile
//! too — every surface generator's output does — because the windows wrap
//! through it.
//!
//! # The oversized example
//!
//! The example may be larger than the output — a 768² scratch for a 512²
//! tile, generated with its feature count scaled to match — and is sampled
//! at native texel density, so each window shows a different two thirds of
//! it.  More example, more distinct windows.  A same-size example works too;
//! there is simply less content to draw from.
//!
//! # Cost
//!
//! Bake time only.  Per channel plane: a 256-entry histogram map, one
//! Gaussianised copy of the example in `f32`, and then, per output texel,
//! three lookups and one inverse lookup, rows in parallel.  Measured
//! numbers are in the CHANGELOG for 0.7.0.

use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureMap, validate_dimensions},
    hex_lattice::HexLattice,
    math::cell_hash,
};

/// Configures [`hex_blend`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HexBlendConfig {
    /// Seed for the per-vertex windows into the example.
    pub seed: u32,
    /// Hexagonal cells down the output tile, rounded to a row count of at
    /// least one; the column count follows from the lattice.  A cell is the
    /// widest structure that survives at full strength — anything wider is
    /// broken up across windows — so set it a little above the size of the
    /// features that should stay whole.
    pub cells: f64,
    /// Exponent on the barycentric weights before normalisation.  `1` is the
    /// plain histogram-preserving blend, right for stochastic materials;
    /// higher values narrow the blend zones, and around `8` a structured
    /// texture keeps its stones whole with narrow seams instead of ghosting.
    pub blend_exponent: f64,
}

impl Default for HexBlendConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            cells: 8.0,
            blend_exponent: 1.0,
        }
    }
}

/// Rebuild `example` as a `width × height` tile with no structure wider than
/// a cell — see the [module docs](self).
///
/// Every plane the example carries is blended the same way: albedo, normal
/// (renormalised afterwards), ORM, and emissive if present.  Only the
/// example's base mip level is read; the result has one level, so run
/// [`TextureMap::with_mips`] on the result, not the example.
///
/// # Errors
///
/// Either size outside the crate's dimension envelope.
pub fn hex_blend(
    example: &TextureMap,
    config: &HexBlendConfig,
    width: u32,
    height: u32,
) -> Result<TextureMap, TextureError> {
    validate_dimensions(width, height)?;
    validate_dimensions(example.width, example.height)?;
    let (sw, sh) = (example.width as usize, example.height as usize);
    let (w, h) = (width as usize, height as usize);
    let base = sw * sh * 4;
    let lat = HexLattice::new(config.cells);

    let plane = |bytes: &[u8]| blend_plane(&bytes[..base], sw, sh, w, h, &lat, config);
    let albedo = plane(&example.albedo);
    let mut normal = plane(&example.normal);
    renormalise(&mut normal);
    let roughness = plane(&example.roughness);
    let emissive = example.emissive.as_deref().map(plane);

    Ok(TextureMap {
        albedo,
        normal,
        roughness,
        emissive,
        width,
        height,
        mip_level_count: 1,
    })
}

/// One output texel's view of the example: three windows and their weights.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frame {
    /// Example-space UV of each tap, wrapped into `[0, 1)`.
    taps: [(f64, f64); 3],
    /// Variance-preserving weights: the barycentric weights, raised to the
    /// blend exponent and renormalised, then divided by the root of their sum
    /// of squares.
    weights: [f64; 3],
}

/// The frame at output UV `(u, v)`.
///
/// `ratio` is `(width / example_width, height / example_height)`: the example
/// is sampled at its native texel density, so a smaller output sees a
/// proportionally smaller window of it.
fn frame_at(
    lat: &HexLattice,
    seed: u32,
    exponent: f64,
    ratio: (f64, f64),
    u: f64,
    v: f64,
) -> Frame {
    let (qf, rf) = lat.axial(u, v);
    let (verts, bary) = lat.triangle(qf, rf);
    let mut taps = [(0.0, 0.0); 3];
    let mut weights = [0.0; 3];
    for i in 0..3 {
        let (q, r) = verts[i];
        let (cq, cr) = lat.canonical(q, r);
        // This vertex's window into the example.
        let ou = cell_hash(cq, cr, seed);
        let ov = cell_hash(cq, cr, seed.wrapping_add(0x9e37_79b9));
        // The texel's offset from its vertex, as axial residues — exact under
        // a whole-tile step — then in UV.
        let (dq, dr) = (qf - q as f64, rf - r as f64);
        let du = dq / lat.cols;
        let dv = (0.5 * dq + dr) / lat.rows;
        taps[i] = (
            (ou + du * ratio.0).rem_euclid(1.0),
            (ov + dv * ratio.1).rem_euclid(1.0),
        );
        weights[i] = if exponent == 1.0 {
            bary[i]
        } else {
            bary[i].max(0.0).powf(exponent)
        };
    }
    if exponent != 1.0 {
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
    }
    let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
    if norm > 0.0 {
        for w in &mut weights {
            *w /= norm;
        }
    }
    Frame { taps, weights }
}

/// Resolution of the inverse map from the Gaussian domain back to bytes.
const BINS: usize = 4096;

/// A channel's histogram, as a map to a Gaussian of mean `0.5` and standard
/// deviation `1/6`, and the way back.
struct ChannelLut {
    to_gauss: [f32; 256],
    from_gauss: Vec<u8>,
}

impl ChannelLut {
    fn build(plane: &[u8], channel: usize) -> Self {
        let mut hist = [0u64; 256];
        for px in plane.chunks_exact(4) {
            hist[usize::from(px[channel])] += 1;
        }
        let n = (plane.len() / 4) as f64;
        let mut cdf = [0.0f64; 256];
        let mut acc = 0u64;
        for (v, c) in cdf.iter_mut().enumerate() {
            acc += hist[v];
            *c = acc as f64 / n;
        }
        // Each byte maps to the Gaussian value at the middle of its own
        // probability mass, so a byte survives the round trip unchanged.
        let mut to_gauss = [0.0f32; 256];
        for (v, g) in to_gauss.iter_mut().enumerate() {
            let p = (cdf[v] - hist[v] as f64 / (2.0 * n)).clamp(1e-6, 1.0 - 1e-6);
            *g = (0.5 + inv_phi(p) / 6.0) as f32;
        }
        let from_gauss = (0..BINS)
            .map(|b| {
                let g = (b as f64 + 0.5) / BINS as f64;
                let p = phi((g - 0.5) * 6.0);
                cdf.partition_point(|&c| c < p).min(255) as u8
            })
            .collect();
        Self {
            to_gauss,
            from_gauss,
        }
    }

    #[inline]
    fn forward(&self, byte: u8) -> f32 {
        self.to_gauss[usize::from(byte)]
    }

    #[inline]
    fn inverse(&self, g: f32) -> u8 {
        let b = (f64::from(g.clamp(0.0, 1.0)) * BINS as f64) as usize;
        self.from_gauss[b.min(BINS - 1)]
    }
}

/// Error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

/// Standard normal CDF.
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal quantile, by bisection — only ever called 256 times per
/// channel, at LUT build.
fn inv_phi(p: f64) -> f64 {
    let (mut lo, mut hi) = (-8.0f64, 8.0f64);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if phi(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The texel of a four-channel `f32` plane under texel coordinates,
/// wrapping.
///
/// Nearest, not bilinear, on purpose.  The example is sampled at native
/// density — every tap steps exactly one example texel per output texel, in
/// every window, whatever the two sizes are — so within a cell a nearest
/// lookup is a pure translation of the example and loses nothing.  A
/// bilinear lookup at the same positions would only interpolate at a fixed
/// sub-texel phase per cell: a blur, which costs variance the histogram
/// then has to give back.
#[inline]
fn nearest(g: &[f32], sw: usize, sh: usize, tx: f64, ty: f64) -> [f32; 4] {
    let x = (tx.floor() as i64).rem_euclid(sw as i64) as usize;
    let y = (ty.floor() as i64).rem_euclid(sh as i64) as usize;
    let i = (y * sw + x) * 4;
    [g[i], g[i + 1], g[i + 2], g[i + 3]]
}

/// Blend one four-channel byte plane of the example into a `w × h` plane.
fn blend_plane(
    example: &[u8],
    sw: usize,
    sh: usize,
    w: usize,
    h: usize,
    lat: &HexLattice,
    cfg: &HexBlendConfig,
) -> Vec<u8> {
    let luts: [ChannelLut; 4] = std::array::from_fn(|c| ChannelLut::build(example, c));
    let gauss: Vec<f32> = example
        .chunks_exact(4)
        .flat_map(|px| {
            [
                luts[0].forward(px[0]),
                luts[1].forward(px[1]),
                luts[2].forward(px[2]),
                luts[3].forward(px[3]),
            ]
        })
        .collect();
    let ratio = (w as f64 / sw as f64, h as f64 / sh as f64);

    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        let v = y as f64 / h as f64;
        for (x, px) in row.chunks_exact_mut(4).enumerate() {
            let u = x as f64 / w as f64;
            let f = frame_at(lat, cfg.seed, cfg.blend_exponent, ratio, u, v);
            let mut acc = [0.5f32; 4];
            for (tap, weight) in f.taps.iter().zip(f.weights) {
                let s = nearest(&gauss, sw, sh, tap.0 * sw as f64, tap.1 * sh as f64);
                let weight = weight as f32;
                for (a, s) in acc.iter_mut().zip(s) {
                    *a += weight * (s - 0.5);
                }
            }
            for (c, slot) in px.iter_mut().enumerate() {
                *slot = luts[c].inverse(acc[c]);
            }
        }
    });
    out
}

/// Blending unit normals does not give unit normals; put them back on the
/// sphere.  Alpha is left alone.
fn renormalise(normal: &mut [u8]) {
    for px in normal.chunks_exact_mut(4) {
        let n = [px[0], px[1], px[2]].map(|b| f32::from(b) / 255.0 * 2.0 - 1.0);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-6 {
            for (slot, c) in px.iter_mut().zip(n) {
                *slot = ((c / len * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::TextureGenerator;
    use crate::pavers::{PaversConfig, PaversGenerator, PaversLayout};

    /// Three rows of hexagonal pavers with loud colour jitter: flat patches a
    /// third of the tile wide, which is exactly the structure the operator
    /// exists to break up.
    fn patchy(size: u32) -> TextureMap {
        PaversGenerator::new(PaversConfig {
            scale: 3.0,
            layout: PaversLayout::Hexagonal,
            cell_variance: 0.8,
            roughness: 0.0,
            ..Default::default()
        })
        .generate(size, size)
        .expect("generate")
    }

    fn luminance(map: &TextureMap) -> Vec<f64> {
        map.albedo
            .chunks_exact(4)
            .map(|p| 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
            .collect()
    }

    fn std_dev(v: &[f64]) -> f64 {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Separable wrapping box blur of half-width `r`.
    fn box_blur(l: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
        let k = (2 * r + 1) as f64;
        let mut tmp = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                tmp[y * w + x] = (0..=2 * r)
                    .map(|d| l[y * w + (x + w + d - r) % w])
                    .sum::<f64>()
                    / k;
            }
        }
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = (0..=2 * r)
                    .map(|d| tmp[((y + h + d - r) % h) * w + x])
                    .sum::<f64>()
                    / k;
            }
        }
        out
    }

    fn cdf(bytes: impl Iterator<Item = u8>) -> [f64; 256] {
        let mut h = [0.0f64; 256];
        let mut n = 0.0;
        for b in bytes {
            h[usize::from(b)] += 1.0;
            n += 1.0;
        }
        let mut acc = 0.0;
        for c in &mut h {
            acc += *c;
            *c = acc / n;
        }
        h
    }

    const SWEEP: u32 = 2048;

    /// The seam identities, bit for bit: `u = 0` and `u = 1` are the same
    /// point of the pattern one tile apart, and so are `v = 0` and `v = 1`.
    #[test]
    fn the_frame_is_periodic_in_u_and_v() {
        let ratio = (2.0 / 3.0, 2.0 / 3.0);
        for cells in 1..=16 {
            let lat = HexLattice::new(f64::from(cells));
            for i in 0..SWEEP {
                let t = (f64::from(i) + 0.5) / f64::from(SWEEP);
                assert_eq!(
                    frame_at(&lat, 7, 1.0, ratio, 0.0, t),
                    frame_at(&lat, 7, 1.0, ratio, 1.0, t),
                    "cells {cells}: the frame steps across the U seam at v = {t}",
                );
                assert_eq!(
                    frame_at(&lat, 7, 1.0, ratio, t, 0.0),
                    frame_at(&lat, 7, 1.0, ratio, t, 1.0),
                    "cells {cells}: the frame steps across the V seam at u = {t}",
                );
            }
        }
    }

    /// Periodicity as a property of the whole function, not just of the two
    /// seam lines.
    #[test]
    fn shifting_by_a_whole_tile_changes_nothing() {
        let ratio = (0.75, 0.5);
        for cells in 1..=12 {
            let lat = HexLattice::new(f64::from(cells));
            for i in 0..97 {
                let u = f64::from(i) * 0.009_137;
                for j in 0..31 {
                    let v = (f64::from(j) + 0.5) / 31.0;
                    let a = frame_at(&lat, 3, 2.0, ratio, u, v);
                    let b = frame_at(&lat, 3, 2.0, ratio, u + 1.0, v);
                    for k in 0..3 {
                        assert!(
                            (a.taps[k].0 - b.taps[k].0).abs() < 1e-12
                                && (a.taps[k].1 - b.taps[k].1).abs() < 1e-12
                                && (a.weights[k] - b.weights[k]).abs() < 1e-12,
                            "cells {cells}: frame at ({u}, {v}) is not the frame one tile over",
                        );
                    }
                }
            }
        }
    }

    /// The output tiles: the join between its last and first column is no
    /// larger a step than any interior pair of columns, and the same for
    /// rows.
    #[test]
    fn the_output_tiles() {
        let example = patchy(96);
        let out = hex_blend(&example, &HexBlendConfig::default(), 64, 64).expect("blend");
        let (w, h) = (64usize, 64usize);
        let l = luminance(&out);
        let col_step = |a: usize, b: usize| {
            (0..h)
                .map(|y| (l[y * w + a] - l[y * w + b]).abs())
                .sum::<f64>()
                / h as f64
        };
        let row_step = |a: usize, b: usize| {
            (0..w)
                .map(|x| (l[a * w + x] - l[b * w + x]).abs())
                .sum::<f64>()
                / w as f64
        };
        let interior_cols = (0..w - 1).map(|x| col_step(x, x + 1)).sum::<f64>() / (w - 1) as f64;
        let interior_rows = (0..h - 1).map(|y| row_step(y, y + 1)).sum::<f64>() / (h - 1) as f64;
        let seam_col = col_step(w - 1, 0);
        let seam_row = row_step(h - 1, 0);
        assert!(
            seam_col < 1.5 * interior_cols,
            "the U seam steps by {seam_col:.2} against an interior {interior_cols:.2}",
        );
        assert!(
            seam_row < 1.5 * interior_rows,
            "the V seam steps by {seam_row:.2} against an interior {interior_rows:.2}",
        );
    }

    /// Every channel of the output has the example's histogram — the
    /// property that keeps the blend from going grey.  Exact to the Gaussian
    /// fit: a continuous histogram (rock) comes back tight; the patchy paver
    /// tile, with a dozen distinct values per channel, is the documented
    /// limit and is held to a looser bound so the number is on record.
    #[test]
    fn the_histogram_is_preserved_per_channel() {
        use crate::rock::{RockConfig, RockGenerator};
        let rock = RockGenerator::new(RockConfig::default())
            .generate(96, 96)
            .expect("generate");
        for (name, example, bound) in [("rock", rock, 0.02), ("patchy", patchy(96), 0.12)] {
            let out = hex_blend(&example, &HexBlendConfig::default(), 96, 96).expect("blend");
            for c in 0..3 {
                let a = cdf(example.albedo.chunks_exact(4).map(|p| p[c]));
                let b = cdf(out.albedo.chunks_exact(4).map(|p| p[c]));
                let worst = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f64, f64::max);
                eprintln!("{name} channel {c}: worst CDF difference {worst:.4}");
                assert!(
                    worst < bound,
                    "{name} channel {c}: the CDFs differ by {worst:.3} somewhere (bound {bound})",
                );
            }
        }
    }

    /// What the operator is for: structure wider than a cell fades, while
    /// the total contrast — and so the detail — is kept.
    #[test]
    fn structure_wider_than_a_cell_fades_while_contrast_survives() {
        let example = patchy(96);
        let out = hex_blend(&example, &HexBlendConfig::default(), 96, 96).expect("blend");
        let (a, b) = (luminance(&example), luminance(&out));
        // A blur wider than two cells (cells = 8 → a cell is 12 texels).
        let (la, lb) = (box_blur(&a, 96, 96, 14), box_blur(&b, 96, 96, 14));
        let low = std_dev(&lb) / std_dev(&la);
        let total = std_dev(&b) / std_dev(&a);
        eprintln!("low-pass contrast ratio {low:.3}, total contrast ratio {total:.3}");
        assert!(
            low < 0.7,
            "large structure survived: low-pass contrast ratio {low:.2}",
        );
        assert!(
            (0.85..=1.15).contains(&total),
            "contrast was not preserved: total ratio {total:.2}",
        );
    }

    /// A channel the example holds constant comes out constant — alpha at
    /// 255, and the metallic channel of a dielectric at 0.
    #[test]
    fn a_constant_channel_passes_through() {
        let example = patchy(64);
        let out = hex_blend(&example, &HexBlendConfig::default(), 64, 64).expect("blend");
        assert!(out.albedo.chunks_exact(4).all(|p| p[3] == 255));
        assert!(example.roughness.chunks_exact(4).all(|p| p[2] == 0));
        assert!(out.roughness.chunks_exact(4).all(|p| p[2] == 0));
    }

    #[test]
    fn normals_stay_unit_length() {
        let example = patchy(64);
        let out = hex_blend(&example, &HexBlendConfig::default(), 64, 64).expect("blend");
        for p in out.normal.chunks_exact(4) {
            let n = [p[0], p[1], p[2]].map(|b| f64::from(b) / 255.0 * 2.0 - 1.0);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 2.5 / 255.0, "|n| = {len}");
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn the_example_and_output_sizes_are_independent() {
        let example = patchy(96);
        let small = hex_blend(&example, &HexBlendConfig::default(), 64, 48).expect("blend");
        assert_eq!(
            (small.width, small.height, small.albedo.len()),
            (64, 48, 64 * 48 * 4)
        );
        let big = hex_blend(&example, &HexBlendConfig::default(), 128, 128).expect("blend");
        assert_eq!(big.normal.len(), 128 * 128 * 4);
        assert!(hex_blend(&example, &HexBlendConfig::default(), 0, 8).is_err());
    }

    #[test]
    fn the_seed_chooses_the_windows() {
        let example = patchy(64);
        let a = hex_blend(&example, &HexBlendConfig::default(), 64, 64).expect("blend");
        let again = hex_blend(&example, &HexBlendConfig::default(), 64, 64).expect("blend");
        let b = hex_blend(
            &example,
            &HexBlendConfig {
                seed: 8,
                ..Default::default()
            },
            64,
            64,
        )
        .expect("blend");
        assert_eq!(a.albedo, again.albedo, "not deterministic");
        assert_ne!(a.albedo, b.albedo, "the seed changed nothing");
    }
}
