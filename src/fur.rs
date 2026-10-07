//! Fur texture generator: strands drawn one by one, from a cow's or a
//! horse's short straight coat to a sheep's curled fleece, with optional
//! pied patches.
//!
//! The algorithm:
//! 1. Root a handful of strands in every cell of a grid over the tile, each
//!    hanging down V at an angle the flow field and its lock give it, with
//!    its own length, and - with curl - bending as it goes, so a fleece's
//!    strands are short tight arcs.
//! 2. Draw every strand into a buffer once, over the texels its segments
//!    can touch, keeping at each texel the topmost strand: its distance gives
//!    the strand's edge, its position along the strand the colour from root
//!    to tip.  Where none reaches, the dark undercoat shows.  A strand that
//!    runs off the tile's edge is drawn on round onto the other side, so the
//!    coat tiles.
//! 3. Gather the coat into locks - cells of a tiling cellular lattice, more
//!    and smaller as the curl rises - darkened at the partings and domed,
//!    strongly for a fleece.
//! 4. Lay a second, pied colour over patches of the coat.

use std::f64::consts::PI;

use noise::{Fbm, MultiFractal, Perlin};

use crate::math::{cell_hash, smoothstep};
use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{CellularParams, ToroidalNoise, cellular, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`FurGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FurConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Strands across the tile \[8, 64\]: how fine the coat is.
    pub density: f64,
    /// Strand length \[0.05, 1\]: from stubble to long hair.
    pub length: f64,
    /// Curl \[0, 1\]: `0` is a straight coat, `1` a sheep's crimped fleece
    /// in small round locks.
    pub curl: f64,
    /// How strongly the strands gather into locks \[0, 1\].
    pub clumping: f64,
    /// How far the strands' direction wanders across the coat \[0, 1\].
    pub flow: f64,
    /// Colour at the roots, and of the undercoat between strands, in linear
    /// RGB \[0, 1\].
    pub color_root: [f32; 3],
    /// Colour at the tips in linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// How much of the coat the pied colour covers \[0, 1\].
    pub patches: f64,
    /// The pied colour in linear RGB \[0, 1\]: a Holstein's white, a
    /// piebald's black.
    pub color_patch: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for FurConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            density: 28.0,
            length: 0.5,
            curl: 0.0,
            clumping: 0.5,
            flow: 0.4,
            color_root: [0.08, 0.05, 0.03],
            color_tip: [0.40, 0.26, 0.14],
            patches: 0.0,
            color_patch: [0.80, 0.78, 0.72],
            normal_strength: 3.0,
        }
    }
}

/// Procedural fur texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`FurConfig`].  Construct
/// via [`FurGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::fur` task for non-blocking generation.
pub struct FurGenerator {
    config: FurConfig,
    flow_noise: ToroidalNoise<Fbm<Perlin>>,
    patch_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl FurGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated calls to
    /// [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: FurConfig) -> Self {
        let seed = config.seed;
        Self {
            flow_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(17)).set_octaves(3), 1.6),
            patch_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(29)).set_octaves(3), 1.3),
            config,
        }
    }
}

/// Strands rooted in each cell of the grid.
const STRANDS_PER_CELL: usize = 12;

/// Segments a strand is drawn as, so that it can bend.
const SEGMENTS: usize = 3;

/// A strand of `length` 1 reaches this many grid cells.
const LONGEST_CELLS: f64 = 2.6;

/// One strand: where it is rooted, in cells, and the points along it.
#[derive(Clone)]
struct Strand {
    /// The points from root to tip, in cell units relative to the cell's
    /// own corner.
    points: [(f64, f64); SEGMENTS + 1],
    /// Its place in the stack: the topmost strand at a texel is drawn.
    z: f64,
}

/// The coat's layout, worked from the config once per bake.
struct Layout {
    /// Grid cells across the tile, a whole number so the coat tiles.
    cells: usize,
    /// Every strand, `STRANDS_PER_CELL` to a cell, cells row by row.
    strands: Vec<Strand>,
    /// Strand half-width at the root, in cell units.
    half_width: f64,
    locks: CellularParams,
}

fn layout(c: &FurConfig, flow: &ToroidalNoise<Fbm<Perlin>>) -> Layout {
    let density = c.density.clamp(4.0, 128.0);
    let curl = c.curl.clamp(0.0, 1.0);
    let cells = (density * 0.5).round().max(2.0) as usize;
    let locks = CellularParams::new(5.0 + 15.0 * curl, c.seed.wrapping_add(5));
    // A fleece's strands are short and tightly bent.
    let reach = LONGEST_CELLS * c.length.clamp(0.05, 1.0) * (1.0 - 0.45 * curl);
    let turn = curl * 2.6;
    let mut strands = Vec::with_capacity(cells * cells * STRANDS_PER_CELL);
    for cy in 0..cells {
        for cx in 0..cells {
            for j in 0..STRANDS_PER_CELL {
                let id = ((cy * cells + cx) * STRANDS_PER_CELL + j) as i64;
                let h = |salt: i64| cell_hash(id, salt, c.seed);
                let (ox, oy) = (h(1), h(2));
                let (ru, rv) = (
                    (cx as f64 + ox) / cells as f64,
                    (cy as f64 + oy) / cells as f64,
                );
                let lock = cellular(ru, rv, locks);
                let lock_turn = (cell_hash(lock.cell_x, lock.cell_y, c.seed) - 0.5) * PI * 0.5;
                let wander = (normalize(flow.get(ru, rv)) - 0.5) * c.flow * PI * 1.2;
                let mut angle = wander
                    + lock_turn * c.clumping.clamp(0.0, 1.0)
                    + (h(3) - 0.5) * (0.3 + curl * PI);
                let length = reach * (0.7 + 0.6 * h(4));
                let bend = turn * if h(5) < 0.5 { -1.0 } else { 1.0 } / SEGMENTS as f64;
                let mut points = [(ox, oy); SEGMENTS + 1];
                for k in 1..=SEGMENTS {
                    let (px, py) = points[k - 1];
                    // Hanging down V, turned `angle` toward +U.
                    let step = length / SEGMENTS as f64;
                    points[k] = (px + angle.sin() * step, py + angle.cos() * step);
                    angle += bend;
                }
                strands.push(Strand { points, z: h(6) });
            }
        }
    }
    Layout {
        cells,
        strands,
        half_width: 0.32 * cells as f64 / density,
        locks,
    }
}

/// What the topmost strand at a texel says: how much of the texel it covers,
/// where along the strand the texel lies (root 0, tip 1), and its place in
/// the stack; `z < 0` where no strand reaches.
#[derive(Clone, Copy)]
struct Hit {
    cover: f32,
    along: f32,
    z: f32,
}

const NO_HIT: Hit = Hit {
    cover: 0.0,
    along: 1.0,
    z: -1.0,
};

/// Draw every strand into a `width x height` buffer of [`Hit`]s, keeping the
/// topmost at each texel.  Texels are measured at their corners, as the
/// surface driver samples them, and a segment's texels wrap round the
/// tile's edges.
fn splat(lay: &Layout, width: usize, height: usize) -> Vec<Hit> {
    let mut hits = vec![NO_HIT; width * height];
    let n = lay.cells as f64;
    let (sx, sy) = (width as f64 / n, height as f64 / n);
    for (s, strand) in lay.strands.iter().enumerate() {
        let cell = s / STRANDS_PER_CELL;
        let (col, row) = ((cell % lay.cells) as f64, (cell / lay.cells) as f64);
        let z = strand.z as f32;
        for k in 0..SEGMENTS {
            let (a, b) = (strand.points[k], strand.points[k + 1]);
            let (a, b) = ((a.0 + col, a.1 + row), (b.0 + col, b.1 + row));
            let w = lay.half_width;
            let x0 = ((a.0.min(b.0) - w) * sx).floor() as i64;
            let x1 = ((a.0.max(b.0) + w) * sx).ceil() as i64;
            let y0 = ((a.1.min(b.1) - w) * sy).floor() as i64;
            let y1 = ((a.1.max(b.1) + w) * sy).ceil() as i64;
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let idx = py.rem_euclid(height as i64) as usize * width
                        + px.rem_euclid(width as i64) as usize;
                    if hits[idx].z >= z {
                        continue;
                    }
                    let p = (px as f64 / sx, py as f64 / sy);
                    let (d, t) = to_segment(p, a, b);
                    let along = (k as f64 + t) / SEGMENTS as f64;
                    // Strands taper from root to tip.
                    let half = w * (1.0 - 0.55 * along);
                    if d < half {
                        hits[idx] = Hit {
                            cover: (1.0 - smoothstep(0.55 * half, half, d)) as f32,
                            along: along as f32,
                            z,
                        };
                    }
                }
            }
        }
    }
    hits
}

/// The distance from `p` to the segment `a`-`b`, and how far along it the
/// nearest point lies, in `[0, 1]`.
fn to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a.0 + dx * t - p.0, a.1 + dy * t - p.1);
    ((qx * qx + qy * qy).sqrt(), t)
}

/// Per-generation sampler: the layout, the drawn strands and the patch grid.
struct FurCell<'a> {
    config: &'a FurConfig,
    layout: &'a Layout,
    hits: &'a [Hit],
    patch_grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for FurCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let lay = self.layout;
        let curl = c.curl.clamp(0.0, 1.0);
        let clump = c.clumping.clamp(0.0, 1.0);

        let lock = cellular(u, v, lay.locks);
        let scale = lay.locks.scale.round().max(1.0);
        // 0 on the parting between two locks, 1 well inside one.
        let inside = smoothstep(0.0, 0.3 / scale, lock.ridge());
        // 1 at the lock's heart, falling to 0 a cell's width out.
        let dome = (1.0 - lock.f1 * scale).clamp(0.0, 1.0);
        let phase = cell_hash(lock.cell_x, lock.cell_y, c.seed);

        let idx = y as usize * self.width + x as usize;
        let hit = self.hits[idx];
        let (cover, along, z) = (hit.cover as f64, hit.along as f64, hit.z.max(0.0) as f64);
        let lock_height = clump * (0.3 + 0.7 * curl) * (dome * 0.7 + inside * 0.3);
        let height = 0.15 + cover * (0.45 + 0.35 * z) + lock_height * 1.4;

        let parting = (1.0 - clump * 0.5 * (1.0 - inside)) as f32;
        let tipness = (along.powf(0.8) * (0.55 + 0.45 * z)) as f32;
        let tone = 1.0 + (phase as f32 - 0.5) * 0.12;
        let t = 0.5 + (0.5 - c.patches.clamp(0.0, 1.0)) * 0.55;
        let pied = smoothstep(t - 0.025, t + 0.025, normalize(self.patch_grid[idx])) as f32;
        let mut color = [0.0f32; 3];
        for (i, slot) in color.iter_mut().enumerate() {
            let (root, tip) = (
                lerp(c.color_root[i], c.color_patch[i] * 0.5, pied),
                lerp(c.color_tip[i], c.color_patch[i], pied),
            );
            let strand = lerp(root, tip, tipness);
            let under = root * 0.75;
            *slot = (lerp(under, strand, cover as f32) * tone * parting).clamp(0.0, 1.0);
        }
        SurfaceSample {
            occlusion: (0.5 + 0.3 * cover + 0.2 * inside) as f32,
            ..SurfaceSample::matte(height, color, 0.84 - 0.12 * cover as f32)
        }
    }
}

impl FurGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;
        let mut patch_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.patch_noise, width, height, &mut patch_grid);
        let lay = layout(c, &self.flow_noise);
        let hits = splat(&lay, width as usize, height as usize);
        let cell = FurCell {
            config: c,
            layout: &lay,
            hits: &hits,
            patch_grid: &patch_grid,
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);
        if let Some(ws) = ws {
            ws.return_grid(patch_grid);
        }
        result
    }
}

impl TextureGenerator for FurGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, None)
    }

    fn generate_with_workspace(
        &self,
        width: u32,
        height: u32,
        workspace: &mut Workspace,
    ) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, Some(workspace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{luminance, wrap_and_inside_steps};

    fn bake(c: FurConfig, size: usize) -> Vec<f64> {
        luminance(
            &FurGenerator::new(c)
                .generate(size as u32, size as u32)
                .expect("a bake"),
        )
    }

    /// A straight coat's strands run down V: the albedo changes far more
    /// from one texel to the next across them than along them.
    #[test]
    fn a_straight_coats_strands_run_down_v() {
        let size = 192;
        let l = bake(
            FurConfig {
                curl: 0.0,
                flow: 0.0,
                clumping: 0.0,
                length: 1.0,
                ..FurConfig::default()
            },
            size,
        );
        let (mut across, mut along) = (0.0, 0.0);
        for y in 0..size - 1 {
            for x in 0..size - 1 {
                across += (l[y * size + x + 1] - l[y * size + x]).abs();
                along += (l[(y + 1) * size + x] - l[y * size + x]).abs();
            }
        }
        assert!(across > along * 2.5, "across {across:.0}, along {along:.0}");
    }

    /// The pied colour covers about as much of the coat as `patches` asks,
    /// none at all at `0`, and the coat wraps in both directions.
    #[test]
    fn patches_cover_what_they_ask_and_the_coat_wraps() {
        let size = 128;
        let patched = |patches: f64| {
            let l = bake(
                FurConfig {
                    patches,
                    color_patch: [1.0, 1.0, 1.0],
                    color_root: [0.0, 0.0, 0.0],
                    color_tip: [0.05, 0.05, 0.05],
                    ..FurConfig::default()
                },
                size,
            );
            l.iter().filter(|&&x| x > 110.0).count() as f64 / l.len() as f64
        };
        assert_eq!(patched(0.0), 0.0);
        let half = patched(0.5);
        assert!(
            (0.15..0.7).contains(&half),
            "half covers {:.0}%",
            half * 100.0
        );
        assert!(patched(1.0) > half, "more patches, more cover");

        let (mut wrap_u, mut inside_u, mut wrap_v, mut inside_v) = (0.0, 0.0, 0.0, 0.0);
        for seed in 1..=4 {
            let l = bake(
                FurConfig {
                    seed,
                    curl: 0.6,
                    ..FurConfig::default()
                },
                size,
            );
            let (w, i) = wrap_and_inside_steps(&l, size, 1, true);
            wrap_u += w;
            inside_u += i;
            let (w, i) = wrap_and_inside_steps(&l, size, 1, false);
            wrap_v += w;
            inside_v += i;
        }
        assert!(
            wrap_u < inside_u * 1.3,
            "a seam in U: {wrap_u:.0} against {inside_u:.0}"
        );
        assert!(
            wrap_v < inside_v * 1.3,
            "a seam in V: {wrap_v:.0} against {inside_v:.0}"
        );
    }
}
