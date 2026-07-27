//! Forest floor: fallen leaf litter lying over dark humus.
//!
//! Litter is stamped rather than sampled from noise.  Leaves are discrete
//! objects that overlap and hide one another, and no amount of fractal noise
//! produces that read — the tell is that noise-based "litter" has no edges you
//! can follow around a single leaf.
//!
//! Each layer places one leaf per lattice cell, rotated and tinted from the
//! cell hash; layers stack with a per-leaf depth so the litter interleaves
//! instead of the last layer simply painting over the others.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, cell_hash, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Most litter layers a config may stack, bounding the per-texel stamp work.
///
/// Each layer walks a 3×3 cell neighbourhood, so cost is linear in this.
pub const MAX_LAYERS: usize = 4;

/// Configures the appearance of a [`ForestFloorGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ForestFloorConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Leaves across the tile in the coarsest litter layer.
    pub litter_scale: f64,
    /// Litter layers stacked on top of one another, clamped to
    /// [`MAX_LAYERS`].  Each successive layer is finer and sits higher.
    pub layers: usize,
    /// Fraction of lattice cells that actually carry a leaf, in `[0, 1]`.
    /// Lower values let the humus show through.
    pub coverage: f64,
    /// Leaf length as a multiple of the lattice cell size.
    pub leaf_length: f64,
    /// Leaf width as a multiple of its length.
    pub leaf_width: f64,
    /// How much each leaf lifts the height field, giving the litter depth.
    pub leaf_thickness: f64,
    /// Darkening of the midrib running down each leaf, in `[0, 1]`.
    pub midrib: f32,
    /// Frequency of the humus grain showing between the leaves.
    pub humus_scale: f64,
    /// Damp humus colour in linear RGB \[0, 1\].
    pub color_humus: [f32; 3],
    /// Fresh-fallen leaf colour in linear RGB \[0, 1\].
    pub color_leaf: [f32; 3],
    /// Older, rotted-down leaf colour in linear RGB \[0, 1\]; each leaf lands
    /// somewhere between this and [`color_leaf`](Self::color_leaf).
    pub color_leaf_old: [f32; 3],
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ForestFloorConfig {
    fn default() -> Self {
        Self {
            seed: 31,
            litter_scale: 7.0,
            layers: 3,
            coverage: 0.85,
            leaf_length: 1.15,
            leaf_width: 0.5,
            leaf_thickness: 0.35,
            midrib: 0.22,
            humus_scale: 14.0,
            color_humus: [0.09, 0.07, 0.05],
            color_leaf: [0.46, 0.31, 0.12],
            color_leaf_old: [0.22, 0.16, 0.09],
            normal_strength: 2.2,
        }
    }
}

/// Procedural forest-floor texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`ForestFloorConfig`].
pub struct ForestFloorGenerator {
    config: ForestFloorConfig,
    humus: ToroidalNoise<Fbm<Perlin>>,
}

impl ForestFloorGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ForestFloorConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(13)).set_octaves(4);
        let humus = ToroidalNoise::new(fbm, config.humus_scale);
        Self { config, humus }
    }
}

/// One leaf resolved at a texel: what it looks like and how high it lies.
struct Leaf {
    depth: f64,
    /// Position across the leaf's width, for the midrib.
    across: f64,
    /// Position along the leaf, `0` at the centre and `1` at the tip.
    along: f64,
    /// How far inside the silhouette, `1` at the middle.
    inside: f64,
    shade: f64,
}

/// Per-generation sampler: humus grid plus the stamped litter layers.
struct ForestFloorCell<'a> {
    config: &'a ForestFloorConfig,
    humus: &'a [f64],
    layers: usize,
    width: usize,
}

/// Shortest signed offset between two UV coordinates on the torus.
#[inline]
fn wrap_delta(a: f64, b: f64) -> f64 {
    let d = a - b;
    d - d.round()
}

impl ForestFloorCell<'_> {
    /// Find the topmost leaf covering `(u, v)`, if any.
    ///
    /// Walks the 3×3 cell neighbourhood of each layer because a leaf is
    /// longer than its cell and therefore reaches into its neighbours.
    fn topmost_leaf(&self, u: f64, v: f64) -> Option<Leaf> {
        let c = self.config;
        let mut best: Option<Leaf> = None;

        for layer in 0..self.layers {
            // Finer, higher layers on top of coarser ones.
            let scale = (c.litter_scale * 1.35_f64.powi(layer as i32))
                .round()
                .clamp(1.0, 512.0);
            let seed = c.seed.wrapping_add(layer as u32 * 97);
            let n = scale as i64;

            let gi = (u * scale).floor() as i64;
            let gj = (v * scale).floor() as i64;
            let half_length = (c.leaf_length.max(0.01) * 0.5) / scale;
            let half_width = half_length * c.leaf_width.clamp(0.05, 1.0);

            for di in -1i64..=1 {
                for dj in -1i64..=1 {
                    let ci = (gi + di).rem_euclid(n);
                    let cj = (gj + dj).rem_euclid(n);

                    // Thin the litter so humus shows through.
                    if cell_hash(ci, cj, seed.wrapping_add(3)) > c.coverage.clamp(0.0, 1.0) {
                        continue;
                    }

                    let jx = 0.1 + 0.8 * cell_hash(ci, cj, seed);
                    let jy = 0.1 + 0.8 * cell_hash(cj, ci, seed.wrapping_add(1));
                    let cx = (ci as f64 + jx) / scale;
                    let cy = (cj as f64 + jy) / scale;

                    let dx = wrap_delta(u, cx);
                    let dy = wrap_delta(v, cy);

                    // Leaves are not stamped from one die: vary each by a
                    // quarter either way, or the litter reads as printed
                    // wallpaper however well the rotations are shuffled.
                    let size = 0.75 + 0.5 * cell_hash(ci, cj, seed.wrapping_add(6));

                    // Leaves land at every angle.
                    let angle = cell_hash(ci, cj, seed.wrapping_add(2)) * TAU;
                    let (sin, cos) = angle.sin_cos();
                    let along = dx * cos + dy * sin;
                    let across = -dx * sin + dy * cos;

                    let nx = along / (half_length * size);
                    let ny = across / (half_width * size);
                    let radial = nx * nx + ny * ny;
                    if radial >= 1.0 {
                        continue;
                    }

                    // Later layers sit higher; the hash breaks ties within a
                    // layer so leaves interleave rather than stacking by rank.
                    let depth =
                        layer as f64 + cell_hash(ci, cj, seed.wrapping_add(4)).clamp(0.0, 0.99);
                    if best.as_ref().is_some_and(|b| b.depth >= depth) {
                        continue;
                    }

                    best = Some(Leaf {
                        depth,
                        across: ny.abs(),
                        along: nx.abs(),
                        inside: 1.0 - radial,
                        shade: cell_hash(ci, cj, seed.wrapping_add(5)),
                    });
                }
            }
        }

        best
    }
}

impl SurfaceCell for ForestFloorCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let humus = normalize(self.humus[y as usize * self.width + x as usize]);

        let Some(leaf) = self.topmost_leaf(u, v) else {
            // Bare humus between the litter.
            let mottle = (humus as f32 - 0.5) * 0.10;
            let color = [
                (c.color_humus[0] + mottle).clamp(0.0, 1.0),
                (c.color_humus[1] + mottle * 0.9).clamp(0.0, 1.0),
                (c.color_humus[2] + mottle * 0.8).clamp(0.0, 1.0),
            ];
            return SurfaceSample {
                height: humus * 0.12,
                color,
                roughness: 0.96,
                metallic: 0.0,
                occlusion: 0.55,
                emissive: [0.0, 0.0, 0.0],
            };
        };

        // Every leaf has weathered a different amount.
        let t = leaf.shade as f32;
        let mut color = [
            lerp(c.color_leaf_old[0], c.color_leaf[0], t),
            lerp(c.color_leaf_old[1], c.color_leaf[1], t),
            lerp(c.color_leaf_old[2], c.color_leaf[2], t),
        ];

        // Midrib: a darker seam down the leaf's spine, fading toward the tip.
        let rib = ((1.0 - (leaf.across / 0.16).min(1.0)) * (1.0 - leaf.along)) as f32;
        let rib_shade = 1.0 - rib * c.midrib.clamp(0.0, 1.0);
        for channel in &mut color {
            *channel = (*channel * rib_shade).clamp(0.0, 1.0);
        }

        // Grain from the humus grid keeps leaves from reading as flat decals.
        let grain = (humus as f32 - 0.5) * 0.06;
        for channel in &mut color {
            *channel = (*channel + grain).clamp(0.0, 1.0);
        }

        // Leaves cup slightly, so the height rises toward the middle, and each
        // layer lies above the last.
        let cup = leaf.inside.sqrt();
        let height = leaf.depth * c.leaf_thickness * 0.35 + cup * c.leaf_thickness;

        SurfaceSample {
            height,
            color,
            roughness: (0.80 + (1.0 - cup as f32) * 0.12).clamp(0.0, 1.0),
            metallic: 0.0,
            // Leaf edges tuck into the litter below them.
            occlusion: lerp(0.7, 1.0, cup as f32),
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

impl ForestFloorGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut humus = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.humus, width, height, &mut humus);

        let cell = ForestFloorCell {
            config: c,
            humus: &humus,
            layers: c.layers.clamp(1, MAX_LAYERS),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(humus);
        }
        result
    }
}

impl TextureGenerator for ForestFloorGenerator {
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

    fn bake(config: ForestFloorConfig) -> TextureMap {
        ForestFloorGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    /// Fraction of the tile dark enough to be bare humus rather than a leaf.
    fn humus_fraction(map: &TextureMap) -> f64 {
        map.albedo.chunks(4).filter(|px| px[0] < 90).count() as f64 / (128 * 128) as f64
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(ForestFloorConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert_eq!(map.normal.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(ForestFloorConfig::default()).albedo,
            bake(ForestFloorConfig::default()).albedo
        );
        assert_ne!(
            bake(ForestFloorConfig::default()).albedo,
            bake(ForestFloorConfig {
                seed: 808,
                ..Default::default()
            })
            .albedo
        );
    }

    /// Litter must cover most of the floor while leaving humus visible.
    #[test]
    fn litter_covers_most_of_the_humus() {
        let humus = humus_fraction(&bake(ForestFloorConfig::default()));
        assert!(
            (0.01..0.5).contains(&humus),
            "humus fraction {humus:.3} — the floor is either bare or fully paved"
        );
    }

    /// `coverage` must actually thin the litter out.
    #[test]
    fn coverage_controls_how_much_humus_shows() {
        let dense = humus_fraction(&bake(ForestFloorConfig {
            coverage: 1.0,
            ..Default::default()
        }));
        let sparse = humus_fraction(&bake(ForestFloorConfig {
            coverage: 0.3,
            ..Default::default()
        }));
        assert!(
            sparse > dense,
            "lowering coverage did not expose more humus ({dense:.3} → {sparse:.3})"
        );
    }

    /// Stacking layers is the point: more layers must change the litter.
    #[test]
    fn layers_stack() {
        let single = bake(ForestFloorConfig {
            layers: 1,
            ..Default::default()
        });
        let stacked = bake(ForestFloorConfig {
            layers: 3,
            ..Default::default()
        });
        assert_ne!(single.albedo, stacked.albedo, "extra layers had no effect");
        assert!(
            humus_fraction(&stacked) < humus_fraction(&single),
            "stacking layers did not bury more humus"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(ForestFloorConfig {
            litter_scale: 0.0,
            layers: 99,
            coverage: 9.0,
            leaf_length: 0.0,
            leaf_width: 0.0,
            humus_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
