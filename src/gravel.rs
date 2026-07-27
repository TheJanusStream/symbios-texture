//! Loose gravel: small packed stones with dust and fines between them.
//!
//! Distinct from [`cobblestone`](crate::cobblestone), which lays a few large
//! set stones with mortar gaps.  Gravel is many small stones at mixed sizes,
//! and its [`metric`](GravelConfig::metric) decides whether they read as
//! water-rounded river shingle or freshly-crushed angular aggregate.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{
        CellMetric, CellularParams, ToroidalNoise, cell_hash, cellular, normalize, sample_grid_into,
    },
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`GravelGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GravelConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Stones across the tile.  Gravel wants a high count — around 20 reads
    /// as roadbase, around 8 as coarse ballast.
    pub scale: f64,
    /// Stone shape: [`CellMetric::Euclidean`] gives water-rounded shingle,
    /// [`CellMetric::Chebyshev`] and [`CellMetric::Manhattan`] give the flat
    /// faces and sharp arrises of crushed aggregate.
    pub metric: CellMetric,
    /// How irregularly stones are packed, in `[0, 1]`.
    pub jitter: f64,
    /// Dome profile exponent: low is flat and pebble-like, high is sharply
    /// peaked.
    pub roundness: f64,
    /// Spread of per-stone size variation in `[0, 1]`; gravel is graded, not
    /// uniform.
    pub size_variance: f64,
    /// Spread of per-stone tint variation in `[0, 1]`.
    pub cell_variance: f32,
    /// How much dust and grit fills the gaps between stones, in `[0, 1]`.
    pub fines_level: f64,
    /// Frequency of the grit grain over stones and fines alike.
    pub grain_scale: f64,
    /// Lit stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Shadowed stone colour in linear RGB \[0, 1\].
    pub color_dark: [f32; 3],
    /// Colour of the dust between stones in linear RGB \[0, 1\].
    pub color_fines: [f32; 3],
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for GravelConfig {
    fn default() -> Self {
        Self {
            seed: 23,
            scale: 20.0,
            metric: CellMetric::Euclidean,
            jitter: 0.9,
            roundness: 1.6,
            size_variance: 0.45,
            cell_variance: 0.13,
            fines_level: 0.55,
            grain_scale: 60.0,
            color_stone: [0.40, 0.38, 0.35],
            color_dark: [0.17, 0.16, 0.15],
            color_fines: [0.26, 0.24, 0.21],
            normal_strength: 2.5,
        }
    }
}

/// Procedural gravel texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`GravelConfig`].
pub struct GravelGenerator {
    config: GravelConfig,
    grain: ToroidalNoise<Fbm<Perlin>>,
}

impl GravelGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GravelConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(5)).set_octaves(2);
        let grain = ToroidalNoise::new(fbm, config.grain_scale);
        Self { config, grain }
    }
}

/// Per-generation sampler: grit grid plus the stone lattice.
struct GravelCell<'a> {
    config: &'a GravelConfig,
    grain: &'a [f64],
    params: CellularParams,
    width: usize,
}

impl SurfaceCell for GravelCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let grit = normalize(self.grain[y as usize * self.width + x as usize]);

        let stone = cellular(u, v, self.params);

        // Measure the stone against the distance to its *neighbour* rather
        // than a fixed radius: `f1 / f2` reaches 1 exactly at the cell wall
        // however large or small that cell happens to be.  A fixed radius
        // suits the average cell and leaves the rest either overlapping or
        // marooned in dust, which reads as pebbles scattered on concrete
        // instead of graded aggregate packed together.
        let size = cell_hash(stone.cell_x, stone.cell_y, c.seed.wrapping_add(7));
        let variance = c.size_variance.clamp(0.0, 1.0);
        let fill = 1.0 - variance + variance * size;

        let t = (stone.f1 / (stone.f2 * fill).max(1e-9)).min(1.0);
        let dome = (1.0 - t * t).max(0.0).powf(c.roundness.max(0.05) * 0.5);
        let on_stone = dome > 0.0;

        // Fines: grit-textured dust packed into the gaps.
        let fines = c.fines_level.clamp(0.0, 1.0);
        let fines_height = (grit - 0.5) * 0.25 * fines;

        let height = if on_stone {
            dome + (grit - 0.5) * 0.10
        } else {
            fines_height
        };

        let (color, roughness) = if on_stone {
            let tint = (cell_hash(stone.cell_x, stone.cell_y, c.seed.wrapping_add(19)) - 0.5)
                as f32
                * 2.0
                * c.cell_variance;
            // Stones differ from each other far more than a single stone
            // varies across its own face, so per-stone tint carries the look
            // and the dome only darkens the last sliver at the rim.  Shading
            // the whole dome into the albedo bakes in lighting the normal map
            // is already providing, and the result reads as soap bubbles.
            let rim = smoothstep(0.0, 0.28, dome) as f32;
            let grit_shade = (grit as f32 - 0.5) * 0.07;
            let base = [
                lerp(c.color_dark[0], c.color_stone[0], rim) + tint + grit_shade,
                lerp(c.color_dark[1], c.color_stone[1], rim) + tint * 0.9 + grit_shade,
                lerp(c.color_dark[2], c.color_stone[2], rim) + tint * 0.8 + grit_shade,
            ];
            let color = [
                base[0].clamp(0.0, 1.0),
                base[1].clamp(0.0, 1.0),
                base[2].clamp(0.0, 1.0),
            ];
            // Wet-worn crowns are marginally smoother than the flanks.
            (
                color,
                (0.88 - dome as f32 * 0.15 + (grit as f32 - 0.5) * 0.08).clamp(0.0, 1.0),
            )
        } else {
            let dust = (grit as f32 - 0.5) * 0.08;
            let color = [
                (c.color_fines[0] + dust).clamp(0.0, 1.0),
                (c.color_fines[1] + dust).clamp(0.0, 1.0),
                (c.color_fines[2] + dust).clamp(0.0, 1.0),
            ];
            (color, 0.97)
        };

        // Gaps between stones sit in their neighbours' shadow.
        let occlusion = if on_stone {
            lerp(0.72, 1.0, dome as f32)
        } else {
            0.6
        };

        SurfaceSample {
            height,
            color,
            roughness,
            metallic: 0.0,
            occlusion,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl GravelGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut grain = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.grain, width, height, &mut grain);

        let cell = GravelCell {
            config: c,
            grain: &grain,
            params: CellularParams::new(c.scale, c.seed)
                .with_jitter(c.jitter)
                .with_metric(c.metric),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(grain);
        }
        result
    }
}

impl TextureGenerator for GravelGenerator {
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

    fn bake(config: GravelConfig) -> TextureMap {
        GravelGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(GravelConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert_eq!(map.normal.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(GravelConfig::default()).albedo,
            bake(GravelConfig::default()).albedo
        );
        assert_ne!(
            bake(GravelConfig::default()).albedo,
            bake(GravelConfig {
                seed: 404,
                ..Default::default()
            })
            .albedo
        );
    }

    /// Stones must cover most of the tile without covering all of it —
    /// gravel is packed stones *with* visible fines, not either alone.
    #[test]
    fn stones_and_fines_both_appear() {
        let map = bake(GravelConfig::default());
        let fines_luma = 140;
        let dark = map.albedo.chunks(4).filter(|px| px[0] < fines_luma).count() as f64
            / (128 * 128) as f64;
        assert!(
            (0.05..0.85).contains(&dark),
            "gravel is all stone or all fines (dark fraction {dark:.3})"
        );
    }

    /// The metric is the shape knob: swapping it must actually change the
    /// stones, not just their shading.
    #[test]
    fn metric_changes_stone_shape() {
        let round = bake(GravelConfig {
            metric: CellMetric::Euclidean,
            ..Default::default()
        });
        let angular = bake(GravelConfig {
            metric: CellMetric::Chebyshev,
            ..Default::default()
        });
        assert_ne!(
            round.normal, angular.normal,
            "metric did not change stone geometry"
        );
    }

    /// Size variance must actually grade the stones; with it off they should
    /// all fill their cells identically.
    #[test]
    fn size_variance_grades_the_stones() {
        let graded = bake(GravelConfig::default());
        let uniform = bake(GravelConfig {
            size_variance: 0.0,
            ..Default::default()
        });
        assert_ne!(graded.albedo, uniform.albedo, "size variance had no effect");
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(GravelConfig {
            scale: 0.0,
            jitter: 5.0,
            roundness: 0.0,
            size_variance: 9.0,
            fines_level: -2.0,
            grain_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
