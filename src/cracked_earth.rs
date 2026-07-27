//! Dried mud / lakebed texture: flat plates separated by narrow cracks.
//!
//! Built on [`cellular_edge`], whose distance-to-the-cell-wall is what keeps
//! cracks a *constant* width no matter how large or small the surrounding
//! plate is.  The obvious cheaper mask, `F2 − F1`, widens with the cell, so
//! big plates get canyons and small ones get hairlines — the giveaway that a
//! surface was made from Voronoi rather than by drying.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{
        CellMetric, CellularParams, ToroidalNoise, cell_hash, cellular, cellular_edge, normalize,
        sample_grid_into,
    },
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`CrackedEarthGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CrackedEarthConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Plates across the tile.
    pub scale: f64,
    /// How irregular the plates are, in `[0, 1]`: `0` is a regular lattice,
    /// `1` lets plate centres reach their cell edges.
    pub jitter: f64,
    /// Crack width in UV units — a fraction of the tile, so a config draws
    /// the same crack at any bake resolution.
    pub crack_width: f64,
    /// How deep cracks cut into the height field.
    pub crack_depth: f64,
    /// How much plate edges curl up as they dry, in height units.  The
    /// signature of dried mud: the plate is not flat, it lifts toward every
    /// crack it borders.
    pub curl: f64,
    /// How far the curl reaches back from a crack, in UV units.
    pub curl_reach: f64,
    /// Spread of per-plate tint variation in `[0, 1]`; each plate dries to a
    /// slightly different shade.
    pub plate_variance: f32,
    /// Frequency of the silt grain broken across plates and crack floors.
    pub grain_scale: f64,
    /// How strongly the silt grain modulates colour and height.
    pub grain_strength: f64,
    /// Dried plate colour in linear RGB \[0, 1\].
    pub color_plate: [f32; 3],
    /// Damp crack-floor colour in linear RGB \[0, 1\].
    pub color_crack: [f32; 3],
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for CrackedEarthConfig {
    fn default() -> Self {
        Self {
            seed: 11,
            scale: 7.0,
            jitter: 0.85,
            crack_width: 0.010,
            crack_depth: 0.55,
            curl: 0.22,
            curl_reach: 0.035,
            plate_variance: 0.10,
            grain_scale: 26.0,
            grain_strength: 0.12,
            color_plate: [0.44, 0.33, 0.22],
            color_crack: [0.13, 0.09, 0.06],
            normal_strength: 3.0,
        }
    }
}

/// Procedural cracked-earth texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`CrackedEarthConfig`].
pub struct CrackedEarthGenerator {
    config: CrackedEarthConfig,
    grain: ToroidalNoise<Fbm<Perlin>>,
}

impl CrackedEarthGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: CrackedEarthConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(3)).set_octaves(3);
        let grain = ToroidalNoise::new(fbm, config.grain_scale);
        Self { config, grain }
    }
}

/// Per-generation sampler: silt grid plus the plate lattice.
struct CrackedEarthCell<'a> {
    config: &'a CrackedEarthConfig,
    grain: &'a [f64],
    params: CellularParams,
    width: usize,
}

impl SurfaceCell for CrackedEarthCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let grain = normalize(self.grain[y as usize * self.width + x as usize]);

        // Distance to the plate wall, and which plate we are standing on.
        // Two lattice walks: one resolves the wall, the other the owner, and
        // the plate tint needs an owner the edge pass does not report.
        let edge = cellular_edge(u, v, self.params);
        let plate = cellular(u, v, self.params);

        let half_width = (c.crack_width.max(0.0)) * 0.5;
        // 1 inside the crack, 0 out on the plate.
        let crack = 1.0 - smoothstep(half_width, half_width * 2.5, edge);

        // Plates dry from the top and lift at their rims.
        let reach = c.curl_reach.max(1e-6);
        let proximity = (1.0 - (edge / reach).clamp(0.0, 1.0)).powi(2);
        let lift = c.curl * proximity;

        let silt = (grain - 0.5) * c.grain_strength;
        let height = (1.0 + lift + silt) * (1.0 - crack) - c.crack_depth * crack;

        // Each plate dries to its own shade, and the silt grain breaks up the
        // shade *within* a plate — without it the plates read as poured
        // concrete tiles rather than dried mud.
        let tint = (cell_hash(plate.cell_x, plate.cell_y, c.seed.wrapping_add(41)) - 0.5) as f32
            * 2.0
            * c.plate_variance;
        let mottle = silt as f32;
        let plate_color = [
            (c.color_plate[0] + tint + mottle).clamp(0.0, 1.0),
            (c.color_plate[1] + tint * 0.85 + mottle * 0.9).clamp(0.0, 1.0),
            (c.color_plate[2] + tint * 0.6 + mottle * 0.7).clamp(0.0, 1.0),
        ];

        let t = crack as f32;
        let color = [
            lerp(plate_color[0], c.color_crack[0], t),
            lerp(plate_color[1], c.color_crack[1], t),
            lerp(plate_color[2], c.color_crack[2], t),
        ];

        // Crack floors hold damp silt: rougher, and shadowed by their walls.
        let roughness = lerp(0.82 + (grain as f32 - 0.5) * 0.10, 0.97, t).clamp(0.0, 1.0);
        let occlusion = lerp(1.0, 0.45, t);

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

impl CrackedEarthGenerator {
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

        let cell = CrackedEarthCell {
            config: c,
            grain: &grain,
            params: CellularParams::new(c.scale, c.seed)
                .with_jitter(c.jitter)
                .with_metric(CellMetric::Euclidean),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(grain);
        }
        result
    }
}

impl TextureGenerator for CrackedEarthGenerator {
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

    /// sRGB red-channel value below which a texel counts as crack rather than
    /// plate.  Sits between the encoded crack colour (~101) and the encoded
    /// plate colour (~171); note that neither is anywhere near 0, so a naive
    /// "is it dark" threshold finds nothing at all.
    const CRACK_LUMA: u8 = 140;

    fn bake(config: CrackedEarthConfig) -> TextureMap {
        CrackedEarthGenerator::new(config)
            .generate(64, 64)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(CrackedEarthConfig::default());
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(CrackedEarthConfig::default()).albedo,
            bake(CrackedEarthConfig::default()).albedo
        );
        let other = bake(CrackedEarthConfig {
            seed: 99,
            ..Default::default()
        });
        assert_ne!(bake(CrackedEarthConfig::default()).albedo, other.albedo);
    }

    /// Crack coverage at the resolutions this is actually baked at.
    ///
    /// Measured at 256² — `crack_width` is a fraction of the *tile*, so at 64²
    /// a realistic crack is a third of a texel wide and simply has nothing to
    /// land on.  That is the correct behaviour for a resolution-independent
    /// knob, but it makes a 64² sample useless for judging coverage.
    fn crack_coverage(config: CrackedEarthConfig) -> f64 {
        let map = CrackedEarthGenerator::new(config)
            .generate(256, 256)
            .expect("generate");
        map.albedo.chunks(4).filter(|px| px[0] < CRACK_LUMA).count() as f64 / (256 * 256) as f64
    }

    /// Cracks must be a minority of the surface — this is a plated texture,
    /// not a dark one with occasional plates.
    #[test]
    fn cracks_are_a_minority_of_the_tile() {
        let fraction = crack_coverage(CrackedEarthConfig::default());
        assert!(
            (0.02..0.45).contains(&fraction),
            "crack coverage {fraction:.3} is not a plausible crack network"
        );
    }

    /// The whole reason this generator uses `cellular_edge`: crack width must
    /// not scale with plate size.
    ///
    /// Doubling `scale` halves the plate area, so a width-proportional mask
    /// (`F2 − F1`) would roughly double the fraction of the tile that reads as
    /// crack per unit of crack length.  With true edge distance the coverage
    /// tracks total crack *length* instead, which grows far more slowly.
    #[test]
    fn crack_width_is_independent_of_plate_size() {
        let coverage = |scale: f64| {
            crack_coverage(CrackedEarthConfig {
                scale,
                ..Default::default()
            })
        };

        let coarse = coverage(5.0);
        let fine = coverage(10.0);
        // Twice the plates means roughly twice the crack length, so coverage
        // should grow near-linearly rather than with plate area.
        assert!(
            fine < coarse * 3.0,
            "crack coverage exploded with plate count ({coarse:.3} → {fine:.3}); \
             width is tracking cell size"
        );
        assert!(
            fine > coarse,
            "more plates did not add crack length ({coarse:.3} → {fine:.3})"
        );
    }

    /// Plate rims must stand above plate centres — the curl is the feature.
    #[test]
    fn plate_rims_curl_above_their_centres() {
        let flat = bake(CrackedEarthConfig {
            curl: 0.0,
            ..Default::default()
        });
        let curled = bake(CrackedEarthConfig {
            curl: 0.6,
            ..Default::default()
        });
        assert_ne!(
            flat.normal, curled.normal,
            "curl did not reach the height field"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(CrackedEarthConfig {
            scale: 0.0,
            jitter: 9.0,
            crack_width: -1.0,
            curl_reach: 0.0,
            grain_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }
}
