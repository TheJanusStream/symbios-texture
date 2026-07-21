//! Lichen-encrustation surface generator.
//!
//! Irregular crustose thalli colonising bare stone: a toroidal FBM field is
//! thresholded by `coverage` into discrete patches, each carrying a pale
//! chalky **margin** at its growing edge, a granular interior, and one of two
//! species tints (a sage grey-green and a rusty orange) selected by a slower
//! field so a rock face shows several colonies rather than one flat wash.
//! Un-colonised texels keep the rock substrate colour.
//!
//! This is the Tundra ground-cover skin and the rock-encrustation overlay;
//! every layer is toroidal, so it tiles over boulders and ground alike.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`LichenGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LichenConfig {
    /// PRNG seed for the deterministic noise pattern.
    pub seed: u32,
    /// Scale of the thallus-patch field — lower is broader colonies.
    pub patch_scale: f64,
    /// Octaves for the patch field; more octaves give more raggedly-lobed
    /// colony outlines.
    pub patch_octaves: usize,
    /// Fraction of the surface colonised `[0, 1]`.  `0` leaves bare rock;
    /// `1` covers it completely.
    pub coverage: f64,
    /// Width of the pale growing margin at a colony's edge, as a fraction of
    /// the patch field `[0, 0.4]`.  `0` disables the rim.
    pub rim_width: f64,
    /// Scale of the species-mix field — lower means larger single-species
    /// areas.
    pub species_scale: f64,
    /// Bare rock substrate colour in linear RGB \[0, 1\].
    pub color_rock: [f32; 3],
    /// First species tint (sage grey-green) in linear RGB \[0, 1\].
    pub color_lichen_a: [f32; 3],
    /// Second species tint (rusty orange) in linear RGB \[0, 1\].
    pub color_lichen_b: [f32; 3],
    /// Pale margin colour in linear RGB \[0, 1\].
    pub color_rim: [f32; 3],
    /// Scale of the granular interior texture — higher is finer grain.
    pub grain_scale: f64,
    /// Strength of the granular interior texture `[0, 1]`.
    pub grain_strength: f64,
    /// Crust relief `[0, 1]` — how far the thallus stands proud of the rock
    /// in the height/normal map.
    pub relief: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for LichenConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            patch_scale: 3.0,
            patch_octaves: 2,
            coverage: 0.45,
            rim_width: 0.06,
            species_scale: 1.8,
            // Linear-RGB values, so these sit well below their sRGB
            // appearance: 0.14 linear reads as a mid sage, not a pale wash.
            color_rock: [0.13, 0.13, 0.12],
            color_lichen_a: [0.14, 0.17, 0.10],
            color_lichen_b: [0.26, 0.13, 0.04],
            color_rim: [0.38, 0.40, 0.32],
            grain_scale: 40.0,
            grain_strength: 0.18,
            relief: 0.5,
            normal_strength: 1.8,
        }
    }
}

/// Procedural lichen-encrustation surface generator.
///
/// See the [module documentation](self) for the visual model.
pub struct LichenGenerator {
    config: LichenConfig,
    patch: ToroidalNoise<Fbm<Perlin>>,
    species: ToroidalNoise<Fbm<Perlin>>,
    grain: ToroidalNoise<Fbm<Perlin>>,
}

impl LichenGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LichenConfig) -> Self {
        let fbm_patch: Fbm<Perlin> =
            Fbm::new(config.seed).set_octaves(config.patch_octaves.clamp(1, 10));
        let patch = ToroidalNoise::new(fbm_patch, config.patch_scale);

        let fbm_species: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(19)).set_octaves(2);
        let species = ToroidalNoise::new(fbm_species, config.species_scale);

        let fbm_grain: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(64)).set_octaves(2);
        let grain = ToroidalNoise::new(fbm_grain, config.grain_scale);

        Self {
            config,
            patch,
            species,
            grain,
        }
    }

    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let cell = LichenCell {
            config: &self.config,
            patch: &self.patch,
            species: &self.species,
            grain: &self.grain,
        };
        generate_surface(width, height, self.config.normal_strength, ws, &cell)
    }
}

/// Contrast expansion applied to the thallus-patch field, so colony interiors
/// sit well clear of the coverage threshold and the rim stays a margin.
const PATCH_CONTRAST: f64 = 2.6;

/// Expand a `[0, 1]` value around its midpoint by `k`, clamping back into
/// range.
#[inline]
fn expand(v: f64, k: f64) -> f64 {
    ((v - 0.5) * k + 0.5).clamp(0.0, 1.0)
}

/// Per-pixel sampler over the patch / species / grain layers.
struct LichenCell<'a> {
    config: &'a LichenConfig,
    patch: &'a ToroidalNoise<Fbm<Perlin>>,
    species: &'a ToroidalNoise<Fbm<Perlin>>,
    grain: &'a ToroidalNoise<Fbm<Perlin>>,
}

impl SurfaceCell for LichenCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

        // FBM output bunches around 0.5. Without expansion nearly every
        // colonised texel sits a hair above the threshold, so the margin test
        // below would classify the whole colony as rim.
        let field = expand(normalize(self.patch.get(u, v)), PATCH_CONTRAST);
        // Colonies live where the field clears the coverage threshold.
        let threshold = 1.0 - c.coverage.clamp(0.0, 1.0);
        let above = field - threshold;

        // Bare rock outside every colony.
        if above <= 0.0 {
            return SurfaceSample::matte(0.0, c.color_rock, 0.88);
        }

        // Interior granularity.
        let grain = normalize(self.grain.get(u, v));
        let gs = c.grain_strength.clamp(0.0, 1.0);

        // Species tint: a slow field picks between the two colours, so whole
        // colonies share a hue instead of speckling per-texel.
        let sp = normalize(self.species.get(u, v)) as f32;
        let body = [
            lerp(c.color_lichen_a[0], c.color_lichen_b[0], sp),
            lerp(c.color_lichen_a[1], c.color_lichen_b[1], sp),
            lerp(c.color_lichen_a[2], c.color_lichen_b[2], sp),
        ];

        // Granular shading within the thallus.
        let g = ((grain - 0.5) * gs) as f32;
        let mut color = [
            (body[0] + g).clamp(0.0, 1.0),
            (body[1] + g).clamp(0.0, 1.0),
            (body[2] + g).clamp(0.0, 1.0),
        ];

        // Pale growing margin: a band just inside the colony boundary.
        let rim = c.rim_width.clamp(0.0, 0.4);
        let mut edge = 0.0f32;
        if rim > 0.0 && above < rim {
            edge = (1.0 - above / rim) as f32;
            color = [
                lerp(color[0], c.color_rim[0], edge),
                lerp(color[1], c.color_rim[1], edge),
                lerp(color[2], c.color_rim[2], edge),
            ];
        }

        // The crust stands proud of the rock, thinning toward its margin.
        let relief = c.relief.clamp(0.0, 1.0);
        let thickness = (above / rim.max(1e-3)).min(1.0);
        let height = relief * thickness + grain * gs * 0.15;

        // Chalky and matte; the rim is the driest part.
        let rough = 0.86 + edge * 0.08;

        SurfaceSample::matte(height, color, rough)
    }
}

impl TextureGenerator for LichenGenerator {
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

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = LichenGenerator::new(LichenConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
        assert!(
            map.albedo.chunks(4).all(|px| px[3] == 255),
            "opaque surface"
        );
    }

    #[test]
    fn zero_coverage_is_bare_rock() {
        let map = LichenGenerator::new(LichenConfig {
            coverage: 0.0,
            ..LichenConfig::default()
        })
        .generate(32, 32)
        .expect("generate failed");
        // Every texel is the substrate colour — one flat rock tone.
        let first = &map.albedo[0..3];
        assert!(
            map.albedo.chunks(4).all(|px| &px[0..3] == first),
            "no colonies should appear at zero coverage"
        );
    }

    #[test]
    fn more_coverage_means_more_lichen() {
        // Palette-independent: count texels that differ from the bare-rock
        // substrate colour. Raising coverage must colonise more of the face.
        let cfg = LichenConfig::default();
        let srgb = crate::generator::linear_to_srgb;
        let rock = [
            srgb(cfg.color_rock[0]),
            srgb(cfg.color_rock[1]),
            srgb(cfg.color_rock[2]),
        ];
        let colonised = |coverage: f64| {
            let m = LichenGenerator::new(LichenConfig {
                coverage,
                ..LichenConfig::default()
            })
            .generate(64, 64)
            .expect("generate failed");
            m.albedo.chunks(4).filter(|px| px[0..3] != rock[..]).count()
        };
        let sparse = colonised(0.1);
        let dense = colonised(0.95);
        assert!(
            dense > sparse,
            "denser colonisation should cover more rock (sparse {sparse}, dense {dense})"
        );
    }

    /// Every layer is toroidal, so the wrap-around column pair is just another
    /// pair of neighbours on the torus. A per-texel closeness check would be
    /// wrong here — `coverage` thresholds hard, so a colony boundary landing on
    /// the seam is a legitimate jump. Instead assert the seam is statistically
    /// no more discontinuous than the worst interior neighbour pair; a genuine
    /// non-tiling generator breaks that badly, a hard edge does not.
    #[test]
    fn tiles_seamlessly() {
        const N: usize = 64;
        let map = LichenGenerator::new(LichenConfig::default())
            .generate(N as u32, N as u32)
            .expect("generate failed");

        // Mean absolute RGB difference between column `a` and column `b`.
        let col_mad = |a: usize, b: usize| -> f64 {
            let mut acc = 0u64;
            for row in 0..N {
                let ia = (row * N + a) * 4;
                let ib = (row * N + b) * 4;
                for ch in 0..3 {
                    acc += map.albedo[ia + ch].abs_diff(map.albedo[ib + ch]) as u64;
                }
            }
            acc as f64 / (N * 3) as f64
        };
        let row_mad = |a: usize, b: usize| -> f64 {
            let mut acc = 0u64;
            for col in 0..N {
                let ia = (a * N + col) * 4;
                let ib = (b * N + col) * 4;
                for ch in 0..3 {
                    acc += map.albedo[ia + ch].abs_diff(map.albedo[ib + ch]) as u64;
                }
            }
            acc as f64 / (N * 3) as f64
        };

        let worst_interior_col = (0..N - 1).map(|x| col_mad(x, x + 1)).fold(0.0f64, f64::max);
        let worst_interior_row = (0..N - 1).map(|y| row_mad(y, y + 1)).fold(0.0f64, f64::max);

        let seam_col = col_mad(0, N - 1);
        let seam_row = row_mad(0, N - 1);

        // The wrap pair is one of N torus-adjacent pairs, so it can legitimately
        // be the largest when a colony edge lands on it. Allow 2x the worst
        // interior pair: a generator that genuinely fails to tile butts two
        // uncorrelated fields together and lands far beyond that.
        const TOLERANCE: f64 = 2.0;
        assert!(
            seam_col <= worst_interior_col.max(1.0) * TOLERANCE,
            "horizontal seam ({seam_col:.2}) far worse than interior pairs ({worst_interior_col:.2})"
        );
        assert!(
            seam_row <= worst_interior_row.max(1.0) * TOLERANCE,
            "vertical seam ({seam_row:.2}) far worse than interior pairs ({worst_interior_row:.2})"
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = LichenGenerator::new(LichenConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = LichenGenerator::new(LichenConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            LichenGenerator::new(LichenConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
