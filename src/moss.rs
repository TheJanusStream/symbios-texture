//! Moss-carpet surface generator.
//!
//! A dense velvety cushion of gametophyte filaments: low-frequency toroidal
//! FBM raises the hummocks a moss colony grows in, a high-frequency layer
//! stipples the individual filament tips, and a third slow layer bleaches
//! scattered patches toward a dry straw tone.  Shaded crevices read deep
//! green while the cushion crowns catch a bright yellow-green.
//!
//! Tileable (toroidal noise on every layer), so it wraps ground planes,
//! boulders, and log surfaces without a seam.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`MossGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MossConfig {
    /// PRNG seed for the deterministic noise pattern.
    pub seed: u32,
    /// Scale of the cushion/hummock layer — lower is broader mounds.
    pub cushion_scale: f64,
    /// Octaves for the cushion layer.
    pub cushion_octaves: usize,
    /// Scale of the fine filament-tip stipple — higher is finer grain.
    pub filament_scale: f64,
    /// Octaves for the filament layer.
    pub filament_octaves: usize,
    /// Blend weight of the filament layer `[0, 1]` — how much the fine
    /// stipple breaks up the broad cushions.
    pub filament_weight: f64,
    /// Deep shaded colour in the cushion crevices, linear RGB \[0, 1\].
    pub color_deep: [f32; 3],
    /// Bright colour on the cushion crowns, linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// Dry / bleached straw colour, linear RGB \[0, 1\].
    pub color_dry: [f32; 3],
    /// Share of the carpet bleached toward `color_dry` `[0, 1]`.  `0` is a
    /// uniformly lush carpet; higher scatters sun-dried patches.
    pub dry_patches: f64,
    /// Scale of the dry-patch layer — lower is larger dry areas.
    pub dry_scale: f64,
    /// Cushion relief `[0, 1]` — how much of the height field comes from the
    /// broad mounds versus the fine filament stipple.
    pub cushion_depth: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for MossConfig {
    fn default() -> Self {
        Self {
            seed: 21,
            cushion_scale: 5.0,
            cushion_octaves: 4,
            filament_scale: 34.0,
            filament_octaves: 3,
            filament_weight: 0.45,
            color_deep: [0.03, 0.09, 0.03],
            color_tip: [0.26, 0.44, 0.10],
            color_dry: [0.38, 0.34, 0.14],
            dry_patches: 0.25,
            dry_scale: 2.5,
            cushion_depth: 0.6,
            normal_strength: 2.4,
        }
    }
}

/// Procedural moss-carpet surface generator.
///
/// See the [module documentation](self) for the visual model.  Noise objects
/// are built in the constructor so repeated `generate` calls (size variants)
/// skip initialisation.
pub struct MossGenerator {
    config: MossConfig,
    cushion: ToroidalNoise<Fbm<Perlin>>,
    filament: ToroidalNoise<Fbm<Perlin>>,
    dry: ToroidalNoise<Fbm<Perlin>>,
}

impl MossGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: MossConfig) -> Self {
        let fbm_cushion: Fbm<Perlin> =
            Fbm::new(config.seed).set_octaves(config.cushion_octaves.clamp(1, 10));
        let cushion = ToroidalNoise::new(fbm_cushion, config.cushion_scale);

        let fbm_filament: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(37))
            .set_octaves(config.filament_octaves.clamp(1, 10));
        let filament = ToroidalNoise::new(fbm_filament, config.filament_scale);

        let fbm_dry: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(91)).set_octaves(3);
        let dry = ToroidalNoise::new(fbm_dry, config.dry_scale);

        Self {
            config,
            cushion,
            filament,
            dry,
        }
    }

    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let cell = MossCell {
            config: &self.config,
            cushion: &self.cushion,
            filament: &self.filament,
            dry: &self.dry,
        };
        generate_surface(width, height, self.config.normal_strength, ws, &cell)
    }
}

/// Contrast expansion applied to the cushion/filament colour mix.
const COLOR_CONTRAST: f64 = 2.1;

/// Contrast expansion applied to the dry-patch field before thresholding.
const DRY_CONTRAST: f64 = 2.4;

/// Expand a `[0, 1]` value around its midpoint by `k`, clamping back into
/// range. FBM output is concentrated near 0.5; this spreads it so thresholds
/// and colour ramps have something to bite on.
#[inline]
fn expand(v: f64, k: f64) -> f64 {
    ((v - 0.5) * k + 0.5).clamp(0.0, 1.0)
}

/// Per-pixel sampler over the cushion / filament / dry layers.
struct MossCell<'a> {
    config: &'a MossConfig,
    cushion: &'a ToroidalNoise<Fbm<Perlin>>,
    filament: &'a ToroidalNoise<Fbm<Perlin>>,
    dry: &'a ToroidalNoise<Fbm<Perlin>>,
}

impl SurfaceCell for MossCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

        let cushion = normalize(self.cushion.get(u, v));
        let filament = normalize(self.filament.get(u, v));

        // Lightness of this texel: broad mound shading modulated by the fine
        // filament stipple.
        let w = c.filament_weight.clamp(0.0, 1.0);
        let t = (cushion * (1.0 - w) + filament * w).clamp(0.0, 1.0);
        // FBM output clusters tightly around 0.5, which flattens the carpet
        // into one mid-tone wash. Expanding around the midpoint is what makes
        // the crevices read dark and the cushion crowns catch light.
        let t = expand(t, COLOR_CONTRAST);
        let tf = t as f32;

        let mut color = [
            lerp(c.color_deep[0], c.color_tip[0], tf),
            lerp(c.color_deep[1], c.color_tip[1], tf),
            lerp(c.color_deep[2], c.color_tip[2], tf),
        ];

        // Bleached patches: a slow layer thresholded by `dry_patches`, so at
        // 0 nothing dries and at 1 the whole carpet is straw.
        let dry_amount = c.dry_patches.clamp(0.0, 1.0);
        if dry_amount > 0.0 {
            // Expanded for the same reason as the colour mix — an unexpanded
            // field almost never clears the threshold, so patches never show.
            let dry_field = expand(normalize(self.dry.get(u, v)), DRY_CONTRAST);
            // Map the field through the threshold with a soft shoulder.
            let d = ((dry_field - (1.0 - dry_amount)) / 0.3).clamp(0.0, 1.0) as f32;
            if d > 0.0 {
                color = [
                    lerp(color[0], c.color_dry[0], d),
                    lerp(color[1], c.color_dry[1], d),
                    lerp(color[2], c.color_dry[2], d),
                ];
            }
        }

        // Height: broad cushions plus filament micro-relief — the mix is what
        // makes the normal map read as velvet rather than as rolling dunes.
        let depth = c.cushion_depth.clamp(0.0, 1.0);
        let height = cushion * depth + filament * (1.0 - depth);

        // Moss is very matte; crowns catch marginally more light.
        let rough = 0.94 - tf * 0.10;

        SurfaceSample::matte(height, color, rough)
    }
}

impl TextureGenerator for MossGenerator {
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
        let map = MossGenerator::new(MossConfig::default())
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
    fn tiles_seamlessly() {
        // Toroidal noise on every layer: opposite edges are one texel apart on
        // the torus, so they must be close in value.
        let map = MossGenerator::new(MossConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        for i in 0..64usize {
            // Left/right edge columns.
            let l = (i * 64) * 4;
            let r = (i * 64 + 63) * 4;
            // Top/bottom edge rows.
            let t = i * 4;
            let b = (63 * 64 + i) * 4;
            for ch in 0..3 {
                assert!(
                    (map.albedo[l + ch] as i32 - map.albedo[r + ch] as i32).abs() < 70,
                    "horizontal seam at row {i} channel {ch}"
                );
                assert!(
                    (map.albedo[t + ch] as i32 - map.albedo[b + ch] as i32).abs() < 70,
                    "vertical seam at col {i} channel {ch}"
                );
            }
        }
    }

    #[test]
    fn dry_patches_shift_colour() {
        let lush = MossGenerator::new(MossConfig {
            dry_patches: 0.0,
            ..MossConfig::default()
        })
        .generate(64, 64)
        .expect("generate failed");
        let parched = MossGenerator::new(MossConfig {
            dry_patches: 1.0,
            ..MossConfig::default()
        })
        .generate(64, 64)
        .expect("generate failed");
        // Straw is much redder than moss green: the red channel mean rises.
        let red_mean = |m: &crate::generator::TextureMap| {
            m.albedo.chunks(4).map(|px| px[0] as u64).sum::<u64>() / (64 * 64)
        };
        assert!(
            red_mean(&parched) > red_mean(&lush),
            "drying the carpet should warm it"
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = MossGenerator::new(MossConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = MossGenerator::new(MossConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            MossGenerator::new(MossConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
