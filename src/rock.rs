//! Rock texture generator using Ridged Multifractal noise.
//!
//! Ridged multifractal noise produces sharp, ridge-like features that mimic
//! the cracked and faceted appearance of stone surfaces.

use noise::{MultiFractal, Perlin, RidgedMulti};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`RockGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RockConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Overall spatial scale.
    pub scale: f64,
    /// Octaves for the ridged multifractal noise (more octaves → finer detail).
    pub octaves: usize,
    /// Attenuation of the ridged multifractal (controls sharpness of ridges).
    pub attenuation: f64,
    /// Base (light) rock colour in linear RGB \[0, 1\].
    pub color_light: [f32; 3],
    /// Shadow (dark) colour in linear RGB \[0, 1\].
    pub color_dark: [f32; 3],
    /// Optional ageing pass: run-off streaks down the face, lichen-dark
    /// crevices, wear on exposed ridges.
    ///
    /// Defaults to disabled, so rock is unchanged until a layer is turned up.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength — larger values produce more pronounced surface detail.
    pub normal_strength: f32,
}

impl Default for RockConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            scale: 3.0,
            octaves: 8,
            attenuation: 2.0,
            color_light: [0.37, 0.42, 0.36],
            color_dark: [0.22, 0.20, 0.18],
            weathering: WeatheringConfig::default(),
            normal_strength: 4.0,
        }
    }
}

/// Procedural rock / stone texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`RockConfig`].  Construct
/// via [`RockGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::rock` task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct RockGenerator {
    config: RockConfig,
    noise: ToroidalNoise<RidgedMulti<Perlin>>,
}

impl RockGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the ridged-multifractal noise object up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: RockConfig) -> Self {
        let ridged: RidgedMulti<Perlin> = RidgedMulti::new(config.seed)
            .set_octaves(config.octaves)
            .set_attenuation(config.attenuation);
        let noise = ToroidalNoise::new(ridged, config.scale);
        Self { config, noise }
    }
}

/// Per-generation sampler: precomputed ridged-multifractal grid + config.
struct RockCell<'a> {
    config: &'a RockConfig,
    grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for RockCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        let raw = self.grid[y as usize * self.width + x as usize];
        let t = normalize(raw) as f32;

        let color = [
            lerp(c.color_dark[0], c.color_light[0], t),
            lerp(c.color_dark[1], c.color_light[1], t),
            lerp(c.color_dark[2], c.color_light[2], t),
        ];

        // Ridges (high t) are slightly smoother (exposed mineral); cracks rougher.
        let rough = (0.75 - t * 0.25).clamp(0.0, 1.0);

        // Height stays raw [-1, 1]; generate_inner compensates with
        // strength × 0.5 instead of normalising the whole grid.
        SurfaceSample::matte(raw, color, rough)
    }
}

impl RockGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        let mut grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.noise, width, height, &mut grid);

        let cell = RockCell {
            config: &self.config,
            grid: &grid,
            width: width as usize,
        };
        // Grid values span [-1, 1] (range 2): halving the strength is
        // equivalent to normalising and avoids a full-sized allocation.
        let result = generate_surface_weathered(
            width,
            height,
            self.config.normal_strength * 0.5,
            ws.as_deref_mut(),
            &cell,
            &self.config.weathering,
        );

        if let Some(ws) = ws {
            ws.return_grid(grid);
        }
        result
    }
}

impl TextureGenerator for RockGenerator {
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
    use crate::weathering::Streaks;

    fn bake(config: RockConfig) -> TextureMap {
        RockGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    fn streaked() -> RockConfig {
        RockConfig {
            weathering: WeatheringConfig {
                seed: 4,
                streaks: Streaks {
                    amount: 0.9,
                    density: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn mean_luma(map: &TextureMap) -> f64 {
        map.albedo.chunks(4).map(|px| px[0] as f64).sum::<f64>() / (128.0 * 128.0)
    }

    /// Weathering is opt-in: the default config must produce exactly the rock
    /// it always did, or every downstream golden hash moves.
    #[test]
    fn weathering_is_off_by_default() {
        assert!(RockConfig::default().weathering.is_noop());
        let plain = bake(RockConfig::default());
        let explicit = bake(RockConfig {
            weathering: WeatheringConfig {
                seed: 99,
                ..Default::default()
            },
            ..Default::default()
        });
        assert_eq!(
            plain.albedo, explicit.albedo,
            "a no-op weathering config still changed the rock"
        );
    }

    /// Run-off stains deposit sediment, so they may only darken the face.
    #[test]
    fn streaks_darken_the_face() {
        let plain = bake(RockConfig::default());
        let aged = bake(streaked());
        assert_ne!(plain.albedo, aged.albedo, "streaks had no effect");
        assert!(
            mean_luma(&aged) < mean_luma(&plain),
            "streaks lightened the rock ({:.1} → {:.1})",
            mean_luma(&plain),
            mean_luma(&aged)
        );
    }

    /// Streaks are a colour/ORM effect; the rock itself is not re-carved.
    #[test]
    fn streaks_leave_the_height_field_alone() {
        assert_eq!(
            bake(RockConfig::default()).normal,
            bake(streaked()).normal,
            "streaks disturbed the height field"
        );
    }

    /// The stain length is tile-relative, so a config must age the same at any
    /// bake resolution rather than shrinking with the texel grid.
    #[test]
    fn streaking_is_resolution_relative() {
        let darkening = |size: u32| {
            let mean = |c: RockConfig| {
                let m = RockGenerator::new(c).generate(size, size).expect("bake");
                m.albedo.chunks(4).map(|px| px[0] as f64).sum::<f64>() / (size * size) as f64
            };
            mean(RockConfig::default()) - mean(streaked())
        };
        let small = darkening(128);
        let large = darkening(256);
        assert!(
            (small - large).abs() < small.abs().max(large.abs()) * 0.4,
            "streak darkening drifted with resolution: {small:.2} at 128\u{b2} vs {large:.2} at 256\u{b2}"
        );
    }
}
