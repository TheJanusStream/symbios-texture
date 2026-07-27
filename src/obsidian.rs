//! Obsidian: polished volcanic glass with flow banding.
//!
//! Near-black and near-mirror, so almost all of its read comes from the
//! banding frozen into the melt — long, smoothly-curving sheets that catch
//! the light at glancing angles.  Built from the warped [`stripe`] field:
//! straight bands bent by toroidal noise, which is exactly the shape a slow
//! viscous flow leaves behind.
//!
//! A brushed-metal generator cannot stand in for this.  Brushing is
//! high-frequency scratch detail with no large-scale structure; obsidian is
//! the opposite — no fine detail at all, and everything happening at the
//! scale of the whole surface.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{StripeParams, StripeProfile, ToroidalNoise, normalize, sample_grid_into, stripe},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of an [`ObsidianGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ObsidianConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Body colour in linear RGB \[0, 1\] — obsidian proper is near-black.
    pub color: [f32; 3],
    /// Colour of the banding catching the light; a cool or iridescent tint
    /// here gives sheen and rainbow obsidian.
    pub color_sheen: [f32; 3],
    /// Whole band cycles across the tile horizontally.  Rounded, because the
    /// underlying field only tiles at whole cycle counts.
    pub band_cycles_u: f64,
    /// Whole band cycles across the tile vertically.
    pub band_cycles_v: f64,
    /// How far the bands are bent from straight, in turns of phase.  This is
    /// what separates flow banding from a striped sheet.
    ///
    /// Keep it well under half a turn: past that the bend exceeds half a band
    /// period, the bands fold back through one another, and smooth flow turns
    /// into a churn.
    pub band_warp: f64,
    /// Frequency of the noise doing the bending; low values give long lazy
    /// curves, high values a churned melt.
    pub band_warp_scale: f64,
    /// How tightly the banding is drawn, in `[0, 1]`.
    pub band_sharpness: f64,
    /// How strongly banding shows against the body, in `[0, 1]`.
    pub band_contrast: f32,
    /// Gloss of the polished face, as roughness in `[0, 1]`.
    pub gloss_roughness: f32,
    /// Metallic value — glass is dielectric, but a little goes a long way
    /// toward the near-mirror look at stylised lighting levels.
    pub metallic: f32,
    /// Relief of the banding in the height field.  Polished obsidian is
    /// almost flat, so this stays small.
    pub relief: f64,
    /// Optional ageing pass — dulled edges, dust in the hollows.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ObsidianConfig {
    fn default() -> Self {
        Self {
            seed: 29,
            color: [0.035, 0.032, 0.045],
            color_sheen: [0.16, 0.15, 0.22],
            band_cycles_u: 5.0,
            band_cycles_v: 2.0,
            band_warp: 0.26,
            band_warp_scale: 1.6,
            band_sharpness: 0.35,
            band_contrast: 0.8,
            gloss_roughness: 0.12,
            metallic: 0.6,
            relief: 0.05,
            weathering: WeatheringConfig::default(),
            normal_strength: 0.8,
        }
    }
}

/// Procedural obsidian texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`ObsidianConfig`].
pub struct ObsidianGenerator {
    config: ObsidianConfig,
    warp: ToroidalNoise<Fbm<Perlin>>,
}

impl ObsidianGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ObsidianConfig) -> Self {
        // Two octaves, not more: flow banding is a slow viscous fold, and
        // the fine detail extra octaves add reads as grain on the glass
        // rather than movement in it.
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(4)).set_octaves(2);
        let warp = ToroidalNoise::new(fbm, config.band_warp_scale.max(0.1));
        Self { config, warp }
    }
}

/// Per-generation sampler: warp grid plus the band field.
struct ObsidianCell<'a> {
    config: &'a ObsidianConfig,
    warp: &'a [f64],
    bands: StripeParams,
    width: usize,
}

impl SurfaceCell for ObsidianCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let warp = (normalize(self.warp[y as usize * self.width + x as usize]) - 0.5) * c.band_warp;

        let band = stripe(u, v, self.bands, warp);
        let t = (band as f32 * c.band_contrast.clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let color = [
            lerp(c.color[0], c.color_sheen[0], t),
            lerp(c.color[1], c.color_sheen[1], t),
            lerp(c.color[2], c.color_sheen[2], t),
        ];

        SurfaceSample {
            // Barely any relief: the banding is *in* the glass, not on it.
            height: band * c.relief,
            color,
            // Sheen bands polish marginally brighter than the body.
            roughness: (c.gloss_roughness * (1.0 - t * 0.25)).clamp(0.0, 1.0),
            metallic: c.metallic.clamp(0.0, 1.0),
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

impl ObsidianGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut warp = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.warp, width, height, &mut warp);

        let cell = ObsidianCell {
            config: c,
            warp: &warp,
            bands: StripeParams::new(
                c.band_cycles_u.round() as i32,
                c.band_cycles_v.round() as i32,
            )
            .with_profile(StripeProfile::Sine)
            .with_sharpness(c.band_sharpness),
            width: width as usize,
        };
        let result = generate_surface_weathered(
            width,
            height,
            c.normal_strength,
            ws.as_deref_mut(),
            &cell,
            &c.weathering,
        );

        if let Some(ws) = ws {
            ws.return_grid(warp);
        }
        result
    }
}

impl TextureGenerator for ObsidianGenerator {
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

    fn bake(config: ObsidianConfig) -> TextureMap {
        ObsidianGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(ObsidianConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(ObsidianConfig::default()).albedo,
            bake(ObsidianConfig::default()).albedo
        );
        assert_ne!(
            bake(ObsidianConfig::default()).albedo,
            bake(ObsidianConfig {
                seed: 777,
                ..Default::default()
            })
            .albedo
        );
    }

    /// Obsidian is dark glass; if the surface reads mid-grey it has stopped
    /// being obsidian.
    #[test]
    fn stays_dark() {
        let map = bake(ObsidianConfig::default());
        let mean = map.albedo.chunks(4).map(|px| px[0] as f64).sum::<f64>() / (128.0 * 128.0);
        assert!(mean < 110.0, "obsidian averaged {mean:.1} — too bright");
    }

    /// Polished glass is smooth everywhere; a rough patch would read as
    /// stone.
    #[test]
    fn is_uniformly_glossy() {
        let map = bake(ObsidianConfig::default());
        let roughest = map
            .roughness
            .chunks(4)
            .map(|px| px[1])
            .max()
            .expect("texels");
        assert!(
            roughest < 90,
            "roughness peaked at {roughest} — not polished"
        );
    }

    /// Warping the bands is what turns stripes into flow; it must reach the
    /// output.
    #[test]
    fn warp_bends_the_banding() {
        let straight = bake(ObsidianConfig {
            band_warp: 0.0,
            ..Default::default()
        });
        let flowing = bake(ObsidianConfig::default());
        assert_ne!(straight.albedo, flowing.albedo, "warp had no effect");

        // Unwarped bands are constant along their own direction; warped ones
        // are not. Sample a row and count distinct-ish runs as a proxy.
        let row_variance = |map: &TextureMap| {
            let row: Vec<f64> = (0..128)
                .map(|x| map.albedo[(64 * 128 + x) * 4] as f64)
                .collect();
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64
        };
        assert!(
            row_variance(&flowing) > 0.0 && row_variance(&straight) > 0.0,
            "banding vanished entirely"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(ObsidianConfig {
            band_cycles_u: 0.0,
            band_cycles_v: 0.0,
            band_warp: 1e6,
            band_warp_scale: 0.0,
            band_sharpness: 9.0,
            band_contrast: -3.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
