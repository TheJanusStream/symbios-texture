//! Enamel and glazed ceramic: a smooth fired finish over a body, optionally
//! crazed with fine crackle.
//!
//! The finish that a brushed-metal generator cannot fake.  Enamel's whole
//! character is the *absence* of directional structure — an even, glossy
//! coat whose only incident is the slight orange-peel of the spray and, on
//! older or deliberately crazed work, a web of hairline cracks in the glaze
//! with the body showing through.
//!
//! Crackle uses [`cellular_edge`], so the craze stays hairline-fine across
//! large and small cells alike; a width that grew with the cell would read as
//! a broken tile rather than a crazed glaze.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{CellularParams, ToroidalNoise, cellular_edge, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of an [`EnamelGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnamelConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Fired glaze colour in linear RGB \[0, 1\].
    pub color: [f32; 3],
    /// Colour showing through the craze — the unglazed body beneath.
    pub color_body: [f32; 3],
    /// Gloss of the coat, as roughness in `[0, 1]`.  Enamel is smooth; much
    /// above `0.4` reads as matt paint instead.
    pub gloss_roughness: f32,
    /// Metallic value.  Enamel is a dielectric coat, so this belongs near
    /// zero unless the finish is deliberately metallic-flake.
    pub metallic: f32,
    /// How much of the glaze is crazed, in `[0, 1]`.  `0` leaves the coat
    /// perfectly clear.
    pub crackle: f32,
    /// Craze cells across the tile — higher is a finer web.
    pub crackle_scale: f64,
    /// Craze line width in UV units, a fraction of the tile so the web looks
    /// the same at any bake resolution.
    pub crackle_width: f64,
    /// Amplitude of the sprayed coat's orange-peel undulation.
    pub orange_peel: f64,
    /// Frequency of the orange-peel undulation.
    pub orange_peel_scale: f64,
    /// Optional ageing pass — chipped edges, grime in the craze.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for EnamelConfig {
    fn default() -> Self {
        Self {
            seed: 17,
            color: [0.62, 0.20, 0.16],
            color_body: [0.80, 0.78, 0.74],
            gloss_roughness: 0.18,
            metallic: 0.0,
            crackle: 0.0,
            crackle_scale: 26.0,
            crackle_width: 0.0025,
            orange_peel: 0.11,
            orange_peel_scale: 34.0,
            weathering: WeatheringConfig::default(),
            normal_strength: 3.5,
        }
    }
}

/// Procedural enamel / glazed-ceramic texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`EnamelConfig`].
pub struct EnamelGenerator {
    config: EnamelConfig,
    peel: ToroidalNoise<Fbm<Perlin>>,
}

impl EnamelGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: EnamelConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(2)).set_octaves(2);
        let peel = ToroidalNoise::new(fbm, config.orange_peel_scale.max(0.1));
        Self { config, peel }
    }
}

/// Per-generation sampler: orange-peel grid plus the craze lattice.
struct EnamelCell<'a> {
    config: &'a EnamelConfig,
    peel: &'a [f64],
    params: CellularParams,
    width: usize,
}

impl SurfaceCell for EnamelCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let peel = normalize(self.peel[y as usize * self.width + x as usize]);

        // The coat's own gentle undulation, always present.
        let mut height = (peel - 0.5) * c.orange_peel;

        let crackle = c.crackle.clamp(0.0, 1.0);
        let craze = if crackle > 0.0 {
            let edge = cellular_edge(u, v, self.params);
            let half = c.crackle_width.max(0.0) * 0.5;
            (1.0 - smoothstep(half, half * 2.5, edge)) * crackle as f64
        } else {
            0.0
        };
        // The craze is a split in the glaze, so it cuts down into the coat.
        height -= craze * 0.5;

        let t = craze as f32;
        let color = [
            lerp(c.color[0], c.color_body[0], t),
            lerp(c.color[1], c.color_body[1], t),
            lerp(c.color[2], c.color_body[2], t),
        ];

        SurfaceSample {
            height,
            color,
            // Glaze is glossy; the exposed body inside a craze is not.
            roughness: lerp(c.gloss_roughness, 0.85, t).clamp(0.0, 1.0),
            metallic: lerp(c.metallic, 0.0, t).clamp(0.0, 1.0),
            occlusion: lerp(1.0, 0.7, t),
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

impl EnamelGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut peel = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.peel, width, height, &mut peel);

        let cell = EnamelCell {
            config: c,
            peel: &peel,
            params: CellularParams::new(c.crackle_scale, c.seed.wrapping_add(9)).with_jitter(0.9),
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
            ws.return_grid(peel);
        }
        result
    }
}

impl TextureGenerator for EnamelGenerator {
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

    fn bake(config: EnamelConfig) -> TextureMap {
        EnamelGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(EnamelConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert_eq!(map.normal.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(EnamelConfig::default()).albedo,
            bake(EnamelConfig::default()).albedo
        );
    }

    /// A clear coat is the default: enamel's character is an *even* finish,
    /// so an unconfigured glaze must not be pre-crazed.
    #[test]
    fn clear_coat_by_default() {
        let map = bake(EnamelConfig::default());
        // Every texel should sit close to the glaze colour.
        let strays = map
            .albedo
            .chunks(4)
            .filter(|px| px[0] < 150 || px[1] > 140)
            .count();
        assert!(
            strays * 20 < 128 * 128,
            "default glaze is not an even coat ({strays} stray texels)"
        );
    }

    /// Crazing must expose the body underneath and stay a minority of the
    /// surface.
    #[test]
    fn crackle_exposes_the_body() {
        let clear = bake(EnamelConfig::default());
        let crazed = bake(EnamelConfig {
            crackle: 1.0,
            ..Default::default()
        });
        assert_ne!(clear.albedo, crazed.albedo, "crackle had no effect");

        // Body is much paler than the glaze, so craze texels read bright.
        let exposed =
            crazed.albedo.chunks(4).filter(|px| px[1] > 150).count() as f64 / (128 * 128) as f64;
        assert!(
            (0.005..0.4).contains(&exposed),
            "craze covers {exposed:.3} of the glaze — not a hairline web"
        );
    }

    /// The reason crackle uses `cellular_edge`: line width must not scale
    /// with cell size.
    #[test]
    fn craze_width_is_independent_of_cell_size() {
        let exposed = |scale: f64| {
            bake(EnamelConfig {
                crackle: 1.0,
                crackle_scale: scale,
                ..Default::default()
            })
            .albedo
            .chunks(4)
            .filter(|px| px[1] > 150)
            .count() as f64
                / (128 * 128) as f64
        };
        let coarse = exposed(12.0);
        let fine = exposed(24.0);
        assert!(
            fine > coarse && fine < coarse * 3.0,
            "craze coverage did not track line length ({coarse:.3} → {fine:.3})"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(EnamelConfig {
            crackle: 9.0,
            crackle_scale: 0.0,
            crackle_width: -1.0,
            orange_peel_scale: 0.0,
            gloss_roughness: 5.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
