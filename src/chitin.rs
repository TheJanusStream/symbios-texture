//! Chitin: the hard glossy plating of a carapace.
//!
//! Built on [`cellular_smooth`] rather than plain Worley `F1`.  A hard
//! minimum meets its neighbours at a crease, which reads as cut stone; a
//! softmin lets the plates *swell into* one another, which is how a shell
//! grown from a single sheet actually looks.  The seam between plates is then
//! drawn back in deliberately, as a sutured line rather than a fold.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{
        CellularParams, ToroidalNoise, cell_hash, cellular, cellular_smooth, normalize,
        sample_grid_into,
    },
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`ChitinGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChitinConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Plates across the tile.
    pub scale: f64,
    /// How irregularly the plates are laid, in `[0, 1]`.
    pub jitter: f64,
    /// How softly plates merge into their neighbours.  Low values swell the
    /// plates together into one organic sheet; high values approach the hard
    /// creases of cut stone.
    pub softness: f64,
    /// How far a plate swells before the seam, in `[0, 1]` of the distance to
    /// its neighbour.
    pub plate_fill: f64,
    /// Doming of each plate in the height field.
    pub plate_relief: f64,
    /// Width of the sutured seam between plates, in UV units.
    pub seam_width: f64,
    /// How dark the seam runs, in `[0, 1]`.
    pub seam_depth: f32,
    /// Spread of per-plate colour variation in `[0, 1]`, applied as a
    /// multiplier — the shifting sheen across a carapace.
    pub iridescence: f32,
    /// Lit shell colour in linear RGB \[0, 1\].
    pub color: [f32; 3],
    /// Deep shell colour in linear RGB \[0, 1\], seen in the seams and on
    /// plates that sit lower.
    pub color_deep: [f32; 3],
    /// Gloss of the shell, as roughness in `[0, 1]`.
    pub gloss_roughness: f32,
    /// Metallic value.
    pub metallic: f32,
    /// Frequency of the fine pitting over the shell.
    pub pit_scale: f64,
    /// Optional ageing pass — scuffed ridges, grime in the seams.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ChitinConfig {
    fn default() -> Self {
        Self {
            seed: 37,
            scale: 6.0,
            jitter: 0.75,
            softness: 24.0,
            plate_fill: 0.9,
            plate_relief: 0.55,
            seam_width: 0.006,
            seam_depth: 0.75,
            iridescence: 0.22,
            color: [0.20, 0.34, 0.20],
            color_deep: [0.05, 0.09, 0.07],
            gloss_roughness: 0.28,
            metallic: 0.45,
            pit_scale: 40.0,
            weathering: WeatheringConfig::default(),
            normal_strength: 2.0,
        }
    }
}

/// Procedural chitin texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`ChitinConfig`].
pub struct ChitinGenerator {
    config: ChitinConfig,
    pits: ToroidalNoise<Fbm<Perlin>>,
}

impl ChitinGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ChitinConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(6)).set_octaves(2);
        let pits = ToroidalNoise::new(fbm, config.pit_scale.max(0.1));
        Self { config, pits }
    }
}

/// Per-generation sampler: pit grid plus the plate lattice.
struct ChitinCell<'a> {
    config: &'a ChitinConfig,
    pits: &'a [f64],
    params: CellularParams,
    width: usize,
}

impl SurfaceCell for ChitinCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let pit = normalize(self.pits[y as usize * self.width + x as usize]);

        let plate = cellular(u, v, self.params);
        // Softmin distance: plates merge instead of creasing.
        let soft = cellular_smooth(u, v, self.params, c.softness.max(1.0));

        let fill = c.plate_fill.clamp(0.05, 1.0);
        let t = (soft / (plate.f2 * fill).max(1e-9)).clamp(0.0, 1.0);
        let dome = 1.0 - t * t;

        // The seam is drawn back in on purpose — the softmin removed it.
        let seam = 1.0 - smoothstep(0.0, c.seam_width.max(1e-9), plate.ridge());

        // Each plate carries its own sheen, applied as a multiplier so dark
        // and light shells vary by the same *proportion*; added directly it
        // pushes plates clean out of the shell's colour family.
        let sheen = (cell_hash(plate.cell_x, plate.cell_y, c.seed.wrapping_add(23)) - 0.5) as f32
            * 2.0
            * c.iridescence;

        let lit = (dome as f32 * 0.75 + 0.25).clamp(0.0, 1.0);
        let mut color = [
            lerp(c.color_deep[0], c.color[0], lit) * (1.0 + sheen),
            lerp(c.color_deep[1], c.color[1], lit) * (1.0 + sheen * 0.7),
            lerp(c.color_deep[2], c.color[2], lit) * (1.0 + sheen * 1.2),
        ];

        // Sink the seam toward the deep colour.
        let seam_t = seam as f32 * c.seam_depth.clamp(0.0, 1.0);
        for (channel, deep) in color.iter_mut().zip(c.color_deep) {
            *channel = lerp(*channel, deep, seam_t).clamp(0.0, 1.0);
        }

        let height = dome * c.plate_relief - seam * c.plate_relief * 0.6 + (pit - 0.5) * 0.06;

        SurfaceSample {
            height,
            color,
            // Shell is glossy; the sutures are matt where they hold grime.
            roughness: lerp(c.gloss_roughness, 0.8, seam_t).clamp(0.0, 1.0),
            metallic: lerp(c.metallic, 0.0, seam_t).clamp(0.0, 1.0),
            occlusion: lerp(1.0, 0.55, seam_t),
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

impl ChitinGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut pits = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.pits, width, height, &mut pits);

        let cell = ChitinCell {
            config: c,
            pits: &pits,
            params: CellularParams::new(c.scale, c.seed).with_jitter(c.jitter),
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
            ws.return_grid(pits);
        }
        result
    }
}

impl TextureGenerator for ChitinGenerator {
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

    fn bake(config: ChitinConfig) -> TextureMap {
        ChitinGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(ChitinConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(ChitinConfig::default()).albedo,
            bake(ChitinConfig::default()).albedo
        );
        assert_ne!(
            bake(ChitinConfig::default()).albedo,
            bake(ChitinConfig {
                seed: 555,
                ..Default::default()
            })
            .albedo
        );
    }

    /// Softness is the whole reason this uses a softmin: turning it down must
    /// visibly change how the plates meet.
    #[test]
    fn softness_changes_how_plates_merge() {
        let merged = bake(ChitinConfig {
            softness: 6.0,
            ..Default::default()
        });
        let creased = bake(ChitinConfig {
            softness: 200.0,
            ..Default::default()
        });
        assert_ne!(
            merged.normal, creased.normal,
            "softness did not change the plate geometry"
        );
    }

    /// Seams must be present but narrow — a carapace is plates, not a net.
    #[test]
    fn seams_are_narrow() {
        let seamed = bake(ChitinConfig::default());
        let seamless = bake(ChitinConfig {
            seam_depth: 0.0,
            ..Default::default()
        });
        assert_ne!(seamed.albedo, seamless.albedo, "seams had no effect");

        let dark =
            seamed.albedo.chunks(4).filter(|px| px[1] < 60).count() as f64 / (128 * 128) as f64;
        assert!(
            dark < 0.35,
            "seams cover {dark:.3} of the shell — too wide to be sutures"
        );
    }

    /// Shell is glossy plating; a matt result would read as bark or hide.
    #[test]
    fn shell_is_glossy() {
        let map = bake(ChitinConfig::default());
        let mean_rough =
            map.roughness.chunks(4).map(|px| px[1] as f64).sum::<f64>() / (128.0 * 128.0);
        assert!(
            mean_rough < 140.0,
            "shell averaged roughness {mean_rough:.1} — not glossy plating"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(ChitinConfig {
            scale: 0.0,
            jitter: 9.0,
            softness: 0.0,
            plate_fill: -1.0,
            seam_width: -1.0,
            seam_depth: 9.0,
            pit_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
