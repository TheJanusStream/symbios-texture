//! Iron grille / portcullis alpha-card texture generator.
//!
//! The algorithm:
//! 1. For each pixel compute the signed distance to every vertical bar and
//!    every horizontal bar.  Keep the minimum (nearest) distance for each axis.
//! 2. A pixel is "on a bar" when the combined SDF (min of the two axes) is ≤ 0.
//! 3. When a pixel lies inside both a vertical and a horizontal bar it is a
//!    joint/intersection — rust accumulates more heavily there.
//! 4. A non-toroidal FBM provides the rust distribution noise.
//! 5. When `round_bars` is true the height profile follows a circular cross-
//!    section (`(1 – |d/r|)²`) so the bars look cylindrical in the normal map.
//! 6. Baked as a card through the [`surface`](crate::surface) driver:
//!    clamp-to-edge normals over a silhouette-dilated height field, so the
//!    openings leave no cliff at the bar edges.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    surface::{SurfaceCell, SurfaceOptions, SurfaceSample, generate_surface_with, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of an [`IronGrilleGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IronGrilleConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Number of vertical bars \[2, 10\].
    pub bars_x: usize,
    /// Number of horizontal bars \[2, 10\].
    pub bars_y: usize,
    /// Bar half-width as fraction of card \[0.02, 0.20\].
    pub bar_width: f64,
    /// Round the bar cross-section (`true`) vs rectangular (`false`).
    pub round_bars: bool,
    /// Rust accumulation at joints \[0, 1\].
    pub rust_level: f64,
    /// Iron colour in linear RGB \[0, 1\].
    pub color_iron: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Optional ageing pass — wear on exposed edges, grime in the
    /// recesses, corrosion and run-off streaks.
    ///
    /// Defaults to disabled, so the surface is unchanged until a layer
    /// is turned up.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for IronGrilleConfig {
    fn default() -> Self {
        Self {
            seed: 71,
            bars_x: 4,
            bars_y: 6,
            bar_width: 0.04,
            round_bars: true,
            rust_level: 0.30,
            color_iron: [0.14, 0.13, 0.13],
            color_rust: [0.42, 0.22, 0.08],
            weathering: WeatheringConfig::default(),
            normal_strength: 3.5,
        }
    }
}

/// Procedural iron grille / portcullis texture generator (alpha-card type).
///
/// Drives [`TextureGenerator::generate`] using an [`IronGrilleConfig`].  Construct
/// via [`IronGrilleGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::iron_grille` task for non-blocking generation.
///
/// The result has per-pixel alpha: fully transparent in the openings between
/// bars, and fully opaque on the bars themselves.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct IronGrilleGenerator {
    config: IronGrilleConfig,
    rust_fbm: Fbm<Perlin>,
}

impl IronGrilleGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: IronGrilleConfig) -> Self {
        let rust_fbm: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(4);
        Self { config, rust_fbm }
    }
}

/// Per-generation sampler: the bar counts and the rust noise.
struct IronGrilleCell<'a> {
    config: &'a IronGrilleConfig,
    rust_fbm: &'a Fbm<Perlin>,
    bars_x: usize,
    bars_y: usize,
}

impl SurfaceCell for IronGrilleCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        // Center in [-0.5, 0.5].
        let px = u - 0.5;
        let py = v - 0.5;

        // Minimum SDF to any vertical bar.
        let mut min_v_sdf = f64::MAX;
        for i in 0..self.bars_x {
            let x_center = (i as f64 + 0.5) / self.bars_x as f64 - 0.5;
            let d = (px - x_center).abs() - c.bar_width;
            if d < min_v_sdf {
                min_v_sdf = d;
            }
        }

        // Minimum SDF to any horizontal bar.
        let mut min_h_sdf = f64::MAX;
        for j in 0..self.bars_y {
            let y_center = (j as f64 + 0.5) / self.bars_y as f64 - 0.5;
            let d = (py - y_center).abs() - c.bar_width;
            if d < min_h_sdf {
                min_h_sdf = d;
            }
        }

        // Combined: a pixel is on the grille when it's inside either axis.
        let bar_sdf = min_v_sdf.min(min_h_sdf);
        if bar_sdf > 0.0 {
            // Fully transparent opening; ORM is a neutral sentinel.
            return SurfaceSample {
                height: 0.0,
                color: [0.0, 0.0, 0.0],
                roughness: 1.0,
                metallic: 0.0,
                occlusion: 1.0,
                emissive: [0.0, 0.0, 0.0],
                alpha: 0.0,
            };
        }

        // Joint: pixel is inside both a vertical and a horizontal bar.
        let is_joint = min_v_sdf <= 0.0 && min_h_sdf <= 0.0;

        // Rust noise — sample at medium frequency so it blotches naturally.
        let rust_raw = self.rust_fbm.get([u * 8.0, v * 8.0]) * 0.5 + 0.5; // [0, 1]
        let joint_boost = if is_joint { 1.5_f64 } else { 0.5_f64 };
        let rust_t = (rust_raw * c.rust_level * joint_boost).clamp(0.0, 1.0);

        // Height: round cross-section profile when enabled.
        let h_val = if c.round_bars {
            // bar_sdf is negative inside; its magnitude / bar_width gives
            // normalised depth from the bar surface inward.
            let depth = (bar_sdf.abs() / c.bar_width).clamp(0.0, 1.0);
            (1.0 - depth).powi(2)
        } else {
            1.0
        };

        // Joints are slightly proud of the bar surface.
        let joint_bump = if is_joint { 0.15_f64 } else { 0.0_f64 };

        // Colour: lerp iron → rust driven by rust_t.  ORM: iron is metallic;
        // rust degrades both smoothness and metallicness.
        SurfaceSample {
            height: (h_val + joint_bump).clamp(0.0, 1.0),
            color: [
                lerp(c.color_iron[0], c.color_rust[0], rust_t as f32),
                lerp(c.color_iron[1], c.color_rust[1], rust_t as f32),
                lerp(c.color_iron[2], c.color_rust[2], rust_t as f32),
            ],
            roughness: (0.35 + rust_t * 0.40) as f32,
            metallic: (0.85 - rust_t * 0.50).max(0.0) as f32,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: 1.0,
        }
    }
}

impl IronGrilleGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        workspace: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;
        let cell = IronGrilleCell {
            config: c,
            rust_fbm: &self.rust_fbm,
            bars_x: c.bars_x.max(1),
            bars_y: c.bars_y.max(1),
        };
        generate_surface_with(
            width,
            height,
            c.normal_strength,
            workspace,
            &cell,
            SurfaceOptions::default()
                .with_card(true)
                .with_weathering(&c.weathering),
        )
    }
}

impl TextureGenerator for IronGrilleGenerator {
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
