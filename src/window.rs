//! Window texture generator using 2-D signed distance functions (SDF).
//!
//! The algorithm:
//! 1. Compute a plain rectangular outer silhouette (full card) and a
//!    rounded-box SDF for the inner glass opening.
//! 2. Classify each pixel as frame, mullion, or glass.
//! 3. Subdivide the glass region into `panes_x × panes_y` panes separated by
//!    mullions using fractional UV within the inner glass area.
//! 4. Add FBM grime noise to the glass surface and roughness map.
//! 5. Produce a card whose alpha carries the glass translucency
//!    (clamp-to-edge sampler via `map_to_images_card`), baked through the
//!    [`surface`](crate::surface) driver as a card: clamp-to-edge normals
//!    over a silhouette-dilated height field.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    surface::{SurfaceCell, SurfaceOptions, SurfaceSample, generate_surface_with},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`WindowGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WindowConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Frame width as a fraction of the card \[0, 0.4\].
    pub frame_width: f64,
    /// Number of panes in the horizontal direction.
    pub panes_x: usize,
    /// Number of panes in the vertical direction.
    pub panes_y: usize,
    /// Mullion/muntin thickness as a fraction of the glass area \[0, 0.2\].
    pub mullion_thickness: f64,
    /// Inner (glass-opening) corner-rounding radius as a fraction of the card \[0, 0.4\].
    /// The outer silhouette is always a plain rectangle.
    pub corner_radius: f64,
    /// Glass opacity \[0 = clear, 1 = frosted/opaque\].
    pub glass_opacity: f64,
    /// Grime/dirt noise intensity on glass \[0, 1\].
    pub grime_level: f64,
    /// Frame and mullion colour in linear RGB \[0, 1\].
    pub color_frame: [f32; 3],
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

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            frame_width: 0.08,
            panes_x: 2,
            panes_y: 3,
            mullion_thickness: 0.025,
            corner_radius: 0.02,
            glass_opacity: 0.30,
            grime_level: 0.15,
            color_frame: [0.85, 0.82, 0.78],
            weathering: WeatheringConfig::default(),
            normal_strength: 3.0,
        }
    }
}

/// Procedural window / glazing texture generator (alpha-card type).
///
/// Drives [`TextureGenerator::generate`] using a [`WindowConfig`].  Construct
/// via [`WindowGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::window` task for non-blocking generation.
///
/// The result has per-pixel alpha: opaque frame and mullions,
/// semi-transparent glass (`glass_opacity`).  The frame band extends to the
/// card edge, so no pixel is fully transparent.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct WindowGenerator {
    config: WindowConfig,
    grime_fbm: Fbm<Perlin>,
}

impl WindowGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: WindowConfig) -> Self {
        let grime_fbm: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(6);
        Self { config, grime_fbm }
    }
}

/// Per-generation sampler: the frame geometry and the grime noise.
struct WindowCell<'a> {
    config: &'a WindowConfig,
    grime_fbm: &'a Fbm<Perlin>,
    /// Inner (glass-opening) half-extent, before corner rounding.
    inner_half: f64,
    /// Inner corner radius, floored so the SDF is well-formed.
    inner_r: f64,
    /// UV span and origin of the glass region.
    glass_span: f64,
    glass_origin: f64,
    mullion_half: f64,
    panes_x: usize,
    panes_y: usize,
    /// The glass alpha byte, as a coverage the driver packs back to it.
    glass_alpha: f64,
}

/// ORM bytes the hand-rolled loop wrote with `(x * 255.0) as u8`, which
/// truncates where [`generate_surface_with`] rounds: `0.85 × 255` is 216.75
/// and would pack as 217 through the driver.  Named as the byte so the port
/// is provably free of visual change; harmonising is #18's family.
const FRAME_ROUGHNESS: f32 = 191.0 / 255.0;
const GLASS_METALLIC: f32 = 216.0 / 255.0;

impl SurfaceCell for WindowCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let px = u - 0.5; // centered in [-0.5, 0.5]
        let py = v - 0.5;

        // The outer silhouette (r=0, half-extents 0.5) covers every pixel of
        // the card: px,py ∈ [-0.5,0.5] so outer_sdf ≤ 0 always.
        let inner_sdf = sdf_rounded_box(
            px,
            py,
            self.inner_half - self.inner_r,
            self.inner_half - self.inner_r,
            self.inner_r,
        );

        if inner_sdf > 0.0 {
            // Frame band between outer and inner SDF.  Height ramps from 0 at
            // the outer edge inward to 1 deep in the frame; distance from the
            // nearest outer edge is min(0.5-|px|, 0.5-|py|).
            let edge_dist = (0.5 - px.abs()).min(0.5 - py.abs());
            let edge_t = (edge_dist / (c.frame_width + 0.005)).clamp(0.0, 1.0);
            return SurfaceSample::matte(edge_t, c.color_frame, FRAME_ROUGHNESS);
        }

        // Inside glass area.  Map pixel to glass-local UV in [0, 1].
        let gu = ((u - self.glass_origin) / self.glass_span).clamp(0.0, 1.0);
        let gv = ((v - self.glass_origin) / self.glass_span).clamp(0.0, 1.0);

        // Fractional position within each pane.
        let pu = (gu * self.panes_x as f64).fract();
        let pv = (gv * self.panes_y as f64).fract();

        // Mullion half-width scaled to pane-local coordinates so that internal
        // mullions have the same physical thickness regardless of how many
        // panes there are.
        let mhx = self.mullion_half * self.panes_x as f64;
        let mhy = self.mullion_half * self.panes_y as f64;

        // Check if we're on an internal mullion line.  The outer boundary
        // lines coincide with the frame so panes_x=1 never produces spurious
        // mullions inside the glass.
        let is_mullion_x = self.panes_x > 1 && (pu < mhx || pu > 1.0 - mhx);
        let is_mullion_y = self.panes_y > 1 && (pv < mhy || pv > 1.0 - mhy);

        // For single-pane axis, still suppress the outer boundary band using
        // the same thickness as internal mullions (in glass-UV).
        let at_x_border = gu < self.mullion_half || gu > 1.0 - self.mullion_half;
        let at_y_border = gv < self.mullion_half || gv > 1.0 - self.mullion_half;

        if is_mullion_x || is_mullion_y || at_x_border || at_y_border {
            // Mullion — treat like frame.
            return SurfaceSample::matte(1.0, c.color_frame, FRAME_ROUGHNESS);
        }

        // Glass pane.
        let grime_raw = self.grime_fbm.get([u * 8.0, v * 8.0]) * 0.5 + 0.5;
        let grime = grime_raw * c.grime_level;

        // Light blue-grey glass tint, darkened slightly by grime.  Low
        // roughness, high metallic (simulates reflection).
        SurfaceSample {
            height: grime * 0.08,
            color: [
                (0.82 - grime as f32 * 0.3).clamp(0.0, 1.0),
                (0.88 - grime as f32 * 0.2).clamp(0.0, 1.0),
                (0.93 - grime as f32 * 0.1).clamp(0.0, 1.0),
            ],
            roughness: (0.05 + grime as f32 * 0.3).clamp(0.0, 1.0),
            metallic: GLASS_METALLIC,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: self.glass_alpha,
        }
    }
}

impl WindowGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        workspace: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Inner half-extent = outer half-extent (0.5) minus frame.  Keep the
        // inner corner radius at least a small value so the SDF is
        // well-formed; the outer silhouette is a plain rectangle.
        let inner_half = 0.5 - c.frame_width;
        let cell = WindowCell {
            config: c,
            grime_fbm: &self.grime_fbm,
            inner_half,
            inner_r: c.corner_radius.min(inner_half * 0.9).max(0.005),
            glass_span: inner_half * 2.0,
            glass_origin: 0.5 - inner_half,
            mullion_half: (c.mullion_thickness * 0.5).min(0.49),
            panes_x: c.panes_x.max(1),
            panes_y: c.panes_y.max(1),
            glass_alpha: f64::from((c.glass_opacity * 255.0).round() as u8) / 255.0,
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

impl TextureGenerator for WindowGenerator {
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

// --- helpers ----------------------------------------------------------------

/// Signed distance to a rounded rectangle centred at the origin.
/// Negative inside, positive outside.
/// `bx`, `by` are inner half-extents (before rounding); `r` is the corner radius.
#[inline]
fn sdf_rounded_box(px: f64, py: f64, bx: f64, by: f64, r: f64) -> f64 {
    let dx = px.abs() - bx;
    let dy = py.abs() - by;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    let inside = dx.max(dy).min(0.0);
    outside + inside - r
}
