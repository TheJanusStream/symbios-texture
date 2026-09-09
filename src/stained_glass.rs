//! Stained-glass alpha-card texture generator.
//!
//! The algorithm:
//! 1. Build a toroidal Voronoi diagram using an `n × n` grid of jittered sites
//!    (where `n = round(sqrt(cell_count))`).  For each pixel compute F1 and F2
//!    distances plus the integer cell ID of the nearest site.
//! 2. Pixels whose `F2 – F1` is below `lead_width / n` lie on a lead came
//!    boundary and are rendered as opaque dark metal.
//! 3. Glass pixels receive a vibrant HSV colour derived from the cell hash.
//!    The saturation parameter scales the chroma.  A grime FBM adds subtle
//!    dirt accumulation.
//! 4. Alpha: lead came = 255 (fully opaque); glass = 180 (semi-transparent).
//! 5. Heights: lead = 1.0 (proud of the glass face); glass surface = grime bump.
//!    Baked as a card through the [`surface`](crate::surface) driver —
//!    clamp-to-edge normals over a silhouette-dilated height field — because
//!    this is a card texture that must not tile.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    surface::{SurfaceCell, SurfaceOptions, SurfaceSample, generate_surface_with},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`StainedGlassGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StainedGlassConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Approximate number of glass cells \[5, 25\].
    pub cell_count: usize,
    /// Lead came (border) width as fraction of cell spacing \[0.02, 0.12\].
    pub lead_width: f64,
    /// Glass colour saturation factor \[0.5, 1.0\].
    pub saturation: f32,
    /// Glass surface roughness (waviness) \[0, 0.15\].
    pub glass_roughness: f64,
    /// Grime/dirt accumulation on glass \[0, 0.5\].
    pub grime_level: f64,
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

impl Default for StainedGlassConfig {
    fn default() -> Self {
        Self {
            seed: 63,
            cell_count: 12,
            lead_width: 0.05,
            saturation: 0.85,
            glass_roughness: 0.06,
            grime_level: 0.12,
            weathering: WeatheringConfig::default(),
            normal_strength: 2.5,
        }
    }
}

/// Procedural stained-glass texture generator (alpha-card type).
///
/// Drives [`TextureGenerator::generate`] using a [`StainedGlassConfig`].  Construct
/// via [`StainedGlassGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::stained_glass` task for non-blocking generation.
///
/// The result has per-pixel alpha: semi-transparent glass cells separated by
/// fully-opaque lead came lines.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct StainedGlassGenerator {
    config: StainedGlassConfig,
    grime_fbm: Fbm<Perlin>,
}

impl StainedGlassGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: StainedGlassConfig) -> Self {
        let grime_fbm: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(11)).set_octaves(5);
        Self { config, grime_fbm }
    }
}

/// Per-generation sampler: the site grid and the grime noise.
struct StainedGlassCell<'a> {
    config: &'a StainedGlassConfig,
    grime_fbm: &'a Fbm<Perlin>,
    /// Sites per axis; `grid_n²` is roughly `cell_count`.
    grid_n: i64,
    /// Lead threshold in UV distance units.
    lead_threshold: f64,
}

/// ORM and alpha bytes the hand-rolled loop wrote with `(x * 255.0) as u8`,
/// which truncates where [`generate_surface_with`] rounds.  `0.70 × 255`
/// lands on 178.5 in `f32` and would pack as 179 through the driver; the
/// other two agree either way but are named the same so the four bytes read
/// as one decision.  Naming the byte keeps the port provably free of visual
/// change; harmonising is #18's family.
const LEAD_ROUGHNESS: f32 = 102.0 / 255.0;
const LEAD_METALLIC: f32 = 204.0 / 255.0;
const GLASS_METALLIC: f32 = 178.0 / 255.0;
const GLASS_ALPHA: f64 = 180.0 / 255.0;

impl SurfaceCell for StainedGlassCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let (f1, f2, ci, cj) = voronoi_f1_f2(u, v, self.grid_n, c.seed);

        if (f2 - f1) < self.lead_threshold {
            // Lead came: opaque dark metal, proud of the glass face.
            return SurfaceSample {
                height: 1.0,
                color: [0.05, 0.05, 0.06],
                roughness: LEAD_ROUGHNESS,
                metallic: LEAD_METALLIC,
                occlusion: 1.0,
                emissive: [0.0, 0.0, 0.0],
                alpha: 1.0,
            };
        }

        // Glass pane: derive vibrant colour from cell hash.
        let hue = cell_hash(ci, cj, c.seed.wrapping_add(100));
        let sat_h = cell_hash(cj, ci, c.seed.wrapping_add(200));
        let saturation = (0.70 + sat_h * 0.30) * c.saturation as f64;
        let glass_rgb = hsv_to_rgb(hue, saturation.clamp(0.0, 1.0), 0.85);

        // Grime: FBM dirt on the glass surface, darkening it slightly.
        let grime_raw = self.grime_fbm.get([u * 6.0, v * 6.0]) * 0.5 + 0.5;
        let grime = (grime_raw * c.grime_level) as f32;

        // Glass ORM: low roughness, high metallic (simulates reflections).
        let glass_rough =
            (c.glass_roughness + grime_raw * c.grime_level * 0.2).clamp(0.0, 1.0) as f32;

        SurfaceSample {
            height: grime_raw * c.grime_level * 0.05,
            color: [
                (glass_rgb[0] - grime * 0.25).clamp(0.0, 1.0),
                (glass_rgb[1] - grime * 0.20).clamp(0.0, 1.0),
                (glass_rgb[2] - grime * 0.15).clamp(0.0, 1.0),
            ],
            roughness: glass_rough,
            metallic: GLASS_METALLIC,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: GLASS_ALPHA,
        }
    }
}

impl StainedGlassGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        workspace: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;
        // Grid size: n×n gives approximately cell_count cells (n² ≈ cell_count).
        let grid_n = ((c.cell_count as f64).sqrt().round() as i64).max(2);
        let cell = StainedGlassCell {
            config: c,
            grime_fbm: &self.grime_fbm,
            grid_n,
            lead_threshold: c.lead_width / grid_n as f64,
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

impl TextureGenerator for StainedGlassGenerator {
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

// --- Voronoi ----------------------------------------------------------------

/// Grid-based toroidal Voronoi returning `(F1, F2, cell_i, cell_j)`.
///
/// The grid has `n × n` cells.  Sites are jittered within `[0.1, 0.9]` of
/// each cell.  Distances wrap toroidally in `[0, 1]²`.
fn voronoi_f1_f2(u: f64, v: f64, n: i64, seed: u32) -> (f64, f64, i64, i64) {
    let su = u * n as f64;
    let sv = v * n as f64;
    let gi = su.floor() as i64;
    let gj = sv.floor() as i64;

    let mut f1 = f64::MAX;
    let mut f2 = f64::MAX;
    let mut bi = gi;
    let mut bj = gj;

    for di in -2i64..=2 {
        for dj in -2i64..=2 {
            let ni = (gi + di).rem_euclid(n);
            let nj = (gj + dj).rem_euclid(n);

            // Jitter within [0.1, 0.9] to avoid degenerate zero-area cells.
            let jx = 0.10 + 0.80 * cell_hash(ni, nj, seed);
            let jy = 0.10 + 0.80 * cell_hash(nj, ni + 1, seed.wrapping_add(31));

            let cx = (ni as f64 + jx) / n as f64;
            let cy = (nj as f64 + jy) / n as f64;

            // Toroidal distance.
            let mut dx = (u - cx).abs();
            let mut dy = (v - cy).abs();
            if dx > 0.5 {
                dx = 1.0 - dx;
            }
            if dy > 0.5 {
                dy = 1.0 - dy;
            }
            let d = (dx * dx + dy * dy).sqrt();

            if d < f1 {
                f2 = f1;
                f1 = d;
                bi = ni;
                bj = nj;
            } else if d < f2 {
                f2 = d;
            }
        }
    }

    (f1, f2, bi, bj)
}

// --- helpers ----------------------------------------------------------------

/// Convert HSV (all in `[0, 1]`) to linear-light RGB `[f32; 3]`.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f32; 3] {
    let h6 = (h * 6.0).rem_euclid(6.0);
    let i = h6.floor() as u32;
    let f = h6.fract();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r as f32, g as f32, b as f32]
}

/// Deterministic integer cell hash → `[0, 1]`.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
