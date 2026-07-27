//! Photovoltaic array: silicon cells behind glass, wired by busbars.
//!
//! A manufactured surface, so it is built from index arithmetic rather than
//! noise: a grid of cells with chamfered corners, the thin fingers that
//! collect current across each cell, and the wider busbars soldered down the
//! column.  Noise appears only as the faint crystalline mottle within the
//! silicon and the grime on the cover glass.
//!
//! What makes this read as a panel rather than a tiled floor is that the
//! wiring is *continuous across* cells while the silicon is not.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, cell_hash, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`SolarPanelGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SolarPanelConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Cells across the tile horizontally.
    pub cells_x: f64,
    /// Cells across the tile vertically.
    pub cells_y: f64,
    /// Gap between cells as a fraction of a cell.
    pub cell_gap: f64,
    /// Corner chamfer as a fraction of a cell — monocrystalline wafers are
    /// cut from a round ingot, so their corners are clipped.
    pub corner_cut: f64,
    /// Busbars running down each cell.
    pub busbars: f64,
    /// Width of each busbar as a fraction of a cell.  Total conductor
    /// coverage is this times [`busbars`](Self::busbars), so a handful of
    /// wide bars quickly swamps the silicon they are collecting from.
    pub busbar_width: f64,
    /// Current-collecting fingers across each cell.
    pub fingers: f64,
    /// Width of each finger as a fraction of a cell.  Fingers are hair-thin
    /// on a real cell — a plausible value times
    /// [`fingers`](Self::fingers) lands a few percent of the surface, not a
    /// tenth of it.
    pub finger_width: f64,
    /// Silicon colour in linear RGB \[0, 1\].
    pub color_cell: [f32; 3],
    /// Backing-sheet colour showing in the gaps between cells.
    pub color_backing: [f32; 3],
    /// Colour of the busbars and fingers — tinned copper.
    pub color_wire: [f32; 3],
    /// Spread of per-cell brightness variation in `[0, 1]`, applied as a
    /// multiplier so it stays proportionate against near-black silicon.
    pub cell_variance: f32,
    /// Strength of the crystalline mottle within each cell.
    pub crystal_mottle: f32,
    /// Frequency of the crystalline mottle.
    pub crystal_scale: f64,
    /// Gloss of the cover glass, as roughness in `[0, 1]`.
    pub glass_roughness: f32,
    /// Optional ageing pass — grime on the glass, corrosion at the wiring.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for SolarPanelConfig {
    fn default() -> Self {
        Self {
            seed: 41,
            cells_x: 4.0,
            cells_y: 4.0,
            cell_gap: 0.06,
            corner_cut: 0.14,
            busbars: 3.0,
            busbar_width: 0.014,
            fingers: 18.0,
            finger_width: 0.003,
            color_cell: [0.020, 0.030, 0.075],
            color_backing: [0.72, 0.72, 0.70],
            color_wire: [0.62, 0.63, 0.65],
            cell_variance: 0.18,
            crystal_mottle: 0.30,
            crystal_scale: 22.0,
            glass_roughness: 0.10,
            weathering: WeatheringConfig::default(),
            normal_strength: 1.0,
        }
    }
}

/// Procedural solar-panel texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`SolarPanelConfig`].
pub struct SolarPanelGenerator {
    config: SolarPanelConfig,
    crystal: ToroidalNoise<Fbm<Perlin>>,
}

impl SolarPanelGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SolarPanelConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(8)).set_octaves(3);
        let crystal = ToroidalNoise::new(fbm, config.crystal_scale.max(0.1));
        Self { config, crystal }
    }
}

/// Per-generation sampler: crystal grid plus the cell layout.
struct SolarPanelCell<'a> {
    config: &'a SolarPanelConfig,
    crystal: &'a [f64],
    cols: f64,
    rows: f64,
    width: usize,
}

impl SurfaceCell for SolarPanelCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let crystal = normalize(self.crystal[y as usize * self.width + x as usize]);

        // Position within the wafer grid.
        let gx = u * self.cols;
        let gy = v * self.rows;
        let (col, row) = (gx.floor(), gy.floor());
        let (fx, fy) = (gx.fract(), gy.fract());

        // Distance from the cell edge, in cell units.
        let gap = c.cell_gap.clamp(0.0, 0.45);
        let edge_x = (fx.min(1.0 - fx) - gap * 0.5).max(-1.0);
        let edge_y = (fy.min(1.0 - fy) - gap * 0.5).max(-1.0);
        // Chamfered corners: clip the diagonal as well as the sides.
        let cut = c.corner_cut.clamp(0.0, 0.5);
        let corner = (fx - 0.5).abs() + (fy - 0.5).abs() - (1.0 - cut);

        let on_cell = edge_x > 0.0 && edge_y > 0.0 && corner < 0.0;

        if !on_cell {
            // Backing sheet between and around the wafers.
            return SurfaceSample {
                height: -0.25,
                color: c.color_backing,
                roughness: 0.55,
                metallic: 0.0,
                occlusion: 0.75,
                emissive: [0.0, 0.0, 0.0],
            };
        }

        // Wiring: busbars run down the cell, fingers run across it.  Both are
        // laid in cell-local space so they line up from wafer to wafer.
        let busbar = stripe_mask(fx, c.busbars, c.busbar_width);
        let finger = stripe_mask(fy, c.fingers, c.finger_width);
        let wire = busbar.max(finger * 0.85);

        // Wafer and crystal variation scale the silicon rather than being
        // added to it.  Silicon is nearly black, so an additive nudge that
        // would be imperceptible on a mid-tone swings a cell from black to
        // pale blue — the panel ends up looking like a faulty one.
        let tint = 1.0
            + (cell_hash(col as i64, row as i64, c.seed.wrapping_add(11)) - 0.5) as f32
                * 2.0
                * c.cell_variance;
        let mottle = 1.0 + (crystal as f32 - 0.5) * c.crystal_mottle.clamp(0.0, 1.0);

        let silicon = [
            (c.color_cell[0] * tint * mottle).clamp(0.0, 1.0),
            (c.color_cell[1] * tint * mottle).clamp(0.0, 1.0),
            (c.color_cell[2] * tint * mottle).clamp(0.0, 1.0),
        ];

        let w = wire as f32;
        let color = [
            lerp(silicon[0], c.color_wire[0], w),
            lerp(silicon[1], c.color_wire[1], w),
            lerp(silicon[2], c.color_wire[2], w),
        ];

        SurfaceSample {
            // Wafer sits proud of the backing; wiring sits proud of the wafer.
            height: 0.25 + wire * 0.12,
            color,
            // Glass is glossy over the cell; solder is duller.
            roughness: lerp(c.glass_roughness, 0.35, w).clamp(0.0, 1.0),
            metallic: lerp(0.35, 0.85, w),
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

/// Repeating band mask along one cell-local axis: `1` on a conductor, `0`
/// between them.
///
/// `width` is the half-extent of a single band in cell units, so the total
/// fraction of the axis covered is `width * count`.
#[inline]
fn stripe_mask(coordinate: f64, count: f64, width: f64) -> f64 {
    let n = count.round().max(0.0);
    if n <= 0.0 || width <= 0.0 {
        return 0.0;
    }
    // Bands sit at the centres of `n` equal divisions, so they never land on
    // the cell edge where the chamfer would clip them.
    let phase = (coordinate * n).fract();
    let distance = (phase - 0.5).abs() * 2.0 / n;
    if distance < width { 1.0 } else { 0.0 }
}

impl SolarPanelGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut crystal = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.crystal, width, height, &mut crystal);

        let cell = SolarPanelCell {
            config: c,
            crystal: &crystal,
            // Whole cells only, so the array tiles without a clipped wafer.
            cols: c.cells_x.round().clamp(1.0, 64.0),
            rows: c.cells_y.round().clamp(1.0, 64.0),
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
            ws.return_grid(crystal);
        }
        result
    }
}

impl TextureGenerator for SolarPanelGenerator {
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

    fn bake(config: SolarPanelConfig) -> TextureMap {
        SolarPanelGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(SolarPanelConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(SolarPanelConfig::default()).albedo,
            bake(SolarPanelConfig::default()).albedo
        );
    }

    /// Dark silicon must dominate, with the pale backing only in the gaps —
    /// the balance that reads as a panel rather than a tiled floor.
    #[test]
    fn silicon_dominates_the_backing() {
        let map = bake(SolarPanelConfig::default());
        let pale =
            map.albedo.chunks(4).filter(|px| px[0] > 170).count() as f64 / (128 * 128) as f64;
        assert!(
            (0.02..0.45).contains(&pale),
            "backing covers {pale:.3} of the panel"
        );
    }

    /// Wafer corners are clipped; squaring them off loses the read.
    #[test]
    fn corner_cut_clips_the_wafers() {
        let clipped = bake(SolarPanelConfig::default());
        let square = bake(SolarPanelConfig {
            corner_cut: 0.0,
            ..Default::default()
        });
        assert_ne!(clipped.albedo, square.albedo, "corner cut had no effect");

        let pale = |m: &TextureMap| m.albedo.chunks(4).filter(|px| px[0] > 170).count();
        assert!(
            pale(&clipped) > pale(&square),
            "clipping corners did not expose more backing"
        );
    }

    /// Busbars and fingers must both reach the surface.
    #[test]
    fn wiring_is_drawn() {
        let wired = bake(SolarPanelConfig::default());
        let bare = bake(SolarPanelConfig {
            busbars: 0.0,
            fingers: 0.0,
            ..Default::default()
        });
        assert_ne!(wired.albedo, bare.albedo, "wiring had no effect");

        let only_bars = bake(SolarPanelConfig {
            fingers: 0.0,
            ..Default::default()
        });
        assert_ne!(wired.albedo, only_bars.albedo, "fingers had no effect");
        assert_ne!(only_bars.albedo, bare.albedo, "busbars had no effect");
    }

    /// The array must tile, so the grid has to divide the tile evenly even
    /// when a fractional cell count is configured.
    #[test]
    fn fractional_cell_counts_snap_to_whole_wafers() {
        assert_eq!(
            bake(SolarPanelConfig {
                cells_x: 4.4,
                ..Default::default()
            })
            .albedo,
            bake(SolarPanelConfig {
                cells_x: 4.0,
                ..Default::default()
            })
            .albedo,
            "fractional cell count did not snap to whole wafers"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(SolarPanelConfig {
            cells_x: 0.0,
            cells_y: 1e9,
            cell_gap: 9.0,
            corner_cut: 9.0,
            busbars: -5.0,
            busbar_width: 9.0,
            fingers: 1e9,
            finger_width: -1.0,
            crystal_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
