//! Clay roof-tile texture generator: barrel tiles (Roman, Spanish, kawara)
//! and pantiles, laid in columns down the slope and courses across it.
//!
//! The algorithm:
//! 1. Divide U into tile columns and V into courses, both rounded so the
//!    tile wraps (courses to an even count, so staggered courses wrap too).
//!    V runs down the slope, as an image does: each course's head lies under
//!    the course above, and its butt - its lower edge - overlaps the course
//!    below.
//! 2. Across each column, shape the tile: a smooth S-wave for pantiles
//!    (`barrel = 0`), blending to convex half-round caps over wide concave
//!    pans for barrel tiles (`barrel = 1`).
//! 3. Down each course, tilt the tile so its butt stands proud of the course
//!    below - the step the normal map draws - and darken the band under each
//!    butt, deepest in the pans.
//! 4. Jitter each tile's colour and seat, add fired-clay grain, and grow moss
//!    in the pans just under the butts, where water lingers.

use noise::{Fbm, MultiFractal, Perlin};

use crate::math::cell_hash;
use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`RoofTileGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RoofTileConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Tile columns across the tile \[2, 16\], rounded so the tile wraps.
    pub columns: f64,
    /// Courses down the tile \[2, 16\], rounded to an even count so that
    /// staggered courses wrap.
    pub courses: f64,
    /// Profile across a column \[0, 1\]: `0` a pantile's smooth S-wave, `1` a
    /// barrel tile's half-round caps over wide concave pans.
    pub barrel: f64,
    /// How far each course's butt stands proud of the course below \[0, 1\]:
    /// the depth of the step between courses.
    pub course_step: f64,
    /// Sideways offset of alternate courses, as a fraction of a column
    /// \[0, 1\].  `0` runs the columns straight down the roof, as clay
    /// tiles are laid.
    pub stagger: f64,
    /// Per-tile colour jitter \[0, 1\]: the patchwork of a kiln's firing.
    pub cell_variance: f64,
    /// Glaze \[0, 1\]: `0` is matte fired clay, `1` a glossy glazed tile.
    pub glaze: f64,
    /// Moss in the pans just under each butt \[0, 1\].
    pub moss_level: f64,
    /// Tile colour in linear RGB \[0, 1\].
    pub color_tile: [f32; 3],
    /// Colour of the shadow under each butt, in linear RGB \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Optional ageing pass - wear on exposed edges, grime in the
    /// recesses, corrosion and run-off streaks.
    ///
    /// Defaults to disabled, so the surface is unchanged until a layer
    /// is turned up.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for RoofTileConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            columns: 6.0,
            courses: 6.0,
            barrel: 0.7,
            course_step: 0.6,
            stagger: 0.0,
            cell_variance: 0.35,
            glaze: 0.0,
            moss_level: 0.1,
            color_tile: [0.50, 0.17, 0.08],
            color_shadow: [0.06, 0.03, 0.02],
            weathering: WeatheringConfig::default(),
            normal_strength: 4.0,
        }
    }
}

/// Procedural clay roof-tile texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`RoofTileConfig`].
/// Construct via [`RoofTileGenerator::new`] and call `generate` directly, or
/// spawn a `PendingTexture::roof_tile` task for non-blocking generation.
pub struct RoofTileGenerator {
    config: RoofTileConfig,
    grain_noise: ToroidalNoise<Fbm<Perlin>>,
    moss_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl RoofTileGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated calls to
    /// [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: RoofTileConfig) -> Self {
        let grain: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(4);
        let grain_noise = ToroidalNoise::new(grain, config.columns.max(2.0) * 2.5);
        let moss: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(300)).set_octaves(3);
        let moss_noise = ToroidalNoise::new(moss, 3.0);
        Self {
            config,
            grain_noise,
            moss_noise,
        }
    }
}

/// How far down each course the shadow under the butt above reaches, as a
/// fraction of the course.
const SHADOW_FRAC: f64 = 0.18;

/// Half the width of a barrel tile's cap, as a fraction of a column.
const CAP_HALF_WIDTH: f64 = 0.18;

/// Height of a barrel tile's pan where it meets the cap: the caps stand on
/// the pans' raised edges.
const PAN_EDGE: f64 = 0.35;

/// The height a column's profile spans, in the height field's units: the
/// curve of a cap and the dip of a pan dominate the tile's relief, and the
/// step between courses sits on top of them.
const PROFILE_RELIEF: f64 = 2.5;

/// The column count and course count the tile actually uses: rounded so it
/// wraps, the courses to an even count so that alternate courses wrap.
fn layout(c: &RoofTileConfig) -> (f64, f64) {
    let columns = c.columns.round().max(1.0);
    let courses = ((c.courses * 0.5).round() * 2.0).max(2.0);
    (columns, courses)
}

/// Height across one column at `t` in `[0, 1)`: an S-wave for pantiles
/// blending to half-round caps over concave pans for barrel tiles; `0` in a
/// pan's (or the wave's) lowest line, `1` along a cap's crown.
fn column_profile(t: f64, barrel: f64) -> f64 {
    let wave = (0.5 - 0.5 * (std::f64::consts::TAU * t).cos()).powf(0.8);
    let from_crown = (t - 0.5).abs();
    let barrel_h = if from_crown < CAP_HALF_WIDTH {
        let x = from_crown / CAP_HALF_WIDTH;
        PAN_EDGE + (1.0 - PAN_EDGE) * (1.0 - x * x).sqrt()
    } else {
        // Across the pan, from its lowest line at the column edge (0) to the
        // cap's foot (1).
        let s = ((0.5 - from_crown) / (0.5 - CAP_HALF_WIDTH)).clamp(0.0, 1.0);
        PAN_EDGE * (1.0 - (1.0 - s * s).max(0.0).sqrt())
    };
    wave + (barrel_h - wave) * barrel.clamp(0.0, 1.0)
}

/// Per-generation sampler: the grain and moss grids and the rounded layout.
struct RoofTileCell<'a> {
    config: &'a RoofTileConfig,
    grain_grid: &'a [f64],
    moss_grid: &'a [f64],
    columns: f64,
    courses: f64,
    width: usize,
}

impl SurfaceCell for RoofTileCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let course_pos = v * self.courses;
        let course = course_pos.floor() as i64;
        // 0 at the head, under the course above; toward 1 at the butt.
        let v_frac = course_pos.fract();
        let offset = if course.rem_euclid(2) == 1 {
            c.stagger
        } else {
            0.0
        };
        let column_pos = (u + offset / self.columns).rem_euclid(1.0) * self.columns;
        let column = column_pos.floor() as i64;
        let t = column_pos.fract();

        let profile = column_profile(t, c.barrel);
        // Each tile sits a little differently on the batten.
        let seat = (cell_hash(column, course, c.seed.wrapping_add(7)) - 0.5) * 0.08;

        let idx = y as usize * self.width + x as usize;
        let grain = normalize(self.grain_grid[idx]);
        let moss_raw = normalize(self.moss_grid[idx]);

        // The butt above throws its shadow down the head of this course,
        // deepest in the pans, where the butt stands highest over them.
        let shadow = (1.0 - v_frac / SHADOW_FRAC).clamp(0.0, 1.0).powf(0.7) * (1.0 - 0.3 * profile);
        let moss =
            (c.moss_level * moss_raw * (1.0 - profile).powi(2) * (1.0 - v_frac).powf(1.5) * 1.6)
                .clamp(0.0, 1.0);

        let height = profile * PROFILE_RELIEF + v_frac * c.course_step * 1.2 + seat + grain * 0.04
            - moss * 0.03;

        // Kiln patchwork: each tile lighter or darker, and redder or browner.
        let tone = (cell_hash(column, course, c.seed) - 0.5) * c.cell_variance * 0.5;
        let hue = (cell_hash(course, column, c.seed.wrapping_add(13)) - 0.5) * c.cell_variance;
        let fired = [
            (c.color_tile[0] * (1.0 + tone as f32 + hue as f32 * 0.15)).clamp(0.0, 1.0),
            (c.color_tile[1] * (1.0 + tone as f32 - hue as f32 * 0.25)).clamp(0.0, 1.0),
            (c.color_tile[2] * (1.0 + tone as f32 - hue as f32 * 0.3)).clamp(0.0, 1.0),
        ];
        // Fired-clay mottling, and the pans a shade darker than the crowns:
        // the grime of the channels water runs down, which also lets the
        // curve read under flat light.
        let mottle = (0.92 + 0.16 * grain as f32) * (0.8 + 0.25 * profile as f32);
        let moss_green = [0.10f32, 0.20, 0.06];
        let mut color = [0.0f32; 3];
        for (i, slot) in color.iter_mut().enumerate() {
            let tile = (fired[i] * mottle).clamp(0.0, 1.0);
            let mossy = lerp(tile, moss_green[i], moss as f32);
            *slot = lerp(mossy, c.color_shadow[i], shadow as f32 * 0.9);
        }

        let roughness =
            (lerp(0.78, 0.28, c.glaze as f32) + (grain as f32 - 0.5) * 0.08 + moss as f32 * 0.15)
                .clamp(0.0, 1.0);
        let occlusion = (1.0 - 0.35 * shadow - 0.15 * (1.0 - profile)) as f32;
        SurfaceSample {
            occlusion,
            ..SurfaceSample::matte(height, color, roughness)
        }
    }
}

impl RoofTileGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut grain_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.grain_noise, width, height, &mut grain_grid);
        let mut moss_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.moss_noise, width, height, &mut moss_grid);

        let (columns, courses) = layout(c);
        let cell = RoofTileCell {
            config: c,
            grain_grid: &grain_grid,
            moss_grid: &moss_grid,
            columns,
            courses,
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
            ws.return_grid(grain_grid);
            ws.return_grid(moss_grid);
        }
        result
    }
}

impl TextureGenerator for RoofTileGenerator {
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
    use crate::test_support::{luminance, wrap_step_ratio};

    /// A barrel tile's cap stands above its pans, and a pantile's wave runs
    /// from a pan's lowest line to its crown: the profile spans `[0, 1]`
    /// with the crown in the middle of the column and the lowest line at
    /// its edge.
    #[test]
    fn the_cap_stands_over_the_pans() {
        for barrel in [0.0, 0.5, 1.0] {
            let crown = column_profile(0.5, barrel);
            let channel = column_profile(0.0, barrel);
            assert!((crown - 1.0).abs() < 1e-9, "barrel {barrel}: crown {crown}");
            assert!(channel.abs() < 1e-9, "barrel {barrel}: channel {channel}");
            for i in 0..=100 {
                let h = column_profile(i as f64 / 100.0, barrel);
                assert!((0.0..=1.0).contains(&h), "barrel {barrel}: {h} at {i}%");
            }
        }
        // Barrel caps are narrow: a quarter of the way across, a barrel
        // tile is down in its pan while a pantile is halfway up its wave.
        assert!(column_profile(0.25, 1.0) < 0.2);
        assert!(column_profile(0.25, 0.0) > 0.4);
    }

    /// The shadow under each butt darkens the head of every course, and the
    /// tile wraps in both directions, staggered courses included. Each
    /// tile's own tone is held at the default's so that a wrap is judged by
    /// the tile's shape, not by which two random tones meet there.
    #[test]
    fn courses_are_shadowed_under_each_butt_and_the_tile_wraps() {
        let size = 192;
        let config = RoofTileConfig {
            stagger: 0.5,
            moss_level: 0.0,
            cell_variance: 0.0,
            ..RoofTileConfig::default()
        };
        let (_, courses) = layout(&config);
        let map = RoofTileGenerator::new(config)
            .generate(size as u32, size as u32)
            .expect("a bake");
        let l = luminance(&map);
        let row_mean = |y: usize| l[y * size..(y + 1) * size].iter().sum::<f64>() / size as f64;
        let course_px = size as f64 / courses;
        let (mut head, mut butt) = (0.0, 0.0);
        for k in 0..courses as usize {
            let top = (k as f64 * course_px) as usize;
            head += row_mean(top + 1);
            butt += row_mean(top + (course_px * 0.7) as usize);
        }
        assert!(head < butt * 0.75, "head {head:.1} against butt {butt:.1}");
        let columns_px = size / layout(&RoofTileConfig::default()).0 as usize;
        let u_ratio = wrap_step_ratio(&l, size, columns_px, true);
        let v_ratio = wrap_step_ratio(&l, size, course_px as usize, false);
        assert!(u_ratio < 1.6, "a seam in U: {u_ratio:.2}");
        assert!(v_ratio < 1.6, "a seam in V: {v_ratio:.2}");
    }
}
