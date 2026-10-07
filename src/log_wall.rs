//! Log-wall texture generator: round logs stacked along V and running the
//! length of U, with chinking between them - a log cabin's wall, or, turned a
//! quarter, a palisade.
//!
//! The algorithm:
//! 1. Divide V into log bands, rounded so the tile wraps.
//! 2. Seat a cylinder in each band: its diameter varies log to log, and its
//!    centre line wanders gently along U, sampled on a circle so it tiles.
//!    Whatever the log leaves of the band is chinking.
//! 3. Dress each log: peeled wood with its grain running along the log and
//!    checks (drying cracks) along it, under patches of fissured bark.
//! 4. Shade the chinking as a rough, recessed daub between the logs.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::math::{cell_hash, smoothstep};
use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`LogWallGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogWallConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Logs stacked up the tile \[2, 12\], rounded so the tile wraps.
    pub log_count: f64,
    /// Chinking between logs as a fraction of a log's band \[0, 0.4\].
    pub chinking_width: f64,
    /// How much logs differ \[0, 1\]: diameter log to log, and how far a
    /// log's centre line wanders along its length.
    pub log_variance: f64,
    /// Bark left on the logs \[0, 1\]: `0` is peeled, smooth wood, `1` logs
    /// still in their bark.
    pub bark: f64,
    /// Grain fineness \[2, 32\]: how many grain lines a log shows across it.
    pub grain_scale: f64,
    /// Drying checks - cracks along the grain - on the peeled wood \[0, 1\].
    pub crack_density: f64,
    /// Light wood colour in linear RGB \[0, 1\].
    pub color_wood_light: [f32; 3],
    /// Dark wood colour in linear RGB \[0, 1\].
    pub color_wood_dark: [f32; 3],
    /// Bark colour in linear RGB \[0, 1\].
    pub color_bark: [f32; 3],
    /// Chinking colour in linear RGB \[0, 1\].
    pub color_chinking: [f32; 3],
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

impl Default for LogWallConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            log_count: 5.0,
            chinking_width: 0.12,
            log_variance: 0.5,
            bark: 0.3,
            grain_scale: 10.0,
            crack_density: 0.4,
            color_wood_light: [0.50, 0.34, 0.18],
            color_wood_dark: [0.26, 0.16, 0.08],
            color_bark: [0.13, 0.09, 0.06],
            color_chinking: [0.52, 0.49, 0.43],
            weathering: WeatheringConfig::default(),
            normal_strength: 6.0,
        }
    }
}

/// Procedural log-wall texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`LogWallConfig`].
/// Construct via [`LogWallGenerator::new`] and call `generate` directly, or
/// spawn a `PendingTexture::log_wall` task for non-blocking generation.
pub struct LogWallGenerator {
    config: LogWallConfig,
    grain_noise: ToroidalNoise<Fbm<Perlin>>,
    bark_noise: ToroidalNoise<Fbm<Perlin>>,
    patch_noise: ToroidalNoise<Fbm<Perlin>>,
    daub_noise: ToroidalNoise<Fbm<Perlin>>,
    check_noise: ToroidalNoise<Fbm<Perlin>>,
    wander_noise: Fbm<Perlin>,
}

impl LogWallGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated calls to
    /// [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: LogWallConfig) -> Self {
        let seed = config.seed;
        Self {
            grain_noise: ToroidalNoise::new(Fbm::new(seed).set_octaves(5), 1.0),
            bark_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(11)).set_octaves(4), 1.0),
            patch_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(23)).set_octaves(3), 2.0),
            daub_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(37)).set_octaves(4), 12.0),
            check_noise: ToroidalNoise::new(Fbm::new(seed.wrapping_add(61)).set_octaves(2), 1.0),
            wander_noise: Fbm::new(seed.wrapping_add(53)).set_octaves(3),
            config,
        }
    }
}

/// The height of a log's crown over the chinking, in the height field's
/// units: the round of the log dominates the wall's relief.
const LOG_RELIEF: f64 = 3.0;

/// The checks' field across a log, as a share of the grain's: a few long
/// cracks to a log rather than one at every turn of the grain.
const CHECK_ACROSS_RATIO: f64 = 0.3;

/// The checks' field along a log, as a share of the grain's: slower still,
/// so a check runs nearly straight down the log, as a drying crack does.
const CHECK_ALONG_RATIO: f64 = 0.4;

/// The radius of the U circle a log's wander is sampled on: about two gentle
/// bends over the tile's width.
const WANDER_RADIUS: f64 = 0.35;

/// How far across the grain noise's V circle one log runs, in turns, and
/// how much more slowly the grain changes along the log than across it.
const GRAIN_TURNS_PER_LOG: f64 = 0.2;
const GRAIN_ALONG_RATIO: f64 = 0.06;

/// One log's place in its band, as the band's fraction: its centre line at
/// each column and its radius.
struct Log {
    centre: Vec<f64>,
    radius: f64,
    /// Where this log starts on the grain noise's V circle, in turns.
    phase: f64,
}

/// Per-generation sampler: the logs, the noise and the daub grid.
struct LogWallCell<'a> {
    config: &'a LogWallConfig,
    generator: &'a LogWallGenerator,
    logs: &'a [Log],
    daub_grid: &'a [f64],
    log_count: f64,
    width: usize,
}

impl SurfaceCell for LogWallCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let band = v * self.log_count;
        let k = (band.floor() as usize).min(self.logs.len() - 1);
        let log = &self.logs[k];
        let across = (band.fract() - log.centre[x as usize]) / log.radius;
        let idx = y as usize * self.width + x as usize;
        let daub = normalize(self.daub_grid[idx]);

        if across.abs() >= 1.0 {
            // Chinking: a rough daub sunk between the logs.
            let color = [
                (c.color_chinking[0] * (0.85 + 0.3 * daub as f32)).clamp(0.0, 1.0),
                (c.color_chinking[1] * (0.85 + 0.3 * daub as f32)).clamp(0.0, 1.0),
                (c.color_chinking[2] * (0.85 + 0.3 * daub as f32)).clamp(0.0, 1.0),
            ];
            let sample = SurfaceSample::matte(0.1 + daub * 0.15, color, 0.92);
            return SurfaceSample {
                occlusion: 0.7,
                ..sample
            };
        }

        // The log's round: 1 along its crown, 0 where it meets the chinking.
        let round = (1.0 - across * across).sqrt();
        let g = self.generator;
        let (su, cu) = (TAU * u).sin_cos();
        let grain_turn = log.phase + (across * 0.5 + 0.5) * GRAIN_TURNS_PER_LOG;
        let (sg, cg) = (TAU * grain_turn).sin_cos();
        let grain_along = c.grain_scale * GRAIN_ALONG_RATIO;
        let grain = normalize(g.grain_noise.get_precomputed(
            cu * grain_along,
            su * grain_along,
            cg * c.grain_scale,
            sg * c.grain_scale,
        ));
        // Bark fissures run along the log too, coarser than the grain.
        let bark_raw = g.bark_noise.get_precomputed(
            cu * 0.8,
            su * 0.8,
            cg * c.grain_scale * 0.6,
            sg * c.grain_scale * 0.6,
        );
        let fissure = smoothstep(0.0, 0.35, bark_raw.abs());
        // Patches where the bark has stayed on: the noise sits mostly
        // between 0.25 and 0.75, so a threshold walked across that span
        // covers about `bark` of the logs.
        let patch = normalize(g.patch_noise.get(u, v + k as f64 * 0.37));
        let threshold = 0.5 + (c.bark - 0.5) * 0.55;
        let on_bark = 1.0 - smoothstep(threshold - 0.04, threshold + 0.04, patch);

        // Checks: drying cracks along the grain on peeled wood, the zero
        // line of their own field. It is coarser across the log than the
        // grain, so a check is a line a pixel or two wide; on the grain's own
        // contour it was a sliver of a pixel, and sampled as scattered dots.
        let crack = if c.crack_density > 0.0 {
            let across = c.grain_scale * CHECK_ACROSS_RATIO;
            let along = grain_along * CHECK_ALONG_RATIO;
            let line = g
                .check_noise
                .get_precomputed(cu * along, su * along, cg * across, sg * across)
                .abs();
            let width = 0.1 * c.crack_density;
            // Broken into long runs by the coarse patch field, read away
            // from where it lays the bark.
            let runs = normalize(g.patch_noise.get(u + 0.5, v + k as f64 * 0.37 + 0.5));
            (1.0 - smoothstep(0.0, width, line)) * smoothstep(0.4, 0.6, runs)
        } else {
            0.0
        };

        let tone = (cell_hash(k as i64, 0, c.seed) - 0.5) * 0.18 * c.log_variance;
        let wood_t = (grain as f32 * 0.8 + tone as f32).clamp(0.0, 1.0);
        let mut color = [0.0f32; 3];
        for (i, slot) in color.iter_mut().enumerate() {
            let wood = lerp(c.color_wood_dark[i], c.color_wood_light[i], wood_t);
            let wood = lerp(wood, c.color_wood_dark[i] * 0.4, crack as f32 * 0.9);
            let bark = c.color_bark[i] * (0.6 + 0.6 * fissure as f32);
            let skin = lerp(wood, bark, on_bark as f32);
            // The round turns away from the light toward the chinking, and
            // gathers grime there.
            *slot = (skin * (0.42 + 0.58 * (round as f32).powf(0.6))).clamp(0.0, 1.0);
        }

        let detail = lerp(
            (grain * 0.08 - crack * 0.25) as f32,
            (fissure * 0.35 - 0.2) as f32,
            on_bark as f32,
        ) as f64;
        let height = 0.25 + round * LOG_RELIEF + detail;
        let roughness = lerp(0.62 + crack as f32 * 0.2, 0.93, on_bark as f32);
        SurfaceSample {
            occlusion: (0.78 + 0.22 * round) as f32,
            ..SurfaceSample::matte(height, color, roughness)
        }
    }
}

impl LogWallGenerator {
    /// Each log's diameter and wandering centre line, one entry per band.
    fn logs(&self, width: usize, log_count: usize) -> Vec<Log> {
        let c = &self.config;
        let fill = (1.0 - c.chinking_width.clamp(0.0, 0.4)) * 0.5;
        (0..log_count)
            .map(|k| {
                let size = 1.0 - (cell_hash(k as i64, 1, c.seed)) * 0.25 * c.log_variance;
                let radius = (fill * size).max(0.05);
                // The wander may take the log no closer than a sliver to its
                // band's edge, so neighbours never overlap.
                let room = (0.5 - radius).max(0.0);
                let centre = (0..width)
                    .map(|x| {
                        let a = TAU * x as f64 / width as f64;
                        let w = self.wander_noise.get([
                            a.cos() * WANDER_RADIUS,
                            a.sin() * WANDER_RADIUS,
                            k as f64 * 3.7,
                        ]);
                        0.5 + (w * 0.9 * c.log_variance).clamp(-1.0, 1.0) * room
                    })
                    .collect();
                Log {
                    centre,
                    radius,
                    phase: cell_hash(k as i64, 2, c.seed),
                }
            })
            .collect()
    }

    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;
        let log_count = c.log_count.round().max(1.0);

        let mut daub_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.daub_noise, width, height, &mut daub_grid);
        let logs = self.logs(width as usize, log_count as usize);
        let cell = LogWallCell {
            config: c,
            generator: self,
            logs: &logs,
            daub_grid: &daub_grid,
            log_count,
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
            ws.return_grid(daub_grid);
        }
        result
    }
}

impl TextureGenerator for LogWallGenerator {
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

    /// Every log stays inside its band: however it wanders, and however its
    /// diameter varies, a sliver of chinking is left between neighbours.
    #[test]
    fn logs_never_overlap_their_neighbours() {
        for variance in [0.0, 0.5, 1.0] {
            let generator = LogWallGenerator::new(LogWallConfig {
                log_variance: variance,
                chinking_width: 0.0,
                ..LogWallConfig::default()
            });
            for log in generator.logs(256, 6) {
                for centre in &log.centre {
                    assert!(
                        centre - log.radius >= -1e-9 && centre + log.radius <= 1.0 + 1e-9,
                        "variance {variance}: a log from {} to {}",
                        centre - log.radius,
                        centre + log.radius
                    );
                }
            }
        }
    }

    /// The chinking reads between the logs (the rows at each band's edge
    /// are the chinking's colour, the rows along each log's crown the
    /// wood's), and the wall tiles in both directions.
    #[test]
    fn chinking_lies_between_the_logs_and_the_wall_wraps() {
        let size = 200;
        let config = LogWallConfig {
            log_variance: 0.0,
            ..LogWallConfig::default()
        };
        let log_count = config.log_count.round() as usize;
        let map = LogWallGenerator::new(config.clone())
            .generate(size as u32, size as u32)
            .expect("a bake");
        let l = luminance(&map);
        let row_mean = |y: usize| l[y * size..(y + 1) * size].iter().sum::<f64>() / size as f64;
        let band = size / log_count;
        let (mut edges, mut crowns) = (0.0, 0.0);
        for k in 0..log_count {
            edges += row_mean(k * band);
            crowns += row_mean(k * band + band / 2);
        }
        // The default chinking is a pale daub against darker wood.
        assert!(
            edges > crowns * 1.3,
            "edges {edges:.0} against crowns {crowns:.0}"
        );
        let u_ratio = wrap_step_ratio(&l, size, 1, true);
        let v_ratio = wrap_step_ratio(&l, size, band, false);
        assert!(u_ratio < 1.6, "a seam in U: {u_ratio:.2}");
        assert!(v_ratio < 1.6, "a seam in V: {v_ratio:.2}");
    }
}
