//! Dry-stone wall texture generator: irregular, roughly flat stones laid in
//! rough courses with deep, unmortared gaps between them - a field wall, a
//! sheepfold, a terrace's retaining wall.
//!
//! The algorithm:
//! 1. Divide V into courses whose heights wander about the mean, the
//!    jitter repeating so the tile wraps.
//! 2. Cut each course into stones of random length, normalised so the cuts
//!    wrap in U, from a random starting offset per course so the joints do
//!    not line up.
//! 3. Shape each stone as a rounded box inset from its cell - by the gap,
//!    and by a random amount at its top, so stones of one course differ in
//!    height - with its outline roughened by noise, its face tilted a
//!    little and bumped.
//! 4. Shade the gaps deep and dark, give each stone its own tone between
//!    two colours, and spot the upper faces with lichen.

use noise::{Fbm, MultiFractal, Perlin};

use crate::math::{cell_hash, smoothstep};
use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`DryStoneGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DryStoneConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Courses of stone up the tile \[3, 20\].
    pub courses: usize,
    /// Stones in a course, on average \[2, 12\].
    pub stones_per_course: usize,
    /// How far course heights wander from the mean \[0, 1\].
    pub course_jitter: f64,
    /// The gap between stones as a fraction of a course's height \[0, 0.3\].
    pub gap_width: f64,
    /// How rough the stones' outlines are and how much their heights differ
    /// within a course \[0, 1\].
    pub irregularity: f64,
    /// Bumps and tilt across each stone's face \[0, 1\].
    pub face_relief: f64,
    /// Stone-to-stone colour variation \[0, 1\]: how far each stone's tone
    /// strays, and how much of the second colour it takes.
    pub cell_variance: f64,
    /// Lichen on the stones' upper faces \[0, 1\].
    pub moss_level: f64,
    /// Main stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Second stone colour, mixed into some stones, in linear RGB \[0, 1\].
    pub color_stone_alt: [f32; 3],
    /// Colour deep in the gaps in linear RGB \[0, 1\].
    pub color_gap: [f32; 3],
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

impl Default for DryStoneConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            courses: 8,
            stones_per_course: 5,
            course_jitter: 0.5,
            gap_width: 0.12,
            irregularity: 0.6,
            face_relief: 0.5,
            cell_variance: 0.5,
            moss_level: 0.15,
            color_stone: [0.36, 0.35, 0.33],
            color_stone_alt: [0.40, 0.33, 0.24],
            color_gap: [0.03, 0.03, 0.03],
            weathering: WeatheringConfig::default(),
            normal_strength: 5.0,
        }
    }
}

/// Procedural dry-stone wall texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`DryStoneConfig`].
/// Construct via [`DryStoneGenerator::new`] and call `generate` directly, or
/// spawn a `PendingTexture::dry_stone` task for non-blocking generation.
pub struct DryStoneGenerator {
    config: DryStoneConfig,
    outline_noise: ToroidalNoise<Fbm<Perlin>>,
    face_noise: ToroidalNoise<Fbm<Perlin>>,
    lichen_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl DryStoneGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated calls to
    /// [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: DryStoneConfig) -> Self {
        let courses = config.courses.max(1) as f64;
        let seed = config.seed;
        Self {
            // Two octaves at a low frequency: a stone's outline is lumpy,
            // not torn.
            outline_noise: ToroidalNoise::new(Fbm::new(seed).set_octaves(2), courses * 0.6),
            face_noise: ToroidalNoise::new(
                Fbm::new(seed.wrapping_add(19)).set_octaves(3),
                courses * 1.6,
            ),
            lichen_noise: ToroidalNoise::new(
                Fbm::new(seed.wrapping_add(41)).set_octaves(4),
                courses * 1.5,
            ),
            config,
        }
    }
}

/// The height of a stone's face over the gaps, in the height field's units.
const STONE_RELIEF: f64 = 1.5;

/// One course: where it starts and ends in V, and its stones.
struct Course {
    v0: f64,
    v1: f64,
    /// Where the course's first cut lies in U.
    offset: f64,
    /// Cut positions relative to `offset`, from `0` to `1`, one more than
    /// there are stones.
    cuts: Vec<f64>,
    stones: Vec<Stone>,
}

/// One stone's own draws.
struct Stone {
    /// How far the stone's top falls short of the course's, as a fraction of
    /// the course's height.
    top_inset: f64,
    /// How far its bottom stands off the course below, likewise.
    bottom_inset: f64,
    /// How round its corners are, as a fraction of its smaller half-extent:
    /// from a squarish block to a cobble.
    rounding: f64,
    /// The tilt of its face, across and up.
    tilt: (f64, f64),
    tone: f64,
    alt: f64,
}

/// The courses of a tile: heights jittered about `1 / count`, repeating so
/// the last course's top is the first's bottom one tile on.
fn courses(c: &DryStoneConfig) -> Vec<Course> {
    let count = c.courses.max(1);
    let jitter = |k: usize| (cell_hash(k as i64, 0, c.seed) - 0.5) * c.course_jitter * 0.6;
    let bounds: Vec<f64> = (0..=count)
        .map(|k| (k as f64 + jitter(k % count)) / count as f64)
        .collect();
    (0..count)
        .map(|k| {
            let mean = c.stones_per_course.max(1) as f64;
            let spread = (cell_hash(k as i64, 1, c.seed) - 0.5) * mean * 0.5 * c.irregularity;
            let n = (mean + spread).round().max(1.0) as usize;
            // Lengths from a quarter of the mean to well over it, so short
            // stones sit between long ones.
            let lengths: Vec<f64> = (0..n)
                .map(|i| {
                    let draw = cell_hash(k as i64, 10 + i as i64, c.seed);
                    1.0 + (draw - 0.5) * 1.5 * c.irregularity.max(0.1)
                })
                .collect();
            let total: f64 = lengths.iter().sum();
            let mut cuts = vec![0.0];
            let mut run = 0.0;
            for l in &lengths {
                run += l / total;
                cuts.push(run);
            }
            if let Some(last) = cuts.last_mut() {
                *last = 1.0;
            }
            let stones = (0..n)
                .map(|i| {
                    let h = |salt: i64| cell_hash(k as i64 * 64 + i as i64, salt, c.seed);
                    Stone {
                        top_inset: h(2) * 0.35 * c.irregularity,
                        bottom_inset: h(7) * 0.08 * c.irregularity,
                        rounding: 0.25 + 0.45 * h(8),
                        tilt: (h(3) - 0.5, h(4) - 0.5),
                        tone: h(5) - 0.5,
                        alt: h(6),
                    }
                })
                .collect();
            Course {
                v0: bounds[k],
                v1: bounds[k + 1],
                offset: cell_hash(k as i64, 2, c.seed),
                cuts,
                stones,
            }
        })
        .collect()
}

/// Per-generation sampler: the courses, the noise grids and the row table.
struct DryStoneCell<'a> {
    config: &'a DryStoneConfig,
    courses: &'a [Course],
    /// Each row's course, by index.
    row_course: &'a [usize],
    outline_grid: &'a [f64],
    face_grid: &'a [f64],
    lichen_grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for DryStoneCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let course = &self.courses[self.row_course[y as usize]];
        // The course may straddle the tile's top or bottom edge.
        let v = if v < course.v0 {
            v + 1.0
        } else if v >= course.v1 {
            v - 1.0
        } else {
            v
        };
        let along = (u - course.offset).rem_euclid(1.0);
        let i = course
            .cuts
            .partition_point(|&cut| cut <= along)
            .saturating_sub(1)
            .min(course.stones.len() - 1);
        let stone = &course.stones[i];
        let (left, right) = (course.cuts[i], course.cuts[i + 1]);

        let height_v = course.v1 - course.v0;
        let mean_h = 1.0 / c.courses.max(1) as f64;
        let gap = c.gap_width.clamp(0.0, 0.3) * mean_h;
        // The stone sits on the course below, or nearly; its top falls
        // short by its own inset.
        let top = course.v0 + stone.top_inset * height_v;
        let bottom = course.v1 - stone.bottom_inset * height_v;
        let (cx, cy) = ((left + right) * 0.5, (top + bottom) * 0.5);
        let hx = ((right - left) * 0.5 - gap * 0.5).max(1e-4);
        let hy = ((bottom - top) * 0.5 - gap * 0.5).max(1e-4);
        let (px, py) = (along - cx, v - cy);
        let radius = hx.min(hy) * stone.rounding;
        let qx = px.abs() - hx + radius;
        let qy = py.abs() - hy + radius;
        let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
        let idx = y as usize * self.width + x as usize;
        let rough = (normalize(self.outline_grid[idx]) - 0.5) * c.irregularity * mean_h * 0.5;
        let d = outside + qx.max(qy).min(0.0) - radius + rough;

        if d >= 0.0 {
            let sample = SurfaceSample::matte(0.0, c.color_gap, 0.95);
            return SurfaceSample {
                occlusion: 0.35,
                ..sample
            };
        }

        // 0 at the stone's outline, 1 most of the way to its middle, on a
        // dome: steepest where the stone's side drops into the gap, rounding
        // over onto its face. (A smoothstep bevel standing on a step put the
        // steepest slope in a band all round the face, and a lit wall drew
        // it as a pale rim on every stone.)
        let dome = (hy * 0.8).max(1e-4);
        let t = (-d / dome).min(1.0);
        let edge = 1.0 - (1.0 - t) * (1.0 - t);
        let bump = normalize(self.face_grid[idx]) - 0.5;
        // A texel's offset across its stone, held to the box: where the box
        // has shrunk to nothing (a stone narrower than its gap) the outline
        // noise still opens blobs, and an unheld offset tilted them sky-high.
        let (ax, ay) = ((px / hx).clamp(-1.0, 1.0), (py / hy).clamp(-1.0, 1.0));
        let tilt = (stone.tilt.0 * ax + stone.tilt.1 * ay) * 0.25;
        let height = edge * STONE_RELIEF * (1.0 + (tilt + bump * 0.5) * c.face_relief);

        // Lichen gathers on the upper half of a stone's face.
        let upper = 1.0 - smoothstep(-0.6, 0.2, py / hy.max(1e-4));
        let lichen = (c.moss_level * 2.0 * upper * edge).min(1.0)
            * smoothstep(0.55, 0.75, normalize(self.lichen_grid[idx]));

        let shade = (1.0 + stone.tone * c.cell_variance * 0.5) as f32;
        let alt = (stone.alt * c.cell_variance) as f32;
        let lichen_colour = [0.42f32, 0.42, 0.22];
        let mut color = [0.0f32; 3];
        for (k, slot) in color.iter_mut().enumerate() {
            let stone_c = lerp(c.color_stone[k], c.color_stone_alt[k], alt) * shade;
            let faced = stone_c * (0.7 + 0.3 * edge as f32) * (0.94 + 0.12 * bump as f32);
            *slot = lerp(faced, lichen_colour[k], lichen as f32 * 0.8).clamp(0.0, 1.0);
        }
        SurfaceSample {
            occlusion: (0.7 + 0.3 * edge) as f32,
            ..SurfaceSample::matte(height, color, 0.86 - 0.06 * edge as f32)
        }
    }
}

/// The course each texel row falls in, `height` rows over the tile.
fn row_courses(courses: &[Course], height: u32) -> Vec<usize> {
    (0..height)
        .map(|y| {
            let v = y as f64 / height as f64;
            courses
                .iter()
                .position(|k| {
                    (k.v0..k.v1).contains(&v)
                        || (k.v0..k.v1).contains(&(v + 1.0))
                        || (k.v0..k.v1).contains(&(v - 1.0))
                })
                .unwrap_or(0)
        })
        .collect()
}

impl DryStoneGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut outline_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.outline_noise, width, height, &mut outline_grid);
        let mut face_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.face_noise, width, height, &mut face_grid);
        let mut lichen_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.lichen_noise, width, height, &mut lichen_grid);

        let courses = courses(c);
        let row_course = row_courses(&courses, height);
        let cell = DryStoneCell {
            config: c,
            courses: &courses,
            row_course: &row_course,
            outline_grid: &outline_grid,
            face_grid: &face_grid,
            lichen_grid: &lichen_grid,
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
            ws.return_grid(outline_grid);
            ws.return_grid(face_grid);
            ws.return_grid(lichen_grid);
        }
        result
    }
}

impl TextureGenerator for DryStoneGenerator {
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
    use crate::test_support::{luminance, wrap_and_inside_steps};

    /// The courses tile V exactly - each one starts where the last ended,
    /// and the last ends one tile on from where the first starts - and each
    /// course's cuts run from 0 to 1 in order.
    #[test]
    fn the_courses_and_their_cuts_tile() {
        for jitter in [0.0, 0.5, 1.0] {
            let c = DryStoneConfig {
                course_jitter: jitter,
                irregularity: 1.0,
                ..DryStoneConfig::default()
            };
            let courses = courses(&c);
            assert_eq!(courses.len(), c.courses);
            for pair in courses.windows(2) {
                assert_eq!(pair[0].v1, pair[1].v0);
                assert!(pair[0].v1 > pair[0].v0, "a course with no height");
            }
            let span = courses[c.courses - 1].v1 - courses[0].v0;
            assert!((span - 1.0).abs() < 1e-12, "jitter {jitter}: span {span}");
            for k in &courses {
                assert_eq!(k.cuts.first(), Some(&0.0));
                assert_eq!(k.cuts.last(), Some(&1.0));
                assert!(k.cuts.windows(2).all(|w| w[0] < w[1]));
                assert_eq!(k.cuts.len(), k.stones.len() + 1);
            }
        }
    }

    /// The gaps are dark and deep, a share of the wall about what the gap
    /// width asks for, and the wall wraps in U: over eight seeds, the step
    /// from the last column back to the first is no bigger than a step
    /// between neighbours inside the tile. (V is the course layout's, which
    /// `the_courses_and_their_cuts_tile` proves exact; a single bake cannot
    /// judge it, since the wrap row always lies near the first course's
    /// jittered boundary.)
    #[test]
    fn the_gaps_are_dark_and_the_wall_wraps() {
        let size = 192;
        let (mut wrap, mut inside) = (0.0, 0.0);
        for seed in 1..=8 {
            let c = DryStoneConfig {
                seed,
                moss_level: 0.0,
                ..DryStoneConfig::default()
            };
            let map = DryStoneGenerator::new(c)
                .generate(size as u32, size as u32)
                .expect("a bake");
            let l = luminance(&map);
            // The gap's linear 0.03 packs to about 48 in the sRGB albedo.
            let dark = l.iter().filter(|&&x| x < 56.0).count() as f64 / l.len() as f64;
            assert!(
                (0.06..0.45).contains(&dark),
                "seed {seed}: {:.0}% of the wall is gap",
                dark * 100.0
            );
            let (w, i) = wrap_and_inside_steps(&l, size, 1, true);
            wrap += w;
            inside += i;
        }
        assert!(
            wrap < inside * 1.3,
            "a seam in U: {wrap:.0} against {inside:.0}"
        );
    }

    /// A stone narrower than its gap - in the envelope: three courses of
    /// twelve stones at the widest gap - stands no higher than the wall's
    /// relief. The tilt read a texel's offset against a box that had
    /// collapsed to nothing, so the blobs the outline noise opens in the
    /// gap stood up to ten times the relief (25.6, against 2.6 for the
    /// default wall).
    #[test]
    fn a_stone_narrower_than_its_gap_stays_within_the_relief() {
        let config = DryStoneConfig {
            courses: 3,
            stones_per_course: 12,
            gap_width: 0.3,
            irregularity: 1.0,
            face_relief: 1.0,
            ..Default::default()
        };
        let generator = DryStoneGenerator::new(config.clone());
        let (w, h) = (128u32, 128u32);
        let grid = |noise: &ToroidalNoise<Fbm<Perlin>>| {
            let mut out = Vec::new();
            sample_grid_into(noise, w, h, &mut out);
            out
        };
        let (outline_grid, face_grid, lichen_grid) = (
            grid(&generator.outline_noise),
            grid(&generator.face_noise),
            grid(&generator.lichen_noise),
        );
        let courses = courses(&config);
        let row_course = row_courses(&courses, h);
        let cell = DryStoneCell {
            config: &config,
            courses: &courses,
            row_course: &row_course,
            outline_grid: &outline_grid,
            face_grid: &face_grid,
            lichen_grid: &lichen_grid,
            width: w as usize,
        };
        let mut highest = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f64 / w as f64, y as f64 / h as f64);
                let height = cell.sample(x, y, u, v).height;
                assert!(height.is_finite(), "texel ({x}, {y}) height {height}");
                highest = highest.max(height.abs());
            }
        }
        assert!(
            highest <= STONE_RELIEF * 2.0,
            "a texel stands {highest:.2} high, past twice the relief ({})",
            STONE_RELIEF * 2.0
        );
    }
}
