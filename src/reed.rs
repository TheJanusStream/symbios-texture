//! Reed / cattail card generator.
//!
//! A shoreline emergent: a few tall, near-vertical strap leaves rising from a
//! common waterline base, optionally topped by the cattail's signature brown
//! catkin — a velvety cylindrical spike on a bare stalk.  The billboard for
//! wetland margins and pond edges.
//!
//! Distinct from the [`grass`](crate::grass) tuft card: reeds are far taller
//! relative to their width, stand almost straight rather than fanning, taper
//! only near the tip, and carry the seed-head spike no grass tuft has.
//!
//! # Coordinate conventions
//! Local cell UV: leaves root at the bottom edge (`v = 1`) and rise toward
//! the tips (`v = 0`) — the same base-down convention as the grass, leaf, and
//! petal cards, so an upright billboard meets the waterline at its base.
//!
//! Upload with `map_to_images_card`; see [`crate::sprite`] for the shared
//! atlas conventions.

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.01;

/// Configures the appearance of a [`ReedGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReedConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Leaves per clump (clamped to `1..=12`).
    pub blade_count: usize,
    /// Colour at the submerged/shaded base in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour at the leaf tips in linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// Colour of the cattail catkin (seed head) in linear RGB \[0, 1\]; also
    /// the transparent-texel halo guard.
    pub color_catkin: [f32; 3],
    /// Half-width of a leaf at its base as a fraction of the cell
    /// `[0.008, 0.08]`.
    pub blade_width: f64,
    /// Shortest leaf height as a fraction of the cell `[0.3, 1]`.
    pub height_min: f64,
    /// Tallest leaf height as a fraction of the cell `[0.3, 1]`; clamped to
    /// be `>= height_min`.
    pub height_max: f64,
    /// Lateral tip lean as a fraction of the cell `[0, 0.3]` — reeds stand
    /// far straighter than grass, so keep this small.
    pub lean: f64,
    /// Fraction of the leaf length over which the tip tapers `[0.05, 0.8]`.
    /// Reed straps hold their width then narrow only near the very tip.
    pub tip_fraction: f64,
    /// Share of stalks bearing a catkin `[0, 1]`.  `0` is a pure reed clump
    /// (no seed heads); `1` gives every stalk a cattail.
    pub catkin_share: f64,
    /// Catkin length as a fraction of the cell `[0, 0.4]`.
    pub catkin_length: f64,
    /// Catkin half-width as a fraction of the cell `[0.005, 0.06]`.
    pub catkin_width: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ReedConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            blade_count: 6,
            color_base: [0.10, 0.16, 0.06],
            color_tip: [0.38, 0.44, 0.16],
            color_catkin: [0.24, 0.13, 0.05],
            blade_width: 0.022,
            height_min: 0.62,
            height_max: 0.98,
            lean: 0.09,
            tip_fraction: 0.28,
            catkin_share: 0.4,
            catkin_length: 0.2,
            catkin_width: 0.022,
            normal_strength: 1.2,
        }
    }
}

/// One stalk of a reed clump, in local cell UV with `y = 1 - v` measured up
/// from the waterline at the bottom edge.
struct Stalk {
    /// Root x-position at the bottom edge.
    base_x: f64,
    /// Signed lateral tip displacement.
    lean: f64,
    /// Total stalk height (`y` at the tip, catkin included).
    tip_y: f64,
    /// Root half-width.
    base_hw: f64,
    /// Catkin length (0 when this stalk bears none).
    catkin: f64,
    /// Per-stalk brightness so overlapping leaves read as separate straps.
    shade: f32,
}

/// One baked reed-clump variant.
pub(crate) struct ReedCell {
    config: ReedConfig,
    stalks: Vec<Stalk>,
}

impl ReedCell {
    pub(crate) fn new(config: &ReedConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let n = config.blade_count.clamp(1, 12);
        let hmin = config.height_min.clamp(0.3, 1.0);
        let hmax = config.height_max.clamp(0.3, 1.0).max(hmin);
        let lean = config.lean.clamp(0.0, 0.3);
        let base_hw = config.blade_width.clamp(0.008, 0.08);
        let share = config.catkin_share.clamp(0.0, 1.0);
        let catkin_len = config.catkin_length.clamp(0.0, 0.4);

        let mut stalks = Vec::with_capacity(n);
        for i in 0..n {
            // Spread the roots across the clump with a jittered even spacing.
            let slot = if n > 1 {
                (i as f64 / (n - 1) as f64) * 2.0 - 1.0
            } else {
                0.0
            };
            let jittered = slot + rng.range(-0.15, 0.15);
            let base_x = 0.5 + jittered * 0.16 + rng.range(-0.015, 0.015);
            stalks.push(Stalk {
                base_x,
                lean: jittered * lean * rng.range(0.6, 1.1),
                tip_y: rng.range(hmin, hmax),
                base_hw: base_hw * rng.range(0.85, 1.1),
                catkin: if rng.next_f64() < share {
                    catkin_len
                } else {
                    0.0
                },
                shade: rng.range(0.85, 1.08) as f32,
            });
        }

        Self {
            config: config.clone(),
            stalks,
        }
    }
}

impl SpriteCell for ReedCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let tip_fraction = c.tip_fraction.clamp(0.05, 0.8);
        let catkin_hw = c.catkin_width.clamp(0.005, 0.06);
        let y = 1.0 - v; // height above the waterline

        let mut best_alpha = 0.0f64;
        let mut best_color = c.color_catkin;
        let mut best_height = 0.0f64;
        let mut best_rough = 0.65f32;

        for s in &self.stalks {
            if s.tip_y <= 0.0 {
                continue;
            }
            let t = y / s.tip_y; // 0 at root, 1 at tip
            let tc = t.clamp(0.0, 1.0);
            let cx = s.base_x + s.lean * tc * tc;
            let lateral = (u - cx).abs();

            // Catkin: a blunt cylinder occupying the top of a bearing stalk.
            let catkin_start = if s.catkin > 0.0 {
                1.0 - (s.catkin / s.tip_y).min(0.9)
            } else {
                2.0
            };

            let (hw, is_catkin) = if t >= catkin_start {
                // Rounded ends on the spike.
                let span = (1.0 - catkin_start).max(1e-6);
                let local = ((t - catkin_start) / span).clamp(0.0, 1.0);
                let cap = (1.0 - (2.0 * local - 1.0).powi(6)).max(0.0).powf(0.35);
                (catkin_hw * cap, true)
            } else {
                // Strap leaf: holds its width, then tapers over the top
                // `tip_fraction` of its length.
                let taper_start = 1.0 - tip_fraction;
                let w = if tc <= taper_start {
                    1.0
                } else {
                    let k = (tc - taper_start) / tip_fraction;
                    (1.0 - k).max(0.0).powf(0.7)
                };
                ((s.base_hw * w).max(0.0012), false)
            };

            let lateral_d = lateral - hw;
            let end_d = if t > 1.0 { (t - 1.0) * s.tip_y } else { 0.0 };
            let d = lateral_d.max(end_d);
            let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
            if alpha <= best_alpha {
                continue;
            }

            let rim = if hw > 1e-9 {
                ((u - cx) / hw).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let dome = 1.0 - rim * rim;

            let (color, height, rough) = if is_catkin {
                // Velvety brown spike: matte, strongly domed.
                let shade = 0.82 + 0.18 * dome;
                (
                    [
                        (c.color_catkin[0] * shade as f32).clamp(0.0, 1.0),
                        (c.color_catkin[1] * shade as f32).clamp(0.0, 1.0),
                        (c.color_catkin[2] * shade as f32).clamp(0.0, 1.0),
                    ],
                    0.35 + 0.65 * dome,
                    0.92f32,
                )
            } else {
                let mut col = lerp_color(c.color_base, c.color_tip, tc as f32);
                // Contact shading near the waterline.
                let ao = (0.70 + 0.30 * tc) as f32;
                let mul = s.shade * ao;
                col = [
                    (col[0] * mul).clamp(0.0, 1.0),
                    (col[1] * mul).clamp(0.0, 1.0),
                    (col[2] * mul).clamp(0.0, 1.0),
                ];
                (col, (0.2 + 0.7 * dome) * (0.65 + 0.35 * (1.0 - tc)), 0.6f32)
            };

            best_alpha = alpha;
            best_color = color;
            best_height = height * alpha;
            best_rough = rough;
        }

        if best_alpha <= 0.0 {
            return SpriteSample {
                color: c.color_catkin,
                alpha: 0.0,
                height: 0.0,
                roughness: 0.65,
            };
        }

        SpriteSample {
            color: best_color,
            alpha: best_alpha,
            height: best_height,
            roughness: best_rough,
        }
    }
}

/// Procedural reed / cattail card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct ReedGenerator {
    config: ReedConfig,
}

impl ReedGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ReedConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for ReedGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| ReedCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single() -> ReedConfig {
        ReedConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..ReedConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = ReedGenerator::new(ReedConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn roots_opaque_corners_transparent() {
        let map = ReedGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        assert!(
            (48..80).any(|x| at(x, 126) == 255),
            "stalk roots at the base must be opaque"
        );
        assert_eq!(at(2, 2), 0, "top-left corner transparent");
        assert_eq!(at(125, 2), 0, "top-right corner transparent");
    }

    #[test]
    fn reeds_are_narrower_than_they_are_tall() {
        // A reed clump is a tall, narrow silhouette: its widest opaque span is
        // a small fraction of its height.
        let map = ReedGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let widest = (0..128)
            .map(|row| {
                (0..128)
                    .filter(|&x| map.albedo[(row * 128 + x) * 4 + 3] > 128)
                    .count()
            })
            .max()
            .unwrap();
        let tallest = (0..128)
            .filter(|&row| (0..128).any(|x| map.albedo[(row * 128 + x) * 4 + 3] > 128))
            .count();
        assert!(
            widest < tallest,
            "reed clump should be taller than wide (w={widest}, h={tallest})"
        );
    }

    #[test]
    fn catkins_add_brown_coverage() {
        let bare = ReedGenerator::new(ReedConfig {
            catkin_share: 0.0,
            ..single()
        })
        .generate(128, 128)
        .expect("generate failed");
        let heads = ReedGenerator::new(ReedConfig {
            catkin_share: 1.0,
            catkin_width: 0.05,
            ..single()
        })
        .generate(128, 128)
        .expect("generate failed");
        // The catkin is a thick brown spike: red-dominant opaque texels appear.
        let brownish = |m: &crate::generator::TextureMap| {
            m.albedo
                .chunks(4)
                .filter(|px| px[3] > 200 && px[0] > px[1] && px[1] >= px[2])
                .count()
        };
        assert!(
            brownish(&heads) > brownish(&bare),
            "cattail heads should add brown coverage"
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = ReedGenerator::new(ReedConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = ReedGenerator::new(ReedConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            ReedGenerator::new(ReedConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
