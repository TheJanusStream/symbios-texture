//! Conifer needle-cluster card generator.
//!
//! A short shoot bearing paired needles: a woody axis runs up the card with
//! pairs of thin, straight needles splaying outward and forward at a fixed
//! angle, shortening toward the shoot tip.  This is the foliage billboard for
//! pine, spruce, and fir — the conifer counterpart of the broadleaf
//! [`twig`](crate::twig) card.
//!
//! Tune it across the conifer genera with three knobs: a wide `needle_angle`
//! with long needles reads as pine, a narrow angle with short needles as
//! spruce, and a high `pair_count` with minimal taper as fir.
//!
//! # Coordinate conventions
//! Local cell UV: the shoot roots at the bottom edge (`v = 1`) and grows
//! toward its tip (`v = 0`) — the same base-down convention as the grass,
//! reed, and petal cards.
//!
//! Upload with `map_to_images_card`; see [`crate::sprite`] for the shared
//! atlas conventions.

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.008;

/// Configures the appearance of a [`NeedleGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeedleConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Needle pairs along the shoot (clamped to `1..=24`).
    pub pair_count: usize,
    /// Colour at the needle bases in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour at the needle tips in linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// Woody shoot colour in linear RGB \[0, 1\]; also the transparent-texel
    /// halo guard.
    pub color_shoot: [f32; 3],
    /// Needle splay from the shoot axis, in degrees `[5, 85]`.  Small angles
    /// hug the shoot (spruce); large angles fan out (pine).
    pub needle_angle: f64,
    /// Needle length as a fraction of the cell `[0.05, 0.6]`.
    pub needle_length: f64,
    /// Needle half-width as a fraction of the cell `[0.002, 0.03]`.
    pub needle_width: f64,
    /// How much shorter the needles get toward the shoot tip `[0, 1]`.
    /// `0` gives a cylindrical spray; `1` tapers to a point.
    pub length_taper: f64,
    /// Shoot length as a fraction of the cell `[0.2, 1]`.
    pub shoot_length: f64,
    /// Shoot half-width as a fraction of the cell `[0.002, 0.04]`.
    pub shoot_width: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            pair_count: 11,
            color_base: [0.05, 0.13, 0.07],
            color_tip: [0.16, 0.31, 0.14],
            color_shoot: [0.21, 0.13, 0.07],
            needle_angle: 42.0,
            needle_length: 0.3,
            needle_width: 0.009,
            length_taper: 0.55,
            shoot_length: 0.9,
            shoot_width: 0.009,
            normal_strength: 1.2,
        }
    }
}

/// Distance from point `p` to segment `a`→`b`, plus the projection parameter
/// along the segment in `[0, 1]`.
#[inline]
fn point_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len2 = abx * abx + aby * aby;
    let s = if len2 <= 1e-12 {
        0.0
    } else {
        (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0)
    };
    let (cx, cy) = (a.0 + abx * s, a.1 + aby * s);
    let (dx, dy) = (p.0 - cx, p.1 - cy);
    ((dx * dx + dy * dy).sqrt(), s)
}

/// One needle of a cluster, as a segment in local cell UV with `y = 1 - v`.
struct Needle {
    a: (f64, f64),
    b: (f64, f64),
    /// Per-needle brightness so crossing needles stay legible.
    shade: f32,
}

/// One baked needle-cluster variant.
pub(crate) struct NeedleCell {
    config: NeedleConfig,
    needles: Vec<Needle>,
    shoot_len: f64,
    shoot_hw: f64,
    needle_hw: f64,
}

impl NeedleCell {
    pub(crate) fn new(config: &NeedleConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let pairs = config.pair_count.clamp(1, 24);
        let angle = config.needle_angle.clamp(5.0, 85.0).to_radians();
        let length = config.needle_length.clamp(0.05, 0.6);
        let taper = config.length_taper.clamp(0.0, 1.0);
        let shoot_len = config.shoot_length.clamp(0.2, 1.0);
        let shoot_hw = config.shoot_width.clamp(0.002, 0.04);
        let needle_hw = config.needle_width.clamp(0.002, 0.03);

        let mut needles = Vec::with_capacity(pairs * 2);
        for i in 0..pairs {
            // Attachment height along the shoot.
            let t = (i as f64 + 0.5) / pairs as f64;
            let ay = t * shoot_len;
            // Needles shorten toward the tip.
            let len = length * (1.0 - taper * t) * rng.range(0.85, 1.05);
            for side in [-1.0f64, 1.0] {
                // Jitter each needle's splay so the pairs aren't a perfect comb.
                let a = angle * rng.range(0.85, 1.15);
                let dir = (side * a.sin(), a.cos());
                needles.push(Needle {
                    a: (0.5, ay),
                    b: (0.5 + dir.0 * len, ay + dir.1 * len),
                    shade: rng.range(0.82, 1.1) as f32,
                });
            }
        }

        Self {
            config: config.clone(),
            needles,
            shoot_len,
            shoot_hw,
            needle_hw,
        }
    }
}

impl SpriteCell for NeedleCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let y = 1.0 - v;
        let p = (u, y);

        let mut best_alpha = 0.0f64;
        let mut best_color = c.color_shoot;
        let mut best_height = 0.0f64;
        let mut best_rough = 0.55f32;

        // Woody shoot: a vertical capsule from the base to the shoot tip.
        {
            let (dist, _) = point_segment(p, (0.5, 0.0), (0.5, self.shoot_len));
            let d = dist - self.shoot_hw;
            let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
            if alpha > 0.0 {
                let rim = (dist / self.shoot_hw.max(1e-9)).clamp(0.0, 1.0);
                best_alpha = alpha;
                best_color = c.color_shoot;
                best_height = (1.0 - rim * rim) * 0.6 * alpha;
                best_rough = 0.8;
            }
        }

        for n in &self.needles {
            let (dist, s) = point_segment(p, n.a, n.b);
            // Needles taper to a point at their tip.
            let hw = (self.needle_hw * (1.0 - 0.8 * s)).max(0.0008);
            let d = dist - hw;
            let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
            if alpha <= best_alpha {
                continue;
            }

            let mut color = lerp_color(c.color_base, c.color_tip, s as f32);
            color = [
                (color[0] * n.shade).clamp(0.0, 1.0),
                (color[1] * n.shade).clamp(0.0, 1.0),
                (color[2] * n.shade).clamp(0.0, 1.0),
            ];

            // Rounded cross-section along the needle.
            let rim = (dist / hw.max(1e-9)).clamp(0.0, 1.0);
            let dome = 1.0 - rim * rim;

            best_alpha = alpha;
            best_color = color;
            best_height = (0.25 + 0.75 * dome) * alpha;
            best_rough = 0.5;
        }

        if best_alpha <= 0.0 {
            return SpriteSample {
                color: c.color_shoot,
                alpha: 0.0,
                height: 0.0,
                roughness: 0.55,
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

/// Procedural conifer needle-cluster card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct NeedleGenerator {
    config: NeedleConfig,
}

impl NeedleGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: NeedleConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for NeedleGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| NeedleCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single() -> NeedleConfig {
        NeedleConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..NeedleConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = NeedleGenerator::new(NeedleConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn shoot_axis_is_opaque_and_corners_are_clear() {
        let map = NeedleGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        // The shoot runs up the middle of the card.
        assert_eq!(at(64, 100), 255, "shoot axis must be opaque");
        assert_eq!(at(2, 2), 0, "top-left corner transparent");
        assert_eq!(at(125, 2), 0, "top-right corner transparent");
    }

    #[test]
    fn wider_angle_spreads_the_spray() {
        let narrow = NeedleGenerator::new(NeedleConfig {
            needle_angle: 10.0,
            ..single()
        })
        .generate(128, 128)
        .expect("generate failed");
        let wide = NeedleGenerator::new(NeedleConfig {
            needle_angle: 80.0,
            ..single()
        })
        .generate(128, 128)
        .expect("generate failed");
        // Widest opaque span across any row.
        let span = |m: &crate::generator::TextureMap| {
            (0..128)
                .map(|row| {
                    (0..128)
                        .filter(|&x| m.albedo[(row * 128 + x) * 4 + 3] > 128)
                        .count()
                })
                .max()
                .unwrap()
        };
        assert!(
            span(&wide) > span(&narrow),
            "a wider splay should broaden the cluster ({} vs {})",
            span(&wide),
            span(&narrow)
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = NeedleGenerator::new(NeedleConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = NeedleGenerator::new(NeedleConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            NeedleGenerator::new(NeedleConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
