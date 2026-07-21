//! Grass-blade tuft card generator.
//!
//! A small clump of upright, tapering blades fanning from a common root line
//! at the bottom of the cell — the workhorse billboard for a ground-cover
//! scatter tier.  Each blade is a curved, tip-tapered ribbon; blades fan
//! symmetrically and jitter their height, lean, curvature, width, and dryness
//! per variant so a field of stamped cards reads as many distinct tufts from
//! one bake.
//!
//! # Coordinate conventions
//! Local cell UV: blades root at the bottom edge (`v = 1`) and grow upward
//! toward the tips (`v = 0`) — the same base-down convention as the leaf and
//! petal cards, so an upright billboard quad meets the ground at its blade
//! roots.
//!
//! Upload with `map_to_images_card`; see [`crate::sprite`] for the shared
//! atlas conventions.

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.012;

/// Configures the appearance of a [`GrassTuftGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GrassTuftConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent tuft variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Blades per tuft (clamped to `1..=24`).
    pub blade_count: usize,
    /// Colour at the blade root / near the ground in linear RGB \[0, 1\] —
    /// typically the darkest, most shaded green.
    pub color_base: [f32; 3],
    /// Colour at the blade tips in linear RGB \[0, 1\] — typically brighter
    /// and warmer than the base.
    pub color_tip: [f32; 3],
    /// Colour of dry/dead blades and of the transparent-texel halo guard in
    /// linear RGB \[0, 1\] — a desaturated straw tone.
    pub color_dry: [f32; 3],
    /// Half-width of a blade at its root as a fraction of the cell
    /// `[0.01, 0.12]`.
    pub blade_width: f64,
    /// Tip taper exponent `[0.5, 4]`.  Higher → the blade narrows to its
    /// point faster (stiffer, more needle-like tips).
    pub blade_taper: f64,
    /// Shortest blade height as a fraction of the cell `[0.2, 1]`.
    pub height_min: f64,
    /// Tallest blade height as a fraction of the cell `[0.2, 1]`; clamped to
    /// be `>= height_min`.
    pub height_max: f64,
    /// Lateral tip fan as a fraction of the cell `[0, 0.5]` — how far the
    /// outermost blade tips splay from the tuft centre.
    pub fan_spread: f64,
    /// Blade curvature `[0, 0.5]` — the outward arc/droop each blade takes on
    /// toward its tip, in the fan direction.
    pub curve: f64,
    /// Horizontal spread of the blade roots at the base as a fraction of the
    /// cell `[0, 0.4]`.
    pub base_spread: f64,
    /// Fraction of blades rendered dry/straw-toned `[0, 1]`.
    pub dry_fraction: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for GrassTuftConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            // A grass tuft's primary use is a single ground-cover billboard —
            // one full tuft per card. Consumers wanting per-particle variety
            // opt into a variant atlas by raising these.
            variant_rows: 1,
            variant_cols: 1,
            blade_count: 9,
            color_base: [0.11, 0.17, 0.06],
            color_tip: [0.36, 0.46, 0.14],
            color_dry: [0.46, 0.39, 0.15],
            blade_width: 0.05,
            blade_taper: 1.3,
            height_min: 0.55,
            height_max: 0.96,
            fan_spread: 0.34,
            curve: 0.14,
            base_spread: 0.16,
            dry_fraction: 0.22,
            normal_strength: 1.2,
        }
    }
}

/// One blade of a tuft, in local cell UV with `y = 1 - v` measured up from
/// the root line at the bottom edge.
struct Blade {
    /// Root x-position at the bottom edge.
    base_x: f64,
    /// Signed lateral tip displacement (fan direction × spread).
    lean: f64,
    /// Additional quadratic bow along the blade, same sign as `lean`.
    curve: f64,
    /// Blade height (`y` at the tip).
    tip_y: f64,
    /// Root half-width.
    base_hw: f64,
    /// Dryness `[0, 1]` — blend weight toward `color_dry`.
    dry: f32,
    /// Per-blade brightness so overlapping blades read as separate ribbons.
    shade: f32,
}

/// One baked tuft variant.
pub(crate) struct GrassTuftCell {
    config: GrassTuftConfig,
    blades: Vec<Blade>,
}

impl GrassTuftCell {
    pub(crate) fn new(config: &GrassTuftConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let n = config.blade_count.clamp(1, 24);

        let hmin = config.height_min.clamp(0.2, 1.0);
        let hmax = config.height_max.clamp(0.2, 1.0).max(hmin);
        let fan = config.fan_spread.clamp(0.0, 0.5);
        let base_spread = config.base_spread.clamp(0.0, 0.4);
        let curve = config.curve.clamp(0.0, 0.5);
        let dry_fraction = config.dry_fraction.clamp(0.0, 1.0);
        let base_hw = config.blade_width.clamp(0.01, 0.12);

        let mut blades = Vec::with_capacity(n);
        for i in 0..n {
            // Symmetric fan position in [-1, 1]; a lone blade sits centred.
            let fan_pos = if n > 1 {
                (i as f64 / (n - 1) as f64) * 2.0 - 1.0
            } else {
                0.0
            };
            // Jitter the fan position slightly so blades do not land on a
            // perfectly even comb.
            let jittered = (fan_pos + rng.range(-0.18, 0.18)).clamp(-1.2, 1.2);
            let base_x = 0.5 + jittered * base_spread * 0.5 + rng.range(-0.02, 0.02);
            let lean = jittered * fan * rng.range(0.7, 1.1);
            let blade_curve = jittered * curve * rng.range(0.6, 1.2);
            // Central blades stand tallest; outer blades fall away.
            let height_bias = 1.0 - 0.35 * jittered.abs();
            let tip_y = (rng.range(hmin, hmax) * height_bias).clamp(0.12, 1.0);
            let dry = if rng.next_f64() < dry_fraction {
                rng.range(0.55, 1.0) as f32
            } else {
                0.0
            };
            let shade = rng.range(0.82, 1.08) as f32;
            blades.push(Blade {
                base_x,
                lean,
                curve: blade_curve,
                tip_y,
                base_hw: base_hw * rng.range(0.8, 1.1),
                dry,
                shade,
            });
        }

        Self {
            config: config.clone(),
            blades,
        }
    }
}

impl SpriteCell for GrassTuftCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let taper = c.blade_taper.clamp(0.5, 4.0);
        let y = 1.0 - v; // height above the root line

        // Winning (frontmost, taken as fullest-coverage) blade.
        let mut best_alpha = 0.0f64;
        let mut best_color = c.color_dry;
        let mut best_height = 0.0f64;

        for b in &self.blades {
            if b.tip_y <= 0.0 {
                continue;
            }
            let s = y / b.tip_y; // 0 at root, 1 at tip
            let sc = s.clamp(0.0, 1.0);

            // Centreline and half-width at this height.
            let cx = b.base_x + b.lean * sc + b.curve * sc * sc;
            let hw = (b.base_hw * (1.0 - sc).powf(taper)).max(0.0015);
            let lateral_d = (u - cx).abs() - hw;
            // Beyond the tip the blade ends; charge the overshoot as distance.
            let end_d = if s > 1.0 { (s - 1.0) * b.tip_y } else { 0.0 };
            let d = lateral_d.max(end_d);
            let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
            if alpha <= best_alpha {
                continue;
            }

            // Colour: root→tip gradient, then dry blend, then per-blade shade
            // and a gentle base darkening (contact shadow near the ground).
            let mut color = lerp_color(c.color_base, c.color_tip, sc as f32);
            if b.dry > 0.0 {
                color = lerp_color(color, c.color_dry, b.dry);
            }
            let ao = (0.72 + 0.28 * sc) as f32;
            let mul = b.shade * ao;
            color = [
                (color[0] * mul).clamp(0.0, 1.0),
                (color[1] * mul).clamp(0.0, 1.0),
                (color[2] * mul).clamp(0.0, 1.0),
            ];

            // Rounded cross-section ridge for the normal map.
            let rim = if hw > 1e-9 {
                ((u - cx) / hw).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let dome = 1.0 - rim * rim;
            let height = (0.2 + 0.7 * dome) * (0.6 + 0.4 * (1.0 - sc));

            best_alpha = alpha;
            best_color = color;
            best_height = height * alpha;
        }

        if best_alpha <= 0.0 {
            // Transparent texel: keep the straw halo colour so bilinear
            // filtering does not pull a dark rim across the blades.
            return SpriteSample {
                color: c.color_dry,
                alpha: 0.0,
                height: 0.0,
                roughness: 0.7,
            };
        }

        SpriteSample {
            color: best_color,
            alpha: best_alpha,
            height: best_height,
            // Tips catch a touch more sheen than the shaded base.
            roughness: 0.7,
        }
    }
}

/// Procedural grass-blade tuft card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct GrassTuftGenerator {
    config: GrassTuftConfig,
}

impl GrassTuftGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GrassTuftConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for GrassTuftGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| GrassTuftCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> GrassTuftConfig {
        GrassTuftConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..GrassTuftConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = GrassTuftGenerator::new(GrassTuftConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn roots_opaque_top_corners_transparent() {
        let map = GrassTuftGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        // The root line at the bottom centre carries blades.
        let base_row_has_blade = (48..80).any(|x| at(x, 124) == 255);
        assert!(base_row_has_blade, "blade roots at the base must be opaque");
        // The very top corners are above the tallest blade → transparent.
        assert_eq!(at(2, 2), 0, "top-left corner must be transparent");
        assert_eq!(at(125, 2), 0, "top-right corner must be transparent");
    }

    #[test]
    fn tuft_narrows_upward() {
        // A tuft fans out: near the base it spans more columns than near the
        // tips, so opaque coverage should shrink with height.
        let map = GrassTuftGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let coverage = |row: usize| {
            (0..128)
                .filter(|&x| map.albedo[(row * 128 + x) * 4 + 3] > 128)
                .count()
        };
        let low = coverage(112); // near the roots
        let high = coverage(24); // near the tips
        assert!(
            low >= high,
            "coverage should not grow toward the tips (low={low}, high={high})"
        );
    }

    #[test]
    fn variants_differ() {
        let atlas = GrassTuftConfig {
            variant_rows: 2,
            variant_cols: 2,
            ..GrassTuftConfig::default()
        };
        let map = GrassTuftGenerator::new(atlas)
            .generate(128, 128)
            .expect("generate failed");
        let differs = (0..64usize).any(|y| {
            (0..64usize).any(|x| {
                let a = ((y * 128) + x) * 4;
                let b = ((y * 128) + x + 64) * 4;
                map.albedo[a..a + 4] != map.albedo[b..b + 4]
            })
        });
        assert!(differs, "tuft atlas cells should bake distinct variants");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = GrassTuftGenerator::new(GrassTuftConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = GrassTuftGenerator::new(GrassTuftConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            GrassTuftGenerator::new(GrassTuftConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
