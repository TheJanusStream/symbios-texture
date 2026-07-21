//! Palmate broadleaf card generator.
//!
//! A leaf built in **polar** coordinates about its petiole attachment, rather
//! than as a midrib-and-envelope blade like the [`leaf`](crate::leaf) card.
//! A radius function with `lobe_count` peaks fanning across `fan_angle`
//! carves the classic palmate silhouettes — maple, sycamore, ivy, fig — with
//! main veins radiating to each lobe tip and an optional cordate (heart)
//! notch at the base.
//!
//! This is the **shape** half of the re-skin lever: the existing `Leaf` card
//! is pinnate with a single midrib, so a material variant that swaps to this
//! generator changes a species' leaf *form*, not merely its colour. Setting
//! `lobe_count` to 1 with a shallow `lobe_depth` yields a plain ovate/cordate
//! blade, so one generator covers both the palmate and simple-broadleaf
//! families.
//!
//! # Coordinate conventions
//! Local cell UV matches the [`leaf`](crate::leaf) card: `u = 0.5` is the
//! midline, `v = 0` is the petiole attachment (base), `v = 1` is the apex.
//!
//! Upload with `map_to_images_card`; see [`crate::sprite`] for the shared
//! atlas conventions.

use std::f64::consts::PI;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.01;

/// Configures the appearance of a [`BroadleafGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadleafConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Blade interior colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour toward the margin in linear RGB \[0, 1\].
    pub color_edge: [f32; 3],
    /// Palmate lobes `[1, 9]`.  `5` is maple/sycamore, `3` is ivy/fig, `1`
    /// with a shallow depth is a plain ovate blade.
    pub lobe_count: f64,
    /// Depth of the sinuses between lobes `[0, 0.8]` — how far the margin
    /// cuts back toward the attachment between lobe tips.
    pub lobe_depth: f64,
    /// Half-angle of the leaf fan in degrees `[30, 110]`.  Wider fans give
    /// the broad, almost circular outline of a sycamore.
    pub fan_angle: f64,
    /// Blade radius as a fraction of the cell `[0.3, 1]`.
    pub radius: f64,
    /// Cordate basal notch depth `[0, 0.5]` — `0` is a wedge (cuneate) base,
    /// larger cuts the heart-shaped sinus of a lime or ivy leaf.
    pub base_notch: f64,
    /// Width of the radiating main veins as a fraction of the blade
    /// `[0.01, 0.2]`.
    pub vein_width: f64,
    /// Fraction of the V axis reserved for the petiole `[0, 0.3]`.
    pub petiole_length: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for BroadleafConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            color_base: [0.13, 0.26, 0.08],
            color_edge: [0.28, 0.38, 0.12],
            lobe_count: 5.0,
            lobe_depth: 0.34,
            fan_angle: 78.0,
            radius: 0.92,
            base_notch: 0.18,
            vein_width: 0.05,
            petiole_length: 0.1,
            normal_strength: 1.4,
        }
    }
}

/// One baked broadleaf variant.
pub(crate) struct BroadleafCell {
    config: BroadleafConfig,
    /// Per-variant radius scale.
    radius: f64,
    /// Per-variant fan half-angle, radians.
    fan: f64,
    /// Signed rotation of the whole blade, radians.
    tilt: f64,
}

impl BroadleafCell {
    pub(crate) fn new(config: &BroadleafConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        Self {
            config: config.clone(),
            radius: config.radius.clamp(0.3, 1.0) * rng.range(0.9, 1.0),
            fan: config.fan_angle.clamp(30.0, 110.0).to_radians() * rng.range(0.94, 1.06),
            tilt: rng.range(-0.06, 0.06),
        }
    }

    /// Margin radius at fan-relative angle `q` in `[-1, 1]`.
    fn margin(&self, q: f64) -> f64 {
        let c = &self.config;
        let lobes = c.lobe_count.clamp(1.0, 9.0);
        let depth = c.lobe_depth.clamp(0.0, 0.8);

        // Lobe modulation: `lobes` peaks across the fan. cos(lobes*π*q) has a
        // maximum at q = 0 and alternates, so an odd lobe count puts a tip on
        // the midline — which is what a maple actually does.
        let lobe = (lobes * PI * q).cos();
        // Overall fan envelope: falls to zero at the fan edges so the blade
        // closes instead of being sliced off square.
        let envelope = (q * PI * 0.5).cos().max(0.0).powf(0.45);

        let mut r = self.radius * envelope * (1.0 - depth * 0.5 * (1.0 - lobe));

        // Cordate basal notch: pull the margin in near the fan extremes.
        let notch = c.base_notch.clamp(0.0, 0.5);
        if notch > 0.0 {
            let t = ((q.abs() - (1.0 - notch)) / notch.max(1e-6)).clamp(0.0, 1.0);
            r *= 1.0 - 0.75 * t * t;
        }
        r.max(0.0)
    }
}

impl SpriteCell for BroadleafCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let out = SpriteSample {
            color: c.color_edge,
            alpha: 0.0,
            height: 0.0,
            roughness: 0.55,
        };

        // Petiole: a narrow stalk below the blade attachment.
        let pet = c.petiole_length.clamp(0.0, 0.3);
        if pet > 0.0 && v < pet {
            let dist = (u - 0.5).abs();
            let hw = 0.008;
            if dist >= hw {
                return out;
            }
            let t = dist / hw;
            return SpriteSample {
                color: c.color_base,
                alpha: 1.0,
                height: (1.0 - t * t).sqrt() * 0.5,
                roughness: 0.6,
            };
        }

        // Polar coordinates about the attachment point at the blade base.
        let ox = u - 0.5;
        let oy = v - pet;
        let r = (ox * ox + oy * oy).sqrt();
        if r <= 1e-6 {
            return out;
        }
        // Angle from straight up (+v), positive toward +u, with the blade's
        // per-variant tilt removed.
        let theta = ox.atan2(oy.max(0.0)) - self.tilt;
        // Behind the attachment: no blade.
        if oy < 0.0 {
            return out;
        }
        // Fan-relative angle in [-1, 1]; outside the fan there is no blade.
        let q = theta / self.fan;
        if q.abs() > 1.0 {
            return out;
        }

        let margin = self.margin(q);
        if margin <= 1e-6 {
            return out;
        }

        let d = r - margin;
        let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return out;
        }

        let edge_frac = (r / margin).clamp(0.0, 1.0);

        // Main veins radiate from the attachment to each lobe tip: the lobe
        // maxima sit where cos(lobes*π*q) peaks, i.e. at even multiples.
        let lobes = c.lobe_count.clamp(1.0, 9.0);
        let vein_phase = (lobes * PI * q).cos().abs();
        let vw = c.vein_width.clamp(0.01, 0.2);
        let vein = vein_phase.powf(1.0 / vw.max(1e-3) * 0.06);

        // Cross-section dome plus the raised vein ribs.
        let dome = 1.0 - edge_frac * edge_frac;
        let height = (dome * 0.45 + vein * 0.45 * (1.0 - edge_frac * 0.5)).clamp(0.0, 1.0);

        let mut color = lerp_color(c.color_base, c.color_edge, edge_frac as f32);
        let vein_light = (vein as f32) * 0.13;
        color = [
            (color[0] + vein_light).min(1.0),
            (color[1] + vein_light * 0.85).min(1.0),
            (color[2] + vein_light * 0.4).min(1.0),
        ];

        SpriteSample {
            color,
            alpha,
            height: height * alpha,
            roughness: 0.55,
        }
    }
}

/// Procedural palmate broadleaf card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct BroadleafGenerator {
    config: BroadleafConfig,
}

impl BroadleafGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: BroadleafConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for BroadleafGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| BroadleafCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single() -> BroadleafConfig {
        BroadleafConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..BroadleafConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = BroadleafGenerator::new(BroadleafConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn blade_is_opaque_and_top_corners_are_clear() {
        let map = BroadleafGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        // Mid-blade on the midline (v ~ 0.45 → row ~58 in top-down UV).
        assert_eq!(at(64, 58), 255, "mid-blade must be opaque");
        // The base corners sit outside the fan.
        assert_eq!(at(3, 125), 0, "bottom-left corner transparent");
        assert_eq!(at(124, 125), 0, "bottom-right corner transparent");
    }

    #[test]
    fn deeper_sinuses_remove_blade_area() {
        let cover = |depth: f64| {
            BroadleafGenerator::new(BroadleafConfig {
                lobe_depth: depth,
                ..single()
            })
            .generate(128, 128)
            .expect("generate failed")
            .albedo
            .chunks(4)
            .filter(|px| px[3] > 128)
            .count()
        };
        assert!(
            cover(0.7) < cover(0.0),
            "cutting deeper sinuses should shrink the blade"
        );
    }

    #[test]
    fn is_shape_distinct_from_the_pinnate_leaf_card() {
        // The point of this generator is a different silhouette. A palmate
        // blade is far wider relative to its height than the narrow pinnate
        // `Leaf` envelope, so compare their widest opaque spans.
        let widest = |m: &crate::generator::TextureMap| {
            (0..128)
                .map(|row| {
                    (0..128)
                        .filter(|&x| m.albedo[(row * 128 + x) * 4 + 3] > 128)
                        .count()
                })
                .max()
                .unwrap()
        };
        let palmate = BroadleafGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let pinnate = crate::leaf::LeafGenerator::new(crate::leaf::LeafConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        assert!(
            widest(&palmate) > widest(&pinnate),
            "palmate blade ({}) should be broader than the pinnate leaf ({})",
            widest(&palmate),
            widest(&pinnate)
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = BroadleafGenerator::new(BroadleafConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = BroadleafGenerator::new(BroadleafConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            BroadleafGenerator::new(BroadleafConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
