//! Frond-pinna card generator.
//!
//! A single leaflet (pinna) of a pinnate frond: a narrow, lanceolate blade
//! with a strong central midrib, shallow pinnate secondary veins, and a
//! margin that ranges from **entire** (a smooth-edged palm leaflet) to
//! **pinnatifid** (the lobed pinnule of a fern) via one depth knob.  It is
//! the drop-in foliage card for the leaflet slot of an L-system palm or fern,
//! whose rachis geometry is drawn by the grammar while each leaflet is
//! stamped as one of these billboards.
//!
//! Distinct from the broad, serrated, lobed [`LeafGenerator`](crate::leaf)
//! card: a frond pinna is several times narrower, has an unbroken (or
//! regularly-lobed, never randomly-toothed) margin, and reads as a single
//! strap rather than a broadleaf.
//!
//! # Coordinate conventions
//! Local cell UV matches the [`leaf`](crate::leaf) card: `u = 0.5` is the
//! midrib, `v = 0` is the base (rachis attachment), `v = 1` is the tip.
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

/// Configures the appearance of a [`FrondGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FrondConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Blade interior colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour toward the margin / tip in linear RGB \[0, 1\].
    pub color_edge: [f32; 3],
    /// Maximum half-width of the pinna as a fraction of the cell
    /// `[0.04, 0.30]` — a frond leaflet is a narrow strap, far below the
    /// broadleaf `LeafConfig` envelope.
    pub width: f64,
    /// Tip acuteness `[0.4, 3]`.  Higher → the blade holds its width longer
    /// then narrows abruptly to a sharper, more drawn-out point.
    pub tip_taper: f64,
    /// Width of the midrib ridge as a fraction of the local half-width
    /// `[0.05, 0.5]`.
    pub midrib_width: f64,
    /// Number of pinnate secondary vein pairs branching from the midrib.
    pub vein_count: f64,
    /// Margin lobe count per side (`0` with `lobe_depth = 0` → entire).
    pub lobe_count: f64,
    /// Margin lobe depth as a fraction of the envelope `[0, 0.6]`.  `0` is an
    /// entire (palm-leaflet) margin; larger cuts the crenate/pinnatifid
    /// margin of a fern pinnule.
    pub lobe_depth: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for FrondConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            color_base: [0.11, 0.30, 0.09],
            color_edge: [0.22, 0.42, 0.13],
            width: 0.13,
            tip_taper: 1.4,
            midrib_width: 0.16,
            vein_count: 9.0,
            lobe_count: 0.0,
            lobe_depth: 0.0,
            normal_strength: 1.3,
        }
    }
}

/// Peak of `sin(πv)·exp(-v)` over `v ∈ [0, 1]` — the un-normalised envelope
/// maximum, divided out so `width` is the true maximum half-width.
const ENVELOPE_PEAK: f64 = 0.606_530_66;

/// One baked pinna variant.
pub(crate) struct FrondCell {
    config: FrondConfig,
    /// Per-variant length scale (fraction of the full cell height reached).
    length: f64,
    /// Per-variant width scale.
    width_scale: f64,
    /// Signed lateral skew of the midrib toward the tip.
    skew: f64,
}

impl FrondCell {
    pub(crate) fn new(config: &FrondConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        Self {
            config: config.clone(),
            length: rng.range(0.9, 1.0),
            width_scale: rng.range(0.85, 1.05),
            skew: rng.range(-0.05, 0.05),
        }
    }

    /// Half-width of the lanceolate envelope at base-relative position `v`.
    fn envelope(&self, v: f64) -> f64 {
        let c = &self.config;
        // Lanceolate: widest near the base third, long taper to the tip.
        let shape = (PI * v).sin() * (-v * c.tip_taper).exp() / ENVELOPE_PEAK;
        c.width.clamp(0.04, 0.30) * self.width_scale * shape
    }
}

impl SpriteCell for FrondCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let out = SpriteSample {
            color: c.color_edge,
            alpha: 0.0,
            height: 0.0,
            roughness: 0.5,
        };

        // Remap V so the pinna reaches only `length` up the cell; beyond that
        // is transparent tip margin.
        if v <= 0.0 || v >= self.length {
            return out;
        }
        let vb = v / self.length;

        let env = self.envelope(vb);
        if env <= 1e-6 {
            return out;
        }

        // Margin lobes (pinnatifid) — a sign-preserving cosine along V scales
        // the envelope, cutting regular notches; entire when depth is 0.
        let eff_env = if c.lobe_count > 0.0 && c.lobe_depth > 0.0 {
            let cos_val = (vb * (2.0 * c.lobe_count + 1.0) * PI).cos();
            let shaped = cos_val.signum() * cos_val.abs();
            (env * (1.0 + shaped * c.lobe_depth.clamp(0.0, 0.6))).max(0.0)
        } else {
            env
        };
        if eff_env <= 1e-6 {
            return out;
        }

        // Midrib skews slightly toward the tip for a natural sweep.
        let centre = 0.5 + self.skew * vb * vb;
        let raw = (u - centre).abs();
        let d = raw - eff_env;
        let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return out;
        }

        let edge_frac = (raw / eff_env).clamp(0.0, 1.0);

        // Cross-section dome, highest at the midrib.
        let dome = 1.0 - edge_frac * edge_frac;

        // Midrib ridge.
        let midrib_norm = (raw / (env * c.midrib_width.clamp(0.05, 0.5))).min(1.0);
        let midrib = (1.0 - midrib_norm).powi(2);

        // Pinnate secondary veins: shallow chevrons from midrib toward tip.
        let vein_freq = c.vein_count.max(0.0) * PI;
        let secondary = (vb * vein_freq - raw * vein_freq * 2.4)
            .sin()
            .abs()
            .powf(4.0);

        let height = (dome * 0.35 + midrib * 0.45 + secondary * 0.2).clamp(0.0, 1.0);

        // Colour: interior toward margin, midrib lightened.
        let mut color = lerp_color(c.color_base, c.color_edge, edge_frac as f32);
        let vein_light = (midrib as f32 * 0.6 + secondary as f32 * 0.4) * 0.14;
        color = [
            (color[0] + vein_light).min(1.0),
            (color[1] + vein_light * 0.85).min(1.0),
            (color[2] + vein_light * 0.4).min(1.0),
        ];

        SpriteSample {
            color,
            alpha,
            height: height * alpha,
            roughness: 0.5,
        }
    }
}

/// Procedural frond-pinna card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct FrondGenerator {
    config: FrondConfig,
}

impl FrondGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: FrondConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for FrondGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| FrondCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single() -> FrondConfig {
        FrondConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..FrondConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = FrondGenerator::new(FrondConfig::default())
            .generate(48, 96)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 48 * 96 * 4);
        assert_eq!(map.normal.len(), 48 * 96 * 4);
        assert_eq!(map.roughness.len(), 48 * 96 * 4);
    }

    #[test]
    fn midrib_opaque_margins_transparent() {
        let map = FrondGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        // The midrib around the widest point (near the base, at the bottom
        // of the cell in generate_atlas coords) is opaque.
        assert_eq!(at(64, 96), 255, "midrib must be opaque");
        // The far side margins are transparent — the pinna is narrow.
        assert_eq!(at(4, 64), 0, "left margin transparent");
        assert_eq!(at(124, 64), 0, "right margin transparent");
    }

    #[test]
    fn pinna_is_narrow() {
        // A frond pinna is much narrower than a broadleaf: at its widest the
        // opaque span is a small fraction of the card.
        let map = FrondGenerator::new(single())
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
        assert!(widest < 64, "pinna should be narrow (widest span {widest})");
        assert!(widest > 4, "pinna should be visible (widest span {widest})");
    }

    #[test]
    fn lobed_margin_removes_coverage() {
        let entire = FrondGenerator::new(single())
            .generate(128, 128)
            .expect("generate failed");
        let lobed = FrondGenerator::new(FrondConfig {
            lobe_count: 6.0,
            lobe_depth: 0.5,
            ..single()
        })
        .generate(128, 128)
        .expect("generate failed");
        let cover =
            |m: &crate::generator::TextureMap| m.albedo.chunks(4).filter(|px| px[3] > 128).count();
        assert!(
            cover(&lobed) < cover(&entire),
            "a pinnatifid margin should remove blade coverage"
        );
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = FrondGenerator::new(FrondConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = FrondGenerator::new(FrondConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            FrondGenerator::new(FrondConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
