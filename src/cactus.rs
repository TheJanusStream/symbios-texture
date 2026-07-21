//! Cactus-skin surface generator.
//!
//! A tileable succulent hide: vertical accordion **ribs** running up the
//! stem, a regular grid of **areoles** (the felted cushions from which spines
//! grow) seated on the rib crests, and short pale **spines** radiating from
//! each areole.  The waxy blue-green skin reads as a barrel/columnar cactus
//! when wrapped around an L-system cactus stem.
//!
//! Tiling: the ribs are periodic in `U` (integer `rib_count`), the areoles
//! are a regular `rib_count × areole_rows` lattice, and the skin mottle uses
//! integer-frequency sinusoids — so albedo and the toroidal normal map wrap
//! seamlessly around the stem circumference and along its height.

use std::f64::consts::{PI, TAU};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`CactusSkinGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CactusSkinConfig {
    /// PRNG seed for the skin mottle phase.
    pub seed: u32,
    /// Number of vertical ribs around the tile (periodic in `U`; clamped to
    /// `3..=40` so the wrap stays seamless).
    pub rib_count: usize,
    /// Rows of areoles up the tile (periodic in `V`; clamped to `2..=40`).
    pub areole_rows: usize,
    /// Rib relief `[0, 1]` — how deep the accordion pleats read in the
    /// height/normal map.
    pub rib_depth: f64,
    /// Rib crest sharpness `[0.3, 3]`.  Higher → sharper, narrower ridge
    /// crests with broader valley floors.
    pub rib_sharpness: f64,
    /// Skin colour on the rib ridges in linear RGB \[0, 1\].
    pub color_skin: [f32; 3],
    /// Skin colour in the shaded rib valleys in linear RGB \[0, 1\].
    pub color_valley: [f32; 3],
    /// Areole felt colour in linear RGB \[0, 1\].
    pub color_areole: [f32; 3],
    /// Spine colour in linear RGB \[0, 1\].
    pub color_spine: [f32; 3],
    /// Areole felt radius in UV units `[0.005, 0.08]`.
    pub areole_size: f64,
    /// Spine reach as a multiple of `areole_size` `[1, 6]`.
    pub spine_reach: f64,
    /// Spines radiating from each areole (clamped to `0..=24`).
    pub spine_count: usize,
    /// Skin gloss `[0, 1]` — higher lowers roughness for a waxier sheen.
    pub waxiness: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for CactusSkinConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            rib_count: 8,
            areole_rows: 9,
            rib_depth: 0.85,
            rib_sharpness: 0.85,
            color_skin: [0.22, 0.42, 0.27],
            color_valley: [0.07, 0.17, 0.11],
            color_areole: [0.55, 0.50, 0.40],
            color_spine: [0.86, 0.82, 0.66],
            areole_size: 0.022,
            spine_reach: 3.2,
            spine_count: 8,
            waxiness: 0.55,
            normal_strength: 1.4,
        }
    }
}

/// A baked cactus-skin sampler.
pub struct CactusSkinCell {
    config: CactusSkinConfig,
    ribs: f64,
    rows: f64,
    /// Deterministic mottle phase offset from the seed.
    phase: f64,
}

impl CactusSkinCell {
    fn new(config: &CactusSkinConfig) -> Self {
        let ribs = config.rib_count.clamp(3, 40) as f64;
        let rows = config.areole_rows.clamp(2, 40) as f64;
        // Bounded, deterministic phase so different seeds decorrelate the
        // mottle without breaking the integer-frequency tiling.
        let phase = (config.seed % 360) as f64 * PI / 180.0;
        Self {
            config: config.clone(),
            ribs,
            rows,
            phase,
        }
    }
}

impl SurfaceCell for CactusSkinCell {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = &self.config;

        // --- Ribs: accordion pleats periodic in U ---
        let rib_c = (u * self.ribs * TAU).cos();
        let ridge = (0.5 + 0.5 * rib_c).powf(c.rib_sharpness.clamp(0.3, 3.0));
        let mut height = ridge * c.rib_depth.clamp(0.0, 1.0);

        // Subtle waxy mottle — integer frequencies keep it toroidal.
        let mottle =
            ((u * self.ribs * TAU + self.phase).sin() * (v * self.rows * 2.0 * TAU).sin()) * 0.03;

        let mut color = [
            lerp(c.color_valley[0], c.color_skin[0], ridge as f32) + mottle as f32,
            lerp(c.color_valley[1], c.color_skin[1], ridge as f32) + mottle as f32,
            lerp(c.color_valley[2], c.color_skin[2], ridge as f32) + mottle as f32,
        ];

        // Waxy skin: glossier on the lit ridges, rougher in the valleys.
        let wax = c.waxiness.clamp(0.0, 1.0) as f32;
        let mut roughness = lerp(0.75, 0.28, ridge as f32 * wax);

        // --- Areole lattice: one cushion per rib crest per row ---
        // Signed UV distance to the nearest lattice point (crest × row).
        let fu = u * self.ribs;
        let du = (fu - fu.round()) / self.ribs;
        let fv = v * self.rows - 0.5;
        let dv = (fv - fv.round()) / self.rows;
        let d = (du * du + dv * dv).sqrt();

        let a_size = c.areole_size.clamp(0.005, 0.08);
        if d < a_size {
            // Felted cushion: a raised, matte, pale-brown patch.
            let t = (1.0 - d / a_size) as f32;
            color = [
                lerp(color[0], c.color_areole[0], t),
                lerp(color[1], c.color_areole[1], t),
                lerp(color[2], c.color_areole[2], t),
            ];
            height += (0.25 * (1.0 - (d / a_size)).max(0.0)) * c.rib_depth.max(0.2);
            roughness = lerp(roughness, 0.85, t);
        } else {
            // --- Spines: pale radial streaks reaching past the areole ---
            let reach = a_size * c.spine_reach.clamp(1.0, 6.0);
            let n = c.spine_count.clamp(0, 24);
            if n > 0 && d < reach {
                let ang = dv.atan2(du); // [-π, π]
                let spoke = (ang / TAU + 0.5) * n as f64; // [0, n)
                let frac = (spoke - spoke.floor() - 0.5).abs() * 2.0; // 0 at spoke centre
                // Narrow streak that fades toward its tip.
                let along = ((reach - d) / (reach - a_size)).clamp(0.0, 1.0);
                let on = ((0.12 - frac) / 0.12).clamp(0.0, 1.0) * along;
                if on > 0.0 {
                    let t = on as f32;
                    color = [
                        lerp(color[0], c.color_spine[0], t),
                        lerp(color[1], c.color_spine[1], t),
                        lerp(color[2], c.color_spine[2], t),
                    ];
                    height += 0.15 * on * c.rib_depth.max(0.2);
                    roughness = lerp(roughness, 0.4, t);
                }
            }
        }

        SurfaceSample {
            height,
            color: [
                color[0].clamp(0.0, 1.0),
                color[1].clamp(0.0, 1.0),
                color[2].clamp(0.0, 1.0),
            ],
            roughness: roughness.clamp(0.0, 1.0),
            metallic: 0.0,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

/// Procedural cactus-skin surface generator.
///
/// See the [module documentation](self) for the visual model.
pub struct CactusSkinGenerator {
    config: CactusSkinConfig,
}

impl CactusSkinGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: CactusSkinConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for CactusSkinGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let cell = CactusSkinCell::new(&self.config);
        generate_surface(width, height, self.config.normal_strength, None, &cell)
    }

    fn generate_with_workspace(
        &self,
        width: u32,
        height: u32,
        workspace: &mut Workspace,
    ) -> Result<TextureMap, TextureError> {
        let cell = CactusSkinCell::new(&self.config);
        generate_surface(
            width,
            height,
            self.config.normal_strength,
            Some(workspace),
            &cell,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = CactusSkinGenerator::new(CactusSkinConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
        // Opaque surface.
        assert!(map.albedo.chunks(4).all(|px| px[3] == 255));
    }

    #[test]
    fn tiles_horizontally() {
        // Integer rib_count → the left and right edges of the tile match, so
        // the skin wraps a cylinder without a visible seam.
        let map = CactusSkinGenerator::new(CactusSkinConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        for y in 0..128usize {
            let left = &map.albedo[(y * 128) * 4..(y * 128) * 4 + 3];
            let right = &map.albedo[(y * 128 + 127) * 4..(y * 128 + 127) * 4 + 3];
            // Adjacent-across-the-seam columns (x=0 and x=127) are one texel
            // apart on the torus, so they are close, not identical.
            for ch in 0..3 {
                assert!(
                    (left[ch] as i32 - right[ch] as i32).abs() < 60,
                    "seam discontinuity at y={y} channel {ch}"
                );
            }
        }
    }

    #[test]
    fn ribs_modulate_brightness() {
        // Across one row there should be both bright ridges and dark valleys.
        let map = CactusSkinGenerator::new(CactusSkinConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let row: Vec<u8> = (0..128)
            .map(|x| map.albedo[(64 * 128 + x) * 4 + 1])
            .collect();
        let min = *row.iter().min().unwrap();
        let max = *row.iter().max().unwrap();
        assert!(max as i32 - min as i32 > 30, "ribs should vary brightness");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = CactusSkinGenerator::new(CactusSkinConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = CactusSkinGenerator::new(CactusSkinConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            CactusSkinGenerator::new(CactusSkinConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
