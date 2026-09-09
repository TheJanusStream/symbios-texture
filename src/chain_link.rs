//! Chain-link fence card generator.
//!
//! A woven diamond mesh: two wire families running at ±45°, rendered as
//! cylindrical profiles on a diagonal lattice.  Crossing parity lifts one
//! wire over the other (`weave_depth`), and rust accumulates at the
//! crossings where water sits.  The alpha channel is transparent everywhere
//! off the wires, so the card reads as a see-through fence.  Baked as a card
//! through the [`surface`](crate::surface) driver: clamp-to-edge normals over
//! a silhouette-dilated height field.
//!
//! Upload with `map_to_images_card`;
//! chain-link cards do not tile.

use noise::Perlin;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    sprite::{fbm2, lerp_color},
    surface::{SurfaceCell, SurfaceOptions, SurfaceSample, generate_surface_with},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`ChainLinkGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChainLinkConfig {
    /// PRNG seed for the rust pattern.
    pub seed: u32,
    /// Diamond cells across the card, rounded and clamped to `[2, 32]`.
    pub cell_count: f64,
    /// Wire radius in lattice units `[0.02, 0.2]` — fraction of a diamond
    /// half-diagonal.
    pub wire_radius: f64,
    /// Over/under weave relief at the crossings `[0, 1]`.
    pub weave_depth: f64,
    /// Rust accumulation at the crossings `[0, 1]`.
    pub rust_level: f64,
    /// Galvanised wire colour in linear RGB \[0, 1\].
    pub color_wire: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Optional ageing pass — wear on exposed edges, grime in the
    /// recesses, corrosion and run-off streaks.
    ///
    /// Defaults to disabled, so the surface is unchanged until a layer
    /// is turned up.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for ChainLinkConfig {
    fn default() -> Self {
        Self {
            seed: 83,
            cell_count: 8.0,
            wire_radius: 0.07,
            weave_depth: 0.6,
            rust_level: 0.2,
            color_wire: [0.62, 0.64, 0.66],
            color_rust: [0.45, 0.24, 0.10],
            weathering: WeatheringConfig::default(),
            normal_strength: 3.0,
        }
    }
}

/// Procedural chain-link fence card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct ChainLinkGenerator {
    config: ChainLinkConfig,
}

impl ChainLinkGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ChainLinkConfig) -> Self {
        Self { config }
    }
}

/// Per-generation sampler: the lattice constants and the rust noise.
struct ChainLinkCell<'a> {
    config: &'a ChainLinkConfig,
    perlin: &'a Perlin,
    /// Diamond cells across the card.
    k: f64,
    /// Wire radius in lattice units.
    wr: f64,
    /// Over/under weave relief.
    lift: f64,
    width: f64,
    height: f64,
}

/// The open mesh's roughness byte — `150`, which the hand-rolled loop wrote
/// directly.  Named as the byte, not a fraction, so the port is provably free
/// of visual change.
const OPEN_MESH_ROUGHNESS: f32 = 150.0 / 255.0;

impl SurfaceCell for ChainLinkCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        // Texel centres, as the hand-rolled loop sampled; the driver's corner
        // convention is for surfaces that tile, and a card does not.
        let u = (f64::from(x) + 0.5) / self.width;
        let v = (f64::from(y) + 0.5) / self.height;

        // Diagonal lattice coordinates: wires run along the integer lines of
        // each family.
        let a = (u + v) * self.k;
        let b = (u - v) * self.k;
        let da = (a - a.round()).abs();
        let db = (b - b.round()).abs();

        // Cylindrical wire profiles.
        let wr = self.wr;
        let profile = |d: f64| {
            if d < wr {
                (1.0 - (d / wr) * (d / wr)).sqrt()
            } else {
                0.0
            }
        };
        let pa = profile(da);
        let pb = profile(db);

        if pa <= 0.0 && pb <= 0.0 {
            // Open mesh: transparent, wire colour in RGB to avoid halos under
            // bilinear filtering.
            return SurfaceSample {
                height: 0.0,
                color: c.color_wire,
                roughness: OPEN_MESH_ROUGHNESS,
                metallic: 0.0,
                occlusion: 1.0,
                emissive: [0.0, 0.0, 0.0],
                alpha: 0.0,
            };
        }

        // Over/under weave: crossing parity lifts one family.
        let lift = self.lift;
        let over_a = ((a.round() + b.round()) as i64).rem_euclid(2) == 0;
        let ha = pa * 0.45 + 0.35 + if over_a { lift } else { -lift };
        let hb = pb * 0.45 + 0.35 + if over_a { -lift } else { lift };
        let (height, prof) = match (pa > 0.0, pb > 0.0) {
            (true, true) => {
                if ha >= hb {
                    (ha, pa)
                } else {
                    (hb, pb)
                }
            }
            (true, false) => (ha, pa),
            _ => (hb, pb),
        };

        // Rust pools where the wires cross.
        let cross = (pa * pb).clamp(0.0, 1.0);
        let rust_n = fbm2(self.perlin, u * 9.0, v * 9.0, 3);
        let rust = (cross * rust_n * 1.8 * c.rust_level.clamp(0.0, 1.0)).clamp(0.0, 1.0) as f32;

        let shade = (0.55 + 0.45 * prof) as f32;
        let base = lerp_color(c.color_wire, c.color_rust, rust);

        SurfaceSample {
            height: height.clamp(0.0, 1.0),
            color: [base[0] * shade, base[1] * shade, base[2] * shade],
            roughness: (0.35 + rust * 0.5).clamp(0.0, 1.0),
            metallic: (0.85 * (1.0 - rust)).clamp(0.0, 1.0),
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: 1.0,
        }
    }
}

impl ChainLinkGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        workspace: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;
        let perlin = Perlin::new(c.seed);
        let cell = ChainLinkCell {
            config: c,
            perlin: &perlin,
            k: c.cell_count.round().clamp(2.0, 32.0),
            wr: c.wire_radius.clamp(0.02, 0.2),
            lift: c.weave_depth.clamp(0.0, 1.0) * 0.18,
            width: f64::from(width),
            height: f64::from(height),
        };
        generate_surface_with(
            width,
            height,
            c.normal_strength,
            workspace,
            &cell,
            SurfaceOptions::default()
                .with_card(true)
                .with_weathering(&c.weathering),
        )
    }
}

impl TextureGenerator for ChainLinkGenerator {
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

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn mesh_is_mostly_open_with_opaque_wires() {
        let map = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let open = map.albedo.chunks(4).filter(|px| px[3] == 0).count();
        let wire = map.albedo.chunks(4).filter(|px| px[3] == 255).count();
        assert!(wire > 0, "wires must be opaque");
        assert!(open > wire, "the mesh should be mostly see-through");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
