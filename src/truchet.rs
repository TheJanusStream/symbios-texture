//! Truchet tiles: hashed quarter-arcs that interlock into continuous curves.
//!
//! Two quarter-arcs joining opposite corners of a square tile, dropped in one
//! of two orientations per cell.  The arcs meet at every tile edge whichever
//! way the neighbour fell, so a grid of random orientations reads as one
//! continuous, deliberately-routed network — circuit traces, pipework, celtic
//! knotwork — from nothing but a per-cell coin flip.
//!
//! The traces can glow.  Emissive is requested at bake time through
//! [`crate::surface::SurfaceOptions`] only when
//! [`emissive_intensity`](TruchetConfig::emissive_intensity) is above zero, so
//! an unlit panel does not carry an emissive buffer it never uses.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, cell_hash, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceOptions, SurfaceSample, generate_surface_with, lerp},
    weathering::WeatheringConfig,
};

/// Configures the appearance of a [`TruchetGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TruchetConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Tiles across the panel.  Rounded to an integer so the network closes
    /// at the tile edge.
    pub scale: f64,
    /// Trace width as a fraction of a tile.
    pub trace_width: f64,
    /// How far the trace stands proud of (or sinks into) the panel.
    pub trace_relief: f64,
    /// Fraction of cells that carry a trace at all, in `[0, 1]`.  Below `1`
    /// the network breaks into runs and stubs rather than covering the panel
    /// evenly.
    pub density: f64,
    /// Panel colour in linear RGB \[0, 1\].
    pub color_panel: [f32; 3],
    /// Trace colour in linear RGB \[0, 1\].
    pub color_trace: [f32; 3],
    /// Glow colour in linear RGB \[0, 1\].
    pub color_glow: [f32; 3],
    /// Glow strength.  At `0` the panel is unlit and the bake skips the
    /// emissive channel entirely.
    pub emissive_intensity: f32,
    /// Roughness of the panel between traces.
    pub panel_roughness: f32,
    /// Roughness of the trace itself — plated conductor is smoother.
    pub trace_roughness: f32,
    /// Metallic value of the trace.
    pub trace_metallic: f32,
    /// Frequency of the fine mottle across the panel.
    pub mottle_scale: f64,
    /// Optional ageing pass — grime between traces, corrosion at the joints.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for TruchetConfig {
    fn default() -> Self {
        Self {
            seed: 47,
            scale: 6.0,
            trace_width: 0.09,
            trace_relief: 0.6,
            density: 0.85,
            color_panel: [0.035, 0.055, 0.050],
            color_trace: [0.16, 0.42, 0.34],
            color_glow: [0.10, 0.85, 0.60],
            emissive_intensity: 1.0,
            panel_roughness: 0.72,
            trace_roughness: 0.30,
            trace_metallic: 0.65,
            mottle_scale: 18.0,
            weathering: WeatheringConfig::default(),
            normal_strength: 1.4,
        }
    }
}

/// Procedural Truchet-tile panel generator.
///
/// Drives [`TextureGenerator::generate`] using a [`TruchetConfig`].
pub struct TruchetGenerator {
    config: TruchetConfig,
    mottle: ToroidalNoise<Fbm<Perlin>>,
}

impl TruchetGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: TruchetConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(14)).set_octaves(3);
        let mottle = ToroidalNoise::new(fbm, config.mottle_scale.max(0.1));
        Self { config, mottle }
    }
}

/// Per-generation sampler: mottle grid plus the tile lattice.
struct TruchetCell<'a> {
    config: &'a TruchetConfig,
    mottle: &'a [f64],
    tiles: f64,
    width: usize,
}

impl TruchetCell<'_> {
    /// Distance from `(u, v)` to the nearest trace centre-line, in tile
    /// units.  Returns `None` where the cell carries no trace.
    fn trace_distance(&self, u: f64, v: f64) -> Option<f64> {
        let n = self.tiles;
        let (su, sv) = (u * n, v * n);
        let (ci, cj) = (su.div_euclid(1.0) as i64, sv.div_euclid(1.0) as i64);
        let (fu, fv) = (su.rem_euclid(1.0), sv.rem_euclid(1.0));

        // Thin the network so it reads as routed runs rather than a full mesh.
        if cell_hash(ci, cj, self.config.seed.wrapping_add(5)) > self.config.density {
            return None;
        }

        // Two quarter-arcs of radius ½, centred on opposite corners.  The coin
        // flip picks which diagonal pair, and either choice still meets every
        // edge at its midpoint — which is exactly why neighbours always
        // connect.
        let flipped = cell_hash(ci, cj, self.config.seed) < 0.5;
        let (first, second) = if flipped {
            ((0.0, 0.0), (1.0, 1.0))
        } else {
            ((1.0, 0.0), (0.0, 1.0))
        };

        let arc = |centre: (f64, f64)| {
            let d = ((fu - centre.0).powi(2) + (fv - centre.1).powi(2)).sqrt();
            (d - 0.5).abs()
        };
        Some(arc(first).min(arc(second)))
    }
}

impl SurfaceCell for TruchetCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let mottle = normalize(self.mottle[y as usize * self.width + x as usize]);

        let half = (c.trace_width.max(0.0) * 0.5).min(0.45);
        let trace = match self.trace_distance(u, v) {
            Some(distance) => 1.0 - smoothstep(half * 0.6, half, distance),
            None => 0.0,
        };

        let t = trace as f32;
        let panel_shade = (mottle as f32 - 0.5) * 0.06;
        let color = [
            (lerp(c.color_panel[0], c.color_trace[0], t) + panel_shade).clamp(0.0, 1.0),
            (lerp(c.color_panel[1], c.color_trace[1], t) + panel_shade).clamp(0.0, 1.0),
            (lerp(c.color_panel[2], c.color_trace[2], t) + panel_shade).clamp(0.0, 1.0),
        ];

        let glow = c.emissive_intensity.max(0.0) * t;
        let emissive = [
            (c.color_glow[0] * glow).clamp(0.0, 1.0),
            (c.color_glow[1] * glow).clamp(0.0, 1.0),
            (c.color_glow[2] * glow).clamp(0.0, 1.0),
        ];

        SurfaceSample {
            height: trace * c.trace_relief + (mottle - 0.5) * 0.05,
            color,
            roughness: lerp(c.panel_roughness, c.trace_roughness, t).clamp(0.0, 1.0),
            metallic: lerp(0.0, c.trace_metallic, t).clamp(0.0, 1.0),
            occlusion: 1.0,
            emissive,
        }
    }
}

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl TruchetGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut mottle = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.mottle, width, height, &mut mottle);

        let cell = TruchetCell {
            config: c,
            mottle: &mottle,
            tiles: c.scale.round().clamp(1.0, 64.0),
            width: width as usize,
        };
        // Only carry an emissive buffer when the panel actually glows.
        let result = generate_surface_with(
            width,
            height,
            c.normal_strength,
            ws.as_deref_mut(),
            &cell,
            SurfaceOptions::default()
                .with_emissive(c.emissive_intensity > 0.0)
                .with_weathering(&c.weathering),
        );

        if let Some(ws) = ws {
            ws.return_grid(mottle);
        }
        result
    }
}

impl TextureGenerator for TruchetGenerator {
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

    fn bake(config: TruchetConfig) -> TextureMap {
        TruchetGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(TruchetConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert_eq!(map.normal.len(), 128 * 128 * 4);
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(TruchetConfig::default()).albedo,
            bake(TruchetConfig::default()).albedo
        );
        assert_ne!(
            bake(TruchetConfig::default()).albedo,
            bake(TruchetConfig {
                seed: 2024,
                ..Default::default()
            })
            .albedo
        );
    }

    /// The emissive buffer is paid for only when the panel glows — the whole
    /// reason this generator picks its passes at runtime.
    #[test]
    fn emissive_is_carried_only_when_lit() {
        assert!(
            bake(TruchetConfig::default()).emissive.is_some(),
            "a lit panel dropped its glow"
        );
        assert!(
            bake(TruchetConfig {
                emissive_intensity: 0.0,
                ..Default::default()
            })
            .emissive
            .is_none(),
            "an unlit panel still carried an emissive buffer"
        );
    }

    /// Glow must land on the traces and nowhere else.
    #[test]
    fn glow_follows_the_traces() {
        let map = bake(TruchetConfig::default());
        let emissive = map.emissive.expect("lit");
        let lit = emissive.chunks(4).filter(|px| px[1] > 40).count() as f64 / (128 * 128) as f64;
        assert!(
            (0.02..0.6).contains(&lit),
            "glow covers {lit:.3} of the panel — not a trace network"
        );
    }

    /// Arcs meet every edge at its midpoint, so a trace crossing a tile
    /// boundary must find a trace waiting on the other side.
    ///
    /// This is the property that makes a grid of coin flips read as a routed
    /// network rather than as confetti.
    #[test]
    fn traces_connect_across_tile_edges() {
        let tiles = 6.0;
        let cell = TruchetCell {
            config: &TruchetConfig {
                density: 1.0,
                ..Default::default()
            },
            mottle: &[0.0],
            tiles,
            width: 1,
        };

        // Walk the vertical edges at their midpoints: both sides must be on a
        // trace centre-line, i.e. at distance ~0.
        for i in 1..tiles as i64 {
            let u = i as f64 / tiles;
            let v = (0.5 + (i as f64 - 1.0)) / tiles;
            let left = cell.trace_distance(u - 1e-6, v).expect("dense");
            let right = cell.trace_distance(u + 1e-6, v).expect("dense");
            assert!(
                left < 1e-3 && right < 1e-3,
                "trace did not meet the edge at u={u}: {left} vs {right}"
            );
        }
    }

    /// Density thins the network out.
    #[test]
    fn density_thins_the_network() {
        let covered = |density| {
            bake(TruchetConfig {
                density,
                ..Default::default()
            })
            .albedo
            .chunks(4)
            .filter(|px| px[1] > 90)
            .count()
        };
        assert!(
            covered(1.0) > covered(0.35),
            "lowering density did not remove traces"
        );
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(TruchetConfig {
            scale: 0.0,
            trace_width: 9.0,
            density: -1.0,
            emissive_intensity: -5.0,
            mottle_scale: 0.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
