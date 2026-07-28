//! Parquet flooring: short boards laid in a repeating figure.
//!
//! Distinct from [`plank`](crate::plank), which runs long boards in one
//! direction.  Parquet's character is that the boards *change direction*, so
//! the grain catches light differently block to block — and that is carried
//! almost entirely by the layout, not by the wood.
//!
//! Each layout maps a texel to the board covering it, giving the board's
//! origin and whether its grain runs across or along.  Grain then comes from
//! a [`stripe`] field rotated into the board's own frame, which is what keeps
//! it running with the board rather than across the whole floor.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{
        StripeParams, StripeProfile, ToroidalNoise, cell_hash, normalize, sample_grid_into, stripe,
    },
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered, lerp},
    weathering::WeatheringConfig,
};

/// How the boards are laid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParquetLayout {
    /// Boards laid in interlocking L-shaped pairs, each pair turned 90° from
    /// its neighbour — the classic zig-zag of a herringbone floor.
    #[default]
    Herringbone,
    /// Square blocks of parallel boards, each block turned 90° from the last,
    /// giving a woven chequerboard.
    Basket,
    /// Short boards in offset courses, like stretcher-bond brickwork.
    Brick,
}

/// Configures the appearance of a [`ParquetGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ParquetConfig {
    /// PRNG seed for the deterministic pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// How the boards are laid.
    pub layout: ParquetLayout,
    /// Board slots across the tile.  Rounded so the figure repeats exactly at
    /// the tile edge — for [`ParquetLayout::Herringbone`] that means rounding
    /// to a whole number of `2 × aspect` repeats.
    pub scale: f64,
    /// Board length as a multiple of its width.  Rounded for herringbone,
    /// whose interlock is defined on whole cells.
    pub aspect: f64,
    /// Width of the joint between boards, as a fraction of a board width.
    pub joint_width: f64,
    /// Depth of the joint in the height field.
    pub joint_depth: f64,
    /// Grain lines along each board.
    pub grain_lines: f64,
    /// How strongly the grain shows, in `[0, 1]`.
    pub grain_contrast: f32,
    /// How much the grain wanders from straight.
    pub grain_warp: f64,
    /// Spread of per-board tint variation in `[0, 1]`; boards are cut from
    /// different parts of the log.
    pub board_variance: f32,
    /// Light wood colour in linear RGB \[0, 1\].
    pub color_wood: [f32; 3],
    /// Dark grain colour in linear RGB \[0, 1\].
    pub color_grain: [f32; 3],
    /// Joint colour in linear RGB \[0, 1\].
    pub color_joint: [f32; 3],
    /// Sheen of the finish, as roughness in `[0, 1]`.
    pub gloss_roughness: f32,
    /// Optional ageing pass — worn edges, grime in the joints.
    #[serde(default)]
    pub weathering: WeatheringConfig,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            seed: 43,
            layout: ParquetLayout::Herringbone,
            scale: 8.0,
            aspect: 4.0,
            joint_width: 0.05,
            joint_depth: 0.5,
            grain_lines: 7.0,
            grain_contrast: 0.35,
            grain_warp: 0.22,
            board_variance: 0.13,
            color_wood: [0.36, 0.20, 0.09],
            color_grain: [0.19, 0.10, 0.04],
            color_joint: [0.07, 0.04, 0.02],
            gloss_roughness: 0.32,
            weathering: WeatheringConfig::default(),
            normal_strength: 1.6,
        }
    }
}

/// Which board covers a texel, and where the texel sits within it.
struct Board {
    /// Position along the board's length, in `[0, 1)`.
    along: f64,
    /// Position across the board's width, in `[0, 1)`.
    across: f64,
    /// Board identity, for per-board tint.
    id: (i64, i64),
}

/// Procedural parquet texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`ParquetConfig`].
pub struct ParquetGenerator {
    config: ParquetConfig,
    warp: ToroidalNoise<Fbm<Perlin>>,
}

impl ParquetGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ParquetConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(config.seed.wrapping_add(12)).set_octaves(2);
        let warp = ToroidalNoise::new(fbm, 6.0);
        Self { config, warp }
    }
}

/// Per-generation sampler: warp grid plus the board layout.
struct ParquetCell<'a> {
    config: &'a ParquetConfig,
    warp: &'a [f64],
    /// Board slots across the tile, rounded so the figure repeats.
    slots: f64,
    /// Board length in board-width cells (herringbone).
    length: i64,
    /// Board-width cells across the tile, a whole number of herringbone
    /// repeats so the figure closes at the tile edge.
    cells: f64,
    grain: StripeParams,
    width: usize,
}

impl ParquetCell<'_> {
    /// Resolve which board covers `(u, v)`.
    fn board_at(&self, u: f64, v: f64) -> Board {
        let n = self.slots;
        let aspect = self.config.aspect.max(1.0);

        match self.config.layout {
            ParquetLayout::Basket => {
                // Square blocks, alternating direction with block parity.
                let (bu, bv) = (u * n, v * n);
                let (bi, bj) = (bu.floor(), bv.floor());
                let (fu, fv) = (bu.fract(), bv.fract());
                let turned = ((bi + bj) as i64).rem_euclid(2) == 1;
                let (along, across) = if turned { (fv, fu) } else { (fu, fv) };
                Board {
                    along,
                    across,
                    id: (bi as i64, bj as i64),
                }
            }
            ParquetLayout::Brick => {
                // Offset courses: each row shifts half a board.
                let rows = n * aspect;
                let row = (v * rows).floor();
                let offset = if (row as i64).rem_euclid(2) == 0 {
                    0.0
                } else {
                    0.5
                };
                let along = (u * n + offset).fract();
                let across = (v * rows).fract();
                Board {
                    along,
                    across,
                    id: ((u * n + offset).floor() as i64, row as i64),
                }
            }
            ParquetLayout::Herringbone => {
                // Work in cells one board *width* square; a board is `length`
                // cells long.  Which way a cell's board runs is decided by
                // `(i − j) mod 2·length`: the first `length` keys form a
                // horizontal board, the rest a vertical one.
                //
                // That single test is the whole tiling.  Because the key
                // shifts by one along both axes, consecutive cells of a
                // horizontal run share a key band while the run above steps
                // sideways — which is precisely the interlocking zig-zag, and
                // why herringbone cannot be built from an axis-aligned grid of
                // blocks the way basket and brick can.
                let length = self.length;
                let cells = self.cells;
                let (su, sv) = (u * cells, v * cells);
                let (i, j) = (su.floor() as i64, sv.floor() as i64);
                let (fu, fv) = (su.fract(), sv.fract());

                let span = length * 2;
                let key = (i - j).rem_euclid(span);
                if key < length {
                    // Horizontal board: the run starts `key` cells back.
                    Board {
                        along: (key as f64 + fu) / length as f64,
                        across: fv,
                        id: (i - key, j),
                    }
                } else {
                    // Vertical board: the key counts down as we climb, so the
                    // foot of the run is `span − 1 − key` cells below.
                    let from_foot = span - 1 - key;
                    Board {
                        along: (from_foot as f64 + fv) / length as f64,
                        across: fu,
                        id: (i, j - from_foot),
                    }
                }
            }
        }
    }
}

impl SurfaceCell for ParquetCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let warp =
            (normalize(self.warp[y as usize * self.width + x as usize]) - 0.5) * c.grain_warp;

        let board = self.board_at(u, v);

        // Joint: a border around every board, measured in board-local units so
        // it stays even whichever way the board is turned.
        let joint_w = c.joint_width.clamp(0.0, 0.4);
        let edge_along = board.along.min(1.0 - board.along);
        let edge_across = board.across.min(1.0 - board.across);
        let joint = 1.0 - smoothstep(joint_w * 0.5, joint_w, edge_along.min(edge_across).max(0.0));

        // Grain runs along the board, in the board's own frame.
        let grain = stripe(board.across, board.along, self.grain, warp);

        let tint = (cell_hash(board.id.0, board.id.1, c.seed.wrapping_add(31)) - 0.5) as f32
            * 2.0
            * c.board_variance;
        let grain_t = grain as f32 * c.grain_contrast.clamp(0.0, 1.0);

        let wood = [
            (lerp(c.color_wood[0], c.color_grain[0], grain_t) + tint).clamp(0.0, 1.0),
            (lerp(c.color_wood[1], c.color_grain[1], grain_t) + tint * 0.8).clamp(0.0, 1.0),
            (lerp(c.color_wood[2], c.color_grain[2], grain_t) + tint * 0.6).clamp(0.0, 1.0),
        ];

        let j = joint as f32;
        let color = [
            lerp(wood[0], c.color_joint[0], j),
            lerp(wood[1], c.color_joint[1], j),
            lerp(wood[2], c.color_joint[2], j),
        ];

        SurfaceSample {
            height: (1.0 - joint * c.joint_depth) + grain * 0.05,
            color,
            // Finish is glossy on the board, matt in the joint.
            roughness: lerp(c.gloss_roughness, 0.9, j).clamp(0.0, 1.0),
            metallic: 0.0,
            occlusion: lerp(1.0, 0.5, j),
            emissive: [0.0, 0.0, 0.0],
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

impl ParquetGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut warp = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.warp, width, height, &mut warp);

        // Herringbone repeats every `2 · length` cells, so the cell count has
        // to be a whole number of those or the figure breaks at the seam.
        let length = c.aspect.round().clamp(2.0, 12.0) as i64;
        let span = (length * 2) as f64;
        let repeats = (c.scale.round().clamp(1.0, 64.0) / span).round().max(1.0);

        let cell = ParquetCell {
            config: c,
            warp: &warp,
            slots: c.scale.round().clamp(1.0, 64.0),
            length,
            cells: repeats * span,
            // Grain runs along the board's length, so the cycles sit on the
            // board-local V axis.
            grain: StripeParams::new(0, c.grain_lines.round().clamp(1.0, 64.0) as i32)
                .with_profile(StripeProfile::Sine)
                .with_sharpness(0.5),
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
            ws.return_grid(warp);
        }
        result
    }
}

impl TextureGenerator for ParquetGenerator {
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

    /// sRGB red-channel value below which a texel counts as joint rather than
    /// board.  Sits between the encoded joint colour (~75) and the wood
    /// (~137); neither is near 0, so a naive "is it dark" test finds nothing.
    const JOINT_LUMA: u8 = 100;

    fn bake(config: ParquetConfig) -> TextureMap {
        ParquetGenerator::new(config)
            .generate(128, 128)
            .expect("generate")
    }

    #[test]
    fn produces_correct_buffer_sizes() {
        let map = bake(ParquetConfig::default());
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
        assert!(map.emissive.is_none());
    }

    #[test]
    fn deterministic_for_same_seed() {
        assert_eq!(
            bake(ParquetConfig::default()).albedo,
            bake(ParquetConfig::default()).albedo
        );
        assert_ne!(
            bake(ParquetConfig::default()).albedo,
            bake(ParquetConfig {
                seed: 1234,
                ..Default::default()
            })
            .albedo
        );
    }

    /// The layouts are the whole point — each must lay a different floor.
    #[test]
    fn layouts_are_distinct() {
        let bake_layout = |layout| {
            bake(ParquetConfig {
                layout,
                ..Default::default()
            })
            .albedo
        };
        let layouts = [
            ParquetLayout::Herringbone,
            ParquetLayout::Basket,
            ParquetLayout::Brick,
        ];
        for (a, b) in layouts.iter().zip(layouts.iter().skip(1)) {
            assert_ne!(bake_layout(*a), bake_layout(*b), "{a:?} matched {b:?}");
        }
    }

    /// Joints must be visible but a minority — this is a floor of boards, not
    /// a grid of grout.
    #[test]
    fn joints_are_a_minority() {
        for layout in [
            ParquetLayout::Herringbone,
            ParquetLayout::Basket,
            ParquetLayout::Brick,
        ] {
            let map = bake(ParquetConfig {
                layout,
                ..Default::default()
            });
            let dark = map.albedo.chunks(4).filter(|px| px[0] < JOINT_LUMA).count() as f64
                / (128 * 128) as f64;
            assert!(
                (0.01..0.5).contains(&dark),
                "{layout:?} joints cover {dark:.3} of the floor"
            );
        }
    }

    /// Widening the joint must expose more of it.
    #[test]
    fn joint_width_controls_coverage() {
        let coverage = |joint_width| {
            bake(ParquetConfig {
                joint_width,
                ..Default::default()
            })
            .albedo
            .chunks(4)
            .filter(|px| px[0] < JOINT_LUMA)
            .count()
        };
        assert!(
            coverage(0.12) > coverage(0.03),
            "wider joints did not show more"
        );
    }

    /// Grain has to reach the surface, or the boards read as flat paint.
    #[test]
    fn grain_shows_on_the_boards() {
        let grained = bake(ParquetConfig::default());
        let bare = bake(ParquetConfig {
            grain_contrast: 0.0,
            ..Default::default()
        });
        assert_ne!(grained.albedo, bare.albedo, "grain had no effect");
    }

    #[test]
    fn extreme_configs_stay_finite() {
        let map = bake(ParquetConfig {
            scale: 0.0,
            aspect: 0.0,
            joint_width: 9.0,
            grain_lines: 0.0,
            grain_contrast: 5.0,
            board_variance: -2.0,
            ..Default::default()
        });
        assert_eq!(map.albedo.len(), 128 * 128 * 4);
    }
}
