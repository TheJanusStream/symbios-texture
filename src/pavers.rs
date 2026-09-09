//! Pavers / tiles texture generator.
//!
//! The algorithm:
//! 1. Classify each pixel as **stone** or **grout** using a grid SDF.
//!    - `Square`: axis-aligned rectangular cells with a rounded-box SDF.
//!    - `Hexagonal`: flat-top hex grid via axial cube-rounding and IQ's hex SDF.
//! 2. Hash each cell's integer ID → per-paver colour variance.
//! 3. Overlay a toroidal FBM for surface micro-detail and bump height.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface_weathered},
    weathering::WeatheringConfig,
};

/// Layout of individual paver stones.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PaversLayout {
    /// Rectangular stones arranged in a regular grid.
    Square,
    /// Flat-top hexagonal stones.
    Hexagonal,
}

/// Configures the appearance of a [`PaversGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaversConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Grid density — roughly the number of pavers across the tile.
    pub scale: f64,
    /// Width-to-height ratio for `Square` stones (ignored for `Hexagonal`).
    pub aspect_ratio: f64,
    /// Grout gap as a fraction of stone size \[0, 0.4\].
    pub grout_width: f64,
    /// Corner bevel radius as a fraction of grout half-width \[0, 1\].
    pub bevel: f64,
    /// Per-paver colour jitter \[0, 1\].
    pub cell_variance: f64,
    /// Surface FBM micro-detail amplitude \[0, 1\].
    pub roughness: f64,
    /// Paving stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Grout / joint colour in linear RGB \[0, 1\].
    pub color_grout: [f32; 3],
    /// Stone layout pattern.
    pub layout: PaversLayout,
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

impl Default for PaversConfig {
    fn default() -> Self {
        Self {
            seed: 23,
            scale: 5.0,
            aspect_ratio: 1.0,
            grout_width: 0.08,
            bevel: 0.5,
            cell_variance: 0.10,
            roughness: 0.30,
            color_stone: [0.48, 0.44, 0.40],
            color_grout: [0.28, 0.27, 0.26],
            layout: PaversLayout::Square,
            weathering: WeatheringConfig::default(),
            normal_strength: 3.5,
        }
    }
}

/// Procedural pavers / tiles texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`PaversConfig`].  Construct
/// via [`PaversGenerator::new`] and call `generate` directly, or spawn a
/// `PendingTexture::pavers` task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct PaversGenerator {
    config: PaversConfig,
    surf_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl PaversGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: PaversConfig) -> Self {
        let fbm: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm, config.scale * 2.0);

        Self { config, surf_noise }
    }
}

/// Per-generation sampler: surface grid + derived SDF layout constants.
struct PaversCell<'a> {
    config: &'a PaversConfig,
    surf_grid: &'a [f64],
    bevel_r: f64,
    /// Half the grout gap, as a fraction of the cell pitch.
    grout_half: f64,
    /// Inner half-extents for the stone SDF (before bevel).
    hx: f64,
    hy: f64,
    /// Integer column / row counts so the grid tiles exactly.
    cols: f64,
    rows: f64,
    width: usize,
}

impl SurfaceCell for PaversCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;
        let raw_surf = normalize(self.surf_grid[idx]);

        let (sdf_val, cell_id_u, cell_id_v) = match c.layout {
            PaversLayout::Square => {
                square_cell(u, v, self.cols, self.rows, self.hx, self.hy, self.bevel_r)
            }
            PaversLayout::Hexagonal => hex_cell(u, v, c.scale, self.grout_half, self.bevel_r),
        };

        let (h_val, color) = if sdf_val < 0.0 {
            // Inside stone.
            let edge_t = ((-sdf_val) / (self.bevel_r + 0.005)).clamp(0.0, 1.0);
            let noise_bump = (raw_surf - 0.5) * c.roughness * 0.4;
            let h_val = (edge_t + noise_bump * edge_t).clamp(0.0, 1.0);

            let cv = cell_hash(cell_id_u, cell_id_v, c.seed);
            let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
            let color = [
                (c.color_stone[0] + jitter as f32).clamp(0.0, 1.0),
                (c.color_stone[1] + jitter as f32 * 0.8).clamp(0.0, 1.0),
                (c.color_stone[2] + jitter as f32 * 0.6).clamp(0.0, 1.0),
            ];
            (h_val, color)
        } else {
            // Grout joint.
            (raw_surf * c.roughness * 0.04, c.color_grout)
        };

        let rough_val = if sdf_val < 0.0 {
            0.70 + raw_surf as f32 * 0.20
        } else {
            0.92
        };

        SurfaceSample::matte(h_val, color, rough_val)
    }
}

impl PaversGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Surface micro-detail FBM.
        let mut surf_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.surf_noise, width, height, &mut surf_grid);

        let grout_half = (c.grout_width * 0.5).clamp(0.0, 0.45);
        let bevel_r = (c.bevel * grout_half).max(0.0);
        let cell = PaversCell {
            config: c,
            surf_grid: &surf_grid,
            bevel_r,
            grout_half,
            hx: (0.5 - grout_half - bevel_r).max(0.0),
            hy: (0.5 - grout_half - bevel_r).max(0.0),
            // At least one column, for the same reason as `brick`: a small
            // enough `aspect_ratio` would round the count to zero and flatten
            // every row into a single constant sample.
            cols: (c.scale * c.aspect_ratio).round().max(1.0),
            rows: c.scale.round(),
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
            ws.return_grid(surf_grid);
        }
        result
    }
}

impl TextureGenerator for PaversGenerator {
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

// --- Square grid ------------------------------------------------------------

/// Returns `(sdf, cell_id_u, cell_id_v)` for a square-grid paver.
///
/// `sdf < 0` means inside the stone; `sdf >= 0` means in the grout.
fn square_cell(
    u: f64,
    v: f64,
    cols: f64,
    rows: f64,
    hx: f64,
    hy: f64,
    bevel_r: f64,
) -> (f64, i64, i64) {
    let u_scaled = u * cols;
    let v_scaled = v * rows;
    let cell_u = u_scaled.floor() as i64;
    let cell_v = v_scaled.floor() as i64;
    let cx = u_scaled.fract() - 0.5;
    let cy = v_scaled.fract() - 0.5;

    // Rounded-box SDF: negative inside, positive outside.
    let dx = cx.abs() - hx;
    let dy = cy.abs() - hy;
    let sdf = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0) - bevel_r;

    (sdf, cell_u, cell_v)
}

// --- Hexagonal grid ---------------------------------------------------------

/// Returns `(sdf, cell_id_q, cell_id_r)` for a flat-top hexagonal paver.
///
/// Uses axial cube-rounding to find the nearest hex center, then IQ's hex SDF.
/// `sdf < 0` inside stone, `sdf >= 0` in grout.  The returned pair is the
/// cell's *canonical* id on the torus, not the raw axial pair — see
/// "Cell identity" below.
///
/// # Tiling
/// A flat-top hex lattice of circumradius `hex_r` steps `1.5 * hex_r` across
/// and `sqrt(3) * hex_r` down.  Fixing `hex_r = 1 / (rows * sqrt(3))` puts
/// exactly `rows` rows in `[0, 1]`, so V tiles by construction.  U needs an
/// integer column count, and it has to be an **even** one: stepping a whole
/// tile sideways maps `(q, r)` to `(q + cols, r - cols / 2)`, which is a
/// lattice vector only when `cols` is even.  So the natural float count
/// `rows * sqrt(3) / 1.5` is rounded to the nearest even integer and `u` is
/// scaled by it directly.  That leaves the hexes slightly non-regular — at
/// worst 15.5% wide (at `scale = 3`), 13.4% narrow (at `scale = 2`, `4`,
/// `6`) — which is imperceptible next to the seam it buys.
///
/// # Cell identity
/// Raw `(q, r)` is not a cell's identity on a torus: `u + 1` shifts it by
/// `(cols, -cols / 2)` and `v + 1` by `(0, rows)`.  Reducing by both periods
/// gives one representative per cell, so a paver cut in half by the tile's
/// own edge hashes once and draws one colour instead of two.
fn hex_cell(u: f64, v: f64, scale: f64, grout_half: f64, bevel_r: f64) -> (f64, i64, i64) {
    const SQRT3: f64 = 1.732_050_807_568_877_3;

    // Vertical tiling requires an integer number of rows; round scale.
    let rows = scale.round().max(1.0);
    // Circumradius so that `rows` hex rows fit exactly across [0, 1].
    let hex_r = 1.0 / (rows * SQRT3);
    // Column count: the natural float value is `1 / (1.5 * hex_r)`, rounded
    // to the nearest even integer for the reason in the tiling note above.
    let cols = ((rows * SQRT3 / 1.5) * 0.5).round().max(1.0) * 2.0;

    // Fractional axial coordinates (flat-top convention), written in terms of
    // the row and column counts rather than through `hex_r`.  A whole-tile
    // step then moves them by exactly `cols` and `-cols / 2`, both integers,
    // which is what makes the seam identities hold bit for bit.
    let qf = u * cols;
    let rf = v * rows - 0.5 * qf;
    let sf = -qf - rf;

    // Cube-round to the nearest hex center and keep the offset within the cell.
    let (q, r, _s) = cube_round(qf, rf, sf);
    let (dq, dr) = (qf - q as f64, rf - r as f64);

    // That offset in stretched UV space, where the SDF is evaluated (the hexes
    // are slightly non-regular there, per the tiling note).
    let dx = 1.5 * hex_r * dq;
    let dy = SQRT3 * hex_r * (dr + 0.5 * dq);

    // The cell's own apothem is half the row pitch; `grout_half` and
    // `bevel_r` are fractions of that pitch, exactly as `square_cell`
    // takes them against a pitch of one.
    let pitch = SQRT3 * hex_r;
    let bevel = bevel_r * pitch;
    let stone = (pitch * (0.5 - grout_half) - bevel).max(0.0);
    let sdf = hex_sdf(dx, dy, stone) - bevel;

    // Canonical cell id: undo `n` whole-tile steps in U, then reduce V.
    let (cols_i, rows_i) = (cols as i64, rows as i64);
    let n = q.div_euclid(cols_i);
    let cell_q = q.rem_euclid(cols_i);
    let cell_r = (r + n * (cols_i / 2)).rem_euclid(rows_i);

    (sdf, cell_q, cell_r)
}

/// Cube-coordinate rounding (standard hex-grid algorithm).
#[inline]
fn cube_round(qf: f64, rf: f64, sf: f64) -> (i64, i64, i64) {
    let (rq, rr, rs) = (qf.round() as i64, rf.round() as i64, sf.round() as i64);
    let (dq, dr, ds) = (
        (rq as f64 - qf).abs(),
        (rr as f64 - rf).abs(),
        (rs as f64 - sf).abs(),
    );
    if dq > dr && dq > ds {
        (-rr - rs, rr, rs)
    } else if dr > ds {
        (rq, -rq - rs, rs)
    } else {
        (rq, rr, -rq - rr)
    }
}

/// IQ's flat-top hexagon SDF.
///
/// `r` is the **apothem** (center → edge midpoint), so the hexagon is `2 * r`
/// tall and `4 * r / sqrt(3)` wide.  Returns negative inside, positive
/// outside.
#[inline]
fn hex_sdf(mut px: f64, mut py: f64, r: f64) -> f64 {
    // k = (-sqrt(3)/2, 0.5, 1/sqrt(3))
    const KX: f64 = -0.866_025_403_784;
    const KY: f64 = 0.5;
    const KZ: f64 = 0.577_350_269_189;

    px = px.abs();
    py = py.abs();
    let d = (KX * px + KY * py).min(0.0);
    px -= 2.0 * d * KX;
    py -= 2.0 * d * KY;
    let qx = px - px.clamp(-KZ * r, KZ * r);
    let qy = py - r;
    (qx * qx + qy * qy).sqrt() * qy.signum()
}

// --- Helpers ----------------------------------------------------------------

/// Deterministic integer cell hash → \[0, 1\].
fn cell_hash(bu: i64, bv: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bu as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (bv as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{TextureGenerator, linear_to_srgb};

    fn albedo_at(map: &TextureMap, width: u32, x: usize, y: usize) -> [u8; 3] {
        let i = (y * width as usize + x) * 4;
        [map.albedo[i], map.albedo[i + 1], map.albedo[i + 2]]
    }

    /// `v` values to sweep a seam over: texel centres of a 2048-tall tile.
    ///
    /// Centres rather than corners on purpose.  Cube-rounding is ambiguous
    /// wherever a fractional axial coordinate lands exactly on `n + 0.5` —
    /// the sample sits on the boundary between two cells and either is a
    /// correct answer — and `rf = v * scale` hits that for corner sampling
    /// (`v = i / 2048`) at, for example, `scale = 8, v = 0.0625`.  With
    /// `v = (i + 0.5) / 2048` it cannot: `(2i + 1) * scale = 2048 * (2n + 1)`
    /// has no solution for any `scale` below 2048, because the left side
    /// carries at most as many factors of two as `scale` does and the right
    /// side carries eleven.  So every sample below is unambiguous, and the
    /// seam identities hold *bit for bit* rather than to a tolerance.
    const SEAM_SWEEP: u32 = 2048;

    /// [`hex_cell`] at the default config's grout and bevel, derived the
    /// way `generate_inner` derives them.
    fn hex_at(u: f64, v: f64, scale: f64) -> (f64, i64, i64) {
        let c = PaversConfig::default();
        let grout_half = (c.grout_width * 0.5).clamp(0.0, 0.45);
        hex_cell(u, v, scale, grout_half, (c.bevel * grout_half).max(0.0))
    }

    fn seam_vs() -> impl Iterator<Item = f64> {
        (0..SEAM_SWEEP).map(|i| (f64::from(i) + 0.5) / f64::from(SEAM_SWEEP))
    }

    /// A tiling generator must agree at `u = 0` and `u = 1`: they are the same
    /// point of the repeating pattern, one tile apart.  Both halves of the cell
    /// answer have to agree — the SDF, or the stone's outline steps at the
    /// join, and the cell id, or its colour does.
    ///
    /// Before the fix only `scale = 3` agreed, and by coincidence: the `u`
    /// stretch was inverted, which happens to land on four columns there.
    /// Measured worst `|sdf(0, v) - sdf(1, v)|`, as a fraction of `hex_r`:
    /// scale 3 → 0%, scale 4 → 35%, scale 5 → 58%, scale 6 → 87%, scale 8 →
    /// 67%.
    #[test]
    fn a_hex_paver_tile_is_periodic_in_u() {
        for scale in 1..=16 {
            let scale = f64::from(scale);
            for v in seam_vs() {
                let (sdf_l, q_l, r_l) = hex_at(0.0, v, scale);
                let (sdf_r, q_r, r_r) = hex_at(1.0, v, scale);
                assert_eq!(
                    sdf_l, sdf_r,
                    "scale {scale}: the hex SDF steps across the U seam at v = {v}",
                );
                assert_eq!(
                    (q_l, r_l),
                    (q_r, r_r),
                    "scale {scale}: the hex cell id changes across the U seam at v = {v}, \
                     so a paver straddling it draws two colours",
                );
            }
        }
    }

    /// The same statement for the other axis.  The geometry always tiled in V
    /// — the row spacing is `1 / scale` by construction — but the cell id did
    /// not: `r` steps by `scale` across the V seam and was hashed raw, so a
    /// paver straddling the top edge drew one colour above and another below.
    #[test]
    fn a_hex_paver_tile_is_periodic_in_v() {
        for scale in 1..=16 {
            let scale = f64::from(scale);
            for u in seam_vs() {
                let (sdf_t, q_t, r_t) = hex_at(u, 0.0, scale);
                let (sdf_b, q_b, r_b) = hex_at(u, 1.0, scale);
                assert_eq!(
                    sdf_t, sdf_b,
                    "scale {scale}: the hex SDF steps across the V seam at u = {u}",
                );
                if sdf_t < 0.0 {
                    assert_eq!(
                        (q_t, r_t),
                        (q_b, r_b),
                        "scale {scale}: the hex cell id changes across the V seam at u = {u}",
                    );
                }
            }
        }
    }

    /// Periodicity as a property of the whole function, not just of the two
    /// seam lines: shifting `u` by one tile must land on the same sample.
    /// This is what stops a fix that special-cases the edges.
    #[test]
    fn shifting_u_by_a_whole_tile_changes_nothing() {
        for scale in 1..=16 {
            let scale = f64::from(scale);
            for i in 0..257 {
                let u = f64::from(i) * 0.003_571;
                for j in 0..64 {
                    let v = (f64::from(j) + 0.5) / 64.0;
                    let (sdf_a, q_a, r_a) = hex_at(u, v, scale);
                    let (sdf_b, q_b, r_b) = hex_at(u + 1.0, v, scale);
                    assert!(
                        (sdf_a - sdf_b).abs() < 1e-12,
                        "scale {scale}: sdf({u}, {v}) = {sdf_a} but sdf({}, {v}) = {sdf_b}",
                        u + 1.0,
                    );
                    if sdf_a < 0.0 {
                        assert_eq!(
                            (q_a, r_a),
                            (q_b, r_b),
                            "scale {scale}: the cell id at ({u}, {v}) is not the cell id one \
                             tile over",
                        );
                    }
                }
            }
        }
    }

    /// The pixel-level statement of the same defect, in the shape of
    /// `brick::a_brick_straddling_the_u_seam_is_one_colour`.
    ///
    /// Column `q = 0` is centred on `u = 0`, so at `scale = 6` the paver whose
    /// centre row is `v = 0.5` is cut in half by the tile's own edge: its right
    /// half draws at `x = 0` and its left half at `x = width - 1`.  Albedo is
    /// flat within a paver — the only term is the per-cell hash — so the two
    /// columns must read the same byte triple.
    #[test]
    fn a_hex_paver_straddling_the_u_seam_is_one_colour() {
        let cfg = PaversConfig {
            scale: 6.0,
            layout: PaversLayout::Hexagonal,
            // Loud jitter so a wrong cell id cannot pass by luck.
            cell_variance: 0.6,
            roughness: 0.0,
            ..Default::default()
        };
        let (w, h) = (256, 256);
        let map = PaversGenerator::new(cfg).generate(w, h).expect("generate");
        let y = h as usize / 2; // v = 0.5, the centre row of column 0
        assert_eq!(
            albedo_at(&map, w, 0, y),
            albedo_at(&map, w, w as usize - 1, y),
            "the two halves of the seam-straddling paver drew different colours",
        );
    }

    /// The fraction of pixels that drew the grout colour, which is flat: the
    /// joint takes `color_grout` unmodified, so an exact byte match counts it.
    fn grout_fraction(layout: PaversLayout, grout_width: f64) -> f64 {
        let cfg = PaversConfig {
            scale: 5.0,
            layout,
            grout_width,
            // Corner rounding would add stone area back and blur the
            // comparison; the question here is the extent, not the corner.
            bevel: 0.0,
            ..PaversConfig::default()
        };
        let joint = cfg.color_grout.map(linear_to_srgb);
        let (w, h) = (256, 256);
        let map = PaversGenerator::new(cfg).generate(w, h).expect("generate");
        let hits = (0..h as usize)
            .flat_map(|y| (0..w as usize).map(move |x| (x, y)))
            .filter(|&(x, y)| albedo_at(&map, w, x, y) == joint)
            .count();
        hits as f64 / f64::from(w * h)
    }

    /// `grout_width` has to open a joint on *both* layouts, and the same one.
    ///
    /// The hexagonal path takes `grout_half` and `bevel_r` as fractions of the
    /// cell pitch exactly as [`square_cell`] does, so a stone of linear extent
    /// `1 - grout_width` covers `(1 - grout_width)²` of its cell whatever the
    /// cell's shape — the two layouts must land on the same area fraction, and
    /// on the closed form.
    ///
    /// Before the fix the hexagonal layout drew no grout at all, at any width:
    /// [`hex_sdf`] takes the apothem and `hex_cell` handed it the circumradius,
    /// so the drawn hexagon strictly contained its own cell and every sample
    /// came out inside the stone.  Measured over a 512² grid, the grout pixel
    /// count was 0.0000% at scales 3, 4, 5, 6 and 8, and the largest SDF value
    /// anywhere was −0.0097 — there was no zero crossing to find (#17).
    #[test]
    fn grout_width_opens_the_same_joint_on_both_layouts() {
        assert!(
            grout_fraction(PaversLayout::Hexagonal, 0.0) < 0.01,
            "a zero-width joint should leave no grout",
        );

        let mut previous = 0.0;
        for width in [0.05, 0.1, 0.2, 0.3] {
            let hex = grout_fraction(PaversLayout::Hexagonal, width);
            let square = grout_fraction(PaversLayout::Square, width);
            let expected = 1.0 - (1.0 - width) * (1.0 - width);
            assert!(
                (hex - expected).abs() < 0.02,
                "grout_width {width} should open {expected:.3} of the tile on hexagons, \
                 not {hex:.3}",
            );
            assert!(
                (hex - square).abs() < 0.02,
                "grout_width {width} opens {hex:.3} of a hexagonal tile but {square:.3} of \
                 a square one; the two layouts disagree about what the width means",
            );
            assert!(
                hex > previous,
                "a wider joint should draw more grout, but {width} gave {hex:.3} after \
                 {previous:.3}",
            );
            previous = hex;
        }
    }
}
