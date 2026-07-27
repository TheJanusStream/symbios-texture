//! Toroidal 4D noise mapping for seamless texture tiling.
//!
//! Maps 2D UV coordinates (in \[0, 1\]) to a 4D point on a torus so that noise
//! sampled there wraps perfectly at all four edges with no seam.
//!
//! The mapping is:
//!   nx = cos(2π·u) · frequency
//!   ny = sin(2π·u) · frequency
//!   nz = cos(2π·v) · frequency
//!   nw = sin(2π·v) · frequency
//!
//! `frequency` is the torus radius in noise-space. Larger values push the
//! sample point further from the origin, crossing more noise-lattice cells and
//! producing higher-frequency / more-detailed patterns.
//!
//! Seam-freedom is guaranteed because cos(0)=cos(2π) and sin(0)=sin(2π), so
//! u=0 and u=1 always resolve to the identical 4D coordinate.
//!
//! # Cellular noise
//!
//! Alongside the lattice-noise wrapper, this module provides the seamless
//! cellular (Worley / Voronoi) family that the cell-decomposition generators
//! are built from.  All three samplers share one [`CellularParams`] layout and
//! tile by the same trick — cell indices wrap modulo the lattice period, and
//! offsets take the shorter way round the torus:
//!
//! - [`cellular`] — the two nearest site distances plus the owning cell, for
//!   domed stones, per-cell variation, and `F2 − F1` crack masks.
//! - [`cellular_edge`] — true distance to the cell wall, for cracks of
//!   *constant* width regardless of cell size.
//! - [`cellular_smooth`] — a softmin `F1` without the faceted creases of plain
//!   Worley, for organic plating.
//!
//! # Oriented stripes
//!
//! [`stripe`] provides directional wave fields — wood grain, brushed metal,
//! fabric ribs, dune ripple — as a cheap stand-in for phasor noise, with the
//! same defining trait: uniform contrast everywhere, rather than the muddy
//! mid-tones stacked FBM settles into.  [`StripeParams`] takes whole cycle
//! counts per axis (which is what keeps the field seamless — see its docs),
//! [`StripeProfile`] reshapes the wave, and a caller-supplied warp bends
//! straight bands into flowing grain.

use noise::NoiseFn;
use rayon::prelude::*;
use std::f64::consts::TAU;

/// Wraps any 4-dimensional noise function and samples it on a torus, producing
/// output that tiles seamlessly when `u` and `v` are each in `[0, 1]`.
pub struct ToroidalNoise<N> {
    noise: N,
    /// Torus radius in noise-space.  Larger → more detail per texture tile.
    pub frequency: f64,
}

impl<N: NoiseFn<f64, 4>> ToroidalNoise<N> {
    /// Wrap a 4-D noise function with a toroidal mapping at the given
    /// `frequency` (torus radius in noise space; larger → more detail per
    /// tile).
    pub fn new(noise: N, frequency: f64) -> Self {
        Self { noise, frequency }
    }

    /// Sample the noise at normalised UV coordinates in [0, 1].
    ///
    /// Both `u` and `v` wrap continuously; there is no seam.
    pub fn get(&self, u: f64, v: f64) -> f64 {
        // Radius = frequency: as u/v sweep [0,1] the 4D point traces a circle
        // of this radius through noise space, giving arc-length = 2π·frequency.
        // With Perlin lattice cells of size 1, a radius of ~1 gives ~6 cells of
        // variation; radius 4 gives ~25.
        let nx = (TAU * u).cos() * self.frequency;
        let ny = (TAU * u).sin() * self.frequency;
        let nz = (TAU * v).cos() * self.frequency;
        let nw = (TAU * v).sin() * self.frequency;
        self.noise.get([nx, ny, nz, nw])
    }

    /// Sample at an offset UV — useful when building domain-warp chains.
    pub fn get_offset(&self, u: f64, v: f64, du: f64, dv: f64) -> f64 {
        self.get(u + du, v + dv)
    }

    /// Sample the noise at pre-projected 4D torus coordinates.
    ///
    /// Use this with lookup tables (see [`sample_grid`]) to avoid recomputing
    /// `sin`/`cos` for every sample in a regular grid.
    #[inline]
    pub fn get_precomputed(&self, nx: f64, ny: f64, nz: f64, nw: f64) -> f64 {
        self.noise.get([nx, ny, nz, nw])
    }
}

/// Sample noise into a pre-allocated buffer, resizing it to `width * height`.
///
/// This is the allocation-friendly variant of [`sample_grid`].  Pass the same
/// `Vec` across multiple generation calls (or via a [`Workspace`]) to reuse
/// its heap allocation rather than allocating a fresh 128 MB buffer per grid
/// at 4096×4096.
///
/// Values are in `[-1, 1]`.  Torus coordinates are precomputed into lookup
/// tables of size `W` and `H`, reducing trigonometric calls from `O(W × H)`
/// to `O(W + H)`.
///
/// Rows are evaluated in parallel on the ambient rayon pool: the crate's
/// private texture pool when called from an async generation task, or the
/// caller's pool (usually the global one) for direct synchronous calls.
/// Output is byte-identical to serial evaluation — each element is a pure
/// function of its coordinates.
///
/// [`Workspace`]: crate::generator::Workspace
pub fn sample_grid_into<N: NoiseFn<f64, 4> + Sync>(
    noise: &ToroidalNoise<N>,
    width: u32,
    height: u32,
    out: &mut Vec<f64>,
) {
    let w = width as usize;
    let h = height as usize;
    let freq = noise.frequency;

    // One sin/cos pair per column and per row instead of per pixel.
    let col_cos: Vec<f64> = (0..w)
        .map(|x| (TAU * x as f64 / w as f64).cos() * freq)
        .collect();
    let col_sin: Vec<f64> = (0..w)
        .map(|x| (TAU * x as f64 / w as f64).sin() * freq)
        .collect();
    let row_cos: Vec<f64> = (0..h)
        .map(|y| (TAU * y as f64 / h as f64).cos() * freq)
        .collect();
    let row_sin: Vec<f64> = (0..h)
        .map(|y| (TAU * y as f64 / h as f64).sin() * freq)
        .collect();

    out.clear();
    out.resize(w * h, 0.0);
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let nz = row_cos[y];
        let nw = row_sin[y];
        for (x, slot) in row.iter_mut().enumerate() {
            *slot = noise.get_precomputed(col_cos[x], col_sin[x], nz, nw);
        }
    });
}

/// Convenience: iterate over a `width × height` grid and collect samples.
///
/// Returns a `Vec<f64>` of length `width * height`, values in `[-1, 1]`.
///
/// For high-resolution textures (≥ 2048) where multiple grids are needed,
/// prefer [`sample_grid_into`] with a [`Workspace`] to reuse allocations.
///
/// [`Workspace`]: crate::generator::Workspace
pub fn sample_grid<N: NoiseFn<f64, 4> + Sync>(
    noise: &ToroidalNoise<N>,
    width: u32,
    height: u32,
) -> Vec<f64> {
    let mut out = Vec::new();
    sample_grid_into(noise, width, height, &mut out);
    out
}

/// Map a raw noise sample from `[-1, 1]` to `[0, 1]`.
#[inline]
pub fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}

/// Bilinearly interpolate a value from a toroidal (seamlessly tiling) grid.
///
/// `u` and `v` are in UV space and may fall outside `[0, 1]`; they are wrapped
/// before sampling so the lookup is always valid.  Used by the domain-warped
/// generators (bark, marble) to fetch the warped base-noise value without
/// additional `sin`/`cos` calls per pixel.
#[inline]
pub fn bilinear_sample_torus(grid: &[f64], w: usize, h: usize, u: f64, v: f64) -> f64 {
    // Wrap UV into [0, 1).
    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);

    // Convert to fractional pixel coordinates.
    let px = u * w as f64;
    let py = v * h as f64;

    let x0 = px as usize % w;
    let y0 = py as usize % h;
    let x1 = (x0 + 1) % w;
    let y1 = (y0 + 1) % h;

    let fx = px.fract();
    let fy = py.fract();

    let v00 = grid[y0 * w + x0];
    let v10 = grid[y0 * w + x1];
    let v01 = grid[y1 * w + x0];
    let v11 = grid[y1 * w + x1];

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Distance metric used to measure the offset from a query point to a
/// cellular-noise site.
///
/// The metric decides the *shape* of a cell far more than its size: Euclidean
/// gives the familiar rounded stone, Manhattan gives diamonds with axis-aligned
/// facets, and Chebyshev gives squared-off blocks that read as tiling or
/// masonry even before any pattern work is layered on top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CellMetric {
    /// Straight-line distance — rounded, organic cells (pebbles, plates).
    #[default]
    Euclidean,
    /// Sum of per-axis offsets — diamond cells with 45° facets.
    Manhattan,
    /// Largest per-axis offset — square, axis-aligned cells.
    Chebyshev,
}

impl CellMetric {
    /// Combine two non-negative per-axis offsets into a single distance.
    #[inline]
    pub fn distance(self, dx: f64, dy: f64) -> f64 {
        match self {
            Self::Euclidean => (dx * dx + dy * dy).sqrt(),
            Self::Manhattan => dx + dy,
            Self::Chebyshev => dx.max(dy),
        }
    }
}

/// Site jitter used by the pre-existing cell-decomposition generators, and the
/// default for [`CellularParams`].
///
/// A site is placed within `±0.35` of its cell centre, i.e. across the middle
/// 70% of the cell.  Holding sites back from the cell corners keeps any two
/// neighbours from landing almost on top of each other, which is what produces
/// degenerate sliver cells.
pub const DEFAULT_JITTER: f64 = 0.70;

/// Upper bound on the cellular lattice resolution, matching the crate's
/// `MAX_DIMENSION`: beyond one cell per texel the pattern is pure aliasing.
const MAX_CELLS: f64 = 4096.0;

/// The 5×5 block of candidate cells searched around every query point.
///
/// Five cells across is enough that a site jittered to the far corner of its
/// cell is still found before a nearer one is missed, for any jitter in
/// `[0, 1]`.
const NEIGHBOURHOOD: usize = 25;

/// Two site vectors closer together than this are treated as the same site.
///
/// Needed because a small lattice (`cells` < 5) wraps the same cell into the
/// 5×5 neighbourhood more than once, and a duplicate of the owning site would
/// otherwise report a zero-width border everywhere.
const SITE_EPSILON: f64 = 1e-12;

/// Smallest usable softmin falloff; guards against a divide-by-zero when a
/// caller passes 0.
const MIN_FALLOFF: f64 = 1e-6;

/// Layout of a cellular (Worley / Voronoi) lattice in UV space.
///
/// Shared by [`cellular`], [`cellular_edge`], and [`cellular_smooth`] so a
/// generator can build the layout once and query it several ways per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellularParams {
    /// Cells across the tile.  Rounded to an integer so the lattice divides
    /// the tile evenly and therefore wraps without a seam.
    pub scale: f64,
    /// PRNG seed for site placement.
    pub seed: u32,
    /// How site distance is measured.
    pub metric: CellMetric,
    /// How far a site may stray from its cell centre, in `[0, 1]`: `0.0` pins
    /// every site dead centre (a perfectly regular lattice), `1.0` lets it
    /// reach the cell edge.  Values near `1.0` produce sliver cells where two
    /// neighbours crowd a shared corner; see [`DEFAULT_JITTER`].
    pub jitter: f64,
}

impl CellularParams {
    /// A lattice of `scale × scale` cells with Euclidean distance and
    /// [`DEFAULT_JITTER`].
    pub fn new(scale: f64, seed: u32) -> Self {
        Self {
            scale,
            seed,
            metric: CellMetric::default(),
            jitter: DEFAULT_JITTER,
        }
    }

    /// Measure site distance with `metric` instead of Euclidean.
    #[must_use]
    pub fn with_metric(mut self, metric: CellMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set how far sites may stray from their cell centres — see
    /// [`jitter`](Self::jitter).
    #[must_use]
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Integer cell count across the tile, clamped to a sane range.
    #[inline]
    fn cells(&self) -> i64 {
        self.scale.round().clamp(1.0, MAX_CELLS) as i64
    }

    /// Site-placement window within a cell as `(low edge, span)`, both in
    /// cell-relative units.
    #[inline]
    fn jitter_bounds(&self) -> (f64, f64) {
        let j = self.jitter.clamp(0.0, 1.0);
        (0.5 - 0.5 * j, j)
    }
}

/// The two nearest sites to a query point, as returned by [`cellular`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellularSample {
    /// Distance to the nearest site, in UV units.
    pub f1: f64,
    /// Distance to the second-nearest site, in UV units.
    pub f2: f64,
    /// Integer x coordinate of the nearest site's cell, in `[0, cells)`.
    pub cell_x: i64,
    /// Integer y coordinate of the nearest site's cell, in `[0, cells)`.
    pub cell_y: i64,
}

impl CellularSample {
    /// `F2 − F1`, the classic "crackle" mask: zero exactly on a cell boundary
    /// and largest at cell centres.
    ///
    /// Cheap, but the band it draws is *wider where cells are larger*.  Use
    /// [`cellular_edge`] instead when a crack needs a constant width.
    #[inline]
    pub fn ridge(&self) -> f64 {
        self.f2 - self.f1
    }
}

/// One candidate site as seen from a query point.
#[derive(Clone, Copy, Debug, Default)]
struct Site {
    /// Per-axis offset magnitudes (always ≥ 0), for distance metrics.
    adx: f64,
    ady: f64,
    /// Signed offset from the query point toward the site, for the border
    /// geometry in [`cellular_edge`].
    sdx: f64,
    sdy: f64,
    /// Wrapped integer cell coordinates.
    ci: i64,
    cj: i64,
}

/// Minimum-image offset between query coordinate `p` and site coordinate `c`
/// on one wrapped axis, as `(magnitude, signed)`.
///
/// The magnitude is computed exactly as the original Voronoi loop did — take
/// `|p − c|`, then fold it through `1 − d` once it exceeds half a tile — so
/// that [`toroidal_voronoi`] keeps producing bit-for-bit identical results.
/// The signed component points from the query point toward the nearest image
/// of the site.
#[inline]
fn wrap_delta(p: f64, c: f64) -> (f64, f64) {
    let raw = p - c;
    let mut m = raw.abs();
    // `raw > 0` means the site sits at the lower coordinate, so the vector
    // pointing at it is negative.
    let mut s = if raw > 0.0 { -1.0 } else { 1.0 };
    if m > 0.5 {
        m = 1.0 - m;
        s = -s;
    }
    (m, s * m)
}

/// Resolved cellular lattice: everything [`visit_sites`] needs to place sites.
///
/// `pos_scale` is the divisor mapping cell indices back into UV space and
/// `jitter_lo`/`jitter_span` are the site-placement window in cell-relative
/// units.  They are stored rather than derived from a [`CellularParams`] so
/// [`toroidal_voronoi`] can supply the literal values its historical output
/// depends on.
#[derive(Clone, Copy, Debug)]
struct Lattice {
    cells: i64,
    pos_scale: f64,
    jitter_lo: f64,
    jitter_span: f64,
    seed: u32,
}

impl Lattice {
    /// The lattice described by `params`, with cell positions derived from the
    /// rounded integer cell count.
    fn new(params: &CellularParams) -> Self {
        let cells = params.cells();
        let (jitter_lo, jitter_span) = params.jitter_bounds();
        Self {
            cells,
            pos_scale: cells as f64,
            jitter_lo,
            jitter_span,
            seed: params.seed,
        }
    }
}

/// Visit every candidate site in the 5×5 neighbourhood around `(u, v)`.
#[inline]
fn visit_sites(u: f64, v: f64, lat: &Lattice, mut visit: impl FnMut(Site)) {
    let gi = (u * lat.pos_scale).floor() as i64;
    let gj = (v * lat.pos_scale).floor() as i64;

    for di in -2i64..=2 {
        for dj in -2i64..=2 {
            let ni = (gi + di).rem_euclid(lat.cells);
            let nj = (gj + dj).rem_euclid(lat.cells);

            let jx = lat.jitter_lo + lat.jitter_span * cell_hash(ni, nj, lat.seed);
            let jy = lat.jitter_lo + lat.jitter_span * cell_hash(nj, ni, lat.seed.wrapping_add(17));

            // Site position in UV space.
            let cx = (ni as f64 + jx) / lat.pos_scale;
            let cy = (nj as f64 + jy) / lat.pos_scale;

            let (adx, sdx) = wrap_delta(u, cx);
            let (ady, sdy) = wrap_delta(v, cy);

            visit(Site {
                adx,
                ady,
                sdx,
                sdy,
                ci: ni,
                cj: nj,
            });
        }
    }
}

/// Sample a seamlessly-tiling cellular (Worley) lattice at `(u, v)`.
///
/// Returns the two nearest site distances and the owning cell — the building
/// blocks for domed stones (`1 − F1`), crack masks
/// ([`ridge`](CellularSample::ridge)), and per-cell colour variation (feed
/// `cell_x`/`cell_y` to [`cell_hash`]).
pub fn cellular(u: f64, v: f64, params: CellularParams) -> CellularSample {
    let lat = Lattice::new(&params);
    let metric = params.metric;

    let mut f1 = f64::MAX;
    let mut f2 = f64::MAX;
    let mut cell_x = 0;
    let mut cell_y = 0;

    visit_sites(u, v, &lat, |s| {
        let d = metric.distance(s.adx, s.ady);
        if d < f1 {
            f2 = f1;
            f1 = d;
            cell_x = s.ci;
            cell_y = s.cj;
        } else if d < f2 {
            f2 = d;
        }
    });

    CellularSample {
        f1,
        f2,
        cell_x,
        cell_y,
    }
}

/// Distance from `(u, v)` to the nearest Voronoi *border*, in UV units.
///
/// Unlike [`ridge`](CellularSample::ridge), this measures the real distance to
/// the cell wall, so thresholding it draws cracks of constant width no matter
/// how large or small the surrounding cells are — the property that makes
/// cracked mud, crackle glaze, and dry lakebeds read correctly.  It costs a
/// second pass over the neighbourhood.
///
/// The border is found as the nearest perpendicular bisector between the
/// owning site and each of its neighbours, following Inigo Quilez's
/// [Voronoi edges](https://iquilezles.org/articles/voronoilines/).
///
/// `params.metric` is deliberately **ignored**: under a non-Euclidean metric a
/// cell wall is not a perpendicular bisector, and constant width — the entire
/// reason to prefer this over `F2 − F1` — is a Euclidean property.  Use
/// [`cellular`] with `ridge()` if you want a non-Euclidean crack mask.
///
/// A single-cell lattice has no borders; that degenerate case returns `0.5`
/// (half a tile away, i.e. "no wall anywhere near").
pub fn cellular_edge(u: f64, v: f64, params: CellularParams) -> f64 {
    let lat = Lattice::new(&params);

    let mut sites = [Site::default(); NEIGHBOURHOOD];
    let mut count = 0usize;
    visit_sites(u, v, &lat, |s| {
        sites[count] = s;
        count += 1;
    });
    let sites = &sites[..count];

    // Pass 1 — the site that owns this pixel.
    let mut owner = 0usize;
    let mut owner_d = f64::MAX;
    for (i, s) in sites.iter().enumerate() {
        let d = CellMetric::Euclidean.distance(s.adx, s.ady);
        if d < owner_d {
            owner_d = d;
            owner = i;
        }
    }
    let (mx, my) = (sites[owner].sdx, sites[owner].sdy);

    // Pass 2 — nearest perpendicular bisector between the owner and any other
    // site: project the midpoint of the two site vectors onto the axis joining
    // them.
    let mut edge = f64::MAX;
    for (i, s) in sites.iter().enumerate() {
        if i == owner {
            continue;
        }
        let rx = s.sdx - mx;
        let ry = s.sdy - my;
        let len = (rx * rx + ry * ry).sqrt();
        if len < SITE_EPSILON {
            // Another wrapped image of the owning site, not a real neighbour.
            continue;
        }
        let d = 0.5 * ((s.sdx + mx) * (rx / len) + (s.sdy + my) * (ry / len));
        if d < edge {
            edge = d;
        }
    }

    if edge == f64::MAX { 0.5 } else { edge.max(0.0) }
}

/// Smooth-minimum variant of [`cellular`]'s `F1`, free of the derivative
/// creases that make plain Worley noise look faceted.
///
/// Blends every site's contribution through `−ln(Σ e^(−k·d)) / k`, so cell
/// boundaries melt into one another instead of meeting at a sharp ridge —
/// suited to organic plating (chitin, leather, weathered stone) where hard
/// cell walls would read as artificial.  `falloff` (`k`) controls the
/// transition: ~8 is very soft, ~64 approaches plain `F1`.
///
/// The result is always ≤ the hard `F1`, since a softmin underestimates.
pub fn cellular_smooth(u: f64, v: f64, params: CellularParams, falloff: f64) -> f64 {
    let lat = Lattice::new(&params);
    let metric = params.metric;
    let k = falloff.max(MIN_FALLOFF);

    let mut acc = 0.0;
    let mut f1 = f64::MAX;

    visit_sites(u, v, &lat, |s| {
        let d = metric.distance(s.adx, s.ady);
        if d < f1 {
            f1 = d;
        }
        acc += (-k * d).exp();
    });

    // With an extreme `falloff` every term can underflow to zero; fall back to
    // the hard minimum rather than returning an infinity.
    if acc > 0.0 { -acc.ln() / k } else { f1 }
}

/// Grid-based toroidal Voronoi in UV space.
///
/// Partitions `[0, 1]²` into `scale × scale` candidate cells and searches a
/// 5×5 neighbourhood around the query point, wrapping toroidally.  Returns
/// `(F1, F2, best_i, best_j)` where F1/F2 are the two nearest site distances
/// in UV units and `(best_i, best_j)` is the integer cell coordinate of the
/// F1 site.  Shared by the cell-decomposition generators (cobblestone,
/// lava).
///
/// Kept as a thin shim over [`visit_sites`] with the exact literals its
/// historical output depends on, so cobblestone and lava stay bit-for-bit
/// stable against their golden hashes.  New generators should call
/// [`cellular`], which additionally offers metrics, a jitter knob, and
/// integer-exact cell positions.
pub(crate) fn toroidal_voronoi(u: f64, v: f64, scale: f64, seed: u32) -> (f64, f64, i64, i64) {
    let n = scale.round().max(1.0) as i64;

    let mut f1 = f64::MAX;
    let mut f2 = f64::MAX;
    // Always overwritten by the first candidate; only a NaN query could leave
    // these at their initial value.
    let mut best_i = 0;
    let mut best_j = 0;

    // Historical jitter window: the middle 70% of each cell, i.e. [0.15, 0.85].
    // Callers pre-round `scale`, so passing it as the position divisor matches
    // `n` exactly.
    let lat = Lattice {
        cells: n,
        pos_scale: scale,
        jitter_lo: 0.15,
        jitter_span: 0.70,
        seed,
    };

    visit_sites(u, v, &lat, |s| {
        let d = (s.adx * s.adx + s.ady * s.ady).sqrt();
        if d < f1 {
            f2 = f1;
            f1 = d;
            best_i = s.ci;
            best_j = s.cj;
        } else if d < f2 {
            f2 = d;
        }
    });

    (f1, f2, best_i, best_j)
}

/// Wave profile applied to a stripe field's phase.
///
/// Separating the profile from the phase is what makes this a usable stand-in
/// for phasor noise: the same oriented field reads as wood grain, a brushed
/// finish, or corduroy purely by reshaping the wave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StripeProfile {
    /// Smooth swell — brushed metal, satin sheen, gentle dune ripple.
    #[default]
    Sine,
    /// Linear rise and fall to a crease — fabric ribs, corduroy.
    Triangle,
    /// Ramp with a hard reset — growth rings, strata, lapped boards.
    Sawtooth,
    /// Flat bands with defined edges — inlay, planking, woven tape.
    Square,
}

impl StripeProfile {
    /// Shape a phase in turns (`[0, 1)`) into a value in `[0, 1]`.
    ///
    /// `sharpness` in `[0, 1]` hardens the pattern: it narrows the edge for
    /// [`Square`](Self::Square) and tightens the crest for the others.
    pub fn shape(self, phase_turns: f64, sharpness: f64) -> f64 {
        let t = phase_turns.rem_euclid(1.0);
        let hardness = sharpness.clamp(0.0, 1.0);
        match self {
            Self::Square => {
                // Edge width shrinks to nothing as sharpness approaches 1.
                let edge = (1.0 - hardness).max(1e-4);
                let s = (TAU * t).sin();
                smoothstep(-edge, edge, s)
            }
            other => {
                let base = match other {
                    Self::Sine => 0.5 - 0.5 * (TAU * t).cos(),
                    Self::Triangle => 1.0 - 2.0 * (t - 0.5).abs(),
                    // Sawtooth is the ramp itself; the reset is the feature.
                    _ => t,
                };
                base.clamp(0.0, 1.0).powf(1.0 + hardness * 3.0)
            }
        }
    }
}

/// A seamless oriented stripe field.
///
/// # Why cycles instead of an angle
///
/// The obvious parameterisation — a direction and a frequency — does not
/// tile.  `sin(dot(p, dir) · f)` only closes on itself when the frequency
/// resolved along *each* axis is a whole number of cycles per tile, and for a
/// free angle it almost never is; the seam shows as a phase jump down one
/// edge.  Naming the per-axis cycle counts instead makes every representable
/// field seamless by construction, with direction and wavelength falling out
/// as derived quantities.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StripeParams {
    /// Whole cycles across the tile horizontally.
    pub cycles_u: i32,
    /// Whole cycles across the tile vertically.
    pub cycles_v: i32,
    /// Wave shape.
    pub profile: StripeProfile,
    /// How hard the pattern's features are, in `[0, 1]`.
    pub sharpness: f64,
}

impl StripeParams {
    /// A sine field running at the given whole cycle counts per axis.
    ///
    /// `(n, 0)` gives vertical bands, `(0, n)` horizontal, `(n, n)` a 45°
    /// diagonal.
    pub fn new(cycles_u: i32, cycles_v: i32) -> Self {
        Self {
            cycles_u,
            cycles_v,
            profile: StripeProfile::default(),
            sharpness: 0.0,
        }
    }

    /// Reshape the wave.
    #[must_use]
    pub fn with_profile(mut self, profile: StripeProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set how hard the pattern's features are — see
    /// [`sharpness`](Self::sharpness).
    #[must_use]
    pub fn with_sharpness(mut self, sharpness: f64) -> Self {
        self.sharpness = sharpness;
        self
    }

    /// Unit vector the stripes advance along, or `None` when the field is
    /// constant.
    pub fn direction(&self) -> Option<(f64, f64)> {
        let (u, v) = (self.cycles_u as f64, self.cycles_v as f64);
        let len = (u * u + v * v).sqrt();
        (len > 0.0).then(|| (u / len, v / len))
    }

    /// Distance between successive bands in UV units, or infinity when the
    /// field is constant.
    pub fn wavelength(&self) -> f64 {
        let (u, v) = (self.cycles_u as f64, self.cycles_v as f64);
        let len = (u * u + v * v).sqrt();
        if len > 0.0 { 1.0 / len } else { f64::INFINITY }
    }
}

/// Phase of a stripe field at `(u, v)`, in turns within `[0, 1)`.
///
/// Useful when a generator wants the raw phase — to drive a hue shift or a
/// second pattern locked to the same bands — rather than a shaped value.
pub fn stripe_phase(u: f64, v: f64, params: StripeParams) -> f64 {
    (params.cycles_u as f64 * u + params.cycles_v as f64 * v).rem_euclid(1.0)
}

/// Sample a stripe field at `(u, v)`, returning a value in `[0, 1]`.
///
/// `warp_turns` bends the bands by shifting the phase; feeding it a toroidal
/// noise sample turns dead-straight stripes into flowing grain while keeping
/// the result seamless, since a periodic phase plus a periodic warp is still
/// periodic.  Pass `0.0` for a perfectly regular field.
///
/// Unlike stacked FBM, the output sweeps the full `[0, 1]` range everywhere
/// rather than clustering around the middle — the uniform contrast that makes
/// oriented patterns read as *material* instead of as haze.
pub fn stripe(u: f64, v: f64, params: StripeParams, warp_turns: f64) -> f64 {
    let phase = params.cycles_u as f64 * u + params.cycles_v as f64 * v + warp_turns;
    params.profile.shape(phase, params.sharpness)
}

/// Hermite smoothstep between two edges, clamped outside them.
#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Deterministic integer hash → \[0, 1\].  Drives Voronoi site jitter and
/// per-cell variance for the cell-decomposition generators.
///
/// Pair it with [`CellularSample`]'s `cell_x`/`cell_y` to give every cell its
/// own stable colour, height, or orientation.  Vary `seed` to decorrelate
/// several such attributes on the same lattice.
pub fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise::Perlin;

    /// Verify that the sampler actually varies across the texture.
    /// With the (broken) inverted formula the stddev was < 0.001; correct
    /// formula gives > 0.1 for frequency=4.
    #[test]
    fn samples_vary_with_frequency() {
        let noise = ToroidalNoise::new(Perlin::new(1), 4.0);
        let samples = sample_grid(&noise, 64, 64);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();
        assert!(
            stddev > 0.1,
            "noise has almost no variation (stddev={stddev:.4}); torus radius is likely wrong"
        );
    }

    /// Verify left/right and top/bottom edges match (seamless tiling).
    #[test]
    fn tiles_seamlessly() {
        let noise = ToroidalNoise::new(Perlin::new(42), 3.0);
        // u=0 and u=1 should give the same value for any v
        for v in [0.0, 0.25, 0.5, 0.75] {
            let at_0 = noise.get(0.0, v);
            let at_1 = noise.get(1.0, v);
            assert!(
                (at_0 - at_1).abs() < 1e-10,
                "horizontal seam at v={v}: {at_0} != {at_1}"
            );
        }
        // v=0 and v=1 should give the same value for any u
        for u in [0.0, 0.25, 0.5, 0.75] {
            let at_0 = noise.get(u, 0.0);
            let at_1 = noise.get(u, 1.0);
            assert!(
                (at_0 - at_1).abs() < 1e-10,
                "vertical seam at u={u}: {at_0} != {at_1}"
            );
        }
    }

    /// Verbatim copy of the Voronoi loop as it stood before it was folded into
    /// [`visit_sites`].  Cobblestone and lava are pinned downstream by golden
    /// image hashes, so the shared walker has to reproduce this *exactly* —
    /// "visually identical" is not good enough.
    fn legacy_toroidal_voronoi(u: f64, v: f64, scale: f64, seed: u32) -> (f64, f64, i64, i64) {
        let n = scale.round().max(1.0) as i64;
        let su = u * scale;
        let sv = v * scale;
        let gi = su.floor() as i64;
        let gj = sv.floor() as i64;

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;
        let mut best_i = gi;
        let mut best_j = gj;

        for di in -2i64..=2 {
            for dj in -2i64..=2 {
                let ni = (gi + di).rem_euclid(n);
                let nj = (gj + dj).rem_euclid(n);

                let jx = 0.15 + 0.70 * cell_hash(ni, nj, seed);
                let jy = 0.15 + 0.70 * cell_hash(nj, ni, seed.wrapping_add(17));

                let cx = (ni as f64 + jx) / scale;
                let cy = (nj as f64 + jy) / scale;

                let mut dx = (u - cx).abs();
                let mut dy = (v - cy).abs();
                if dx > 0.5 {
                    dx = 1.0 - dx;
                }
                if dy > 0.5 {
                    dy = 1.0 - dy;
                }
                let d = (dx * dx + dy * dy).sqrt();

                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    best_i = ni;
                    best_j = nj;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }

        (f1, f2, best_i, best_j)
    }

    #[test]
    fn toroidal_voronoi_matches_legacy_bit_for_bit() {
        // Integer scales only: every caller pre-rounds, which is what lets the
        // shared walker divide by the integer cell count.
        for scale in [1.0, 2.0, 6.0, 7.0, 16.0] {
            for seed in [0u32, 7, 1234, u32::MAX] {
                for yi in 0..29 {
                    for xi in 0..29 {
                        let u = xi as f64 / 29.0;
                        let v = yi as f64 / 29.0;
                        let (f1, f2, ci, cj) = toroidal_voronoi(u, v, scale, seed);
                        let (lf1, lf2, lci, lcj) = legacy_toroidal_voronoi(u, v, scale, seed);
                        assert_eq!(
                            f1.to_bits(),
                            lf1.to_bits(),
                            "F1 drift at ({u}, {v}) scale={scale} seed={seed}: {f1} vs {lf1}"
                        );
                        assert_eq!(
                            f2.to_bits(),
                            lf2.to_bits(),
                            "F2 drift at ({u}, {v}) scale={scale} seed={seed}: {f2} vs {lf2}"
                        );
                        assert_eq!((ci, cj), (lci, lcj), "cell drift at ({u}, {v})");
                    }
                }
            }
        }
    }

    /// Every metric, and all three samplers, must wrap at the tile edges.
    #[test]
    fn cellular_tiles_seamlessly() {
        for metric in [
            CellMetric::Euclidean,
            CellMetric::Manhattan,
            CellMetric::Chebyshev,
        ] {
            let params = CellularParams::new(5.0, 99).with_metric(metric);
            for t in [0.0, 0.17, 0.5, 0.83] {
                // Horizontal seam.
                let a = cellular(0.0, t, params);
                let b = cellular(1.0, t, params);
                assert!(
                    (a.f1 - b.f1).abs() < 1e-12 && (a.f2 - b.f2).abs() < 1e-12,
                    "{metric:?} horizontal seam at v={t}: {a:?} vs {b:?}"
                );
                // Vertical seam.
                let a = cellular(t, 0.0, params);
                let b = cellular(t, 1.0, params);
                assert!(
                    (a.f1 - b.f1).abs() < 1e-12 && (a.f2 - b.f2).abs() < 1e-12,
                    "{metric:?} vertical seam at u={t}: {a:?} vs {b:?}"
                );

                assert!(
                    (cellular_edge(0.0, t, params) - cellular_edge(1.0, t, params)).abs() < 1e-12,
                    "edge horizontal seam at v={t}"
                );
                assert!(
                    (cellular_edge(t, 0.0, params) - cellular_edge(t, 1.0, params)).abs() < 1e-12,
                    "edge vertical seam at u={t}"
                );
                assert!(
                    (cellular_smooth(0.0, t, params, 32.0) - cellular_smooth(1.0, t, params, 32.0))
                        .abs()
                        < 1e-12,
                    "smooth horizontal seam at v={t}"
                );
            }
        }
    }

    /// With jitter disabled the lattice is perfectly regular, so a cell centre
    /// sits exactly on its site.
    #[test]
    fn zero_jitter_pins_sites_to_cell_centres() {
        let cells = 4.0;
        let params = CellularParams::new(cells, 3).with_jitter(0.0);
        for i in 0..4 {
            for j in 0..4 {
                let u = (i as f64 + 0.5) / cells;
                let v = (j as f64 + 0.5) / cells;
                let s = cellular(u, v, params);
                assert!(
                    s.f1 < 1e-15,
                    "site not at cell centre ({i},{j}): F1={}",
                    s.f1
                );
                assert_eq!((s.cell_x, s.cell_y), (i, j), "wrong owning cell");
            }
        }
    }

    /// Jitter actually moves sites off the centres it pins them to at 0.
    #[test]
    fn jitter_displaces_sites() {
        let cells = 4.0;
        let centre = 0.5 / cells;
        let jittered = CellularParams::new(cells, 11).with_jitter(1.0);
        let displaced = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|&(i, j)| {
                let u = (i as f64 + 0.5) / cells;
                let v = (j as f64 + 0.5) / cells;
                cellular(u, v, jittered).f1 > 1e-6
            })
            .count();
        assert_eq!(
            displaced, 16,
            "full jitter left some sites at their centres"
        );
        // …and no site escapes its own cell.
        for i in 0..4 {
            let u = (i as f64 + 0.5) / cells;
            assert!(cellular(u, u, jittered).f1 < centre * 1.5);
        }
    }

    /// The whole point of `cellular_edge`: it is the true distance to the cell
    /// wall, so on a regular lattice the centre of a cell is exactly half a
    /// cell from its border — and that holds at every lattice resolution.
    #[test]
    fn edge_measures_true_distance_to_the_wall() {
        for cells in [4.0, 8.0, 16.0] {
            let params = CellularParams::new(cells, 5).with_jitter(0.0);
            let half = 0.5 / cells;

            let centre = cellular_edge(0.5 / cells, 0.5 / cells, params);
            assert!(
                (centre - half).abs() < 1e-9,
                "cells={cells}: centre-to-wall {centre} != {half}"
            );

            // Straddling the wall between cell 0 and cell 1 → distance ~0.
            let on_wall = cellular_edge(1.0 / cells, 0.5 / cells, params);
            assert!(on_wall < 1e-9, "cells={cells}: on-wall distance {on_wall}");
        }
    }

    /// A one-cell lattice has no walls at all; the degenerate case must not
    /// leak a sentinel or a NaN.
    #[test]
    fn edge_handles_single_cell_lattice() {
        let params = CellularParams::new(1.0, 2);
        let d = cellular_edge(0.3, 0.7, params);
        assert!(d.is_finite() && d > 0.0, "single-cell edge was {d}");
    }

    /// Chebyshev ≤ Euclidean ≤ Manhattan holds for each individual site, so it
    /// survives the minimum regardless of which site wins under each metric.
    #[test]
    fn metrics_are_ordered() {
        for i in 0..17 {
            for j in 0..17 {
                let (u, v) = (i as f64 / 17.0, j as f64 / 17.0);
                let base = CellularParams::new(6.0, 404);
                let cheb = cellular(u, v, base.with_metric(CellMetric::Chebyshev)).f1;
                let eucl = cellular(u, v, base.with_metric(CellMetric::Euclidean)).f1;
                let manh = cellular(u, v, base.with_metric(CellMetric::Manhattan)).f1;
                assert!(
                    cheb <= eucl + 1e-12 && eucl <= manh + 1e-12,
                    "metric order violated at ({u}, {v}): {cheb} / {eucl} / {manh}"
                );
            }
        }
    }

    /// A softmin always undershoots the hard minimum, and tightens toward it
    /// as the falloff grows.
    #[test]
    fn smooth_converges_to_f1() {
        let params = CellularParams::new(6.0, 77);
        for i in 0..13 {
            for j in 0..13 {
                let (u, v) = (i as f64 / 13.0, j as f64 / 13.0);
                let f1 = cellular(u, v, params).f1;
                let soft = cellular_smooth(u, v, params, 16.0);
                let sharper = cellular_smooth(u, v, params, 256.0);
                assert!(soft <= f1 + 1e-12, "softmin exceeded F1 at ({u}, {v})");
                assert!(
                    (sharper - f1).abs() <= (soft - f1).abs() + 1e-12,
                    "higher falloff did not tighten toward F1 at ({u}, {v})"
                );
            }
        }
    }

    /// An extreme falloff underflows every exponential term; the fallback must
    /// still return the hard minimum rather than an infinity.
    #[test]
    fn smooth_survives_extreme_falloff() {
        let params = CellularParams::new(6.0, 5);
        for falloff in [0.0, 1e-9, 1e5, 1e12] {
            let d = cellular_smooth(0.31, 0.62, params, falloff);
            assert!(d.is_finite(), "falloff={falloff} produced {d}");
        }
    }

    /// Degenerate configs are clamped rather than panicking or dividing by
    /// zero, and reported cells always index a real lattice slot.
    #[test]
    fn params_clamp_degenerate_input() {
        for scale in [-4.0, 0.0, 0.4] {
            let s = cellular(0.5, 0.5, CellularParams::new(scale, 1));
            assert!(s.f1.is_finite(), "scale={scale} produced {}", s.f1);
            assert_eq!(
                (s.cell_x, s.cell_y),
                (0, 0),
                "scale={scale} left the lattice"
            );
        }
        // Out-of-range jitter clamps to the [0, 1] window.
        let wild = CellularParams::new(4.0, 1).with_jitter(9.0);
        let full = CellularParams::new(4.0, 1).with_jitter(1.0);
        assert_eq!(cellular(0.3, 0.7, wild).f1, cellular(0.3, 0.7, full).f1);

        let cells = 8;
        let params = CellularParams::new(cells as f64, 12);
        for i in 0..20 {
            let s = cellular(i as f64 / 20.0, 0.42, params);
            assert!(
                (0..cells).contains(&s.cell_x) && (0..cells).contains(&s.cell_y),
                "cell id out of range: {s:?}"
            );
        }
    }

    const PROFILES: [StripeProfile; 4] = [
        StripeProfile::Sine,
        StripeProfile::Triangle,
        StripeProfile::Sawtooth,
        StripeProfile::Square,
    ];

    /// Whole cycle counts exist precisely so the field closes on itself; a
    /// free angle would leave a phase jump down one edge.
    #[test]
    fn stripe_tiles_seamlessly() {
        for profile in PROFILES {
            for (cu, cv) in [(4, 0), (0, 3), (3, 3), (5, -2), (1, 7)] {
                let params = StripeParams::new(cu, cv)
                    .with_profile(profile)
                    .with_sharpness(0.6);
                for t in [0.0, 0.13, 0.5, 0.77] {
                    let (left, right) = (stripe(0.0, t, params, 0.0), stripe(1.0, t, params, 0.0));
                    assert!(
                        (left - right).abs() < 1e-9,
                        "{profile:?} ({cu},{cv}) horizontal seam at v={t}: {left} vs {right}"
                    );
                    let (top, bottom) = (stripe(t, 0.0, params, 0.0), stripe(t, 1.0, params, 0.0));
                    assert!(
                        (top - bottom).abs() < 1e-9,
                        "{profile:?} ({cu},{cv}) vertical seam at u={t}: {top} vs {bottom}"
                    );
                }
            }
        }
    }

    /// Bands must be constant along their own direction and vary across it.
    #[test]
    fn stripe_runs_perpendicular_to_its_cycles() {
        // Vertical bands: constant as v sweeps.
        let vertical = StripeParams::new(4, 0);
        let reference = stripe(0.3, 0.0, vertical, 0.0);
        for v in [0.2, 0.45, 0.9] {
            assert!(
                (stripe(0.3, v, vertical, 0.0) - reference).abs() < 1e-12,
                "vertical bands varied along v"
            );
        }
        assert!(
            (stripe(0.3, 0.0, vertical, 0.0) - stripe(0.42, 0.0, vertical, 0.0)).abs() > 1e-6,
            "vertical bands did not vary across u"
        );

        // Diagonal bands: constant along the anti-diagonal u + v = k.
        let diagonal = StripeParams::new(3, 3);
        let along = stripe(0.1, 0.6, diagonal, 0.0);
        for (u, v) in [(0.2, 0.5), (0.3, 0.4), (0.55, 0.15)] {
            assert!(
                (stripe(u, v, diagonal, 0.0) - along).abs() < 1e-12,
                "diagonal bands varied along their own direction at ({u}, {v})"
            );
        }
    }

    /// Reshaping the wave is the point — the profiles must not collapse into
    /// one another.
    #[test]
    fn profiles_are_distinct() {
        let sample = |profile| {
            (0..32)
                .map(|i| {
                    stripe(
                        i as f64 / 32.0,
                        0.0,
                        StripeParams::new(2, 0).with_profile(profile),
                        0.0,
                    )
                })
                .collect::<Vec<_>>()
        };
        for (a, b) in [
            (StripeProfile::Sine, StripeProfile::Triangle),
            (StripeProfile::Triangle, StripeProfile::Sawtooth),
            (StripeProfile::Sawtooth, StripeProfile::Square),
            (StripeProfile::Square, StripeProfile::Sine),
        ] {
            let (xs, ys) = (sample(a), sample(b));
            let spread = xs
                .iter()
                .zip(&ys)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            assert!(spread > 0.05, "{a:?} and {b:?} produce the same field");
        }
    }

    /// A sharp square wave should sit at its two levels, not ramp between.
    #[test]
    fn square_profile_is_bimodal_when_sharp() {
        let params = StripeParams::new(3, 0)
            .with_profile(StripeProfile::Square)
            .with_sharpness(0.97);
        let samples = 400;
        let midband = (0..samples)
            .map(|i| stripe(i as f64 / samples as f64, 0.0, params, 0.0))
            .filter(|v| (0.1..0.9).contains(v))
            .count();
        assert!(
            midband * 10 < samples,
            "sharp square spent {midband}/{samples} of its range mid-transition"
        );
    }

    /// The sawtooth's hard reset is its defining feature.
    #[test]
    fn sawtooth_ramps_then_resets() {
        let params = StripeParams::new(1, 0).with_profile(StripeProfile::Sawtooth);
        let mut previous = stripe(0.0, 0.0, params, 0.0);
        for i in 1..20 {
            let value = stripe(i as f64 / 20.0, 0.0, params, 0.0);
            assert!(value > previous, "sawtooth did not rise at step {i}");
            previous = value;
        }
        // …and drops back at the wrap rather than easing down.
        assert!(
            stripe(0.999, 0.0, params, 0.0) - stripe(0.001, 0.0, params, 0.0) > 0.9,
            "sawtooth did not reset across the period boundary"
        );
    }

    /// Uniform contrast is the property that separates an oriented field from
    /// stacked FBM, which clusters around its midpoint.
    #[test]
    fn stripe_sweeps_the_full_range() {
        for profile in PROFILES {
            let params = StripeParams::new(5, 0).with_profile(profile);
            let values: Vec<f64> = (0..200)
                .map(|i| stripe(i as f64 / 200.0, 0.37, params, 0.0))
                .collect();
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // Not quite `0..1`: the sawtooth resets exactly at the period
            // boundary, so its supremum is approached but never sampled.
            assert!(
                min < 0.05 && max > 0.95,
                "{profile:?} only spanned {min:.3}..{max:.3}"
            );
        }
    }

    /// Warping bends the bands, and a periodic warp keeps the seam closed.
    #[test]
    fn warp_bends_the_field_but_keeps_the_seam() {
        let params = StripeParams::new(4, 0);
        // A warp that is itself periodic across the tile.
        let warp = |u: f64, v: f64| 0.3 * (TAU * u).sin() * (TAU * v).cos();

        let mut bent = 0;
        for i in 0..40 {
            let (u, v) = (i as f64 / 40.0, 0.3);
            let straight = stripe(u, v, params, 0.0);
            let warped = stripe(u, v, params, warp(u, v));
            if (straight - warped).abs() > 1e-6 {
                bent += 1;
            }
        }
        assert!(bent > 20, "warp barely moved the field ({bent}/40)");

        for v in [0.0, 0.2, 0.65] {
            let left = stripe(0.0, v, params, warp(0.0, v));
            let right = stripe(1.0, v, params, warp(1.0, v));
            assert!(
                (left - right).abs() < 1e-9,
                "warped field broke the seam at v={v}"
            );
        }
    }

    /// Direction and wavelength must describe the field the cycles actually
    /// produce, including the degenerate constant case.
    #[test]
    fn direction_and_wavelength_follow_the_cycles() {
        let horizontal = StripeParams::new(4, 0);
        assert_eq!(horizontal.direction(), Some((1.0, 0.0)));
        assert!((horizontal.wavelength() - 0.25).abs() < 1e-12);

        let diagonal = StripeParams::new(3, 4);
        let (dx, dy) = diagonal.direction().expect("non-constant");
        assert!((dx - 0.6).abs() < 1e-12 && (dy - 0.8).abs() < 1e-12);
        assert!((diagonal.wavelength() - 0.2).abs() < 1e-12);

        let constant = StripeParams::new(0, 0);
        assert_eq!(constant.direction(), None);
        assert!(constant.wavelength().is_infinite());
    }

    /// A constant field and out-of-range sharpness must stay well-defined.
    #[test]
    fn degenerate_stripe_params_stay_finite() {
        let constant = StripeParams::new(0, 0);
        let a = stripe(0.2, 0.8, constant, 0.0);
        let b = stripe(0.9, 0.1, constant, 0.0);
        assert!(
            a.is_finite() && (a - b).abs() < 1e-12,
            "constant field varied"
        );

        for sharpness in [-5.0, 0.0, 1.0, 9.0] {
            for profile in PROFILES {
                let params = StripeParams::new(3, 1)
                    .with_profile(profile)
                    .with_sharpness(sharpness);
                let value = stripe(0.31, 0.62, params, 0.0);
                assert!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "{profile:?} at sharpness {sharpness} produced {value}"
                );
            }
        }
    }

    /// Same seed → same lattice; different seed → different lattice.
    #[test]
    fn cellular_is_seed_deterministic() {
        let a = CellularParams::new(6.0, 1000);
        let b = CellularParams::new(6.0, 1001);
        let mut differed = 0;
        for i in 0..23 {
            let (u, v) = (i as f64 / 23.0, 0.37);
            assert_eq!(
                cellular(u, v, a).f1,
                cellular(u, v, a).f1,
                "not deterministic"
            );
            if (cellular(u, v, a).f1 - cellular(u, v, b).f1).abs() > 1e-9 {
                differed += 1;
            }
        }
        assert!(
            differed > 15,
            "seed barely changed the lattice ({differed}/23)"
        );
    }
}
