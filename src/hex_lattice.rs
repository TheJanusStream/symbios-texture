//! The flat-top hexagonal lattice on the unit torus, shared by the hexagonal
//! paver layout and the hex-tile blend operator.
//!
//! One home for the rule that makes it tile.  `rows` hex rows fit `[0, 1]`
//! vertically by choice of circumradius, and the column count is the natural
//! float count rounded to the nearest **even** integer: a whole-tile step
//! sideways maps `(q, r)` to `(q + cols, r − cols / 2)`, which is a lattice
//! vector only when `cols` is even.  The measurement behind that, and the
//! exact-arithmetic form of the axial coordinates, are documented on
//! `pavers::hex_cell`.

pub(crate) const SQRT3: f64 = 1.732_050_807_568_877_3;

/// A tiling flat-top hex lattice over `[0, 1]²`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HexLattice {
    /// Hex rows down the tile — the rounded scale, at least one.
    pub rows: f64,
    /// Hex columns across the tile; always even.
    pub cols: f64,
    /// Circumradius in V units.  The lattice is scaled in U so that `cols`
    /// columns fit exactly, which leaves the hexes slightly non-regular.
    pub hex_r: f64,
}

impl HexLattice {
    pub(crate) fn new(scale: f64) -> Self {
        // Vertical tiling requires an integer number of rows; round scale.
        let rows = scale.round().max(1.0);
        // Circumradius so that `rows` hex rows fit exactly across [0, 1].
        let hex_r = 1.0 / (rows * SQRT3);
        // Column count: the natural float value is `1 / (1.5 * hex_r)`,
        // rounded to the nearest even integer.
        let cols = ((rows * SQRT3 / 1.5) * 0.5).round().max(1.0) * 2.0;
        Self { rows, cols, hex_r }
    }

    /// Fractional axial coordinates of `(u, v)` (flat-top convention),
    /// written in terms of the row and column counts rather than through
    /// `hex_r`, so that a whole-tile step moves them by exactly `cols` and
    /// `−cols / 2` — both integers — and every seam identity holds bit for
    /// bit.
    #[inline]
    pub(crate) fn axial(&self, u: f64, v: f64) -> (f64, f64) {
        let qf = u * self.cols;
        (qf, v * self.rows - 0.5 * qf)
    }

    /// The canonical representative of cell `(q, r)` on the torus: undo `n`
    /// whole-tile steps in U, then reduce V.
    #[inline]
    pub(crate) fn canonical(&self, q: i64, r: i64) -> (i64, i64) {
        let (cols, rows) = (self.cols as i64, self.rows as i64);
        let n = q.div_euclid(cols);
        (q.rem_euclid(cols), (r + n * (cols / 2)).rem_euclid(rows))
    }

    /// The lattice triangle containing axial `(qf, rf)`: its three vertices
    /// and the barycentric weight of each.
    ///
    /// The axial unit cell is a 60° parallelogram; its short diagonal splits
    /// it into two equilateral triangles, and `fa + fb < 1` picks the lower
    /// one.
    #[inline]
    pub(crate) fn triangle(&self, qf: f64, rf: f64) -> ([(i64, i64); 3], [f64; 3]) {
        let (fi, fj) = (qf.floor(), rf.floor());
        let (fa, fb) = (qf - fi, rf - fj);
        let (i, j) = (fi as i64, fj as i64);
        if fa + fb < 1.0 {
            ([(i, j), (i + 1, j), (i, j + 1)], [1.0 - fa - fb, fa, fb])
        } else {
            (
                [(i + 1, j + 1), (i, j + 1), (i + 1, j)],
                [fa + fb - 1.0, 1.0 - fa, 1.0 - fb],
            )
        }
    }
}

/// Cube-coordinate rounding (standard hex-grid algorithm).
#[inline]
pub(crate) fn cube_round(qf: f64, rf: f64, sf: f64) -> (i64, i64, i64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_column_count_is_always_even() {
        for scale in 1..=32 {
            let lat = HexLattice::new(f64::from(scale));
            assert_eq!(lat.cols as i64 % 2, 0, "scale {scale}: cols {}", lat.cols);
            assert!(lat.cols >= 2.0);
        }
    }

    #[test]
    fn barycentric_weights_sum_to_one_and_name_their_vertex() {
        let lat = HexLattice::new(5.0);
        for i in 0..64 {
            for j in 0..64 {
                let (qf, rf) = (f64::from(i) * 0.137, f64::from(j) * 0.091);
                let (verts, w) = lat.triangle(qf, rf);
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(w.iter().all(|&x| (-1e-12..=1.0 + 1e-12).contains(&x)));
                // The weighted vertices reconstruct the point.
                let q: f64 = verts.iter().zip(&w).map(|(v, w)| v.0 as f64 * w).sum();
                let r: f64 = verts.iter().zip(&w).map(|(v, w)| v.1 as f64 * w).sum();
                assert!((q - qf).abs() < 1e-9 && (r - rf).abs() < 1e-9);
            }
        }
    }
}
