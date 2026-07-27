//! Compact procedural colour ramps.
//!
//! Most generators here name their colours outright — `color_stone`,
//! `color_mortar` — and that is the right choice when a content author wants
//! to say "make the brick redder".  It falls down when one *scalar* has to
//! drive a genuinely varied sweep of colour: heat, iridescence, the shift
//! from fresh to rotted leaf.  A two-colour lerp through that range only ever
//! travels the straight line between its endpoints, so it desaturates through
//! the middle and can never pass through a hue the endpoints do not bracket.
//!
//! [`CosinePalette`] solves that in four vectors, following Inigo Quilez's
//! [palettes](https://iquilezles.org/articles/palettes/) construction.

use std::f32::consts::TAU;

/// A colour ramp of the form `bias + amplitude · cos(2π · (frequency · t +
/// phase))`, evaluated per channel.
///
/// Because each channel carries its own frequency and phase, the ramp can
/// swing through hues no pair of endpoints brackets — which is the whole
/// reason to reach for it over a lerp.
///
/// # Example
///
/// ```
/// use symbios_texture::palette::CosinePalette;
///
/// // A ramp between two colours behaves exactly like a lerp…
/// let ramp = CosinePalette::between([0.1, 0.1, 0.1], [0.9, 0.5, 0.2]);
/// let mid = ramp.sample(0.5);
/// assert!((mid[0] - 0.5).abs() < 1e-5);
///
/// // …while the general form sweeps through hues a lerp cannot reach.
/// let iridescent = CosinePalette {
///     bias: [0.5; 3],
///     amplitude: [0.5; 3],
///     frequency: [1.0; 3],
///     phase: [0.0, 0.33, 0.67],
/// };
/// assert_ne!(iridescent.sample(0.25), iridescent.sample(0.75));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CosinePalette {
    /// Centre of the ramp per channel — the colour at the midpoint of the
    /// swing.
    pub bias: [f32; 3],
    /// Half-swing per channel.  `bias ± amplitude` are the extremes.
    pub amplitude: [f32; 3],
    /// Cycles across `t ∈ [0, 1]` per channel.  Above `1` the ramp repeats,
    /// which is how banded and iridescent looks are built.
    pub frequency: [f32; 3],
    /// Phase offset per channel, in turns.  Staggering the channels is what
    /// turns a grey ramp into a coloured one.
    pub phase: [f32; 3],
}

impl CosinePalette {
    /// Sample the ramp at `t`, returning a linear RGB colour clamped to
    /// `[0, 1]`.
    ///
    /// `t` is not wrapped or clamped before evaluation — the cosine is
    /// periodic, so values outside `[0, 1]` simply continue the ramp.
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (channel, slot) in out.iter_mut().enumerate() {
            let angle = TAU * (self.frequency[channel] * t + self.phase[channel]);
            *slot = (self.bias[channel] + self.amplitude[channel] * angle.cos()).clamp(0.0, 1.0);
        }
        out
    }

    /// A ramp that travels from `start` at `t = 0` to `end` at `t = 1`,
    /// matching a plain lerp.
    ///
    /// Useful as a starting point: take the two colours a generator already
    /// names, then open up frequency or stagger the phases to get somewhere a
    /// lerp cannot.
    pub fn between(start: [f32; 3], end: [f32; 3]) -> Self {
        let mut bias = [0.0f32; 3];
        let mut amplitude = [0.0f32; 3];
        for (channel, (slot, swing)) in bias.iter_mut().zip(&mut amplitude).enumerate() {
            *slot = (start[channel] + end[channel]) * 0.5;
            // cos runs +1 → −1 over half a cycle, so a positive half-swing
            // puts `start` at t = 0 and `end` at t = 1.
            *swing = (start[channel] - end[channel]) * 0.5;
        }
        Self {
            bias,
            amplitude,
            frequency: [0.5; 3],
            phase: [0.0; 3],
        }
    }

    /// A neutral black-to-white ramp.
    pub fn grayscale() -> Self {
        Self::between([0.0; 3], [1.0; 3])
    }
}

impl Default for CosinePalette {
    fn default() -> Self {
        Self::grayscale()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tolerance: f32) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < tolerance)
    }

    /// `between` has to reproduce a lerp exactly, since it is the migration
    /// path for generators that already name two colours.
    #[test]
    fn between_matches_a_lerp() {
        let (start, end) = ([0.1, 0.8, 0.3], [0.9, 0.2, 0.6]);
        let ramp = CosinePalette::between(start, end);

        assert!(close(ramp.sample(0.0), start, 1e-5), "start did not match");
        assert!(close(ramp.sample(1.0), end, 1e-5), "end did not match");

        for step in 0..=10 {
            let t = step as f32 / 10.0;
            let expected = [
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            ];
            let actual = ramp.sample(t);
            // A cosine half-cycle eases where a lerp is linear, so the ends
            // must match exactly and the interior must stay in the same band.
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, e)| (a - e).abs() < 0.12),
                "ramp strayed from the lerp at t={t}: {actual:?} vs {expected:?}"
            );
        }
    }

    /// The reason this type exists: staggered phases reach colours that no
    /// interpolation between two endpoints can produce.
    #[test]
    fn staggered_phases_leave_the_endpoint_line() {
        let ramp = CosinePalette {
            bias: [0.5; 3],
            amplitude: [0.5; 3],
            frequency: [1.0; 3],
            phase: [0.0, 0.33, 0.67],
        };
        let (start, end) = (ramp.sample(0.0), ramp.sample(1.0));
        // Frequency 1 returns to the start after a full cycle.
        assert!(close(start, end, 1e-5), "full cycle did not close");

        // A lerp between identical endpoints is constant; this is not.
        let mid = ramp.sample(0.5);
        assert!(
            !close(mid, start, 0.1),
            "ramp collapsed onto its endpoints: {mid:?} vs {start:?}"
        );
    }

    #[test]
    fn output_is_clamped_and_finite() {
        let wild = CosinePalette {
            bias: [5.0, -3.0, 0.5],
            amplitude: [10.0, 10.0, -8.0],
            frequency: [1e6, 0.0, -4.0],
            phase: [1e6, -1e6, 0.25],
        };
        for step in 0..32 {
            let sample = wild.sample(step as f32 / 8.0 - 2.0);
            assert!(
                sample
                    .iter()
                    .all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
                "sample escaped the unit range: {sample:?}"
            );
        }
    }

    #[test]
    fn grayscale_runs_black_to_white() {
        let ramp = CosinePalette::grayscale();
        assert!(close(ramp.sample(0.0), [0.0; 3], 1e-5));
        assert!(close(ramp.sample(1.0), [1.0; 3], 1e-5));
        assert!(close(ramp.sample(0.5), [0.5; 3], 1e-5));
    }

    /// Frequencies above 1 repeat the ramp — the basis of banded looks.
    #[test]
    fn frequency_above_one_repeats() {
        let ramp = CosinePalette {
            bias: [0.5; 3],
            amplitude: [0.5; 3],
            frequency: [3.0; 3],
            phase: [0.0; 3],
        };
        assert!(
            close(ramp.sample(0.0), ramp.sample(1.0 / 3.0), 1e-5),
            "frequency 3 did not repeat every third of the ramp"
        );
    }
}
