//! The envelope: the range every config field is allowed to hold, and one
//! operation that puts a config back inside it.
//!
//! A generator's fields are bounded three times over in a typical
//! application — once by the search that evolves them, once by the slider a
//! person drags, and once by whatever an application does to a config that
//! arrives from somewhere it does not trust. Those three used to be three
//! hand-written tables that disagreed. The envelope is the single answer:
//! the ranges in
//! [`for_each_texture_field!`](crate::for_each_texture_field) bound the
//! record, and [`ClampToEnvelope`] applies them.
//!
//! # What it is for
//!
//! Every count-shaped field here is an inner-loop bound for a per-pixel
//! kernel. An octave count of four billion is not a strange-looking texture,
//! it is a wedged worker. An application decoding a config from an untrusted
//! peer should call [`ClampToEnvelope::clamp_to_envelope`] before handing it
//! to a generator, and then it needs no cost bounds of its own.
//!
//! # What it deliberately is not
//!
//! * **It is not the mutation range.** Some fields are evolved inside a band
//!   narrower than the envelope, because a search that can reach a mortar
//!   width of zero spends its time on brick walls with no mortar. Those rows
//!   carry an `explore` clause; a person is still free to author the
//!   degenerate value, so the envelope admits it.
//! * **It does not normalise.** Rounding a field a stepped slider would have
//!   stepped, or re-snapping a tiling invariant, would rewrite values a
//!   person authored on purpose. Clamping an in-envelope config is a no-op,
//!   bit for bit, and `envelope_clamp_is_a_no_op_inside_the_envelope` pins
//!   that.
//! * **It is not a schema check.** A field's *type* already constrains it;
//!   the envelope only narrows the numeric range.
//!
//! NaN is the one non-range case handled here: it belongs to no range, and
//! `f64::clamp` propagates it rather than resolving it, so it is mapped to
//! the envelope minimum. Infinities need no special case — they clamp.

use crate::for_each_texture_field;

/// Put every field of a config back inside its envelope.
///
/// Implemented for every config in
/// [`for_each_texture_field!`](crate::for_each_texture_field), including the
/// weathering layers, and recursing into nested configs.
pub trait ClampToEnvelope {
    /// Clamp each numeric field to the range the registry gives it.
    ///
    /// Idempotent, and a no-op on a config that is already inside.
    fn clamp_to_envelope(&mut self);
}

/// Clamp with NaN resolved to `min`; `f64::clamp` alone propagates NaN.
#[inline]
fn clamp_f64(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() {
        min
    } else {
        value.clamp(min, max)
    }
}

/// `f32` counterpart of [`clamp_f64`].
#[inline]
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value.is_nan() {
        min
    } else {
        value.clamp(min, max)
    }
}

/// Generates [`ClampToEnvelope`] for every config in the per-field registry.
///
/// Kinds with no numeric range — `seed`, `bool`, `enum_pick` — expand to
/// nothing; `nested` recurses; the rest clamp. The `explore` clause, the UI
/// columns and the `layout` list are matched and discarded.
macro_rules! impl_clamp_to_envelope {
    (
        $(
            $Config:ty, $header:literal, $editor:ident
            $(, fixup $fixup:ident )?
            { $(
                $kind:ident ( $field:ident $($rest:tt)* )
                $( explore ( $xlo:expr, $xhi:expr ) )?
            ),+ $(,)? }
            $( layout { $($layout:tt)* } )?
        ),+ $(,)?
    ) => {
        $(
            impl ClampToEnvelope for $Config {
                fn clamp_to_envelope(&mut self) {
                    $( impl_clamp_to_envelope!(@row self, $kind ($field $($rest)*)); )+
                }
            }
        )+
    };

    (@row $s:ident, f64 ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
        $s.$f = clamp_f64($s.$f, $min, $max);
    };
    (@row $s:ident, f64_round ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
        $s.$f = clamp_f64($s.$f, $min, $max);
    };
    (@row $s:ident, f32 ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
        $s.$f = clamp_f32($s.$f, $min, $max);
    };
    (@row $s:ident, usize ($f:ident, $l:tt, $min:expr, $max:expr)) => {
        $s.$f = $s.$f.clamp($min, $max);
    };
    (@row $s:ident, color3 ($f:ident, $l:tt, $half:expr)) => {
        for channel in $s.$f.iter_mut() {
            *channel = clamp_f32(*channel, 0.0, 1.0);
        }
    };
    (@row $s:ident, nested ($f:ident, $sub:ty, $editor:ident, $salt:literal)) => {
        $s.$f.clamp_to_envelope();
    };
    // Kinds a range cannot narrow: any `u32` is a valid seed, any `bool` a
    // valid flag, and an enum can only hold a variant it declares.
    (@row $s:ident, seed ($f:ident, $l:tt)) => {};
    (@row $s:ident, bool ($f:ident, $l:tt)) => {};
    (@row $s:ident, enum_pick ($f:ident, $l:tt, [ $(($v:expr, $btn:literal)),+ ])) => {};
}

for_each_texture_field!(impl_clamp_to_envelope);

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};
    use symbios_genetics::Genotype;

    use super::*;

    /// Every numeric row in the registry, pushed past both of its bounds and
    /// handed a NaN. Generated from the same rows the clamp is, so a field
    /// added tomorrow is covered without a second list.
    #[test]
    fn envelope_clamps_every_numeric_row() {
        let mut checked = 0usize;

        macro_rules! check_rows {
            (
                $(
                    $Config:ty, $header:literal, $editor:ident
                    $(, fixup $fixup:ident )?
                    { $(
                        $kind:ident ( $field:ident $($rest:tt)* )
                        $( explore ( $xlo:expr, $xhi:expr ) )?
                    ),+ $(,)? }
                    $( layout { $($layout:tt)* } )?
                ),+ $(,)?
            ) => {
                $( $( check_rows!(@check $Config, $kind ($field $($rest)*)); )+ )+
            };

            (@check $C:ty, f64 ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
                check_rows!(@float $C, $f, $min, $max, f64);
            };
            (@check $C:ty, f64_round ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
                check_rows!(@float $C, $f, $min, $max, f64);
            };
            (@check $C:ty, f32 ($f:ident, $l:tt, $min:expr, $max:expr, $half:expr)) => {
                check_rows!(@float $C, $f, $min, $max, f32);
            };
            (@float $C:ty, $f:ident, $min:expr, $max:expr, $ty:ident) => {{
                let name = concat!(stringify!($C), ".", stringify!($f));
                let mut low = <$C>::default();
                low.$f = $ty::MIN;
                low.clamp_to_envelope();
                assert_eq!(low.$f, $min, "{name} below the envelope");
                let mut high = <$C>::default();
                high.$f = $ty::MAX;
                high.clamp_to_envelope();
                assert_eq!(high.$f, $max, "{name} above the envelope");
                let mut nan = <$C>::default();
                nan.$f = $ty::NAN;
                nan.clamp_to_envelope();
                assert_eq!(nan.$f, $min, "{name} given a NaN");
                checked += 1;
            }};
            (@check $C:ty, usize ($f:ident, $l:tt, $min:expr, $max:expr)) => {{
                let name = concat!(stringify!($C), ".", stringify!($f));
                let mut low = <$C>::default();
                low.$f = usize::MIN;
                low.clamp_to_envelope();
                assert_eq!(low.$f, $min, "{name} below the envelope");
                let mut high = <$C>::default();
                high.$f = usize::MAX;
                high.clamp_to_envelope();
                assert_eq!(high.$f, $max, "{name} above the envelope");
                checked += 1;
            }};
            (@check $C:ty, color3 ($f:ident, $l:tt, $half:expr)) => {{
                let name = concat!(stringify!($C), ".", stringify!($f));
                let mut cfg = <$C>::default();
                cfg.$f = [-5.0, 2.0, f32::NAN];
                cfg.clamp_to_envelope();
                assert_eq!(cfg.$f, [0.0, 1.0, 0.0], "{name} out of the unit cube");
                checked += 1;
            }};
            // Kinds a range cannot narrow, and nested configs, which are
            // covered by their own registry entry.
            (@check $C:ty, seed ($f:ident, $l:tt)) => {};
            (@check $C:ty, bool ($f:ident, $l:tt)) => {};
            (@check $C:ty, enum_pick ($f:ident, $l:tt, [ $(($v:expr, $b:literal)),+ ])) => {};
            (@check $C:ty, nested ($f:ident, $sub:ty, $ed:ident, $salt:literal)) => {};
        }

        for_each_texture_field!(check_rows);

        assert_eq!(
            checked, 585,
            "585 numeric rows: 272 f64 + 126 color3 + 97 f32 + 67 usize + 23 f64_round. \
             A row that stopped being checked means an arm matched the wrong kind."
        );
    }

    /// One `Vec` entry per config, so the whole-config properties below can
    /// be asserted without a third hand-written roster.
    fn every_config_clamped_twice() -> Vec<(&'static str, String, String, String)> {
        let mut out = Vec::new();

        macro_rules! collect {
            (
                $(
                    $Config:ty, $header:literal, $editor:ident
                    $(, fixup $fixup:ident )?
                    { $( $kind:ident ( $field:ident $($rest:tt)* )
                         $( explore ( $xlo:expr, $xhi:expr ) )? ),+ $(,)? }
                    $( layout { $($layout:tt)* } )?
                ),+ $(,)?
            ) => {
                $({
                    let json = |c: &$Config| serde_json::to_string(c).expect("serialises");

                    // A default config, and a mutated one — mutation stays
                    // inside the explore band, which the envelope contains.
                    let mut rng = StdRng::seed_from_u64(0xE14E_10DE);
                    let mut evolved = <$Config>::default();
                    for _ in 0..64 {
                        evolved.mutate(&mut rng, 1.0);
                    }
                    let mut default_cfg = <$Config>::default();
                    let before_default = json(&default_cfg);
                    let before_evolved = json(&evolved);
                    default_cfg.clamp_to_envelope();
                    evolved.clamp_to_envelope();

                    // And a config clamped twice, from a deliberately hostile
                    // starting point: every field of the mutated one shoved
                    // outward by the clamp is already at a bound.
                    let mut twice = evolved.clone();
                    twice.clamp_to_envelope();

                    out.push((
                        stringify!($Config),
                        format!("{before_default}\u{1}{}", json(&default_cfg)),
                        format!("{before_evolved}\u{1}{}", json(&evolved)),
                        format!("{}\u{1}{}", json(&evolved), json(&twice)),
                    ));
                })+
            };
        }

        for_each_texture_field!(collect);
        out
    }

    /// Clamping a config that is already inside its envelope must not move a
    /// single bit — otherwise loading a record would silently rewrite it.
    #[test]
    fn envelope_clamp_is_a_no_op_inside_the_envelope() {
        for (name, default_pair, evolved_pair, _) in every_config_clamped_twice() {
            let (before, after) = default_pair.split_once('\u{1}').expect("pair");
            assert_eq!(before, after, "{name}: clamping a default config moved it");
            let (before, after) = evolved_pair.split_once('\u{1}').expect("pair");
            assert_eq!(before, after, "{name}: clamping an evolved config moved it");
        }
    }

    /// A second clamp must change nothing — the downstream round trip clamps,
    /// quantises and clamps again.
    #[test]
    fn envelope_clamp_is_idempotent() {
        for (name, _, _, twice_pair) in every_config_clamped_twice() {
            let (once, twice) = twice_pair.split_once('\u{1}').expect("pair");
            assert_eq!(once, twice, "{name}: clamping twice differs from once");
        }
    }

    /// Every config in the registry reaches the assertions above.
    #[test]
    fn every_config_has_an_envelope() {
        assert_eq!(every_config_clamped_twice().len(), 62);
    }
}
