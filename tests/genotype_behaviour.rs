//! A blessed fingerprint of every config's [`Genotype`] behaviour.
//!
//! `mutate` and `crossover` are generated from a per-field table. Any change
//! to how that table is *authored* — including moving it behind a shared
//! registry — must leave the generated behaviour bit for bit identical: the
//! same field kinds, the same half-ranges, the same clamps, and above all the
//! same sequence of RNG draws. Field order is load-bearing, not cosmetic:
//! `mutate` draws once per field in declaration order, so a reordered table
//! silently gives every seeded search a different population.
//!
//! Each case walks a config from its default to its clamp bounds under a fixed
//! `StdRng` (rate 1.0 pins every arm's clamp; rate 0.35 pins the rate gate's
//! own draw), crosses the two walks (pinning the 50/50 pick order and the
//! per-channel colour crossover), and hashes the JSON of all three.
//!
//! Re-bless with `GENOTYPE_BLESS=1 cargo test --test genotype_behaviour`, and
//! only when the behaviour is *meant* to move — re-blessing to make a port
//! pass defeats the point.

use std::fmt::Write as _;

use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use symbios_genetics::Genotype;
use symbios_texture::for_each_generator;

/// Where the blessed lines live, relative to the crate root.
const FIXTURE: &str = "tests/fixtures/genotype_behaviour.txt";

/// FNV-1a over the JSON, so the fixture needs no hashing dependency.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Walk `steps` mutations from the default under a fixed seed.
fn walk<T: Genotype + Default>(seed: u64, steps: usize, rate: f32) -> T {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = T::default();
    for _ in 0..steps {
        cfg.mutate(&mut rng, rate);
    }
    cfg
}

/// One fixture line: the two walks and their child, hashed together.
fn fingerprint<T: Genotype + Default + Serialize>(label: &str) -> String {
    let a: T = walk(0x00C0_FFEE, 200, 1.0);
    let b: T = walk(0x0BAD_5EED, 200, 0.35);
    let child = a.crossover(&b, &mut StdRng::seed_from_u64(0x5EED_1304));
    let json = serde_json::to_string(&(&a, &b, &child)).expect("config serialises");
    format!("{label} {:016x}", fnv1a_64(json.as_bytes()))
}

/// One `fingerprint` call per registry row; `for_each_generator!` supplies
/// them, so a new generator joins this gate without a second hand list.
macro_rules! generator_cases {
    ( $( ( $Variant:ident, $module:ident, $Config:ty, $Gen:ty, $Kind:ident ) ),+ $(,)? ) => {
        vec![ $( fingerprint::<$Config>(stringify!($Variant)) ),+ ]
    };
}

/// The weathering layers have `Genotype` impls of their own but no registry
/// row (they are embedded, not spawned), so they are named here.
fn weathering_cases() -> Vec<String> {
    use symbios_texture::weathering::{
        Corrosion, CreviceDirt, EdgeWear, Streaks, WeatheringConfig,
    };
    vec![
        fingerprint::<WeatheringConfig>("WeatheringConfig"),
        fingerprint::<EdgeWear>("EdgeWear"),
        fingerprint::<Corrosion>("Corrosion"),
        fingerprint::<CreviceDirt>("CreviceDirt"),
        fingerprint::<Streaks>("Streaks"),
    ]
}

fn all_cases() -> Vec<String> {
    let mut lines: Vec<String> = for_each_generator!(generator_cases);
    lines.extend(weathering_cases());
    lines
}

#[test]
fn genotype_behaviour_is_unchanged() {
    let lines = all_cases();
    let mut rendered = String::new();
    for line in &lines {
        writeln!(rendered, "{line}").expect("string write");
    }

    if std::env::var("GENOTYPE_BLESS").is_ok() {
        std::fs::create_dir_all("tests/fixtures").expect("fixture dir");
        std::fs::write(FIXTURE, &rendered).expect("write fixture");
        return;
    }

    let blessed = std::fs::read_to_string(FIXTURE).unwrap_or_else(|e| {
        panic!("{FIXTURE} missing ({e}) — bless it with GENOTYPE_BLESS=1 on the OLD code")
    });
    let expected: Vec<&str> = blessed.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        expected.len(),
        lines.len(),
        "config count moved: {} blessed, {} generated",
        expected.len(),
        lines.len()
    );
    let moved: Vec<String> = expected
        .iter()
        .zip(&lines)
        .filter(|(want, got)| **want != got.as_str())
        .map(|(want, got)| format!("  blessed {want}\n  got     {got}"))
        .collect();
    assert!(
        moved.is_empty(),
        "{} config(s) changed their mutate/crossover behaviour:\n{}",
        moved.len(),
        moved.join("\n")
    );
}

/// The fixture is only evidence while it covers every config that has a
/// `Genotype` impl.
#[test]
fn every_config_is_fingerprinted() {
    assert_eq!(
        all_cases().len(),
        62,
        "62 configs implement Genotype: 57 registry rows + 5 weathering structs"
    );
}
