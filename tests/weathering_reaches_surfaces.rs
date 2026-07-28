//! Every built surface must be able to age, and must be untouched until it is
//! asked to.
//!
//! Both halves matter. Routing a generator through the weathered driver is
//! easy to do without actually passing its config through — the code compiles
//! and the golden hashes still pass, because a no-op config is indeed a no-op.
//! The only thing that catches a mis-wired generator is turning a layer *on*
//! and checking something moved.

use symbios_texture::{
    generator::TextureGenerator,
    weathering::{Corrosion, CreviceDirt, EdgeWear, Streaks, WeatheringConfig},
};

/// Every layer at once, so a generator fails this regardless of which part of
/// its surface the weathering happens to land on.
fn heavy() -> WeatheringConfig {
    WeatheringConfig {
        seed: 5,
        edge_wear: EdgeWear {
            amount: 1.0,
            ..Default::default()
        },
        corrosion: Corrosion {
            amount: 1.0,
            coverage: 0.45,
            ..Default::default()
        },
        crevice_dirt: CreviceDirt {
            amount: 1.0,
            ..Default::default()
        },
        streaks: Streaks {
            amount: 1.0,
            density: 0.8,
            ..Default::default()
        },
    }
}

/// Bake each generator twice — untouched, and with every weathering layer on —
/// and assert the pair differs.
macro_rules! assert_surfaces_age {
    ($( ($module:ident, $cfg:ident, $generator:ident) ),+ $(,)?) => {
        $({
            use symbios_texture::$module::{$cfg, $generator};

            let plain = $generator::new($cfg::default())
                .generate(96, 96)
                .expect("plain bake");
            let aged = $generator::new($cfg {
                weathering: heavy(),
                ..Default::default()
            })
            .generate(96, 96)
            .expect("aged bake");

            assert_ne!(
                plain.albedo,
                aged.albedo,
                concat!(stringify!($cfg), " ignored its weathering config")
            );

            // A no-op config must leave the surface exactly as it was, which
            // is what makes the field safe to add to an existing generator.
            let untouched = $generator::new($cfg {
                weathering: WeatheringConfig::default(),
                ..Default::default()
            })
            .generate(96, 96)
            .expect("untouched bake");
            assert_eq!(
                plain.albedo,
                untouched.albedo,
                concat!(stringify!($cfg), " changed under a no-op weathering config")
            );
        })+
    };
}

#[test]
fn every_built_surface_can_age() {
    assert_surfaces_age![
        (ashlar, AshlarConfig, AshlarGenerator),
        (asphalt, AsphaltConfig, AsphaltGenerator),
        (brick, BrickConfig, BrickGenerator),
        (cobblestone, CobblestoneConfig, CobblestoneGenerator),
        (concrete, ConcreteConfig, ConcreteGenerator),
        (corrugated, CorrugatedConfig, CorrugatedGenerator),
        (encaustic, EncausticConfig, EncausticGenerator),
        (fabric, FabricConfig, FabricGenerator),
        (marble, MarbleConfig, MarbleGenerator),
        (metal, MetalConfig, MetalGenerator),
        (pavers, PaversConfig, PaversGenerator),
        (shingle, ShingleConfig, ShingleGenerator),
        (stucco, StuccoConfig, StuccoGenerator),
        (thatch, ThatchConfig, ThatchGenerator),
        (wainscoting, WainscotingConfig, WainscotingGenerator),
    ];
}

/// The surfaces that already carried a weathering block, kept here so a
/// refactor cannot quietly drop one.
#[test]
fn the_original_weatherable_surfaces_still_age() {
    assert_surfaces_age![
        (rock, RockConfig, RockGenerator),
        (enamel, EnamelConfig, EnamelGenerator),
        (obsidian, ObsidianConfig, ObsidianGenerator),
        (chitin, ChitinConfig, ChitinGenerator),
        (solar_panel, SolarPanelConfig, SolarPanelGenerator),
        (parquet, ParquetConfig, ParquetGenerator),
        (truchet, TruchetConfig, TruchetGenerator),
    ];
}
