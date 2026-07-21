//! Canonical generator registry — the single source of truth for the
//! generator roster.
//!
//! Each row is `(Variant, module, ConfigType, GeneratorType, Kind)`:
//!
//! * `Variant` — the `TextureConfig` enum (defined in the
//!   `bevy_symbios_texture` wrapper crate) variant name; doubles as the
//!   UI / cache-key label.
//! * `module` — the crate module name; doubles as the `PendingTexture`
//!   (wrapper crate) constructor name.
//! * `ConfigType` / `GeneratorType` — full paths to the config struct and
//!   its [`TextureGenerator`](crate::generator::TextureGenerator) impl.
//! * `Kind` — `Surface` (tileable, repeat sampler, opaque) or `Card`
//!   (alpha-masked, clamp-to-edge sampler, alpha-blended/masked).
//!
//! Consumers invoke [`for_each_generator!`](crate::for_each_generator) with a
//! callback macro that receives every row:
//!
//! * `define_texture_config` (`material.rs`) — the `TextureConfig` enum,
//!   labels, render properties, spawn dispatch, and cache fingerprints.
//! * `define_pending_constructors` (`async_gen.rs`) — the per-generator
//!   `PendingTexture` constructors.
//!
//! # Adding a generator
//!
//! 1. Write the module: config struct (serde + `Default`) and a
//!    `TextureGenerator` impl, with tests.
//! 2. Add one row to the table below (pick `Surface` or `Card`).
//! 3. Add the per-field tables: `impl_genotype!` in `genetics.rs` and
//!    `impl_config_editor!` in `ui.rs` (behind the `egui` feature).
//! 4. Document the config in the README.
//!
//! Everything else — async constructor, enum variant, label, render
//! properties, dispatch, and fingerprint — derives from the row.

/// Invoke `$callback!` with the full generator table (see module docs for
/// the row format).
#[macro_export]
macro_rules! for_each_generator {
    ($callback:ident) => {
        $callback! {
            (Leaf, leaf, $crate::leaf::LeafConfig, $crate::leaf::LeafGenerator, Card),
            (Twig, twig, $crate::twig::TwigConfig, $crate::twig::TwigGenerator, Card),
            (Bark, bark, $crate::bark::BarkConfig, $crate::bark::BarkGenerator, Surface),
            (Window, window, $crate::window::WindowConfig, $crate::window::WindowGenerator, Card),
            (StainedGlass, stained_glass, $crate::stained_glass::StainedGlassConfig, $crate::stained_glass::StainedGlassGenerator, Card),
            (IronGrille, iron_grille, $crate::iron_grille::IronGrilleConfig, $crate::iron_grille::IronGrilleGenerator, Card),
            (ChainLink, chain_link, $crate::chain_link::ChainLinkConfig, $crate::chain_link::ChainLinkGenerator, Card),
            (LogEnd, log_end, $crate::log_end::LogEndConfig, $crate::log_end::LogEndGenerator, Card),
            (Ground, ground, $crate::ground::GroundConfig, $crate::ground::GroundGenerator, Surface),
            (Rock, rock, $crate::rock::RockConfig, $crate::rock::RockGenerator, Surface),
            (Brick, brick, $crate::brick::BrickConfig, $crate::brick::BrickGenerator, Surface),
            (Plank, plank, $crate::plank::PlankConfig, $crate::plank::PlankGenerator, Surface),
            (Shingle, shingle, $crate::shingle::ShingleConfig, $crate::shingle::ShingleGenerator, Surface),
            (Stucco, stucco, $crate::stucco::StuccoConfig, $crate::stucco::StuccoGenerator, Surface),
            (Concrete, concrete, $crate::concrete::ConcreteConfig, $crate::concrete::ConcreteGenerator, Surface),
            (Metal, metal, $crate::metal::MetalConfig, $crate::metal::MetalGenerator, Surface),
            (Pavers, pavers, $crate::pavers::PaversConfig, $crate::pavers::PaversGenerator, Surface),
            (Ashlar, ashlar, $crate::ashlar::AshlarConfig, $crate::ashlar::AshlarGenerator, Surface),
            (Cobblestone, cobblestone, $crate::cobblestone::CobblestoneConfig, $crate::cobblestone::CobblestoneGenerator, Surface),
            (Thatch, thatch, $crate::thatch::ThatchConfig, $crate::thatch::ThatchGenerator, Surface),
            (Marble, marble, $crate::marble::MarbleConfig, $crate::marble::MarbleGenerator, Surface),
            (Corrugated, corrugated, $crate::corrugated::CorrugatedConfig, $crate::corrugated::CorrugatedGenerator, Surface),
            (Asphalt, asphalt, $crate::asphalt::AsphaltConfig, $crate::asphalt::AsphaltGenerator, Surface),
            (Wainscoting, wainscoting, $crate::wainscoting::WainscotingConfig, $crate::wainscoting::WainscotingGenerator, Surface),
            (Encaustic, encaustic, $crate::encaustic::EncausticConfig, $crate::encaustic::EncausticGenerator, Surface),
            (Fabric, fabric, $crate::fabric::FabricConfig, $crate::fabric::FabricGenerator, Surface),
            (Sand, sand, $crate::sand::SandConfig, $crate::sand::SandGenerator, Surface),
            (Snow, snow, $crate::snow::SnowConfig, $crate::snow::SnowGenerator, Surface),
            (Ice, ice, $crate::ice::IceConfig, $crate::ice::IceGenerator, Surface),
            (Lava, lava, $crate::lava::LavaConfig, $crate::lava::LavaGenerator, Surface),
            (SoftDisc, soft_disc, $crate::soft_disc::SoftDiscConfig, $crate::soft_disc::SoftDiscGenerator, Card),
            (Spark, spark, $crate::spark::SparkConfig, $crate::spark::SparkGenerator, Card),
            (Snowflake, snowflake, $crate::snowflake::SnowflakeConfig, $crate::snowflake::SnowflakeGenerator, Card),
            (Puff, puff, $crate::puff::PuffConfig, $crate::puff::PuffGenerator, Card),
            (Ring, ring, $crate::ring::RingConfig, $crate::ring::RingGenerator, Card),
            (Petal, petal, $crate::petal::PetalConfig, $crate::petal::PetalGenerator, Card),
            (Shard, shard, $crate::shard::ShardConfig, $crate::shard::ShardGenerator, Card),
            (LeafSprite, leaf_sprite, $crate::leaf_sprite::LeafSpriteConfig, $crate::leaf_sprite::LeafSpriteGenerator, Card),
            (Flame, flame, $crate::flame::FlameConfig, $crate::flame::FlameGenerator, Card),
            (Flower, flower, $crate::flower::FlowerConfig, $crate::flower::FlowerGenerator, Card),
            (GrassTuft, grass, $crate::grass::GrassTuftConfig, $crate::grass::GrassTuftGenerator, Card),
            (Frond, frond, $crate::frond::FrondConfig, $crate::frond::FrondGenerator, Card),
            (CactusSkin, cactus, $crate::cactus::CactusSkinConfig, $crate::cactus::CactusSkinGenerator, Surface),
            (Moss, moss, $crate::moss::MossConfig, $crate::moss::MossGenerator, Surface),
            (Lichen, lichen, $crate::lichen::LichenConfig, $crate::lichen::LichenGenerator, Surface),
            (Reed, reed, $crate::reed::ReedConfig, $crate::reed::ReedGenerator, Card),
            (Needle, needle, $crate::needle::NeedleConfig, $crate::needle::NeedleGenerator, Card),
            (Broadleaf, broadleaf, $crate::broadleaf::BroadleafConfig, $crate::broadleaf::BroadleafGenerator, Card),
        }
    };
}

// `#[macro_export]` places `for_each_generator!` at the core crate root, so it
// is reachable as `symbios_texture::for_each_generator!` from the
// `bevy_symbios_texture` wrapper (its only consumer) and as
// `crate::for_each_generator!` within this crate.
