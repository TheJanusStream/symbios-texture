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
//! # The second table
//!
//! [`for_each_texture_field!`](crate::for_each_texture_field) is the
//! per-*field* companion: every field of every config, once, with its
//! envelope, its mutation step, and its UI label. It generates the
//! [`Genotype`](symbios_genetics::Genotype) impls and
//! [`clamp_to_envelope`](crate::envelope::ClampToEnvelope::clamp_to_envelope)
//! here, the config editors in the `bevy_symbios_texture` wrapper, and — for
//! an application that mirrors these configs onto a wire format — its mirror
//! and its record sanitiser. Four hand-synced copies of the same field list
//! used to disagree; a range is now authored once.
//!
//! # Adding a generator
//!
//! 1. Write the module: config struct (serde + `Default`) and a
//!    `TextureGenerator` impl, with tests.
//! 2. Add one row to the table below (pick `Surface` or `Card`).
//! 3. Add one entry per field to
//!    [`for_each_texture_field!`](crate::for_each_texture_field).
//! 4. Add the generator to the roster in `README.md`.
//!
//! Everything else — async constructor, enum variant, label, render
//! properties, dispatch, fingerprint, mutation, crossover, the envelope
//! clamp, and the editor panel — derives from those two rows.

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
            (CrackedEarth, cracked_earth, $crate::cracked_earth::CrackedEarthConfig, $crate::cracked_earth::CrackedEarthGenerator, Surface),
            (Gravel, gravel, $crate::gravel::GravelConfig, $crate::gravel::GravelGenerator, Surface),
            (ForestFloor, forest_floor, $crate::forest_floor::ForestFloorConfig, $crate::forest_floor::ForestFloorGenerator, Surface),
            (Enamel, enamel, $crate::enamel::EnamelConfig, $crate::enamel::EnamelGenerator, Surface),
            (Obsidian, obsidian, $crate::obsidian::ObsidianConfig, $crate::obsidian::ObsidianGenerator, Surface),
            (Chitin, chitin, $crate::chitin::ChitinConfig, $crate::chitin::ChitinGenerator, Surface),
            (SolarPanel, solar_panel, $crate::solar_panel::SolarPanelConfig, $crate::solar_panel::SolarPanelGenerator, Surface),
            (Parquet, parquet, $crate::parquet::ParquetConfig, $crate::parquet::ParquetGenerator, Surface),
            (Truchet, truchet, $crate::truchet::TruchetConfig, $crate::truchet::TruchetGenerator, Surface),
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

/// Invoke `$callback!` with the full per-field table: every field of every
/// texture config, once, with everything the three consumers need.
///
/// One entry per config:
///
/// ```text
/// ConfigPath, "Panel Header", editor_fn [, fixup fixup_fn] {
///     kind(field, label, …) [explore(min, max)],
///     …
/// } [layout { field, …, separator(), text("…"), … }],
/// ```
///
/// # Row kinds
///
/// The mutation clamp is the `explore` band where a row has one, and the
/// envelope otherwise.
///
/// | Row | Mutation | Envelope | Widget |
/// |-----|----------|----------|--------|
/// | `seed(f, label)` | replaced with a random `u32` | none | drag |
/// | `f64(f, label, min, max, half)` | uniform ±`half`, clamped | `min..=max` | slider |
/// | `f64_round(f, label, min, max, half)` | as `f64`, then `.round()` | `min..=max` | slider stepping by 1 |
/// | `f32(f, label, min, max, half)` | uniform ±`half`, clamped | `min..=max` | slider |
/// | `usize(f, label, min, max)` | ±1 step, clamped | `min..=max` | integer slider |
/// | `color3(f, label, half)` | per channel, uniform ±`half` | each channel `0..=1` | colour picker |
/// | `bool(f, label)` | flipped | none | checkbox |
/// | `enum_pick(f, label, [(Variant, "Btn"), …])` | cycles to the next variant | none | selectable row |
/// | `nested(f, SubConfig, editor_fn, "salt")` | delegates to the sub-config | delegates | sub-panel |
///
/// # The four things a row carries that are not obvious
///
/// **A row's range is the envelope**, the range the field is *allowed* to
/// hold — see
/// [`ClampToEnvelope`](crate::envelope::ClampToEnvelope), which is what an
/// application applies to a config arriving from a peer it does not trust.
/// Before this table existed the range was written out three more times, and
/// the three disagreed with each other on 64 fields.
///
/// **`explore(min, max)` narrows what the *search* may reach**, and only that.
/// Some fields have a band the genetic operators should stay inside even
/// though a person may legitimately author outside it: a mortar width of zero
/// is a wall someone might want, and a random walk that finds it is a wasted
/// search. Without the clause the search uses the envelope. Fifty rows carry
/// one; thirty-two of those are a `normal_strength` floor.
///
/// **Row order is behaviour, not layout.** `mutate` draws once per field in
/// this order, so moving a row gives every seeded search a different
/// population from the same seed — `tests/genotype_behaviour.rs` fails if one
/// moves. The optional `layout { … }` list is the *display* order, and it is
/// the only place `separator()` and `text("…")` may appear; without it a
/// panel lists its fields in table order.
///
/// **`label` is `none` for a field with no widget.** Six fields are evolvable
/// but not editable; a `none` label keeps them out of the generated panel
/// rather than silently adding a control.
///
/// A `fixup` names a private `genetics` function run after the per-field pass
/// on both `mutate` and `crossover`, for an invariant coupling two fields
/// that no per-field row can express (the two tiling snaps). The envelope
/// clamp deliberately does *not* run it: clamping must be a no-op on a config
/// already inside the envelope, and a fixup would rewrite hand-authored
/// values that are merely un-snapped.
#[macro_export]
macro_rules! for_each_texture_field {
    ($callback:ident) => {
        $callback! {
        $crate::bark::BarkConfig, "Bark Config", bark_config_editor {
            seed(seed, none),
            f64(scale, "Scale", 0.5, 16.0, 1.0),
            usize(octaves, "Octaves", 1, 12),
            usize(warp_octaves, "Warp Octaves", 1, 6),
            f64(warp_u, "Warp H", 0.0, 1.0, 0.1),
            f64(warp_v, "Warp V", 0.0, 2.0, 0.2),
            color3(color_light, "Light Color", 0.07),
            color3(color_dark, "Dark Color", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
            f64(furrow_multiplier, "Furrow Blend", 0.0, 1.0, 0.2),
            f64(furrow_scale_u, "Plate Width", 0.5, 6.0, 0.5),
            f64(furrow_scale_v, "Plate Length", 0.05, 1.0, 0.1),
            f64(furrow_shape, "Plate Shape", 0.1, 2.0, 0.15),
        } layout {
            color_light, color_dark, scale, warp_u, warp_v, normal_strength,
            octaves, warp_octaves, separator(), text("Rhytidome Plates:"),
            furrow_multiplier, furrow_scale_u, furrow_scale_v, furrow_shape,
        },
        $crate::rock::RockConfig, "Rock Config", rock_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 0.5, 12.0, 0.75),
            usize(octaves, "Octaves", 1, 14),
            f64(attenuation, "Attenuation", 0.5, 6.0, 0.25) explore(1.0, 4.0),
            color3(color_light, "Color Gaps", 0.07),
            color3(color_dark, "Color Stone", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "rock_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::weathering::EdgeWear, "Edge Wear", edge_wear_editor {
            f32(amount, "Amount", 0.0, 1.0, 0.1),
            color3(color, "Substrate Color", 0.06),
            f64(threshold, "Threshold", 0.0, 1.0, 0.08),
            f64(breakup_scale, "Breakup Scale", 1.0, 32.0, 1.5),
            f32(roughness, "Roughness", 0.0, 1.0, 0.08),
            f32(metallic, "Metallic", 0.0, 1.0, 0.1),
        },
        $crate::weathering::Corrosion, "Corrosion", corrosion_editor {
            f32(amount, "Amount", 0.0, 1.0, 0.1),
            color3(color, "Corrosion Color", 0.06),
            f32(coverage, "Coverage", 0.0, 1.0, 0.08),
            f64(spread, "Spread", 0.0, 0.25, 0.02),
            f64(barrier_scale, "Barrier Scale", 1.0, 24.0, 1.0),
            f64(relief, "Crust Relief", 0.0, 0.5, 0.02),
            f32(roughness, "Roughness", 0.0, 1.0, 0.08),
            f32(metallic, "Metallic", 0.0, 1.0, 0.08),
        },
        $crate::weathering::CreviceDirt, "Crevice Dirt", crevice_dirt_editor {
            f32(amount, "Amount", 0.0, 1.0, 0.1),
            color3(color, "Grime Color", 0.05),
            f64(depth, "Depth", 0.005, 0.25, 0.01),
            f32(gravity, "Gravity", 0.0, 1.0, 0.1),
            f32(roughness, "Roughness", 0.0, 1.0, 0.06),
            f32(occlusion, "Occlusion", 0.0, 1.0, 0.08),
        },
        $crate::weathering::Streaks, "Streaks", streaks_editor {
            f32(amount, "Amount", 0.0, 1.0, 0.1),
            color3(color, "Stain Color", 0.05),
            f32(density, "Density", 0.0, 1.0, 0.1),
            f64(length, "Length", 0.0, 1.0, 0.06),
            f64(wander, "Wander", 0.0, 4.0, 0.15),
            f32(roughness, "Roughness", 0.0, 1.0, 0.06),
        },
        $crate::weathering::WeatheringConfig, "Weathering", weathering_config_editor {
            seed(seed, "Seed"),
            nested(edge_wear, $crate::weathering::EdgeWear, edge_wear_editor, "weather_wear"),
            nested(corrosion, $crate::weathering::Corrosion, corrosion_editor, "weather_corrosion"),
            nested(crevice_dirt, $crate::weathering::CreviceDirt, crevice_dirt_editor, "weather_dirt"),
            nested(streaks, $crate::weathering::Streaks, streaks_editor, "weather_streaks"),
        },
        $crate::ground::GroundConfig, "Ground Config", ground_config_editor {
            seed(seed, "Seed"),
            f64(macro_scale, "Macro Scale", 0.5, 8.0, 0.5),
            usize(macro_octaves, "Macro Octaves", 1, 10),
            f64(micro_scale, "Micro Scale", 1.0, 20.0, 1.0),
            usize(micro_octaves, "Micro Octaves", 1, 10),
            f64(micro_weight, "Micro Weight", 0.0, 1.0, 0.1),
            color3(color_dry, "Color Dry", 0.07),
            color3(color_moist, "Color Moist", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::leaf::LeafConfig, "Leaf Config", leaf_config_editor {
            seed(seed, none),
            color3(color_base, "Base Color", 0.07),
            color3(color_edge, "Edge Color", 0.07),
            f64(serration_strength, "Serration", 0.0, 0.5, 0.01) explore(0.0, 0.15),
            f64(vein_angle, "Vein Angle", 0.5, 6.0, 0.3),
            f64(micro_detail, "Micro Detail", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.3) explore(0.5, 6.0),
            f64(lobe_count, "Lobe Count", 0.0, 10.0, 1.0),
            f64(lobe_depth, "Lobe Depth", 0.0, 1.0, 0.15),
            f64(lobe_sharpness, none, 0.1, 5.0, 0.4),
            f64(petiole_length, "Petiole", 0.0, 0.3, 0.02) explore(0.0, 0.25),
            f64(petiole_width, none, 0.008, 0.05, 0.003),
            f64(midrib_width, none, 0.03, 0.35, 0.02),
            f64(vein_count, "Vein Count", 2.0, 14.0, 1.0),
            f64(venule_strength, none, 0.0, 1.0, 0.1),
        } layout {
            color_base, color_edge, serration_strength, vein_angle,
            vein_count, lobe_count, lobe_depth, micro_detail, normal_strength,
            petiole_length,
        },
        $crate::needle::NeedleConfig, "Needle Config", needle_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            usize(pair_count, "Pair Count", 1, 24),
            color3(color_base, "Base Color", 0.05),
            color3(color_tip, "Tip Color", 0.06),
            color3(color_shoot, "Shoot Color", 0.06),
            f64(needle_angle, "Needle Angle", 5.0, 85.0, 6.0),
            f64(needle_length, "Needle Length", 0.05, 0.6, 0.05),
            f64(needle_width, "Needle Width", 0.002, 0.03, 0.003),
            f64(length_taper, "Length Taper", 0.0, 1.0, 0.1),
            f64(shoot_length, "Shoot Length", 0.2, 1.0, 0.08),
            f64(shoot_width, "Shoot Width", 0.002, 0.04, 0.004),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::broadleaf::BroadleafConfig, "Broadleaf Config", broadleaf_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_base, "Base Color", 0.05),
            color3(color_edge, "Edge Color", 0.07),
            f64_round(lobe_count, "Lobe Count", 1.0, 9.0, 1.0),
            f64(lobe_depth, "Lobe Depth", 0.0, 0.8, 0.1),
            f64(fan_angle, "Fan Angle", 30.0, 110.0, 8.0),
            f64(radius, "Radius", 0.3, 1.0, 0.08),
            f64(base_notch, "Base Notch", 0.0, 0.5, 0.06),
            f64(vein_width, "Vein Width", 0.01, 0.2, 0.02),
            f64(petiole_length, "Petiole Length", 0.0, 0.3, 0.03),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::moss::MossConfig, "Moss Config", moss_config_editor {
            seed(seed, "Seed"),
            f64(cushion_scale, "Cushion Scale", 0.5, 16.0, 1.0),
            usize(cushion_octaves, "Cushion Octaves", 1, 10),
            f64(filament_scale, "Filament Scale", 4.0, 64.0, 4.0),
            usize(filament_octaves, "Filament Octaves", 1, 10),
            f64(filament_weight, "Filament Weight", 0.0, 1.0, 0.1),
            color3(color_deep, "Deep Color", 0.05),
            color3(color_tip, "Tip Color", 0.07),
            color3(color_dry, "Dry Color", 0.07),
            f64(dry_patches, "Dry Patches", 0.0, 1.0, 0.1),
            f64(dry_scale, "Dry Scale", 0.5, 12.0, 0.5),
            f64(cushion_depth, "Cushion Depth", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::lichen::LichenConfig, "Lichen Config", lichen_config_editor {
            seed(seed, "Seed"),
            f64(patch_scale, "Patch Scale", 0.5, 16.0, 1.0),
            usize(patch_octaves, "Patch Octaves", 1, 10),
            f64(coverage, "Coverage", 0.0, 1.0, 0.1),
            f64(rim_width, "Rim Width", 0.0, 0.4, 0.03),
            f64(species_scale, "Species Scale", 0.3, 8.0, 0.4),
            color3(color_rock, "Rock Color", 0.05),
            color3(color_lichen_a, "Lichen Color A", 0.07),
            color3(color_lichen_b, "Lichen Color B", 0.07),
            color3(color_rim, "Rim Color", 0.06),
            f64(grain_scale, "Grain Scale", 4.0, 80.0, 5.0),
            f64(grain_strength, "Grain Strength", 0.0, 1.0, 0.1),
            f64(relief, "Relief", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::reed::ReedConfig, "Reed Config", reed_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            usize(blade_count, "Blade Count", 1, 12),
            color3(color_base, "Base Color", 0.05),
            color3(color_tip, "Tip Color", 0.07),
            color3(color_catkin, "Catkin Color", 0.06),
            f64(blade_width, "Blade Width", 0.008, 0.08, 0.005),
            f64(height_min, "Height Min", 0.3, 1.0, 0.1),
            f64(height_max, "Height Max", 0.3, 1.0, 0.1),
            f64(lean, "Lean", 0.0, 0.3, 0.03),
            f64(tip_fraction, "Tip Fraction", 0.05, 0.8, 0.08),
            f64(catkin_share, "Catkin Share", 0.0, 1.0, 0.15),
            f64(catkin_length, "Catkin Length", 0.0, 0.4, 0.05),
            f64(catkin_width, "Catkin Width", 0.005, 0.06, 0.005),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::cactus::CactusSkinConfig, "Cactus Skin Config", cactus_config_editor {
            seed(seed, "Seed"),
            usize(rib_count, "Rib Count", 3, 40),
            usize(areole_rows, "Areole Rows", 2, 40),
            f64(rib_depth, "Rib Depth", 0.0, 1.0, 0.1),
            f64(rib_sharpness, "Rib Sharpness", 0.3, 3.0, 0.3),
            color3(color_skin, "Skin Color", 0.06),
            color3(color_valley, "Valley Color", 0.05),
            color3(color_areole, "Areole Color", 0.05),
            color3(color_spine, "Spine Color", 0.05),
            f64(areole_size, "Areole Size", 0.005, 0.08, 0.005),
            f64(spine_reach, "Spine Reach", 1.0, 6.0, 0.5),
            usize(spine_count, "Spine Count", 0, 24),
            f64(waxiness, "Waxiness", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::frond::FrondConfig, "Frond Config", frond_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_base, "Base Color", 0.05),
            color3(color_edge, "Edge Color", 0.07),
            f64(width, "Width", 0.04, 0.3, 0.02),
            f64(tip_taper, "Tip Taper", 0.4, 3.0, 0.3),
            f64(midrib_width, "Midrib Width", 0.05, 0.5, 0.05),
            f64(vein_count, "Vein Count", 0.0, 20.0, 1.0),
            f64(lobe_count, "Lobe Count", 0.0, 12.0, 1.0),
            f64(lobe_depth, "Lobe Depth", 0.0, 0.6, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::grass::GrassTuftConfig, "Grass Tuft Config", grass_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            usize(blade_count, "Blade Count", 1, 24),
            color3(color_base, "Base Color", 0.05),
            color3(color_tip, "Tip Color", 0.07),
            color3(color_dry, "Dry Color", 0.07),
            f64(blade_width, "Blade Width", 0.01, 0.12, 0.01),
            f64(blade_taper, "Blade Taper", 0.5, 4.0, 0.3),
            f64(height_min, "Height Min", 0.2, 1.0, 0.1),
            f64(height_max, "Height Max", 0.2, 1.0, 0.1),
            f64(fan_spread, "Fan Spread", 0.0, 0.5, 0.05),
            f64(curve, "Curve", 0.0, 0.5, 0.05),
            f64(base_spread, "Base Spread", 0.0, 0.4, 0.04),
            f64(dry_fraction, "Dry Fraction", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.3) explore(0.5, 6.0),
        },
        $crate::twig::TwigConfig, "Twig Config", twig_config_editor {
            nested(leaf, $crate::leaf::LeafConfig, leaf_config_editor, "twig_leaf"),
            color3(stem_color, "Stem Color", 0.07),
            f64(stem_half_width, "Stem Width", 0.005, 0.05, 0.005),
            usize(leaf_pairs, "Leaf Pairs", 1, 8),
            f64(leaf_angle, "Leaf Angle", 0.0, std::f64::consts::PI, 0.15),
            f64(leaf_scale, "Leaf Scale", 0.1, 0.6, 0.05) explore(0.15, 0.6),
            f64(stem_curve, "Stem Curve", 0.0, 0.2, 0.02),
            bool(sympodial, "Sympodial"),
        } layout {
            stem_color, stem_half_width, leaf_pairs, leaf_angle, leaf_scale,
            stem_curve, sympodial, leaf,
        },
        $crate::brick::BrickConfig, "Brick Config", brick_config_editor, fixup snap_row_offset {
            seed(seed, "Seed"),
            f64_round(scale, "Scale (Rows)", 1.0, 16.0, 1.0) explore(1.0, 12.0),
            f64(row_offset, "Row Offset", 0.0, 1.0, 0.1),
            f64(aspect_ratio, "Aspect Ratio", 0.1, 4.0, 0.3),
            f64(mortar_size, "Mortar Size", 0.0, 0.4, 0.03) explore(0.01, 0.35),
            f64(bevel, "Bevel", 0.0, 1.0, 0.2),
            f64(cell_variance, "Color Variance", 0.0, 1.0, 0.1) explore(0.0, 0.8),
            f64(roughness, "Surface Roughness", 0.0, 1.0, 0.1),
            color3(color_brick, "Brick Color", 0.07),
            color3(color_mortar, "Mortar Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "brick_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::window::WindowConfig, "Window Config", window_config_editor {
            seed(seed, "Seed"),
            f64(frame_width, "Frame Width", 0.0, 0.4, 0.02) explore(0.02, 0.4),
            usize(panes_x, "Panes X", 1, 16) explore(1, 6),
            usize(panes_y, "Panes Y", 1, 16) explore(1, 8),
            f64(mullion_thickness, "Mullion Thickness", 0.0, 0.2, 0.005) explore(0.005, 0.15),
            f64(corner_radius, "Corner Radius", 0.0, 0.4, 0.01) explore(0.0, 0.35),
            f64(glass_opacity, "Glass Opacity", 0.0, 1.0, 0.1),
            f64(grime_level, "Grime", 0.0, 1.0, 0.1),
            color3(color_frame, "Frame Color", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 6.0),
        },
        $crate::plank::PlankConfig, "Plank Config", plank_config_editor {
            seed(seed, "Seed"),
            f64_round(plank_count, "Plank Count", 1.0, 16.0, 1.0) explore(2.0, 12.0),
            f64(grain_scale, "Grain Scale", 2.0, 32.0, 2.0) explore(4.0, 24.0),
            f64(joint_width, "Joint Width", 0.0, 0.3, 0.02) explore(0.01, 0.25),
            f64(stagger, "Stagger", 0.0, 1.0, 0.15),
            f64(knot_density, "Knot Density", 0.0, 1.0, 0.1),
            f64(grain_warp, "Grain Warp", 0.0, 1.0, 0.1),
            color3(color_wood_light, "Wood Light", 0.07),
            color3(color_wood_dark, "Wood Dark", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 6.0),
        },
        $crate::shingle::ShingleConfig, "Shingle Config", shingle_config_editor, fixup snap_stagger {
            seed(seed, "Seed"),
            f64_round(scale, "Scale (Rows)", 2.0, 16.0, 1.0) explore(2.0, 12.0),
            f64(shape_profile, "Shape (Square→Scallop)", 0.0, 1.0, 0.2),
            f64(overlap, "Overlap", 0.0, 0.8, 0.1),
            f64(stagger, "Stagger", 0.0, 1.0, 0.15),
            f64(moss_level, "Moss", 0.0, 1.0, 0.1),
            color3(color_tile, "Tile Color", 0.07),
            color3(color_grout, "Grout Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "shingle_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::stucco::StuccoConfig, "Stucco Config", stucco_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 1.0, 20.0, 1.5),
            usize(octaves, "Octaves", 1, 10),
            f64(roughness, "Roughness", 0.0, 1.0, 0.1),
            color3(color_base, "Base Color", 0.07),
            color3(color_shadow, "Shadow Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "stucco_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::concrete::ConcreteConfig, "Concrete Config", concrete_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 1.0, 16.0, 1.0),
            usize(octaves, "Octaves", 1, 10),
            f64(roughness, "Roughness", 0.0, 1.0, 0.1),
            f64_round(formwork_lines, "Formwork Lines", 0.0, 12.0, 1.0),
            f64(formwork_depth, "Formwork Depth", 0.0, 0.5, 0.05),
            f64(pit_density, "Pit Density", 0.0, 0.45, 0.04),
            color3(color_base, "Base Color", 0.07),
            color3(color_pit, "Pit Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "concrete_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::metal::MetalConfig, "Metal Config", metal_config_editor {
            seed(seed, "Seed"),
            enum_pick(style, "Style:", [($crate::metal::MetalStyle::Brushed, "Brushed"), ($crate::metal::MetalStyle::StandingSeam, "Standing Seam"), ($crate::metal::MetalStyle::Hammered, "Hammered"), ($crate::metal::MetalStyle::DiamondPlate, "Diamond Plate"), ($crate::metal::MetalStyle::Riveted, "Riveted"), ($crate::metal::MetalStyle::Perforated, "Perforated")]),
            f64(rivet_size, "Rivet Size", 0.05, 0.9, 0.05),
            f64(hole_size, "Hole Size", 0.05, 0.9, 0.05),
            f64(scale, "Scale", 1.0, 16.0, 1.0),
            f64_round(seam_count, "Seam Count", 1.0, 16.0, 1.0),
            f64(seam_sharpness, "Seam Sharpness", 0.5, 6.0, 0.5),
            f64(brush_stretch, "Brush Stretch", 1.0, 20.0, 1.5),
            f64(roughness, "Roughness", 0.0, 1.0, 0.1),
            f32(metallic, "Metallic", 0.0, 1.0, 0.1),
            f64(rust_level, "Rust", 0.0, 1.0, 0.1),
            color3(color_metal, "Metal Color", 0.07),
            color3(color_rust, "Rust Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "metal_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        } layout {
            seed, style, scale, seam_count, seam_sharpness, brush_stretch,
            rivet_size, hole_size, roughness, metallic, rust_level,
            color_metal, color_rust, weathering, normal_strength,
        },
        $crate::pavers::PaversConfig, "Pavers Config", pavers_config_editor {
            seed(seed, "Seed"),
            enum_pick(layout, "Layout:", [($crate::pavers::PaversLayout::Square, "Square"), ($crate::pavers::PaversLayout::Hexagonal, "Hexagonal")]),
            f64_round(scale, "Scale", 1.0, 16.0, 1.0),
            f64(aspect_ratio, "Aspect Ratio", 0.5, 3.0, 0.2),
            f64(grout_width, "Grout Width", 0.0, 0.35, 0.02) explore(0.01, 0.35),
            f64(bevel, "Bevel", 0.0, 1.0, 0.15),
            f64(cell_variance, "Color Variance", 0.0, 0.8, 0.08),
            f64(roughness, "Surface Roughness", 0.0, 1.0, 0.1),
            color3(color_stone, "Stone Color", 0.07),
            color3(color_grout, "Grout Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "pavers_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::ashlar::AshlarConfig, "Ashlar Config", ashlar_config_editor {
            seed(seed, "Seed"),
            usize(rows, "Rows", 2, 8),
            usize(cols, "Cols", 2, 6),
            f64(mortar_size, "Mortar Size", 0.005, 0.15, 0.02),
            f64(bevel, "Bevel", 0.0, 1.0, 0.2),
            f64(cell_variance, "Color Variance", 0.0, 1.0, 0.1) explore(0.0, 0.8),
            f64(chisel_depth, "Chisel Depth", 0.0, 1.0, 0.1),
            f64(roughness, "Roughness", 0.0, 1.0, 0.1),
            color3(color_stone, "Stone Color", 0.07),
            color3(color_mortar, "Mortar Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "ashlar_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::cobblestone::CobblestoneConfig, "Cobblestone Config", cobblestone_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 2.0, 14.0, 1.0),
            f64(gap_width, "Gap Width", 0.01, 0.3, 0.03),
            f64(cell_variance, "Color Variance", 0.0, 1.0, 0.1) explore(0.0, 0.8),
            f64(roundness, "Roundness", 0.3, 2.5, 0.15),
            color3(color_stone, "Stone Color", 0.07),
            color3(color_mud, "Mud Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "cobblestone_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::thatch::ThatchConfig, "Thatch Config", thatch_config_editor {
            seed(seed, "Seed"),
            f64(density, "Fibre Density", 3.0, 24.0, 2.0),
            f64(anisotropy, "Anisotropy", 2.0, 20.0, 1.0),
            f64(warp_strength, "Warp", 0.0, 0.6, 0.05),
            f64(layer_count, "Layer Count", 2.0, 20.0, 1.0),
            f64(layer_shadow, "Layer Shadow", 0.0, 1.0, 0.1),
            color3(color_straw, "Straw Color", 0.07),
            color3(color_shadow, "Shadow Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "thatch_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::marble::MarbleConfig, "Marble Config", marble_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 0.5, 10.0, 1.0),
            usize(octaves, "Octaves", 2, 10),
            usize(warp_octaves, "Warp Octaves", 1, 6),
            f64(warp_strength, "Warp Strength", 0.0, 2.0, 0.15),
            f64(vein_frequency, "Vein Frequency", 0.5, 10.0, 0.5),
            f64(vein_sharpness, "Vein Sharpness", 0.3, 8.0, 0.5),
            f64(roughness, "Roughness", 0.0, 0.4, 0.05),
            color3(color_base, "Base Color", 0.07),
            color3(color_vein, "Vein Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "marble_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::corrugated::CorrugatedConfig, "Corrugated Metal Config", corrugated_config_editor {
            seed(seed, "Seed"),
            f64_round(ridges, "Ridges", 2.0, 32.0, 2.0) explore(2.0, 20.0),
            f64(ridge_depth, "Ridge Depth", 0.3, 2.5, 0.2),
            f64(roughness, "Roughness", 0.0, 1.0, 0.1),
            f64(rust_level, "Rust", 0.0, 1.0, 0.1),
            f32(metallic, "Metallic", 0.0, 1.0, 0.1),
            color3(color_metal, "Metal Color", 0.07),
            color3(color_rust, "Rust Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "corrugated_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::asphalt::AsphaltConfig, "Asphalt Config", asphalt_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 1.0, 14.0, 1.0),
            f64(aggregate_density, "Aggregate Density", 0.02, 0.5, 0.05),
            f64(aggregate_scale, "Aggregate Scale", 4.0, 40.0, 3.0),
            f64(roughness, "Roughness", 0.5, 1.0, 0.05),
            f64(stain_level, "Stain Level", 0.0, 1.0, 0.1),
            color3(color_base, "Base Color", 0.05),
            color3(color_aggregate, "Aggregate Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "asphalt_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::wainscoting::WainscotingConfig, "Wainscoting Config", wainscoting_config_editor {
            seed(seed, "Seed"),
            usize(panels_x, "Panels X", 1, 4),
            usize(panels_y, "Panels Y", 1, 4),
            f64(frame_width, "Frame Width", 0.05, 0.4, 0.05),
            f64(panel_inset, "Panel Inset", 0.0, 0.2, 0.02),
            f64(grain_scale, "Grain Scale", 4.0, 28.0, 2.0),
            f64(grain_warp, "Grain Warp", 0.0, 1.0, 0.1),
            color3(color_wood_light, "Wood Light", 0.07),
            color3(color_wood_dark, "Wood Dark", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "wainscoting_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 8.0, 0.5) explore(0.5, 8.0),
        },
        $crate::stained_glass::StainedGlassConfig, "Stained Glass Config", stained_glass_config_editor {
            seed(seed, "Seed"),
            usize(cell_count, "Cell Count", 3, 30),
            f64(lead_width, "Lead Width", 0.01, 0.15, 0.01),
            f32(saturation, "Saturation", 0.3, 1.0, 0.1),
            f64(glass_roughness, "Glass Roughness", 0.0, 0.2, 0.02),
            f64(grime_level, "Grime", 0.0, 0.6, 0.05),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::iron_grille::IronGrilleConfig, "Iron Grille Config", iron_grille_config_editor {
            seed(seed, "Seed"),
            usize(bars_x, "Bars X", 1, 12),
            usize(bars_y, "Bars Y", 1, 12),
            f64(bar_width, "Bar Width", 0.01, 0.25, 0.02),
            bool(round_bars, "Round Bars"),
            f64(rust_level, "Rust", 0.0, 1.0, 0.1),
            color3(color_iron, "Iron Color", 0.07),
            color3(color_rust, "Rust Color", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::encaustic::EncausticConfig, "Encaustic Tile Config", encaustic_config_editor {
            seed(seed, "Seed"),
            f64_round(scale, "Scale", 1.0, 12.0, 1.0),
            enum_pick(pattern, "Pattern:", [($crate::encaustic::EncausticPattern::Checkerboard, "Checker"), ($crate::encaustic::EncausticPattern::Octagon, "Octagon"), ($crate::encaustic::EncausticPattern::Diamond, "Diamond")]),
            f64(grout_width, "Grout Width", 0.01, 0.2, 0.02),
            f64(glaze_roughness, "Glaze Roughness", 0.0, 0.15, 0.02),
            color3(color_a, "Color A", 0.07),
            color3(color_b, "Color B", 0.07),
            color3(color_grout, "Grout Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "encaustic_weather"),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.5) explore(0.5, 6.0),
        },
        $crate::soft_disc::SoftDiscConfig, "Soft Disc Config", soft_disc_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_core, "Core Color", 0.07),
            color3(color_halo, "Halo Color", 0.07),
            f64(core_radius, "Core Radius", 0.0, 0.9, 0.05),
            f64(falloff, "Falloff", 0.3, 8.0, 0.5),
            f64(ellipticity, "Ellipticity", 0.0, 0.6, 0.08),
            f64(scale_jitter, "Scale Jitter", 0.0, 0.5, 0.05),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::spark::SparkConfig, "Spark Config", spark_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            usize(points, "Points", 2, 12),
            color3(color_core, "Core Color", 0.07),
            color3(color_tip, "Tip Color", 0.07),
            f64(core_radius, "Core Radius", 0.02, 0.5, 0.03),
            f64(arm_sharpness, "Arm Sharpness", 0.5, 10.0, 0.8),
            f64(falloff, "Falloff", 0.5, 6.0, 0.4),
            f64(length_jitter, "Length Jitter", 0.0, 0.8, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::snowflake::SnowflakeConfig, "Snowflake Config", snowflake_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            usize(arms, "Arms", 3, 8),
            color3(color, "Color", 0.05),
            f64(core_radius, "Core Radius", 0.0, 0.4, 0.03),
            f64(arm_width, "Arm Width", 0.01, 0.12, 0.01),
            usize(branch_pairs, "Branch Pairs", 0, 5),
            f64(branch_angle, "Branch Angle", 0.3, 1.4, 0.1),
            f64(branch_scale, "Branch Scale", 0.1, 1.0, 0.1),
            f64(softness, "Softness", 0.005, 0.08, 0.005),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::puff::PuffConfig, "Puff Config", puff_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_base, "Base Color", 0.07),
            color3(color_shadow, "Shadow Color", 0.07),
            f64(noise_scale, "Noise Scale", 1.0, 8.0, 0.7),
            usize(octaves, "Octaves", 1, 8),
            f64(warp, "Warp", 0.0, 1.5, 0.15),
            f64(density, "Density", 0.0, 1.0, 0.1),
            f64(edge_falloff, "Edge Falloff", 0.5, 6.0, 0.5),
            f64(contrast, "Contrast", 0.5, 4.0, 0.3),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::ring::RingConfig, "Ring Config", ring_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color, "Color", 0.07),
            f64(radius, "Radius", 0.1, 0.9, 0.08),
            f64(thickness, "Thickness", 0.01, 0.5, 0.04),
            f64(falloff, "Falloff", 0.5, 6.0, 0.5),
            f64(waviness, "Waviness", 0.0, 0.3, 0.04),
            usize(wave_count, "Wave Count", 2, 16),
            f64(radius_jitter, "Radius Jitter", 0.0, 0.4, 0.05),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::petal::PetalConfig, "Petal Config", petal_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_base, "Base Color", 0.07),
            color3(color_edge, "Edge Color", 0.07),
            color3(color_throat, "Throat Color", 0.07),
            f64(length, "Length", 0.4, 1.0, 0.06),
            f64(width, "Width", 0.15, 0.95, 0.08),
            f64(peak, "Peak", 0.3, 0.9, 0.07),
            f64(tip_notch, "Tip Notch", 0.0, 0.25, 0.03),
            f64(curl, "Curl", 0.0, 1.0, 0.1),
            f64(asymmetry, "Asymmetry", 0.0, 0.4, 0.05),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.3),
        },
        $crate::shard::ShardConfig, "Shard Config", shard_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Variant Rows", 1, 16),
            usize(variant_cols, "Variant Cols", 1, 16),
            color3(color_base, "Base Color", 0.07),
            color3(color_edge, "Edge Color", 0.07),
            usize(sides, "Sides", 3, 9),
            f64(irregularity, "Irregularity", 0.0, 0.9, 0.1),
            f64(edge_band, "Edge Band", 0.02, 0.5, 0.05),
            f64(grain, "Grain", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 6.0, 0.4),
        },
        $crate::log_end::LogEndConfig, "Log End Config", log_end_config_editor {
            seed(seed, "Seed"),
            f64_round(ring_count, "Ring Count", 4.0, 30.0, 2.0),
            f64(ring_warp, "Ring Warp", 0.0, 1.0, 0.1),
            f64(ring_contrast, "Ring Contrast", 0.5, 4.0, 0.3),
            f64_round(crack_count, "Crack Count", 0.0, 12.0, 1.0),
            f64(bark_width, "Bark Width", 0.02, 0.2, 0.02),
            color3(color_early, "Earlywood", 0.06),
            color3(color_late, "Latewood", 0.06),
            color3(color_bark, "Bark", 0.06),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::chain_link::ChainLinkConfig, "Chain-Link Config", chain_link_config_editor {
            seed(seed, "Seed"),
            f64_round(cell_count, "Cell Count", 4.0, 16.0, 1.0),
            f64(wire_radius, "Wire Radius", 0.02, 0.2, 0.015),
            f64(weave_depth, "Weave Depth", 0.0, 1.0, 0.1),
            f64(rust_level, "Rust Level", 0.0, 1.0, 0.1),
            color3(color_wire, "Wire Color", 0.06),
            color3(color_rust, "Rust Color", 0.06),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::lava::LavaConfig, "Lava Config", lava_config_editor {
            seed(seed, "Seed"),
            f64(plate_scale, "Plate Scale", 3.0, 12.0, 0.7),
            f64(crack_width, "Crack Width", 0.02, 0.3, 0.03),
            f64(glow_falloff, "Glow Falloff", 0.5, 4.0, 0.3),
            color3(color_crust, "Crust Color", 0.04),
            color3(color_glow, "Glow Color", 0.07),
            f32(emissive_intensity, "Emissive Intensity", 0.0, 4.0, 0.3),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::cracked_earth::CrackedEarthConfig, "Cracked Earth Config", cracked_earth_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Plates", 2.0, 20.0, 0.8),
            f64(jitter, "Jitter", 0.0, 1.0, 0.1),
            f64(crack_width, "Crack Width", 0.001, 0.03, 0.002),
            f64(crack_depth, "Crack Depth", 0.0, 1.5, 0.1),
            f64(curl, "Curl", 0.0, 0.8, 0.05),
            f64(curl_reach, "Curl Reach", 0.005, 0.12, 0.008),
            f32(plate_variance, "Plate Variance", 0.0, 0.4, 0.03),
            f64(grain_scale, "Grain Scale", 4.0, 64.0, 3.0),
            f64(grain_strength, "Grain Strength", 0.0, 0.5, 0.03),
            color3(color_plate, "Plate Color", 0.06),
            color3(color_crack, "Crack Color", 0.05),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::gravel::GravelConfig, "Gravel Config", gravel_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Stones", 4.0, 64.0, 2.0),
            enum_pick(metric, "Stone Shape:", [($crate::noise::CellMetric::Euclidean, "Rounded"), ($crate::noise::CellMetric::Manhattan, "Diamond"), ($crate::noise::CellMetric::Chebyshev, "Angular")]),
            f64(jitter, "Jitter", 0.0, 1.0, 0.1),
            f64(roundness, "Roundness", 0.2, 4.0, 0.2),
            f64(size_variance, "Size Variance", 0.0, 1.0, 0.1),
            f32(cell_variance, "Tint Variance", 0.0, 0.5, 0.03),
            f64(fines_level, "Fines", 0.0, 1.0, 0.1),
            f64(grain_scale, "Grain Scale", 8.0, 128.0, 6.0),
            color3(color_stone, "Stone Color", 0.06),
            color3(color_dark, "Shadow Color", 0.05),
            color3(color_fines, "Fines Color", 0.05),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        } layout {
            seed, metric, scale, jitter, roundness, size_variance,
            cell_variance, fines_level, grain_scale, color_stone, color_dark,
            color_fines, normal_strength,
        },
        $crate::forest_floor::ForestFloorConfig, "Forest Floor Config", forest_floor_config_editor {
            seed(seed, "Seed"),
            f64(litter_scale, "Litter Scale", 2.0, 24.0, 0.8),
            usize(layers, "Layers", 1, 4),
            f64(coverage, "Coverage", 0.0, 1.0, 0.1),
            f64(leaf_length, "Leaf Length", 0.3, 2.5, 0.15),
            f64(leaf_width, "Leaf Width", 0.1, 1.0, 0.08),
            f64(leaf_thickness, "Leaf Thickness", 0.0, 1.0, 0.06),
            f32(midrib, "Midrib", 0.0, 1.0, 0.06),
            f64(humus_scale, "Humus Scale", 2.0, 48.0, 2.0),
            color3(color_humus, "Humus Color", 0.04),
            color3(color_leaf, "Fresh Leaf", 0.07),
            color3(color_leaf_old, "Old Leaf", 0.06),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::enamel::EnamelConfig, "Enamel Config", enamel_config_editor {
            seed(seed, "Seed"),
            color3(color, "Glaze Color", 0.07),
            color3(color_body, "Body Color", 0.05),
            f32(gloss_roughness, "Gloss", 0.0, 1.0, 0.05),
            f32(metallic, "Metallic", 0.0, 1.0, 0.08),
            f32(crackle, "Crackle", 0.0, 1.0, 0.1),
            f64(crackle_scale, "Crackle Scale", 4.0, 64.0, 3.0),
            f64(crackle_width, "Crackle Width", 0.0005, 0.02, 0.001),
            f64(orange_peel, "Orange Peel", 0.0, 0.4, 0.02),
            f64(orange_peel_scale, "Peel Scale", 4.0, 80.0, 4.0),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "enamel_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.2) explore(0.1, 4.0),
        },
        $crate::obsidian::ObsidianConfig, "Obsidian Config", obsidian_config_editor {
            seed(seed, "Seed"),
            color3(color, "Body Color", 0.03),
            color3(color_sheen, "Sheen Color", 0.06),
            f64_round(band_cycles_u, "Band Cycles U", 0.0, 24.0, 1.5),
            f64_round(band_cycles_v, "Band Cycles V", 0.0, 24.0, 1.5),
            f64(band_warp, "Band Warp", 0.0, 2.0, 0.1),
            f64(band_warp_scale, "Warp Scale", 0.5, 12.0, 0.5),
            f64(band_sharpness, "Band Sharpness", 0.0, 1.0, 0.1),
            f32(band_contrast, "Band Contrast", 0.0, 1.0, 0.1),
            f32(gloss_roughness, "Gloss", 0.0, 1.0, 0.04),
            f32(metallic, "Metallic", 0.0, 1.0, 0.08),
            f64(relief, "Relief", 0.0, 0.5, 0.02),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "obsidian_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.2) explore(0.1, 4.0),
        },
        $crate::chitin::ChitinConfig, "Chitin Config", chitin_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Plates", 2.0, 24.0, 0.8),
            f64(jitter, "Jitter", 0.0, 1.0, 0.1),
            f64(softness, "Softness", 2.0, 200.0, 4.0),
            f64(plate_fill, "Plate Fill", 0.05, 1.0, 0.08),
            f64(plate_relief, "Plate Relief", 0.0, 1.5, 0.08),
            f64(seam_width, "Seam Width", 0.0005, 0.05, 0.002),
            f32(seam_depth, "Seam Depth", 0.0, 1.0, 0.08),
            f32(iridescence, "Iridescence", 0.0, 0.6, 0.04),
            color3(color, "Shell Color", 0.07),
            color3(color_deep, "Deep Color", 0.05),
            f32(gloss_roughness, "Gloss", 0.0, 1.0, 0.06),
            f32(metallic, "Metallic", 0.0, 1.0, 0.08),
            f64(pit_scale, "Pit Scale", 6.0, 96.0, 5.0),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "chitin_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.3),
        },
        $crate::solar_panel::SolarPanelConfig, "Solar Panel Config", solar_panel_config_editor {
            seed(seed, "Seed"),
            f64_round(cells_x, "Cells X", 1.0, 16.0, 1.0),
            f64_round(cells_y, "Cells Y", 1.0, 16.0, 1.0),
            f64(cell_gap, "Cell Gap", 0.0, 0.4, 0.02),
            f64(corner_cut, "Corner Cut", 0.0, 0.5, 0.03),
            f64_round(busbars, "Busbars", 0.0, 8.0, 1.0),
            f64(busbar_width, "Busbar Width", 0.0, 0.2, 0.01),
            f64_round(fingers, "Fingers", 0.0, 60.0, 3.0),
            f64(finger_width, "Finger Width", 0.0, 0.1, 0.004),
            color3(color_cell, "Silicon Color", 0.03),
            color3(color_backing, "Backing Color", 0.05),
            color3(color_wire, "Wire Color", 0.05),
            f32(cell_variance, "Cell Variance", 0.0, 0.5, 0.02) explore(0.0, 0.3),
            f32(crystal_mottle, "Crystal Mottle", 0.0, 1.0, 0.08),
            f64(crystal_scale, "Crystal Scale", 4.0, 64.0, 3.0),
            f32(glass_roughness, "Glass Gloss", 0.0, 1.0, 0.04),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "solar_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.2) explore(0.1, 4.0),
        },
        $crate::parquet::ParquetConfig, "Parquet Config", parquet_config_editor {
            seed(seed, "Seed"),
            enum_pick(layout, "Layout:", [($crate::parquet::ParquetLayout::Herringbone, "Herringbone"), ($crate::parquet::ParquetLayout::Basket, "Basket"), ($crate::parquet::ParquetLayout::Brick, "Brick")]),
            f64_round(scale, "Boards", 2.0, 32.0, 1.0),
            f64(aspect, "Aspect", 1.0, 12.0, 0.5),
            f64(joint_width, "Joint Width", 0.0, 0.3, 0.01),
            f64(joint_depth, "Joint Depth", 0.0, 1.5, 0.08),
            f64_round(grain_lines, "Grain Lines", 1.0, 32.0, 1.0),
            f32(grain_contrast, "Grain Contrast", 0.0, 1.0, 0.06),
            f64(grain_warp, "Grain Warp", 0.0, 1.0, 0.04),
            f32(board_variance, "Board Variance", 0.0, 0.5, 0.03),
            color3(color_wood, "Wood Color", 0.06),
            color3(color_grain, "Grain Color", 0.05),
            color3(color_joint, "Joint Color", 0.04),
            f32(gloss_roughness, "Gloss", 0.0, 1.0, 0.06),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "parquet_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.2) explore(0.1, 5.0),
        },
        $crate::truchet::TruchetConfig, "Truchet Config", truchet_config_editor {
            seed(seed, "Seed"),
            f64_round(scale, "Tiles", 1.0, 32.0, 1.0),
            f64(trace_width, "Trace Width", 0.01, 0.45, 0.015),
            f64(trace_relief, "Trace Relief", 0.0, 1.5, 0.08),
            f64(density, "Density", 0.0, 1.0, 0.08),
            color3(color_panel, "Panel Color", 0.04),
            color3(color_trace, "Trace Color", 0.06),
            color3(color_glow, "Glow Color", 0.07),
            f32(emissive_intensity, "Glow", 0.0, 4.0, 0.2),
            f32(panel_roughness, "Panel Roughness", 0.0, 1.0, 0.06),
            f32(trace_roughness, "Trace Roughness", 0.0, 1.0, 0.06),
            f32(trace_metallic, "Trace Metallic", 0.0, 1.0, 0.08),
            f64(mottle_scale, "Mottle Scale", 2.0, 48.0, 2.0),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "truchet_weather"),
            f32(normal_strength, "Normal Strength", 0.1, 6.0, 0.2) explore(0.1, 5.0),
        },
        $crate::ice::IceConfig, "Ice Config", ice_config_editor {
            seed(seed, "Seed"),
            f64(scale, "Scale", 1.0, 8.0, 0.6),
            f64(crack_density, "Crack Density", 1.0, 8.0, 0.7),
            f64(vein_sharpness, "Vein Sharpness", 2.0, 12.0, 1.0),
            f64(frost_level, "Frost Level", 0.0, 1.0, 0.1),
            color3(color_ice, "Ice Color", 0.05),
            color3(color_crack, "Crack Color", 0.05),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.25),
        },
        $crate::snow::SnowConfig, "Snow Config", snow_config_editor {
            seed(seed, "Seed"),
            f64(drift_scale, "Drift Scale", 1.0, 6.0, 0.5),
            usize(drift_octaves, "Drift Octaves", 2, 6),
            f64(sparkle_density, "Sparkle Density", 0.0, 0.5, 0.04),
            f64(crust_roughness, "Crust Roughness", 0.5, 1.0, 0.06),
            color3(color_snow, "Snow Color", 0.05),
            color3(color_shadow, "Shadow Color", 0.05),
            f32(normal_strength, "Normal Strength", 0.5, 5.0, 0.3),
        },
        $crate::sand::SandConfig, "Sand Config", sand_config_editor {
            seed(seed, "Seed"),
            f64_round(ripple_count, "Ripple Count", 4.0, 24.0, 2.0),
            f64(ripple_warp, "Ripple Warp", 0.0, 1.5, 0.15),
            f64(grain_density, "Grain Density", 0.0, 0.5, 0.05),
            f64(grain_scale, "Grain Scale", 8.0, 48.0, 4.0),
            color3(color_crest, "Crest Color", 0.07),
            color3(color_trough, "Trough Color", 0.07),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.3),
        },
        $crate::fabric::FabricConfig, "Fabric Config", fabric_config_editor {
            seed(seed, "Seed"),
            enum_pick(weave, "Weave:", [($crate::fabric::WeaveKind::Plain, "Plain"), ($crate::fabric::WeaveKind::Twill, "Twill"), ($crate::fabric::WeaveKind::Satin, "Satin"), ($crate::fabric::WeaveKind::Basket, "Basket")]),
            f64_round(thread_count, "Thread Count", 8.0, 64.0, 4.0),
            f64(thread_width, "Thread Width", 0.3, 0.98, 0.08),
            f64(weave_contrast, "Weave Contrast", 0.0, 1.0, 0.12),
            f64(fuzz, "Fuzz", 0.0, 1.0, 0.1),
            color3(color_warp, "Warp Color", 0.07),
            color3(color_weft, "Weft Color", 0.07),
            nested(weathering, $crate::weathering::WeatheringConfig, weathering_config_editor, "fabric_weather"),
            f32(normal_strength, "Normal Strength", 0.5, 6.0, 0.4),
        },
        $crate::flower::FlowerConfig, "Flower Config", flower_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Atlas Rows", 1, 16),
            usize(variant_cols, "Atlas Cols", 1, 16),
            nested(petal, $crate::petal::PetalConfig, petal_config_editor, "flower_petal"),
            usize(petal_count, "Petal Count", 4, 12),
            f64(center_radius, "Center Radius", 0.05, 0.3, 0.03),
            color3(center_color, "Center Color", 0.07),
            f64(dot_density, "Dot Density", 0.0, 1.0, 0.1),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.2),
        } layout {
            seed, variant_rows, variant_cols, petal_count, center_radius,
            center_color, dot_density, normal_strength, petal,
        },
        $crate::flame::FlameConfig, "Flame Config", flame_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Atlas Rows", 1, 16),
            usize(variant_cols, "Atlas Cols", 1, 16),
            f64(elongation, "Elongation", 1.0, 3.0, 0.2),
            f64(turbulence, "Turbulence", 0.0, 1.5, 0.15),
            f64(lean_jitter, "Lean Jitter", 0.0, 0.5, 0.05),
            f64(falloff, "Falloff", 0.5, 4.0, 0.3),
            color3(color_core, "Core Color", 0.07),
            color3(color_mid, "Mid Color", 0.07),
            color3(color_tip, "Tip Color", 0.07),
            f32(normal_strength, "Normal Strength", 0.0, 4.0, 0.2),
        },
        $crate::leaf_sprite::LeafSpriteConfig, "Leaf Sprite Config", leaf_sprite_config_editor {
            seed(seed, "Seed"),
            usize(variant_rows, "Atlas Rows", 1, 16),
            usize(variant_cols, "Atlas Cols", 1, 16),
            nested(leaf, $crate::leaf::LeafConfig, leaf_config_editor, "leaf_sprite_leaf"),
            f64(shape_jitter, "Shape Jitter", 0.0, 1.0, 0.1),
            f32(tint_jitter, "Tint Jitter", 0.0, 1.0, 0.08),
        } layout {
            seed, variant_rows, variant_cols, shape_jitter, tint_jitter, leaf,
        },
        }
    };
}

// `for_each_texture_field!` is exported for the same reason as
// `for_each_generator!`: the `bevy_symbios_texture` wrapper generates its
// config editors from it, and an application generating a serialisation
// mirror needs the same rows. Consumers that ignore the UI columns match
// `label`, `editor_fn` and `"salt"` as opaque token trees.
