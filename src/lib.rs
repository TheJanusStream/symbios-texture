//! `symbios-texture` — the Bevy-free core of `bevy_symbios_texture`.
//!
//! This crate holds every pure procedural-generation module (bark, rock,
//! brick, leaf, sprite atlases, …), their `Config`/`Generator` types, the
//! [`TextureGenerator`] trait, the raw [`TextureMap`] pixel/mip pipeline, the
//! [`ToroidalNoise`] seamless-tiling wrapper, the shared
//! [`sprite`]/[`surface`] sampling kits, the per-config
//! [`symbios_genetics::Genotype`] implementations, and the canonical
//! generator [`registry`].
//!
//! It depends only on bevy-free crates (`noise`, `rand`, `rayon`, `serde`,
//! `symbios-genetics`).  The Bevy-coupled plugin, async generation pool, asset
//! adapters, resource cache, and egui UI live in the `bevy_symbios_texture`
//! wrapper crate, which re-exports this crate wholesale so the public API is
//! unchanged.
//!
//! [`TextureGenerator`]: generator::TextureGenerator
//! [`TextureMap`]: generator::TextureMap
//! [`ToroidalNoise`]: noise::ToroidalNoise

pub mod ashlar;
pub mod asphalt;
pub mod bark;
pub mod brick;
pub mod broadleaf;
pub mod cactus;
pub mod chain_link;
pub mod cobblestone;
pub mod concrete;
pub mod corrugated;
pub mod encaustic;
pub mod fabric;
pub mod fingerprint;
pub mod flame;
pub mod flower;
pub mod frond;
pub mod generator;
pub mod genetics;
pub mod grass;
pub mod ground;
pub mod ice;
pub mod iron_grille;
pub mod lava;
pub mod leaf;
pub mod leaf_sprite;
pub mod lichen;
pub mod log_end;
pub mod marble;
pub mod metal;
pub mod moss;
pub mod needle;
pub mod noise;
pub mod normal;
pub mod pavers;
pub mod petal;
pub mod plank;
pub mod puff;
pub mod reed;
pub mod registry;
pub mod ring;
pub mod rock;
pub mod sand;
pub mod shard;
pub mod shingle;
pub mod snow;
pub mod snowflake;
pub mod soft_disc;
pub mod spark;
pub mod sprite;
pub mod stained_glass;
pub mod stucco;
pub mod surface;
pub mod thatch;
pub mod twig;
pub mod wainscoting;
pub mod window;

pub use broadleaf::{BroadleafConfig, BroadleafGenerator};
pub use cactus::{CactusSkinConfig, CactusSkinGenerator};
pub use frond::{FrondConfig, FrondGenerator};
pub use generator::{
    MAX_DIMENSION, TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions,
};
pub use grass::{GrassTuftConfig, GrassTuftGenerator};
pub use leaf::{LeafConfig, LeafGenerator, LeafSample, LeafSampler, sample_leaf};
pub use lichen::{LichenConfig, LichenGenerator};
pub use moss::{MossConfig, MossGenerator};
pub use needle::{NeedleConfig, NeedleGenerator};
pub use noise::ToroidalNoise;
pub use reed::{ReedConfig, ReedGenerator};
pub use sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas};
pub use surface::{SurfaceCell, SurfaceSample, generate_surface};
pub use twig::{TwigConfig, TwigGenerator};
