# symbios-texture

Procedural PBR texture generation, engine-free. This crate is the core of
`bevy_symbios_texture`: every generator here is pure CPU code with no Bevy
dependency, so it can be used from any Rust project (or headless tooling)
that wants deterministic, seamlessly tiling textures.

The Bevy-coupled plugin — async generation pool, asset adapters, resource
cache, and egui editor UI — lives in the `bevy_symbios_texture` wrapper
crate, which re-exports this crate wholesale.

## What you get

- **48 generators** covering building materials (brick, ashlar, plank,
  stucco, metal, …), natural surfaces (rock, ground, sand, snow, ice,
  lava, …), vegetation (bark, leaf, grass, moss, lichen, cactus, …), and
  particle sprites (spark, puff, snowflake, flame, …).
- **Full PBR output**: every generator produces an albedo map (sRGB), a
  tangent-space normal map, and an ORM (occlusion/roughness/metallic) map;
  glowing materials (lava) add an emissive map. Optional CPU-side mipmap
  chains with type-correct filtering (linear-light for sRGB, renormalising
  for normals).
- **Seamless tiling**: surface generators sample noise on a 4-D torus
  ([`ToroidalNoise`]), so every tileable texture wraps perfectly at all four
  edges — no seams, no mirroring tricks.
- **Sprite atlases**: card generators can bake an `N × M` atlas of shape
  variants of the same configuration in a single image, for per-particle
  variety from one texture.
- **Evolvable configs**: every config struct implements
  [`symbios_genetics::Genotype`] (mutation + crossover), so textures can be
  bred with `SimpleGA`, `Nsga2`, or `MapElites`.
- **Deterministic**: the same config and size always produce the same bytes,
  across runs and platforms. Parallel generation (rayon) is byte-identical
  to serial. Configs are serde-serializable and structurally fingerprintable
  for cache keys.

## Quick start

```rust
use symbios_texture::{TextureGenerator, TextureMap};
use symbios_texture::rock::{RockConfig, RockGenerator};

fn main() -> Result<(), symbios_texture::TextureError> {
    let generator = RockGenerator::new(RockConfig::default());

    // Albedo + normal + ORM, RGBA8, row-major.
    let map: TextureMap = generator.generate(512, 512)?;
    assert_eq!(map.albedo.len(), 512 * 512 * 4);

    // Optionally append the full mipmap chain on the CPU.
    let map = map.with_mips();
    assert!(map.mip_level_count > 1);
    Ok(())
}
```

Dimensions are validated: zero or anything above
[`MAX_DIMENSION`](https://docs.rs/symbios-texture) (4096 per side, chosen to
bound peak memory) returns a `TextureError` instead of panicking.

### Reusing allocations

At high resolutions each intermediate `f64` grid is large (128 MB at
4096²). A [`Workspace`] pools those buffers across generations:

```rust
use symbios_texture::{TextureGenerator, Workspace};
use symbios_texture::thatch::{ThatchConfig, ThatchGenerator};

let generator = ThatchGenerator::new(ThatchConfig::default());
let mut workspace = Workspace::new();

// First call allocates; subsequent calls reuse the same heap buffers.
let a = generator.generate_with_workspace(1024, 1024, &mut workspace).unwrap();
let b = generator.generate_with_workspace(1024, 1024, &mut workspace).unwrap();
assert_eq!(a.albedo, b.albedo);
```

### Evolving configs

```rust
use rand::rng;
use symbios_genetics::Genotype;
use symbios_texture::bark::BarkConfig;

let mut parent_a = BarkConfig::default();
let parent_b = BarkConfig { seed: 7, ..BarkConfig::default() };

parent_a.mutate(&mut rng(), 0.2);            // perturb ~20% of fields
let child = parent_a.crossover(&parent_b, &mut rng());
```

Mutation respects each field's natural range (and post-hooks re-snap
tiling invariants like brick row offsets), so evolved configs always stay
valid.

## Generator roster

The canonical list lives in [`src/registry.rs`](src/registry.rs), which the
wrapper crate consumes via the `for_each_generator!` macro. Two kinds:

- **Surface** — tileable, opaque, meant for repeat samplers.
- **Card** — alpha-masked cutouts (foliage, windows, particles), meant for
  clamp-to-edge samplers; most can bake variant atlases.

- **Surface** (26): ashlar, asphalt, bark, brick, cactus (skin),
  cobblestone, concrete, corrugated, encaustic, fabric, ground, ice, lava,
  lichen, marble, metal, moss, pavers, plank, rock, sand, shingle, snow,
  stucco, thatch, wainscoting
- **Card** (22): broadleaf, chain link, flame, flower, frond, grass tuft,
  iron grille, leaf, leaf sprite, log end, needle, petal, puff, reed, ring,
  shard, snowflake, soft disc, spark, stained glass, twig, window

Each module follows the same shape: a serde-serializable `*Config` struct
with documented fields and sensible `Default`s, plus a `*Generator` that
implements [`TextureGenerator`].

## Crate layout

- [`generator`](src/generator.rs) — the [`TextureGenerator`] trait,
  [`TextureMap`] output type, mipmap generation, dimension validation.
- [`noise`](src/noise.rs) — [`ToroidalNoise`] seamless-tiling wrapper,
  grid samplers, toroidal Voronoi.
- [`surface`](src/surface.rs) / [`sprite`](src/sprite.rs) — shared drivers
  for the tileable-surface and sprite-atlas families (buffer packing,
  normal derivation, atlas layout).
- [`normal`](src/normal.rs) — heightmap → tangent-space normal map.
- [`genetics`](src/genetics.rs) — `Genotype` impls for all configs.
- [`fingerprint`](src/fingerprint.rs) — stable structural hashing of
  configs for cache keys.
- [`registry`](src/registry.rs) — the generator table macro.
- One module per generator (`bark`, `rock`, `brick`, …).

## Adding a generator

1. Write the module: a config struct (serde + `Default`) and a
   `TextureGenerator` impl, with tests.
2. Add one row to the table in [`src/registry.rs`](src/registry.rs)
   (pick `Surface` or `Card`).
3. Add the per-field `impl_genotype!` table in
   [`src/genetics.rs`](src/genetics.rs), and the `impl_config_editor!`
   table in the wrapper crate's `ui.rs`.
4. Add it to the roster table above.

## License

MIT — see [LICENSE](LICENSE).

[`ToroidalNoise`]: src/noise.rs
[`TextureGenerator`]: src/generator.rs
[`TextureMap`]: src/generator.rs
[`Workspace`]: src/generator.rs
[`symbios_genetics::Genotype`]: https://crates.io/crates/symbios-genetics
