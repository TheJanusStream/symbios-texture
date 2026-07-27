//! Shared scaffolding for tileable surface generators.
//!
//! Mirrors the [`sprite`](crate::sprite) architecture for the surface
//! family: each generator module defines a *cell sampler* — a struct
//! implementing [`SurfaceCell`] that captures the per-generation state
//! (config, precomputed noise grids) and answers point queries — and the
//! shared [`generate_surface`] driver owns buffer allocation, sRGB albedo
//! packing, ORM packing, and the toroidal normal-map derivation.
//!
//! # Height-field convention
//!
//! [`SurfaceSample::height`] feeds [`height_to_normal`] **unmodified**: the
//! driver neither normalises nor clamps it, and `normal_strength` is passed
//! straight through.  Generators therefore control the gradient scale
//! themselves — e.g. a generator emitting raw `[-1, 1]` noise passes
//! `config.normal_strength * 0.5` to compensate for the doubled range,
//! exactly as the hand-rolled implementations did.
//!
//! # UV convention
//!
//! Cells are sampled at `u = x / width`, `v = y / height` (texel corner),
//! matching the toroidal grid samplers in [`crate::noise`], so cells that
//! index a precomputed [`sample_grid_into`](crate::noise::sample_grid_into)
//! buffer and cells that evaluate analytically agree on coordinates.

use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureMap, Workspace, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, height_to_normal},
    weathering::WeatheringConfig,
};

/// One point sample of a tileable surface.
///
/// `color` is linear RGB `[0, 1]` (the driver sRGB-encodes it);
/// `roughness` / `metallic` / `occlusion` land in the ORM green / blue /
/// red channels respectively; `height` feeds the normal-map derivation
/// (see the module docs for the scale convention).
pub struct SurfaceSample {
    /// Height value handed to [`height_to_normal`] unmodified.
    pub height: f64,
    /// Linear RGB albedo in `[0, 1]`.
    pub color: [f32; 3],
    /// PBR roughness `[0, 1]` (ORM green channel).
    pub roughness: f32,
    /// PBR metallic `[0, 1]` (ORM blue channel).
    pub metallic: f32,
    /// Ambient occlusion `[0, 1]` (ORM red channel).
    pub occlusion: f32,
    /// Emissive (glow) colour in linear RGB `[0, 1]`.  Ignored by
    /// [`generate_surface`]; collected into the texture map's emissive
    /// channel by [`generate_surface_emissive`].
    pub emissive: [f32; 3],
}

impl SurfaceSample {
    /// A dielectric, un-occluded sample — the common case for natural
    /// materials (`metallic = 0`, `occlusion = 1`).
    #[inline]
    pub fn matte(height: f64, color: [f32; 3], roughness: f32) -> Self {
        Self {
            height,
            color,
            roughness,
            metallic: 0.0,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

/// The shading half of a [`SurfaceSample`], retained in float form between
/// sampling and byte packing.
///
/// [`generate_surface_weathered`] keeps a grid of these so the
/// [`weathering`](crate::weathering) post-pass can blend against real linear
/// values; the unweathered driver packs straight to bytes and never
/// materialises the grid.  Height lives outside this struct because the
/// driver already keeps a height grid of its own for normal derivation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceField {
    /// Linear RGB albedo in `[0, 1]`.
    pub color: [f32; 3],
    /// PBR roughness `[0, 1]` (ORM green channel).
    pub roughness: f32,
    /// PBR metallic `[0, 1]` (ORM blue channel).
    pub metallic: f32,
    /// Ambient occlusion `[0, 1]` (ORM red channel).
    pub occlusion: f32,
}

impl From<&SurfaceSample> for SurfaceField {
    fn from(s: &SurfaceSample) -> Self {
        Self {
            color: s.color,
            roughness: s.roughness,
            metallic: s.metallic,
            occlusion: s.occlusion,
        }
    }
}

/// Pack one shaded texel into its albedo and ORM slots.
///
/// Shared by both drivers so the sRGB encoding and O/R/M channel order have
/// exactly one definition.
#[inline]
fn pack_texel(f: &SurfaceField, albedo_px: &mut [u8], orm_px: &mut [u8]) {
    albedo_px[0] = linear_to_srgb(f.color[0]);
    albedo_px[1] = linear_to_srgb(f.color[1]);
    albedo_px[2] = linear_to_srgb(f.color[2]);
    albedo_px[3] = 255;

    orm_px[0] = (f.occlusion.clamp(0.0, 1.0) * 255.0).round() as u8;
    orm_px[1] = (f.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
    orm_px[2] = (f.metallic.clamp(0.0, 1.0) * 255.0).round() as u8;
    orm_px[3] = 255;
}

/// A fully-instantiated surface sampler: configuration plus any precomputed
/// per-generation state (noise grids, lookup tables), ready to answer point
/// queries.
///
/// Implementations are constructed once per `generate()` call and sampled
/// for every texel by [`generate_surface`].
pub trait SurfaceCell {
    /// Sample the surface at texel `(x, y)` / UV `(u, v)`.
    ///
    /// Both coordinate forms are provided so grid-backed cells can index
    /// `y * width + x` directly (keeping the trigonometry-free fast path of
    /// precomputed toroidal grids) while analytic cells use UV.
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample;
}

/// Render a tileable `width × height` surface through `cell`.
///
/// The driver:
///
/// 1. samples every texel via [`SurfaceCell::sample`],
/// 2. packs albedo (sRGB-encoded, opaque) and ORM
///    (occlusion / roughness / metallic from the sample),
/// 3. derives the tangent-space normal map from the height field with
///    toroidal ([`BoundaryMode::Wrap`]) neighbours, so normals tile
///    seamlessly alongside the colour data.
///
/// Rows are sampled in parallel — `cell` must be `Sync`.  Work runs on the
/// ambient rayon pool: async generation tasks already execute on the
/// crate's private pool, so nested row-parallelism work-steals across that
/// pool's threads and `AsyncTextureConfig::pool_threads` remains the
/// effective CPU cap; direct synchronous calls parallelise on the caller's
/// pool (usually the global one).  Output is byte-identical to serial
/// evaluation — every sample is a pure function of its coordinates.
///
/// `workspace` (optional) pools the height-field buffer across calls; pass
/// the same [`Workspace`] from
/// [`generate_with_workspace`](crate::generator::TextureGenerator::generate_with_workspace)
/// to avoid re-allocating large grids at high resolutions.
///
/// (`AsyncTextureConfig` lives in the `bevy_symbios_texture` wrapper crate.)
pub fn generate_surface<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    workspace: Option<&mut Workspace>,
    cell: &C,
) -> Result<TextureMap, TextureError> {
    generate_surface_impl(width, height, normal_strength, workspace, cell, false)
}

/// [`generate_surface`] variant that also collects the per-sample
/// [`emissive`](SurfaceSample::emissive) colour into the texture map's
/// emissive channel (sRGB-encoded, like albedo).
///
/// Use this for glowing materials (lava, embers, neon); the polling systems
/// assign the resulting map to `StandardMaterial::emissive_texture`, where
/// it is multiplied by the material's emissive colour factor.
pub fn generate_surface_emissive<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    workspace: Option<&mut Workspace>,
    cell: &C,
) -> Result<TextureMap, TextureError> {
    generate_surface_impl(width, height, normal_strength, workspace, cell, true)
}

/// [`generate_surface`] variant that ages the result through the
/// [`weathering`](crate::weathering) post-pass before packing.
///
/// Weathering masks are neighbourhood operations on the height field, so this
/// driver samples into float fields, ages them, and packs afterwards — the
/// normal map is derived from the *aged* height field, which is what lets
/// corrosion crust read as raised.
///
/// A no-op [`WeatheringConfig`] delegates straight to [`generate_surface`],
/// so a generator can route through this unconditionally and pay nothing
/// until a layer is switched on.
pub fn generate_surface_weathered<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    workspace: Option<&mut Workspace>,
    cell: &C,
    weathering: &WeatheringConfig,
) -> Result<TextureMap, TextureError> {
    generate_surface_weathered_impl(
        width,
        height,
        normal_strength,
        workspace,
        cell,
        weathering,
        false,
    )
}

/// [`generate_surface_weathered`] with the emissive channel collected, for
/// glowing materials that also age.
///
/// Weathering never touches emissive: a stain dulls the surface around a glow
/// without dimming the glow itself.
pub fn generate_surface_weathered_emissive<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    workspace: Option<&mut Workspace>,
    cell: &C,
    weathering: &WeatheringConfig,
) -> Result<TextureMap, TextureError> {
    generate_surface_weathered_impl(
        width,
        height,
        normal_strength,
        workspace,
        cell,
        weathering,
        true,
    )
}

fn generate_surface_weathered_impl<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    mut workspace: Option<&mut Workspace>,
    cell: &C,
    weathering: &WeatheringConfig,
    emit: bool,
) -> Result<TextureMap, TextureError> {
    if weathering.is_noop() {
        return generate_surface_impl(width, height, normal_strength, workspace, cell, emit);
    }
    validate_dimensions(width, height)?;

    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut heights = workspace
        .as_deref_mut()
        .map_or_else(Vec::new, |ws| ws.take_grid());
    heights.clear();
    heights.resize(n, 0.0);

    // Every slot is overwritten during sampling; this is just the allocation.
    let mut fields = vec![
        SurfaceField {
            color: [0.0; 3],
            roughness: 0.0,
            metallic: 0.0,
            occlusion: 1.0,
        };
        n
    ];
    let mut emissive = if emit { vec![0u8; n * 4] } else { Vec::new() };

    // Phase 1 — sample into float fields, retaining shading for the post-pass.
    let sample_into = |x: usize,
                       y: usize,
                       field_slot: &mut SurfaceField,
                       height_slot: &mut f64,
                       emissive_px: Option<&mut [u8]>| {
        let u = x as f64 / w as f64;
        let v = y as f64 / h as f64;
        let s = cell.sample(x as u32, y as u32, u, v);

        *height_slot = s.height;
        *field_slot = SurfaceField::from(&s);

        if let Some(e) = emissive_px {
            e[0] = linear_to_srgb(s.emissive[0]);
            e[1] = linear_to_srgb(s.emissive[1]);
            e[2] = linear_to_srgb(s.emissive[2]);
            e[3] = 255;
        }
    };

    if emit {
        fields
            .par_chunks_mut(w)
            .zip(heights.par_chunks_mut(w))
            .zip(emissive.par_chunks_mut(w * 4))
            .enumerate()
            .for_each(|(y, ((field_row, height_row), emissive_row))| {
                for (x, (field_slot, height_slot)) in
                    field_row.iter_mut().zip(height_row.iter_mut()).enumerate()
                {
                    let ei = x * 4;
                    sample_into(
                        x,
                        y,
                        field_slot,
                        height_slot,
                        Some(&mut emissive_row[ei..ei + 4]),
                    );
                }
            });
    } else {
        fields
            .par_chunks_mut(w)
            .zip(heights.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (field_row, height_row))| {
                for (x, (field_slot, height_slot)) in
                    field_row.iter_mut().zip(height_row.iter_mut()).enumerate()
                {
                    sample_into(x, y, field_slot, height_slot, None);
                }
            });
    }

    // Phase 2 — age the shading, and the height field the normals come from.
    crate::weathering::apply(&mut fields, &mut heights, width, height, weathering);

    // Phase 3 — pack the aged fields.
    let mut albedo = vec![0u8; n * 4];
    let mut roughness = vec![0u8; n * 4];
    albedo
        .par_chunks_mut(w * 4)
        .zip(roughness.par_chunks_mut(w * 4))
        .zip(fields.par_chunks(w))
        .for_each(|((albedo_row, orm_row), field_row)| {
            for (x, f) in field_row.iter().enumerate() {
                let ai = x * 4;
                pack_texel(f, &mut albedo_row[ai..ai + 4], &mut orm_row[ai..ai + 4]);
            }
        });

    let normal = height_to_normal(&heights, width, height, normal_strength, BoundaryMode::Wrap);

    if let Some(ws) = workspace {
        ws.return_grid(heights);
    }

    Ok(TextureMap {
        albedo,
        normal,
        roughness,
        width,
        height,
        mip_level_count: 1,
        emissive: if emit { Some(emissive) } else { None },
    })
}

fn generate_surface_impl<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    mut workspace: Option<&mut Workspace>,
    cell: &C,
    emit: bool,
) -> Result<TextureMap, TextureError> {
    validate_dimensions(width, height)?;

    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut heights = workspace
        .as_deref_mut()
        .map_or_else(Vec::new, |ws| ws.take_grid());
    heights.clear();
    heights.resize(n, 0.0);

    let mut albedo = vec![0u8; n * 4];
    let mut roughness = vec![0u8; n * 4];
    let mut emissive = if emit { vec![0u8; n * 4] } else { Vec::new() };

    // Pack one sample into the albedo/ORM (and optional emissive) row slots.
    // A closure (not a fn) so it captures `cell`/`w`/`h` and stays under the
    // argument-count lint while serving both the emissive and plain paths.
    let write_pixel = |x: usize,
                       y: usize,
                       height_slot: &mut f64,
                       albedo_row: &mut [u8],
                       orm_row: &mut [u8],
                       emissive_px: Option<&mut [u8]>| {
        let u = x as f64 / w as f64;
        let v = y as f64 / h as f64;
        let s = cell.sample(x as u32, y as u32, u, v);

        *height_slot = s.height;

        let ai = x * 4;
        pack_texel(
            &SurfaceField::from(&s),
            &mut albedo_row[ai..ai + 4],
            &mut orm_row[ai..ai + 4],
        );

        if let Some(e) = emissive_px {
            e[0] = linear_to_srgb(s.emissive[0]);
            e[1] = linear_to_srgb(s.emissive[1]);
            e[2] = linear_to_srgb(s.emissive[2]);
            e[3] = 255;
        }
    };

    if emit {
        heights
            .par_chunks_mut(w)
            .zip(albedo.par_chunks_mut(w * 4))
            .zip(roughness.par_chunks_mut(w * 4))
            .zip(emissive.par_chunks_mut(w * 4))
            .enumerate()
            .for_each(|(y, (((height_row, albedo_row), orm_row), emissive_row))| {
                for (x, height_slot) in height_row.iter_mut().enumerate() {
                    let ai = x * 4;
                    write_pixel(
                        x,
                        y,
                        height_slot,
                        albedo_row,
                        orm_row,
                        Some(&mut emissive_row[ai..ai + 4]),
                    );
                }
            });
    } else {
        heights
            .par_chunks_mut(w)
            .zip(albedo.par_chunks_mut(w * 4))
            .zip(roughness.par_chunks_mut(w * 4))
            .enumerate()
            .for_each(|(y, ((height_row, albedo_row), orm_row))| {
                for (x, height_slot) in height_row.iter_mut().enumerate() {
                    write_pixel(x, y, height_slot, albedo_row, orm_row, None);
                }
            });
    }

    let normal = height_to_normal(&heights, width, height, normal_strength, BoundaryMode::Wrap);

    if let Some(ws) = workspace {
        ws.return_grid(heights);
    }

    Ok(TextureMap {
        albedo,
        normal,
        roughness,
        width,
        height,
        mip_level_count: 1,
        emissive: if emit { Some(emissive) } else { None },
    })
}

/// Linear interpolation between two `f32` values with `t` clamped to
/// `[0, 1]` — the shared home for the helper every surface module used to
/// define locally.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant-output cell for driver-level assertions.
    struct Flat;

    impl SurfaceCell for Flat {
        fn sample(&self, _x: u32, _y: u32, _u: f64, _v: f64) -> SurfaceSample {
            SurfaceSample {
                height: 0.5,
                color: [1.0, 0.0, 0.0],
                roughness: 0.5,
                metallic: 1.0,
                occlusion: 0.0,
                emissive: [0.0, 0.0, 0.0],
            }
        }
    }

    #[test]
    fn driver_packs_orm_channels() {
        let map = generate_surface(4, 4, 1.0, None, &Flat).expect("generate");
        // O=R, R=G, M=B channel order.
        assert_eq!(map.roughness[0], 0, "occlusion 0.0 → 0");
        assert_eq!(map.roughness[1], 128, "roughness 0.5 → 128");
        assert_eq!(map.roughness[2], 255, "metallic 1.0 → 255");
        assert_eq!(map.roughness[3], 255);
        // Albedo is opaque and sRGB-encoded.
        assert_eq!(map.albedo[0], 255);
        assert_eq!(map.albedo[3], 255);
        // Flat height field → neutral normal (128, 128, 255).
        assert_eq!(&map.normal[0..4], &[128, 128, 255, 255]);
    }

    #[test]
    fn driver_rejects_invalid_dimensions() {
        assert!(generate_surface(0, 4, 1.0, None, &Flat).is_err());
        assert!(generate_surface(4, 0, 1.0, None, &Flat).is_err());
    }

    #[test]
    fn driver_reuses_workspace_buffers() {
        let mut ws = Workspace::new();
        let a = generate_surface(8, 8, 1.0, Some(&mut ws), &Flat).expect("first");
        let b = generate_surface(8, 8, 1.0, Some(&mut ws), &Flat).expect("second");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn matte_sets_dielectric_defaults() {
        let s = SurfaceSample::matte(0.3, [0.1, 0.2, 0.3], 0.7);
        assert_eq!(s.metallic, 0.0);
        assert_eq!(s.occlusion, 1.0);
    }

    #[test]
    fn lerp_clamps_t() {
        assert_eq!(lerp(0.0, 1.0, -1.0), 0.0);
        assert_eq!(lerp(0.0, 1.0, 2.0), 1.0);
        assert_eq!(lerp(0.0, 1.0, 0.5), 0.5);
    }

    /// A ridged cell, so the weathering masks have convexity and cavity to
    /// find; a flat surface would age to nothing.
    struct Ridged;

    impl SurfaceCell for Ridged {
        fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
            use std::f64::consts::TAU;
            let height = (TAU * 3.0 * u).sin() * (TAU * 3.0 * v).sin();
            SurfaceSample {
                height,
                color: [0.4, 0.42, 0.45],
                roughness: 0.6,
                metallic: 0.0,
                occlusion: 1.0,
                emissive: [0.2, 0.0, 0.0],
            }
        }
    }

    fn weathered_config() -> crate::weathering::WeatheringConfig {
        use crate::weathering::{Corrosion, EdgeWear, WeatheringConfig};
        WeatheringConfig {
            seed: 5,
            edge_wear: EdgeWear {
                amount: 1.0,
                ..Default::default()
            },
            corrosion: Corrosion {
                amount: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// A no-op weathering config must take the plain path and produce exactly
    /// the same bytes — that is what makes it safe to route a generator
    /// through the weathered driver unconditionally.
    #[test]
    fn noop_weathering_matches_the_plain_driver() {
        let plain = generate_surface(32, 32, 1.0, None, &Ridged).expect("plain");
        let weathered = generate_surface_weathered(
            32,
            32,
            1.0,
            None,
            &Ridged,
            &crate::weathering::WeatheringConfig::default(),
        )
        .expect("weathered");

        assert_eq!(plain.albedo, weathered.albedo, "albedo drifted");
        assert_eq!(plain.normal, weathered.normal, "normal drifted");
        assert_eq!(plain.roughness, weathered.roughness, "ORM drifted");
        assert!(weathered.emissive.is_none());
    }

    /// Enabled weathering must reach the packed output, and must re-derive
    /// normals from the aged height field so corrosion crust is lit.
    #[test]
    fn weathering_changes_albedo_and_normals() {
        let plain = generate_surface(48, 48, 1.0, None, &Ridged).expect("plain");
        let aged = generate_surface_weathered(48, 48, 1.0, None, &Ridged, &weathered_config())
            .expect("weathered");

        assert_ne!(plain.albedo, aged.albedo, "weathering never reached albedo");
        assert_ne!(
            plain.roughness, aged.roughness,
            "weathering never reached ORM"
        );
        assert_ne!(
            plain.normal, aged.normal,
            "corrosion relief was not folded into the normal map"
        );
        assert_eq!(aged.width, 48);
        assert_eq!(aged.albedo.len(), 48 * 48 * 4);
    }

    /// The weathered emissive path collects glow, and weathering leaves that
    /// glow alone.
    #[test]
    fn weathered_emissive_preserves_glow() {
        let plain = generate_surface_emissive(32, 32, 1.0, None, &Ridged).expect("plain");
        let aged =
            generate_surface_weathered_emissive(32, 32, 1.0, None, &Ridged, &weathered_config())
                .expect("weathered");

        let plain_emissive = plain.emissive.expect("plain emissive");
        let aged_emissive = aged.emissive.expect("aged emissive");
        assert_eq!(
            plain_emissive, aged_emissive,
            "weathering dimmed the emissive channel"
        );
        assert_ne!(plain.albedo, aged.albedo, "weathering never reached albedo");
    }

    #[test]
    fn weathered_driver_validates_dimensions_and_reuses_workspace() {
        assert!(
            generate_surface_weathered(0, 4, 1.0, None, &Ridged, &weathered_config()).is_err(),
            "zero width accepted"
        );

        let mut ws = Workspace::new();
        let a =
            generate_surface_weathered(16, 16, 1.0, Some(&mut ws), &Ridged, &weathered_config())
                .expect("first");
        let b =
            generate_surface_weathered(16, 16, 1.0, Some(&mut ws), &Ridged, &weathered_config())
                .expect("second");
        assert_eq!(a.albedo, b.albedo, "pooled buffers changed the result");
        assert_eq!(a.normal, b.normal);
    }
}
