//! Post-pass that ages a generated surface: edge wear, corrosion, crevice
//! dirt, and gravity streaks.
//!
//! # Why this is a post-pass
//!
//! Every mask here is a *neighbourhood* operation on the height field —
//! a Laplacian for convexity, a wide blur for cavity, a swept distance field
//! for corrosion growth, a running accumulation down each column for streaks.
//! None of them can be answered by [`SurfaceCell::sample`], which is a pure
//! point query with no view of its neighbours.  So the weathered driver
//! ([`generate_surface_weathered`]) samples the surface into float fields
//! first, ages those fields here, and only then packs to bytes.
//!
//! [`SurfaceCell::sample`]: crate::surface::SurfaceCell::sample
//! [`generate_surface_weathered`]: crate::surface::generate_surface_weathered
//!
//! # Layer order
//!
//! Layers are applied in the order material actually ages, so enabling
//! several at once compounds the way a real surface does:
//!
//! 1. **Edge wear** — raised edges rub back to the substrate underneath.
//! 2. **Corrosion** — rust or patina creeps out of crevices, adding its own
//!    crust relief.
//! 3. **Crevice dirt** — grime settles into recesses and onto upward faces.
//! 4. **Streaks** — runoff draws down from ledges over everything above.
//!
//! # Tiling
//!
//! Every pass wraps at the tile edges (blur, Laplacian, growth front, and the
//! downward accumulation all use modular neighbours), so a weathered surface
//! tiles exactly as seamlessly as the surface it aged.
//!
//! # Cost
//!
//! Each enabled layer allocates one or two `f64` grids and walks the tile a
//! handful of times — a fixed handful, independent of how far corrosion
//! spreads or how wide a crevice counts.  At the catalogue's 256² and splat's
//! 512² this is a few milliseconds.  Grids are
//! allocated only for layers that are switched on, and
//! [`WeatheringConfig::default`] switches every layer off — so a config that
//! embeds weathering but never enables it costs nothing at all.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::SurfaceField,
};

/// How far a blur may reach when a knob is at its maximum, as a fraction of
/// the shorter tile edge.  Caps the radius so an extreme config cannot turn
/// the whole tile into one smear.
const MAX_RADIUS_FRACTION: f64 = 0.25;

/// Forward/backward sweep pairs used to propagate the corrosion distance
/// field.
///
/// A single pair resolves any front that only travels "downstream", but the
/// grid wraps, so a blotch spreading across the tile seam needs distance to
/// travel back around; three pairs settles that for reaches up to the
/// [`MAX_RADIUS_FRACTION`] cap.
const GROWTH_SWEEPS: usize = 3;

/// Percentile of a mask's magnitude that [`percentile_scaled`] maps to 1.
///
/// Just short of the maximum, so a handful of spike texels cannot flatten
/// everything else, but high enough that genuine features still reach full
/// strength.
const NORMALISATION_PERCENTILE: f64 = 0.98;

/// Share of [`Corrosion::coverage`] seeded as nuclei, with growth expected to
/// supply the remainder.
///
/// Well under 1 so blotches are mostly *grown* rather than raw noise
/// thresholded to shape.
const NUCLEI_SHARE: f64 = 0.35;

/// Ages a surface after generation — see the [module docs](self).
///
/// [`Default`] leaves every layer switched off, which makes this safe to
/// embed in an existing generator config behind `#[serde(default)]`: old
/// saved configs keep deserialising, and output is unchanged until a layer is
/// explicitly turned up.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeatheringConfig {
    /// PRNG seed for the blotch, breakup, and streak patterns.
    pub seed: u32,
    /// Substrate showing through on rubbed edges.
    pub edge_wear: EdgeWear,
    /// Rust or patina creeping out of crevices.
    pub corrosion: Corrosion,
    /// Grime settled into recesses.
    pub crevice_dirt: CreviceDirt,
    /// Runoff drawn down from ledges.
    pub streaks: Streaks,
}

impl WeatheringConfig {
    /// Whether every layer is switched off, in which case ageing can be
    /// skipped outright.
    pub fn is_noop(&self) -> bool {
        self.edge_wear.amount <= 0.0
            && self.corrosion.amount <= 0.0
            && self.crevice_dirt.amount <= 0.0
            && self.streaks.amount <= 0.0
    }
}

/// Wear on raised edges, exposing the material beneath a finish.
///
/// Keyed on convexity, so it lands exactly where a surface would be rubbed:
/// brick arrises, plank edges, the crowns of rivets.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeWear {
    /// Strength of the layer; `0` disables it entirely.
    pub amount: f32,
    /// Linear RGB of the substrate revealed — bare metal under paint, pale
    /// stone under a weathered finish.
    pub color: [f32; 3],
    /// How convex a texel must be before it starts wearing, in `[0, 1]` of
    /// the tile's typical peak convexity (the 98th percentile, not the
    /// outright maximum).  Higher confines wear to the sharpest edges only.
    pub threshold: f64,
    /// Frequency of the noise that breaks wear into patches rather than an
    /// even outline.  Higher is more speckled.
    pub breakup_scale: f64,
    /// Roughness of the exposed substrate — rubbed metal is *smoother* than
    /// the finish that covered it.
    pub roughness: f32,
    /// Metallic value of the exposed substrate.
    pub metallic: f32,
}

impl Default for EdgeWear {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: [0.55, 0.55, 0.58],
            threshold: 0.35,
            breakup_scale: 8.0,
            roughness: 0.35,
            metallic: 0.9,
        }
    }
}

/// Rust or patina, seeded in crevices and grown outward.
///
/// The front spreads at a noise-modulated rate, so blotches come out ragged
/// and irregular rather than as tidy discs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Corrosion {
    /// Strength of the layer; `0` disables it entirely.
    pub amount: f32,
    /// Linear RGB of the corrosion product — orange-brown for rust,
    /// green-cyan for verdigris.
    pub color: [f32; 3],
    /// How much of the tile ends up corroded, in `[0, 1]`.
    ///
    /// A ceiling rather than a raw threshold: corrosion never claims more
    /// than this however generous [`spread`](Self::spread) is, and reaches it
    /// as long as spread gives the blotches room to grow.  Thresholding the
    /// noise directly would make the knob almost binary — FBM values cluster
    /// hard around the midpoint — and seeding this fraction and then growing
    /// it would flood the tile.
    pub coverage: f32,
    /// How far corrosion creeps out from its seeds, as a fraction of the
    /// shorter tile edge.
    ///
    /// Expressed relative to the tile rather than in texels or iterations so
    /// a config looks the same baked at 256² and at 512².
    pub spread: f64,
    /// Frequency of the noise that modulates how fast the front advances;
    /// lower gives larger, smoother blotches.
    pub barrier_scale: f64,
    /// Crust height added where corrosion is thickest, in height-field units.
    /// Feeds the normal map, so corroded areas read as raised.
    pub relief: f64,
    /// Roughness of the corroded surface — corrosion product is matte.
    pub roughness: f32,
    /// Metallic value of the corroded surface.  Rust is an oxide, so this
    /// should stay near zero even on steel.
    pub metallic: f32,
}

impl Default for Corrosion {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: [0.34, 0.13, 0.05],
            coverage: 0.18,
            spread: 0.08,
            barrier_scale: 5.0,
            relief: 0.04,
            roughness: 0.92,
            metallic: 0.0,
        }
    }
}

/// Grime settled into recesses and onto upward-facing ledges.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CreviceDirt {
    /// Strength of the layer; `0` disables it entirely.
    pub amount: f32,
    /// Linear RGB of the grime.
    pub color: [f32; 3],
    /// How wide a recess counts as a crevice, as a fraction of the tile.
    /// Larger values catch broad hollows as well as tight seams.
    pub depth: f64,
    /// How strongly upward-facing surfaces are favoured over recesses alone,
    /// in `[0, 1]`.  At `0` dirt fills every hollow evenly; at `1` it collects
    /// only where a surface would actually catch falling grime.
    pub gravity: f32,
    /// Roughness of the grime — dirt is matte.
    pub roughness: f32,
    /// Occlusion written where grime is thickest; below `1` this darkens
    /// crevices, which is what sells the depth.
    pub occlusion: f32,
}

impl Default for CreviceDirt {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: [0.10, 0.09, 0.07],
            depth: 0.04,
            gravity: 0.6,
            roughness: 0.95,
            occlusion: 0.55,
        }
    }
}

/// Runoff stains drawn downward from ledges.
///
/// Tile space is treated as upright with **+V pointing down**, which matches
/// how these textures sit on a wall.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Streaks {
    /// Strength of the layer; `0` disables it entirely.
    pub amount: f32,
    /// Linear RGB of the stain.
    pub color: [f32; 3],
    /// Fraction of candidate ledges that actually run, in `[0, 1]`.
    pub density: f32,
    /// How far a stain runs before fading out, as a fraction of the tile
    /// height.
    ///
    /// Expressed relative to the tile, like [`Corrosion::spread`], so a
    /// config draws the same-looking runs at any bake resolution — a per-row
    /// decay factor would silently halve the run length every time the
    /// texture size doubled.
    pub length: f64,
    /// How far a streak may wander sideways as it descends, in texels per
    /// row.
    pub wander: f64,
    /// Roughness of the stained surface.
    pub roughness: f32,
}

impl Default for Streaks {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: [0.12, 0.11, 0.10],
            density: 0.25,
            length: 0.35,
            wander: 0.5,
            roughness: 0.85,
        }
    }
}

/// Age `fields` in place, using `heights` as the shape the masks are derived
/// from.
///
/// `heights` is taken mutably because corrosion adds crust relief; the caller
/// derives the normal map *after* this returns so the added relief is lit.
///
/// A no-op config returns immediately without allocating.  Mismatched buffer
/// lengths are ignored rather than panicking, since a partially-aged surface
/// would be worse than an un-aged one.
pub fn apply(
    fields: &mut [SurfaceField],
    heights: &mut [f64],
    width: u32,
    height: u32,
    cfg: &WeatheringConfig,
) {
    if cfg.is_noop() {
        return;
    }

    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if n == 0 || fields.len() != n || heights.len() != n {
        return;
    }

    // Shared geometry: convexity drives wear and streak sources, cavity
    // drives dirt and biases where corrosion takes hold.
    let needs_convexity = cfg.edge_wear.amount > 0.0 || cfg.streaks.amount > 0.0;
    let needs_cavity = cfg.crevice_dirt.amount > 0.0 || cfg.corrosion.amount > 0.0;

    let convex = needs_convexity.then(|| percentile_scaled(convexity(heights, w, h)));
    let hollow = needs_cavity.then(|| {
        let radius = radius_texels(cfg.crevice_dirt.depth, w, h);
        percentile_scaled(cavity(heights, w, h, radius))
    });

    if let Some(convex) = convex.as_deref().filter(|_| cfg.edge_wear.amount > 0.0) {
        apply_edge_wear(fields, convex, w, h, cfg);
    }
    if let Some(hollow) = hollow.as_deref().filter(|_| cfg.corrosion.amount > 0.0) {
        apply_corrosion(fields, heights, hollow, w, h, cfg);
    }
    if let Some(hollow) = hollow.as_deref().filter(|_| cfg.crevice_dirt.amount > 0.0) {
        apply_crevice_dirt(fields, hollow, heights, w, h, cfg);
    }
    if let Some(convex) = convex.as_deref().filter(|_| cfg.streaks.amount > 0.0) {
        apply_streaks(fields, convex, heights, w, h, cfg);
    }
}

// ── Layers ────────────────────────────────────────────────────────────────

fn apply_edge_wear(
    fields: &mut [SurfaceField],
    convex: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) {
    let c = &cfg.edge_wear;
    let breakup = noise_grid(w, h, cfg.seed.wrapping_add(0x5EA1), c.breakup_scale, 3);
    let threshold = c.threshold.clamp(0.0, 0.999);
    let amount = c.amount.clamp(0.0, 1.0);

    for (i, f) in fields.iter_mut().enumerate() {
        // Only the convex side wears; recesses are protected.
        let exposure = smoothstep(threshold, 1.0, convex[i].max(0.0));
        if exposure <= 0.0 {
            continue;
        }
        // Break the outline into patches so wear is not a uniform pinstripe.
        let patch = smoothstep(0.35, 0.85, normalize(breakup[i]));
        let m = (exposure * patch) as f32 * amount;
        if m <= 0.0 {
            continue;
        }
        f.color = lerp3(f.color, c.color, m);
        f.roughness = lerp(f.roughness, c.roughness, m);
        f.metallic = lerp(f.metallic, c.metallic, m);
    }
}

fn apply_corrosion(
    fields: &mut [SurfaceField],
    heights: &mut [f64],
    hollow: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) {
    let c = &cfg.corrosion;
    let amount = c.amount.clamp(0.0, 1.0);
    let barrier = noise_grid(w, h, cfg.seed.wrapping_add(0x7005), c.barrier_scale, 3);
    let mask = corrosion_mask(hollow, &barrier, w, h, cfg);

    for (i, f) in fields.iter_mut().enumerate() {
        let m = mask[i] as f32 * amount;
        if m <= 0.0 {
            continue;
        }
        f.color = lerp3(f.color, c.color, m);
        f.roughness = lerp(f.roughness, c.roughness, m);
        f.metallic = lerp(f.metallic, c.metallic, m);
        // Corrosion product is bulkier than the metal it replaced, so the
        // crust stands slightly proud and catches light.
        heights[i] += c.relief * mask[i] * normalize(barrier[i]);
    }
}

fn apply_crevice_dirt(
    fields: &mut [SurfaceField],
    hollow: &[f64],
    heights: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) {
    let c = &cfg.crevice_dirt;
    let amount = c.amount.clamp(0.0, 1.0);
    let gravity = c.gravity.clamp(0.0, 1.0) as f64;
    let facing = upward_facing(heights, w, h);

    for (i, f) in fields.iter_mut().enumerate() {
        let recess = hollow[i].max(0.0);
        if recess <= 0.0 {
            continue;
        }
        // Blend "fills every hollow" with "only what would catch grime".
        let m = (recess * (1.0 - gravity + gravity * facing[i])) as f32 * amount;
        if m <= 0.0 {
            continue;
        }
        f.color = lerp3(f.color, c.color, m);
        f.roughness = lerp(f.roughness, c.roughness, m);
        f.occlusion = lerp(f.occlusion, c.occlusion, m);
    }
}

fn apply_streaks(
    fields: &mut [SurfaceField],
    convex: &[f64],
    heights: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) {
    let c = &cfg.streaks;
    let amount = c.amount.clamp(0.0, 1.0);
    let mask = streak_mask(convex, heights, w, h, cfg);

    for (i, f) in fields.iter_mut().enumerate() {
        let m = mask[i] as f32 * amount;
        if m <= 0.0 {
            continue;
        }
        f.color = lerp3(f.color, c.color, m);
        f.roughness = lerp(f.roughness, c.roughness, m);
    }
}

// ── Masks ─────────────────────────────────────────────────────────────────

/// Discrete Laplacian, negated so **positive means convex** (ridges, edges,
/// crowns) and negative means concave (seams, pits).
///
/// Neighbours wrap, so the result tiles.
fn convexity(heights: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let above = ((y + h - 1) % h) * w;
        let below = ((y + 1) % h) * w;
        let row = y * w;
        for x in 0..w {
            let left = (x + w - 1) % w;
            let right = (x + 1) % w;
            let laplacian = heights[row + left]
                + heights[row + right]
                + heights[above + x]
                + heights[below + x]
                - 4.0 * heights[row + x];
            out[row + x] = -laplacian;
        }
    }
    out
}

/// How recessed each texel is relative to its wider surroundings: positive in
/// hollows, negative on raised ground.
///
/// A cheap stand-in for a baked cavity map — `blur(height) − height`, with the
/// blur wrapping so the result tiles.
fn cavity(heights: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let blurred = box_blur_wrapped(heights, w, h, radius);
    blurred.iter().zip(heights).map(|(b, s)| b - s).collect()
}

/// Weight in `[0, 1]` for surfaces that face "up" in tile space, where +V
/// points down.
///
/// A stylisation, not a physical normal: a height field applied to a wall has
/// no true world-space up, so this reads the vertical gradient and treats the
/// upper lip of a step — where height falls away as V increases — as the face
/// that would catch falling grime.
fn upward_facing(heights: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    let mut peak = 0.0f64;
    for y in 0..h {
        let above = ((y + h - 1) % h) * w;
        let below = ((y + 1) % h) * w;
        let row = y * w;
        for x in 0..w {
            let slope = heights[above + x] - heights[below + x];
            out[row + x] = slope.max(0.0);
            peak = peak.max(out[row + x]);
        }
    }
    if peak > 0.0 {
        for v in &mut out {
            *v = smoothstep(0.0, 1.0, *v / peak);
        }
    }
    out
}

/// Seed corrosion in the most vulnerable `coverage` fraction of the tile,
/// then creep outward from those seeds.
///
/// Growth is a chamfer distance transform with a spatially-varying step cost:
/// the front advances slowly through high-cost noise and races through
/// low-cost noise, which is what makes blotch outlines ragged instead of
/// circular.  Sweeping the grid a few times costs `O(n)` no matter how far
/// corrosion reaches — the obvious alternative, one grid walk per texel of
/// reach, would make cost scale with resolution for no visual gain.
///
/// Neighbour lookups wrap, so blotches cross the tile edge intact; a handful
/// of sweeps is what lets distance propagate the long way around the torus.
fn corrosion_mask(
    hollow: &[f64],
    barrier: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) -> Vec<f64> {
    let c = &cfg.corrosion;
    let n = w * h;
    // Seed noise runs *below* the barrier frequency, with few octaves: nuclei
    // have to come out as connected patches, because growth only dilates what
    // it is given.  High-frequency seeds drop a nucleus every few texels, and
    // no amount of growing or trimming turns that back into blotches — it
    // just dithers.
    let seeds = noise_grid(
        w,
        h,
        cfg.seed.wrapping_add(0x0BAD),
        c.barrier_scale * 0.5,
        2,
    );

    // Vulnerability: mostly noise, nudged toward recesses that hold moisture.
    //
    // The nudge is deliberately weak.  Letting geometry lead looks right until
    // it meets masonry, where the recesses form one connected mortar network:
    // the seeds then *become* that network, and any spread at all floods the
    // whole tile.  Noise-led seeding keeps corrosion patchy along the seams
    // instead of tracing every one of them.
    let vulnerability: Vec<f64> = (0..n)
        .map(|i| normalize(seeds[i]) * (0.8 + 0.2 * hollow[i].clamp(-1.0, 1.0)))
        .collect();

    let coverage = c.coverage.clamp(0.0, 1.0) as f64;
    if coverage <= 0.0 {
        return vec![0.0; n];
    }

    // Nuclei take only a share of the target extent, so growth supplies the
    // rest and the trim below has something to choose from.  Seeding the full
    // `coverage` as nuclei would leave the plateau alone filling the quota,
    // and the trim would discard every grown texel — blotches with no growth
    // in them at all.
    let nuclei = ((coverage * NUCLEI_SHARE * n as f64).round() as usize).clamp(1, n);
    let nucleus_floor = *vulnerability
        .clone()
        .select_nth_unstable_by(n - nuclei, f64::total_cmp)
        .1;

    let reach = c.spread.clamp(0.0, MAX_RADIUS_FRACTION) * w.min(h) as f64;
    let mut dist: Vec<f64> = vulnerability
        .iter()
        .map(|v| if *v >= nucleus_floor { 0.0 } else { f64::MAX })
        .collect();
    if reach <= 0.0 {
        // Nothing creeps: corrosion is confined to the nuclei themselves.
        return dist.iter().map(|d| f64::from(*d == 0.0)).collect();
    }

    // Step cost is the reciprocal of a local speed, so distance stays in
    // "effective texels" and `reach` keeps its tile-relative meaning.
    let speed: Vec<f64> = barrier
        .iter()
        .map(|b| 0.35 + 0.65 * normalize(*b))
        .collect();
    let diagonal = std::f64::consts::SQRT_2;

    for _ in 0..GROWTH_SWEEPS {
        // Forward: west, north, and the two northern diagonals.
        for y in 0..h {
            let above = ((y + h - 1) % h) * w;
            let row = y * w;
            for x in 0..w {
                let west = (x + w - 1) % w;
                let east = (x + 1) % w;
                let step = 1.0 / speed[row + x];
                let best = (dist[row + west] + step)
                    .min(dist[above + x] + step)
                    .min(dist[above + west] + step * diagonal)
                    .min(dist[above + east] + step * diagonal);
                if best < dist[row + x] {
                    dist[row + x] = best;
                }
            }
        }
        // Backward: east, south, and the two southern diagonals.
        for y in (0..h).rev() {
            let below = ((y + 1) % h) * w;
            let row = y * w;
            for x in (0..w).rev() {
                let west = (x + w - 1) % w;
                let east = (x + 1) % w;
                let step = 1.0 / speed[row + x];
                let best = (dist[row + east] + step)
                    .min(dist[below + x] + step)
                    .min(dist[below + east] + step * diagonal)
                    .min(dist[below + west] + step * diagonal);
                if best < dist[row + x] {
                    dist[row + x] = best;
                }
            }
        }
    }

    // Fade out over the reach, so blotch edges feather rather than cut off.
    let grown: Vec<f64> = dist
        .iter()
        .map(|d| smoothstep(0.0, 1.0, 1.0 - d / reach))
        .collect();

    // Finally, pin the *visible* extent to `coverage` by remapping the grown
    // field above its own quantile.  Seeding the requested fraction directly
    // and then letting it spread does not work: growth only ever adds area, so
    // any generous `spread` floods the tile and the knob stops meaning
    // anything.  Remapping keeps the organic blob shapes growth produces while
    // making `coverage` describe what is actually seen.
    let keep = ((coverage * n as f64).round() as usize).clamp(1, n);
    let floor = *grown
        .clone()
        .select_nth_unstable_by(n - keep, f64::total_cmp)
        .1;
    if floor >= 1.0 {
        // The fully-corroded plateau already fills the quota; keep just that.
        return grown.iter().map(|v| f64::from(*v >= 1.0)).collect();
    }
    grown.iter().map(|v| smoothstep(floor, 1.0, *v)).collect()
}

/// Draw runoff downward from convex ledges.
///
/// Each column carries a running stain that fades over `length` and may
/// wander sideways, refreshed wherever a ledge feeds it.  The recurrence is
/// run twice over the tile so a streak leaving the bottom edge re-enters the
/// top carrying the value it actually had — one pass alone would show a seam.
fn streak_mask(
    convex: &[f64],
    heights: &[f64],
    w: usize,
    h: usize,
    cfg: &WeatheringConfig,
) -> Vec<f64> {
    let c = &cfg.streaks;
    // Convert a tile-relative run length into the per-row survival factor the
    // recurrence needs: after `rows` rows a stain should have faded to ~1/e.
    let rows = c.length.clamp(0.0, 1.0) * h as f64;
    let decay = if rows > 0.0 { (-1.0 / rows).exp() } else { 0.0 };
    let density = c.density.clamp(0.0, 1.0) as f64;
    let wander = noise_grid(w, h, cfg.seed.wrapping_add(0x0D19), 6.0, 2);
    let gate = noise_grid(w, h, cfg.seed.wrapping_add(0x57EA), 3.0, 1);

    // Sources: convex *horizontal* lips, thinned so only some actually run.
    //
    // Gating on convexity alone sources a streak from every vertical joint as
    // well, which on any masonry pattern draws a run down each perpend and
    // reads as corduroy.  Requiring an upward-facing gradient keeps sources on
    // the lips that would really shed water.
    let lips = upward_facing(heights, w, h);
    let mut source = vec![0.0; w * h];
    for (i, slot) in source.iter_mut().enumerate() {
        let ledge = smoothstep(0.45, 0.95, convex[i].max(0.0)) * lips[i];
        if ledge > 0.0 && normalize(gate[i]) < density {
            *slot = ledge;
        }
    }

    let mut mask = source.clone();
    for _ in 0..2 {
        for y in 0..h {
            let row = y * w;
            let prev = ((y + h - 1) % h) * w;
            for x in 0..w {
                // Wander: inherit from a neighbouring column so runs drift
                // instead of falling in dead-straight lines.
                let drift = (wander[row + x] * c.wander).round() as i64;
                let sx = (x as i64 + drift).rem_euclid(w as i64) as usize;
                mask[row + x] = source[row + x].max(mask[prev + sx] * decay);
            }
        }
    }

    mask
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Seamless FBM grid in `[-1, 1]`, matching the crate's toroidal convention.
fn noise_grid(w: usize, h: usize, seed: u32, scale: f64, octaves: usize) -> Vec<f64> {
    let fbm = Fbm::<Perlin>::new(seed).set_octaves(octaves.clamp(1, 8));
    let noise = ToroidalNoise::new(fbm, scale.max(0.1));
    let mut out = Vec::new();
    sample_grid_into(&noise, w as u32, h as u32, &mut out);
    out
}

/// Rescale a signed grid so a high percentile of its magnitude sits at 1,
/// leaving sign intact and clamping the outliers beyond it.
///
/// Masks are derived from height fields whose amplitude is entirely up to the
/// generator — the driver deliberately does not normalise height — so knobs
/// like `threshold` would otherwise mean something different for every
/// generator.
///
/// Scaling by a percentile rather than the outright maximum matters more than
/// it looks: a Laplacian over a surface with one hard step (mortar seams,
/// plank joints) spikes on a one-texel line, and dividing by that spike pushes
/// every gentler feature to near zero — which reads as "the layer does
/// nothing" even at full strength.
fn percentile_scaled(mut grid: Vec<f64>) -> Vec<f64> {
    let n = grid.len();
    if n == 0 {
        return grid;
    }
    let mut magnitudes: Vec<f64> = grid.iter().map(|v| v.abs()).collect();
    let idx = (((n - 1) as f64) * NORMALISATION_PERCENTILE).round() as usize;
    let scale = *magnitudes.select_nth_unstable_by(idx, f64::total_cmp).1;
    if scale > 0.0 {
        let inv = 1.0 / scale;
        for v in &mut grid {
            *v = (*v * inv).clamp(-1.0, 1.0);
        }
    }
    grid
}

/// Blur radius in texels for a knob expressed as a fraction of the tile.
fn radius_texels(fraction: f64, w: usize, h: usize) -> usize {
    let short = w.min(h) as f64;
    let r = fraction.clamp(0.0, MAX_RADIUS_FRACTION) * short;
    (r.round() as usize).max(1)
}

/// Separable box blur with wrapping neighbours, so the result tiles.
fn box_blur_wrapped(src: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let span = (radius * 2 + 1) as f64;

    // `radius` never exceeds a quarter of the shorter edge (see
    // `radius_texels`), so `+ w` / `+ h` is enough to keep these unsigned
    // subtractions above zero.
    let mut horizontal = vec![0.0; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0;
            for d in 0..=(radius * 2) {
                acc += src[row + (x + w + d - radius) % w];
            }
            horizontal[row + x] = acc / span;
        }
    }

    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for d in 0..=(radius * 2) {
                acc += horizontal[((y + h + d - radius) % h) * w + x];
            }
            out[y * w + x] = acc / span;
        }
    }

    out
}

/// Hermite smoothstep between two edges, clamped outside them.
#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_COLOR: [f32; 3] = [0.4, 0.5, 0.6];

    fn field() -> SurfaceField {
        SurfaceField {
            color: BASE_COLOR,
            roughness: 0.5,
            metallic: 0.0,
            occlusion: 1.0,
        }
    }

    fn fields(n: usize) -> Vec<SurfaceField> {
        (0..n).map(|_| field()).collect()
    }

    /// A bumpy but seamless height field: peaks and valleys that wrap.
    fn bumpy(w: usize, h: usize) -> Vec<f64> {
        use std::f64::consts::TAU;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
                (TAU * 2.0 * x).sin() * (TAU * 2.0 * y).sin()
            })
            .collect()
    }

    fn only_wear() -> WeatheringConfig {
        WeatheringConfig {
            seed: 3,
            edge_wear: EdgeWear {
                amount: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn only_corrosion() -> WeatheringConfig {
        WeatheringConfig {
            seed: 3,
            corrosion: Corrosion {
                amount: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn only_dirt() -> WeatheringConfig {
        WeatheringConfig {
            seed: 3,
            crevice_dirt: CreviceDirt {
                amount: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn only_streaks() -> WeatheringConfig {
        WeatheringConfig {
            seed: 3,
            streaks: Streaks {
                amount: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn all_layers() -> WeatheringConfig {
        WeatheringConfig {
            seed: 9,
            edge_wear: only_wear().edge_wear,
            corrosion: only_corrosion().corrosion,
            crevice_dirt: only_dirt().crevice_dirt,
            streaks: only_streaks().streaks,
        }
    }

    /// Cyclically shift a grid; used to prove the neighbourhood passes wrap.
    fn shift(grid: &[f64], w: usize, h: usize, dx: usize, dy: usize) -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                out[((y + dy) % h) * w + (x + dx) % w] = grid[y * w + x];
            }
        }
        out
    }

    #[test]
    fn default_config_is_a_noop() {
        let cfg = WeatheringConfig::default();
        assert!(cfg.is_noop());

        let (w, h) = (16, 16);
        let mut f = fields(w * h);
        let mut heights = bumpy(w, h);
        let before = heights.clone();

        apply(&mut f, &mut heights, w as u32, h as u32, &cfg);

        assert_eq!(heights, before, "no-op config disturbed the height field");
        assert!(
            f.iter().all(|f| f.color == BASE_COLOR
                && f.roughness == 0.5
                && f.metallic == 0.0
                && f.occlusion == 1.0),
            "no-op config disturbed the surface fields"
        );
    }

    /// Each layer must reach the output on its own, so a generator can enable
    /// exactly one and see it — and none may repaint the whole tile.
    #[test]
    fn every_layer_changes_the_surface_alone() {
        let (w, h) = (32, 32);
        let base = bumpy(w, h);

        for (name, cfg) in [
            ("edge_wear", only_wear()),
            ("corrosion", only_corrosion()),
            ("crevice_dirt", only_dirt()),
            ("streaks", only_streaks()),
        ] {
            assert!(!cfg.is_noop(), "{name} config reported itself a no-op");
            let mut f = fields(w * h);
            let mut heights = base.clone();
            apply(&mut f, &mut heights, w as u32, h as u32, &cfg);

            let touched = f
                .iter()
                .filter(|f| f.color != BASE_COLOR || f.roughness != 0.5)
                .count();
            assert!(touched > 0, "{name} left every texel untouched");
            assert!(
                touched < w * h,
                "{name} covered the whole tile — it is a mask, not a repaint"
            );
        }
    }

    /// Only corrosion is allowed to disturb the height field, since its crust
    /// stands proud; the other layers are colour/ORM only.
    #[test]
    fn only_corrosion_adds_relief() {
        let (w, h) = (24, 24);
        let base = bumpy(w, h);

        for (name, cfg, expect_change) in [
            ("edge_wear", only_wear(), false),
            ("crevice_dirt", only_dirt(), false),
            ("streaks", only_streaks(), false),
            ("corrosion", only_corrosion(), true),
        ] {
            let mut f = fields(w * h);
            let mut heights = base.clone();
            apply(&mut f, &mut heights, w as u32, h as u32, &cfg);
            assert_eq!(heights != base, expect_change, "{name} relief behaviour");
        }
    }

    /// The geometric masks must commute with a cyclic shift — the proof that
    /// every neighbourhood lookup wraps rather than clamping at the edge.
    #[test]
    fn geometric_masks_wrap_at_the_tile_edge() {
        let (w, h) = (16, 16);
        let base = bumpy(w, h);
        let shifted = shift(&base, w, h, 5, 7);

        for (name, direct, of_shifted) in [
            (
                "convexity",
                shift(&convexity(&base, w, h), w, h, 5, 7),
                convexity(&shifted, w, h),
            ),
            (
                "cavity",
                shift(&cavity(&base, w, h, 3), w, h, 5, 7),
                cavity(&shifted, w, h, 3),
            ),
            (
                "upward_facing",
                shift(&upward_facing(&base, w, h), w, h, 5, 7),
                upward_facing(&shifted, w, h),
            ),
        ] {
            for (i, (a, b)) in direct.iter().zip(&of_shifted).enumerate() {
                assert!(
                    (a - b).abs() < 1e-12,
                    "{name} is not shift-equivariant at {i}: {a} vs {b} — a neighbour lookup is clamping"
                );
            }
        }
    }

    /// Convexity must separate peaks from valleys with the sign convention
    /// the layers rely on.
    #[test]
    fn convexity_is_positive_on_peaks() {
        let (w, h) = (16, 16);
        let heights = bumpy(w, h);
        let convex = convexity(&heights, w, h);

        let peak = heights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;
        let valley = heights
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;

        assert!(convex[peak] > 0.0, "peak was not convex: {}", convex[peak]);
        assert!(
            convex[valley] < 0.0,
            "valley was not concave: {}",
            convex[valley]
        );
    }

    /// Cavity must be positive in hollows and negative on raised ground.
    #[test]
    fn cavity_is_positive_in_hollows() {
        let (w, h) = (16, 16);
        let heights = bumpy(w, h);
        let hollow = cavity(&heights, w, h, 3);

        let peak = heights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;
        let valley = heights
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;

        assert!(hollow[valley] > 0.0, "valley did not read as recessed");
        assert!(hollow[peak] < 0.0, "peak did not read as raised");
    }

    /// Growth must spread the seeded area, and spread further with more
    /// iterations.
    #[test]
    fn corrosion_growth_spreads_from_seeds() {
        let (w, h) = (32, 32);
        let heights = bumpy(w, h);
        let hollow = percentile_scaled(cavity(&heights, w, h, 3));

        let covered = |spread: f64| {
            let cfg = WeatheringConfig {
                seed: 4,
                corrosion: Corrosion {
                    amount: 1.0,
                    coverage: 0.15,
                    spread,
                    ..Default::default()
                },
                ..Default::default()
            };
            let barrier = noise_grid(w, h, cfg.seed.wrapping_add(0x7005), 5.0, 3);
            corrosion_mask(&hollow, &barrier, w, h, &cfg)
                .iter()
                .filter(|v| **v > 0.01)
                .count()
        };

        let seeded = covered(0.0);
        let grown = covered(0.05);
        let grown_more = covered(0.15);

        assert!(seeded > 0, "no corrosion seeds took hold");
        assert!(grown > seeded, "growth did not spread ({seeded} → {grown})");
        assert!(
            grown_more > grown,
            "more spread did not reach further ({grown} → {grown_more})"
        );
    }

    /// `coverage` must describe the *visible* extent, not the seeded extent.
    ///
    /// This is the knob's whole contract: growth only ever adds area, so
    /// seeding the requested fraction and letting it spread would flood the
    /// tile at any generous `spread` — and the failure is invisible to a test
    /// that only checks "some texels changed".
    #[test]
    fn coverage_bounds_the_visible_extent() {
        let (w, h) = (64, 64);
        let n = w * h;
        let heights = bumpy(w, h);
        let hollow = percentile_scaled(cavity(&heights, w, h, 3));

        for coverage in [0.1f32, 0.25, 0.5] {
            for spread in [0.02, 0.08, 0.2] {
                let cfg = WeatheringConfig {
                    seed: 6,
                    corrosion: Corrosion {
                        amount: 1.0,
                        coverage,
                        spread,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let barrier = noise_grid(w, h, cfg.seed.wrapping_add(0x7005), 5.0, 3);
                let corroded = corrosion_mask(&hollow, &barrier, w, h, &cfg)
                    .iter()
                    .filter(|v| **v > 0.0)
                    .count();
                let fraction = corroded as f64 / n as f64;
                // The ceiling is the contract: growth must never flood past
                // what was asked for, however generous the spread.
                assert!(
                    fraction <= coverage as f64 + 0.03,
                    "coverage {coverage} at spread {spread} flooded to {fraction:.3} of the tile"
                );
                // Given room to grow, it should also actually get there.
                if spread >= 0.08 {
                    assert!(
                        fraction >= coverage as f64 - 0.03,
                        "coverage {coverage} at spread {spread} only reached {fraction:.3}"
                    );
                }
            }
        }
    }

    /// A config must look the same baked at two resolutions — the reason
    /// spread is a tile fraction rather than an iteration count.
    #[test]
    fn corrosion_reach_is_resolution_independent() {
        let cfg = WeatheringConfig {
            seed: 8,
            corrosion: Corrosion {
                amount: 1.0,
                coverage: 0.2,
                spread: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };

        let fraction_at = |size: usize| {
            let heights = bumpy(size, size);
            let hollow = percentile_scaled(cavity(&heights, size, size, 3));
            let barrier = noise_grid(size, size, cfg.seed.wrapping_add(0x7005), 5.0, 3);
            let covered = corrosion_mask(&hollow, &barrier, size, size, &cfg)
                .iter()
                .filter(|v| **v > 0.01)
                .count();
            covered as f64 / (size * size) as f64
        };

        let small = fraction_at(64);
        let large = fraction_at(128);
        assert!(
            (small - large).abs() < 0.1,
            "corroded fraction drifted with resolution: {small:.3} at 64² vs {large:.3} at 128²"
        );
    }

    /// Streaks run downward (+V) and nowhere else.
    #[test]
    fn streaks_run_downhill_only() {
        let (w, h) = (32, 32);
        // A single bright ledge across one row, nothing elsewhere, over a
        // height field that steps down there so it reads as an upward-facing
        // lip rather than a vertical joint.
        let ledge_row = 8;
        let mut convex = vec![0.0; w * h];
        for x in 0..w {
            convex[ledge_row * w + x] = 1.0;
        }
        let heights: Vec<f64> = (0..w * h)
            .map(|i| if i / w <= ledge_row { 1.0 } else { 0.0 })
            .collect();

        let cfg = WeatheringConfig {
            seed: 2,
            streaks: Streaks {
                amount: 1.0,
                density: 1.0,
                length: 0.5,
                wander: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mask = streak_mask(&convex, &heights, w, h, &cfg);
        let row_total = |y: usize| (0..w).map(|x| mask[y * w + x]).sum::<f64>();

        assert!(
            row_total(ledge_row) > 0.0,
            "the ledge itself is not stained"
        );
        assert!(
            row_total(ledge_row + 1) > 0.0,
            "nothing ran down from the ledge"
        );
        assert!(
            row_total(ledge_row + 1) < row_total(ledge_row),
            "the stain did not fade as it ran"
        );
        // Just above the ledge is the far end of the run, so it must be the
        // faintest point — proof the flow is directional.
        assert!(
            row_total(ledge_row - 1) < row_total(ledge_row + 1),
            "stain above the ledge beats the stain below it — flow is not downward"
        );
    }

    /// Same seed → same ageing; different seed → different ageing.
    #[test]
    fn weathering_is_seed_deterministic() {
        let (w, h) = (24, 24);
        let base = bumpy(w, h);

        let run = |seed: u32| {
            let mut f = fields(w * h);
            let mut heights = base.clone();
            let cfg = WeatheringConfig {
                seed,
                ..all_layers()
            };
            apply(&mut f, &mut heights, w as u32, h as u32, &cfg);
            f.iter().map(|f| f.color[0]).collect::<Vec<_>>()
        };

        assert_eq!(run(11), run(11), "ageing is not deterministic");
        assert_ne!(run(11), run(12), "seed did not change the ageing");
    }

    /// Degenerate and out-of-range configs must not panic or produce NaN.
    #[test]
    fn extreme_configs_stay_finite() {
        let (w, h) = (16, 16);
        let cfg = WeatheringConfig {
            seed: 7,
            edge_wear: EdgeWear {
                amount: 9.0,
                threshold: -3.0,
                breakup_scale: 0.0,
                ..Default::default()
            },
            corrosion: Corrosion {
                amount: 9.0,
                coverage: 5.0,
                spread: 1e6,
                barrier_scale: 0.0,
                relief: 1e6,
                ..Default::default()
            },
            crevice_dirt: CreviceDirt {
                amount: -1.0,
                depth: 100.0,
                gravity: 4.0,
                ..Default::default()
            },
            streaks: Streaks {
                amount: 9.0,
                density: 9.0,
                length: 5.0,
                wander: 1e6,
                ..Default::default()
            },
        };

        let mut f = fields(w * h);
        let mut heights = bumpy(w, h);
        apply(&mut f, &mut heights, w as u32, h as u32, &cfg);

        assert!(
            heights.iter().all(|v| v.is_finite()),
            "height field went non-finite"
        );
        assert!(
            f.iter().all(|f| f.color.iter().all(|c| c.is_finite())
                && f.roughness.is_finite()
                && f.metallic.is_finite()
                && f.occlusion.is_finite()),
            "surface fields went non-finite"
        );
    }

    /// A mismatched buffer length is ignored rather than panicking.
    #[test]
    fn mismatched_buffers_are_rejected() {
        let mut f = fields(4);
        let mut heights = vec![0.0; 9];
        apply(&mut f, &mut heights, 3, 3, &all_layers());
        assert!(f.iter().all(|f| f.color == BASE_COLOR));
    }
}
