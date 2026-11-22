# HDR+ Web (proof of concept)

A browser-based proof-of-concept port of [Burst Photo](https://github.com/martin-marek/hdr-plus-swift)
(Google's HDR+ burst-photography pipeline in Swift/Metal), rebuilt with:

- **HTML5** — UI, drag & drop, canvas preview
- **flashraw** — RAW/DNG decoding to Bayer mosaic pixels (WASM, fast)
- **WebGPU** — all align+merge compute on the GPU (WGSL compute shaders)

## Status

This is a **proof of concept**: Bayer sensors, uniform-exposure bursts,
spatial-domain merge only, no hot-pixel suppression, no exposure bracketing,
no frequency-domain merge. See "PoC scope" below.

The flashraw integration and the DNG writer are verified headlessly
(`npm test`: synthetic Bayer frame → DNG → flashraw decode → exact pixel +
metadata round-trip). The WGSL kernels are careful line-by-line ports of the
Metal sources but have **not yet executed on a GPU** — the first browser run
is the remaining validation step.

## flashraw integration notes (verified against v0.4.0 source)

- `FlashRaw.open(ArrayBuffer)` → `raw.decode({ backend: 'cpu', output: 'raw-u16', crop: 'active' })`
  → `frame.data: Uint16Array` (Bayer mosaic of the DNG ActiveArea), never
  falling back to the embedded preview, keeping raw black-offset sensor values.
- `frame.blackLevel: {rows, cols, samples, values[], mean}` is anchored at the
  **sensor buffer** origin while the cropped frame's `cfa.pattern` is anchored
  at the frame origin — `src/raw.js` pre-rotates the black pattern by
  `frame.origin` so the GPU kernels can assume plane-anchored 2×2 levels.
- The `.wasm` binaries are `new URL(..., import.meta.url)` assets → local
  `npm install` + the bundled static server is the reliable loading path
  (CDN ESM fallbacks are attempted but best-effort).
- flashraw also runs in Node, which is what makes `npm test` possible.

## Running

```bash
cd hdr-plus-web
npm install       # fetches flashraw (~1 MB JS + WASM binaries served locally)
npm start         # zero-dependency static server on http://localhost:8080
```

Open http://localhost:8080 in a WebGPU-capable browser (Chrome/Edge 113+,
Safari 26, Firefox with WebGPU). Drop ≥2 DNGs from the same burst, press
*Process burst*, get a preview and a 16-bit uncompressed Bayer DNG.

A local flashraw install is the reliable path: its `.wasm` binaries are
`new URL(..., import.meta.url)` assets, which the bundled `server.mjs` serves
with the correct MIME type from `node_modules/`. CDN imports (jsDelivr/esm.sh)
are attempted as fallback but are best-effort.

Headless check of the DNG writer + flashraw integration (no browser needed,
flashraw also runs in Node):

```bash
node test/roundtrip.mjs
```

## Architecture

```
index.html            UI: drop zone, settings, progress, canvas preview, download
src/main.js           orchestration: load → decode → pipeline → preview/DNG
src/raw.js            flashraw wrapper: DNG → {Bayer u16, black/white level, CFA, color matrix}
src/pipeline.js       port of merge/spatial.swift + align/align.swift orchestration
src/kernels.js        WGSL compute kernels (ports of the Metal kernels)
src/gpu.js            WebGPU helpers (buffers, dispatch, readback, uniform packing)
src/dng_writer.js     minimal 16-bit uncompressed Bayer DNG writer (phone-DNG style)
server.mjs            zero-dependency static server
```

Images are handled as **storage buffers** (`array<f32>` planes), not textures:
the Metal code reads/writes texels only via explicit `.read()/.write()` (no
sampler hardware), so buffers map 1:1, dodge WebGPU storage-texture format
restrictions, and have no `maxTextureDimension` size limit. Raw sensor data is
uploaded as a packed `array<u32>` (2×16-bit pixels per u32, zero-copy view).

## Kernel mapping (Metal → WGSL)

Full Metal kernel census of Burst Photo (52 kernels) and their PoC status:

### align/align.metal (9)

| Kernel | Purpose | PoC |
|---|---|---|
| `avg_pool` | 2D average-pooling for pyramid levels | ✅ ported |
| `avg_pool_normalization` | same + per-channel color-factor normalization (bracketed/non-Bayer) | ❌ not needed |
| `compute_tile_differences` | per-tile, per-candidate-offset cost (generic search distance), L1/L2 | ✅ ported |
| `compute_tile_differences25` | optimized variant for the always-25-candidate case | ❌ optimization only |
| `compute_tile_differences_exposure25` | 25-candidate variant with exposure weighting (bracketing) | ❌ bracketing out of scope |
| `correct_upsampling_error` | re-tests 3 candidate alignments per tile to fix upsampled borders | ❌ skipped (quality TODO) |
| `find_best_tile_alignment` | argmin over candidate costs → alignment vector | ✅ ported |
| `warp_texture_bayer` | warp comparison frame by alignment grid (bilinear blend) | ✅ ported |
| `warp_texture_xtrans` | same for Fujifilm X-Trans 6×6 sensors | ❌ Bayer only |

### merge/spatial.metal (2)

| Kernel | Purpose | PoC |
|---|---|---|
| `color_difference` | per-superpixel SAD between two frames (robustness metric) | ✅ ported |
| `compute_merge_weight` | per-superpixel merge weight from diff vs noise estimate | ✅ ported |

### merge/frequency.metal (12) — "Higher quality" merge

| Kernel | Purpose | PoC |
|---|---|---|
| `forward_fft` / `backward_fft` | radix-2 tiled 2D DFT (4 channels packed in RGBA) | ❌ |
| `forward_dft` / `backward_dft` | generic tiled 2D DFT | ❌ |
| `merge_frequency_domain` | robust merge per frequency tile + subpixel alignment via Fourier shift theorem | ❌ |
| `calculate_rms_rgba` | per-tile shot-noise estimate | ❌ |
| `calculate_mismatch_rgba` | per-tile motion-mismatch estimate | ❌ |
| `normalize_mismatch` | global mismatch normalization (reduction) | ❌ |
| `calculate_abs_diff_rgba` | per-tile abs diff helper | ❌ |
| `calculate_highlights_norm_rgba` | per-tile clipped-highlight weight | ❌ |
| `deconvolute_frequency_domain` | Wiener-style deconvolution of merged Fourier data | ❌ |
| `reduce_artifacts_tile_border` | smooths tile-border artifacts of windowed inverse FFT | ❌ |

### texture/texture.metal (25)

| Kernel | Purpose | PoC |
|---|---|---|
| `add_texture` | accumulator: `out += in / n_frames` | ✅ ported (`accumulate`) |
| `add_texture_weighted` | per-pixel weighted average of ref/aligned frame | ✅ ported |
| `blur_mosaic_texture` | separable binomial filter (pyramid + robustness blur) | ✅ ported (`blur`) |
| `prepare_texture_bayer` | raw u16 → f32 plane: hot pixels, black level, exposure diff, zero-pad | ✅ ported (no hot pixels / exposure diff) |
| `prepare_texture_xtrans` | same for X-Trans | ❌ |
| `crop_texture` | remove alignment padding | ✅ ported (`crop`) |
| `fill_with_zeros` | clear plane | ✅ via `writeBuffer(0)` |
| `copy_texture` | plane copy | ✅ via `copyBufferToBuffer` |
| `convert_float_to_uint16` | final float → 16-bit with black/white level scaling | ✅ ported (`convert_to_u16`) |
| `convert_to_rgba` | pack 2×2 CFA quad → RGBA (UI preview / helpers) | ✅ done on CPU in preview |
| `convert_to_bayer` | RGBA → mosaic (used by other modes) | ❌ not needed |
| `add_texture_uint16` | accumulate raw u16 plane (temporal-average path) | ❌ |
| `add_texture_exposure` | exposure-weighted accumulation (bracketing) | ❌ |
| `add_texture_highlights` | highlight reconstruction from R/B (bracketing) | ❌ |
| `calculate_weight_highlights` | highlight-clipping weight map (bracketing) | ❌ |
| `find_hotpixels_bayer` | per-frame-static hot pixel detection | ❌ |
| `find_hotpixels_xtrans` | same for X-Trans | ❌ |
| `normalize_texture` | `out /= norm` (bracketed temporal-average path) | ❌ |
| `sum_rect_columns_float` / `sum_rect_columns_uint` | column sums for global reductions | ✅ replaced by 2-stage workgroup reduction (`sum_partials`/`sum_final`) |
| `sum_row` | row sum (reduction stage 2) | ✅ same |
| `divide_buffer` / `sum_divide_buffer` | buffer-wide divide / mean (reduction stage 3) | ✅ done on CPU (one scalar) |
| `upsample_bilinear_float` | bilinear upsampling (merge weight map) | ✅ ported (`upsample_bilinear`) |
| `upsample_nearest_int` | nearest upsampling of alignment grid | ✅ ported (`upsample_nearest_i32`) |

### exposure/exposure.metal (4)

| Kernel | Purpose | PoC |
|---|---|---|
| `correct_exposure` / `correct_exposure_linear` | shadow lift, linear/curve tone control | ❌ |
| `max_x` / `max_y` | 1D argmax helpers (curve construction) | ❌ |

**PoC uses 15 of 52 kernels** (11 ported verbatim-ish + copy/fill via buffer ops
+ reduction re-implemented). The 37 excluded kernels belong to: X-Trans sensors,
exposure bracketing, the frequency-domain ("higher quality") merge, hot-pixel
suppression, exposure correction, and performance-optimized variants of ported
kernels.

## PoC simplifications (vs Burst Photo)

1. Spatial merge only (the app's "Fast" mode). The frequency-domain merge
   (12 kernels incl. tiled FFT) is the natural next milestone — WGSL workgroup
   storage maps well to the Metal threadgroup FFT.
2. No `correct_upsampling_error` — alignment quality at moving-object borders
   is slightly worse (one of the three candidate re-tests in Hasinoff 2016).
3. Generic `compute_tile_differences` instead of the 25-candidate fast path
   (correct but slower; costs ~25× full-tile evaluations per level).
4. No hot-pixel suppression, no exposure correction, no bracketing.
5. Out-of-bounds reads clamp to edge (Metal returns 0) — border pixels only.
6. Output DNG is a minimal phone-style single-IFD Bayer DNG: preserves
   resolution/bit-depth/CFA/black/white level/color matrix, but not lens
   profiles, opcodes, or full EXIF.

## Next steps

- Decode in Web Workers (parallel flashraw instances) + OPFS caching
- Frequency-domain merge (tiled FFT) for "higher quality" mode
- `correct_upsampling_error` + `compute_tile_differences25`
- Exposure bracketing + exposure correction
- Hot-pixel suppression
