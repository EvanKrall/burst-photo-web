// WebGPU port of Burst Photo's frequency-domain align+merge pipeline
// (burstphoto/merge/frequency.swift + burstphoto/merge/frequency.metal) — the app's
// "Higher quality" merging algorithm.
//
// Deviations from the Swift original (documented where they occur):
//  - Uniform-exposure bursts only: exposure_corr1/corr2 are 1, the highlights-norm
//    kernel is dropped (it writes 1.0 for uniform exposure), and the mismatch
//    kernel's exposure_factor is 1.
//  - The alignment runs ONCE per comparison frame on the shared alignment padding
//    (deriveAlignmentSetup — identical to the spatial pipeline and the inspection
//    session), instead of four times on per-iteration asymmetric padding. The 8-px
//    tile_size_merge shifts of the four iterations are realized by re-PLACING the
//    prepared/aligned frames into per-iteration merge canvases (place_f32, zero-filled)
//    — the same coordinates Swift's prepare_texture produces, but the alignment grids
//    stay identical to the inspection session's, so retained grids can be reused
//    bit-identically (and the pyramid search runs 1× instead of 4×).
//  - tile_size_merge is pinned to 8 (the Swift value), so only the FFT path exists.
//
// Buffer layout convention: f32 planes use Plane.size for their byte length; vec4
// planes are allocated with explicit w*h*16 byte counts (Plane.size is f32-only).
//
// frames: [{ width, height, data: Uint16Array (Bayer mosaic), blackLevel: number[4],
//            whiteLevel: number }] — all frames must share dimensions, reference first.
// options: { tileSize: 16|32|64, searchDistance: 32|64|128, noiseReduction: 0..23,
//            prealigned?: [{texels, w, h}|null] } — prealigned grids from an inspection
//            session (entry 0 = the reference = null), as in alignMergeSpatial.
//
// returns { width, height, pixels: Uint16Array, timings: {...} }

import { packUniforms } from './gpu.js';
import {
  Plane, groups, prepareFrame, cropRectFn, alignTextureEx, warpPlane, buildPyramid,
  alignmentGridGeometry, deriveAlignmentSetup, sumToScalar,
} from './pipeline.js';

const TILE_MERGE = 8; // merge tile size in RGBA px (fixed by the Swift implementation)

// Per-iteration zero-padding shifts, port of:
//   shift_left   = i % 2 == 0 ? ts : 0;  shift_right = i % 2 == 1 ? ts : 0;
//   shift_top    = i < 3       ? ts : 0;  shift_bottom = i >= 3      ? ts : 0;
// The four displaced merge-tile grids suppress frequency-tiling artifacts; the four
// runs' results are averaged.
const ITERATION_SHIFTS = [
  { left: 0, right: TILE_MERGE, top: TILE_MERGE, bottom: 0 },
  { left: TILE_MERGE, right: 0, top: TILE_MERGE, bottom: 0 },
  { left: 0, right: TILE_MERGE, top: 0, bottom: TILE_MERGE },
  { left: TILE_MERGE, right: 0, top: 0, bottom: TILE_MERGE },
];

/**
 * Merge-stage geometry, port of the padding math in align_merge_frequency_domain. The
 * alignment pyramid uses deriveAlignmentSetup (shared with the spatial pipeline); the
 * merge canvas fits image + tile_size_merge plus enough margin that
 * (W + 8 + 2·pad_merge) is a multiple of the 16-px merge-tile period. Bayer-pixel dims.
 */
export function deriveFrequencySetup(width, height, alignment) {
  const tileFactor = alignment.tileSizeArray[alignment.nLevels - 1]
    * alignment.downscaleFactorArray.reduce((a, b) => a * b, 1);
  const padAlignX = Math.trunc((Math.ceil((width + TILE_MERGE) / tileFactor) * tileFactor - width - TILE_MERGE) / 2);
  const padAlignY = Math.trunc((Math.ceil((height + TILE_MERGE) / tileFactor) * tileFactor - height - TILE_MERGE) / 2);
  // Swift: crop_merge = floor(pad_align / (2·tile_size_merge)) · 2·tile_size_merge.
  // The canvas here already IS the post-crop size, so only the remainder survives as
  // pad_merge (the 0..15 px margin each side inside the canvas).
  const padMergeX = padAlignX % (2 * TILE_MERGE);
  const padMergeY = padAlignY % (2 * TILE_MERGE);
  const canvasW = width + TILE_MERGE + 2 * padMergeX;
  const canvasH = height + TILE_MERGE + 2 * padMergeY;
  const nTilesX = canvasW / (2 * TILE_MERGE);
  const nTilesY = canvasH / (2 * TILE_MERGE);
  if (!Number.isInteger(nTilesX) || !Number.isInteger(nTilesY)) {
    throw new Error(`frequency merge geometry: canvas ${canvasW}×${canvasH} is not a multiple of the ${2 * TILE_MERGE}-px merge-tile period`);
  }
  return { canvasW, canvasH, rgbaW: canvasW / 2, rgbaH: canvasH / 2, nTilesX, nTilesY, padMergeX, padMergeY };
}

// Robustness parameter derivation, port of the uniform-exposure path of
// align_merge_frequency_domain (exposure_corr1 == exposure_corr2 == 1).
export function frequencyMergeParams(noiseReduction) {
  // two steps in noise_reduction → factor 2 in the robustness norm (uniform-exposure constant 26.5)
  const robustnessRev = 0.5 * (26.5 - Math.round(noiseReduction));
  const robustnessNorm = Math.pow(2.0, -robustnessRev + 7.5);
  // read noise grows ~3× per iso level (1.6 power) → more smoothing in the shadows
  const readNoise = Math.pow(Math.pow(2.0, -robustnessRev + 10.0), 1.6);
  // stronger denoising where alignment is good (daylight 1 … extreme low-light 25 in
  // the Google paper — a continuous curve derived from the robustness value here)
  const maxMotionNorm = Math.max(1.0, Math.pow(1.3, 11.0 - robustnessRev));
  return { robustnessRev, robustnessNorm, readNoise, maxMotionNorm };
}

// ---- thin GPU-stage wrappers around the WGSL kernels -----------------------

function placeF32(gpu, src, outW, outH, ox, oy) {
  const out = new Plane(gpu.zeros(`place-${ox},${oy}`, outW * outH * 4), outW, outH);
  gpu.dispatch('place_f32',
    [{ buffer: src.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'], [src.w, src.h, outW, outH, ox, oy]) }],
    groups(outW, outH));
  return out;
}

function convertToRGBA(gpu, canvas) {
  const out = new Plane(gpu.zeros('rgba', canvas.w * canvas.h * 4), canvas.w / 2, canvas.h / 2); // vec4: ×16 B/px = ×4/px of the f32 canvas
  gpu.dispatch('convert_to_rgba',
    [{ buffer: canvas.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32'], [canvas.w, out.w, out.h]) }],
    groups(out.w, out.h));
  return out;
}

function convertToBayer(gpu, rgbaPlane) {
  const outW = rgbaPlane.w * 2, outH = rgbaPlane.h * 2;
  const out = new Plane(gpu.zeros('bayer', outW * outH * 4), outW, outH);
  gpu.dispatch('convert_to_bayer',
    [{ buffer: rgbaPlane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32'], [outW]) }],
    groups(outW / 2, outH / 2));
  return out;
}

function calculateRms(gpu, refRGBA, nTiles) {
  const out = new Plane(gpu.zeros('rms', nTiles.x * nTiles.y * 16), nTiles.x, nTiles.y); // vec4
  gpu.dispatch('calculate_rms_rgba',
    [{ buffer: refRGBA.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32'], [refRGBA.w, nTiles.x, nTiles.y]) }],
    groups(nTiles.x, nTiles.y));
  return out;
}

function calculateAbsDiff(gpu, refRGBA, compRGBA) {
  const out = new Plane(gpu.zeros('absdiff', refRGBA.w * refRGBA.h * 16), refRGBA.w, refRGBA.h); // vec4
  gpu.dispatch('calculate_abs_diff_rgba',
    [{ buffer: refRGBA.buf }, { buffer: compRGBA.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32'], [refRGBA.w, refRGBA.h]) }],
    groups(refRGBA.w, refRGBA.h));
  return out;
}

function calculateMismatch(gpu, absDiff, rms, nTiles) {
  const out = new Plane(gpu.zeros('mismatch', nTiles.x * nTiles.y * 4), nTiles.x, nTiles.y);
  gpu.dispatch('calculate_mismatch_rgba',
    [{ buffer: absDiff.buf }, { buffer: rms.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32'], [absDiff.w, absDiff.h, nTiles.x]) }],
    groups(nTiles.x, nTiles.y));
  return out;
}

function normalizeMismatch(gpu, mismatch, mean) {
  gpu.dispatch('normalize_mismatch',
    [{ buffer: mismatch.buf }, { buffer: packUniforms(['u32', 'f32'], [mismatch.w * mismatch.h, mean]) }],
    [Math.ceil((mismatch.w * mismatch.h) / 64), 1, 1]);
}

function forwardFft(gpu, rgbaPlane, nTiles) {
  const ftW = rgbaPlane.w * 2; // interleaved Re/Im complex pairs
  const out = new Plane(gpu.zeros('ft', ftW * rgbaPlane.h * 16), ftW, rgbaPlane.h); // vec4
  gpu.dispatch('forward_fft',
    [{ buffer: rgbaPlane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32'], [rgbaPlane.w, ftW, nTiles.x, nTiles.y, 0]) }],
    groups(nTiles.x, nTiles.y));
  return out;
}

function backwardFft(gpu, ftPlane, rgbaW, nTiles, nTextures) {
  const out = new Plane(gpu.zeros('bft', rgbaW * ftPlane.h * 16), rgbaW, ftPlane.h); // vec4
  gpu.dispatch('backward_fft',
    [{ buffer: ftPlane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32'],
       [rgbaW, ftPlane.w, nTiles.x, nTiles.y, nTextures * TILE_MERGE * TILE_MERGE]) }],
    groups(nTiles.x, nTiles.y));
  return out;
}

function mergeFrequencyDomain(gpu, refFT, compFT, outFT, rms, mismatch, params, nTiles) {
  gpu.dispatch('merge_frequency_domain',
    [{ buffer: refFT.buf }, { buffer: compFT.buf }, { buffer: outFT.buf },
     { buffer: rms.buf }, { buffer: mismatch.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32', 'f32', 'f32'],
       [refFT.w, nTiles.x, nTiles.y, params.robustnessNorm, params.readNoise, params.maxMotionNorm]) }],
    groups(nTiles.x, nTiles.y));
}

function deconvolute(gpu, finalFT, totalMismatch, nTiles) {
  gpu.dispatch('deconvolute_frequency_domain',
    [{ buffer: finalFT.buf }, { buffer: totalMismatch.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32', 'f32', 'f32'],
       [finalFT.w, nTiles.x, nTiles.y, 0, 0, 0]) }], // only dims are read
    groups(nTiles.x, nTiles.y));
}

function reduceArtifactsTileBorder(gpu, outRGBA, refRGBA, blackLevel, nTiles) {
  gpu.dispatch('reduce_artifacts_tile_border',
    [{ buffer: outRGBA.buf }, { buffer: refRGBA.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32', 'f32', 'f32', 'f32'],
       [outRGBA.w, nTiles.x, nTiles.y, blackLevel[0], blackLevel[1], blackLevel[2], blackLevel[3]]) }],
    groups(nTiles.x, nTiles.y));
}

function accumulate(gpu, outPlane, plane, n) {
  gpu.dispatch('accumulate',
    [{ buffer: plane.buf }, { buffer: outPlane.buf },
     { buffer: packUniforms(['i32', 'i32', 'f32'], [plane.w, plane.h, n]) }], groups(plane.w, plane.h));
}

// ---- main pipeline ----------------------------------------------------------

/**
 * Frequency-domain align+merge (Burst Photo "Higher quality" algorithm).
 * Same signature and return shape as alignMergeSpatial.
 */
export async function alignMergeFrequencyDomain(gpu, frames, options, onProgress) {
  const t0 = performance.now();
  const { device } = gpu;
  const nImages = frames.length;
  const { width: W, height: H } = frames[0];
  const { tileSize, searchDistance, noiseReduction } = options;

  const alignment = deriveAlignmentSetup(W, H, tileSize, searchDistance);
  const freq = deriveFrequencySetup(W, H, alignment);
  const params = frequencyMergeParams(noiseReduction);
  const nTiles = { x: freq.nTilesX, y: freq.nTilesY };
  const blackMean = (f) => Math.max(0, f.blackLevel.reduce((a, b) => a + b, 0) / f.blackLevel.length);

  // Warp geometry of the level-0 alignment grid — computed on the pyramid base, which
  // for Bayer is the padded frame downscaled by the mosaic width (2): identical numbers
  // to the spatial pipeline / inspection session.
  const gridGeometry = alignmentGridGeometry(
    Math.floor((W + 2 * alignment.padX) / 2), Math.floor((H + 2 * alignment.padY) / 2),
    alignment.downscaleFactorArray, alignment.tileSizeArray);

  // ---- reference pyramid for the fresh-alignment path (identical setup to the spatial
  // pipeline and the inspection session, so fresh grids match retained grids exactly) ----
  const refPadded = await prepareFrame(gpu, frames[0], alignment.padX, alignment.padY);
  const refPyramid = buildPyramid(gpu, refPadded, alignment.downscaleFactorArray, blackMean(frames[0]));
  refPadded.buf.destroy(); // the pyramid holds its own pooled copies

  // ---- pass 1: alignment grid per comparison frame (fresh pyramid search, or an
  // inspection session's retained grid — same geometry either way) -----------------
  const prealigned = options.prealigned ?? null;
  const gridBuffers = new Map(); // frames[] index (≥ 1) -> {buf, w, h}
  const tAlignStart = performance.now();
  for (let f = 1; f < nImages; f++) {
    onProgress?.({ stage: 'align', frame: f, total: nImages - 1 });
    const pre = prealigned ? prealigned[f] : null;
    if (pre) {
      gridBuffers.set(f, { buf: gpu.storageFromData(`falign-reuse-${f}`, pre.texels), w: pre.w, h: pre.h });
    } else {
      const compPadded = await prepareFrame(gpu, frames[f], alignment.padX, alignment.padY);
      const { warped, alignment: grid } = alignTextureEx(gpu, refPyramid, compPadded, {
        nLevels: alignment.nLevels, downscaleFactorArray: alignment.downscaleFactorArray,
        tileSizeArray: alignment.tileSizeArray, compBlackMean: blackMean(frames[f]),
      });
      warped.buf.destroy();
      compPadded.buf.destroy();
      gridBuffers.set(f, { buf: grid.buf, w: grid.w, h: grid.h });
    }
    await device.queue.onSubmittedWorkDone();
  }
  const tAlignEnd = performance.now();
  for (const p of refPyramid) p.buf.destroy(); // grids are retained; the pyramid is not needed

  // ---- burst accumulator ---------------------------------------------------------
  const finalPlane = new Plane(gpu.zeros('freq-accumulator', W * H * 4), W, H);

  // ---- four displaced merge iterations, summed (each is already a burst average) --
  const iterationMs = [];
  for (let iter = 0; iter < ITERATION_SHIFTS.length; iter++) {
    const tIter = performance.now();
    const shift = ITERATION_SHIFTS[iter];
    const ox = freq.padMergeX + shift.left;  // content offset in the merge canvas (bayer px)
    const oy = freq.padMergeY + shift.top;

    // reference: prepare → place → rgba → per-tile noise → spectrum → accumulator seed
    const refPadded = await prepareFrame(gpu, frames[0], alignment.padX, alignment.padY);
    const refCanvas = placeF32(gpu, refPadded, freq.canvasW, freq.canvasH, ox - alignment.padX, oy - alignment.padY);
    refPadded.buf.destroy();
    const refRGBA = convertToRGBA(gpu, refCanvas);
    refCanvas.buf.destroy();
    const rms = calculateRms(gpu, refRGBA, nTiles);
    const refFT = forwardFft(gpu, refRGBA, nTiles);
    const finalFT = new Plane(gpu.copyBuffer('final-ft', refFT.buf, refFT.w * refFT.h * 16), refFT.w, refFT.h);
    const totalMismatch = new Plane(gpu.zeros('total-mismatch', nTiles.x * nTiles.y * 4), nTiles.x, nTiles.y);

    // per comparison frame: prepare → warp (alignment padding) → place → rgba →
    // mismatch → spectrum → merge into the accumulator
    for (let f = 1; f < nImages; f++) {
      onProgress?.({ stage: `merge ${iter + 1}/4`, frame: f, total: nImages - 1 });
      const compPadded = await prepareFrame(gpu, frames[f], alignment.padX, alignment.padY);
      const grid = gridBuffers.get(f);
      const alignedPadded = warpPlane(gpu, compPadded, grid, gridGeometry);
      compPadded.buf.destroy();
      const compCanvas = placeF32(gpu, alignedPadded, freq.canvasW, freq.canvasH, ox - alignment.padX, oy - alignment.padY);
      alignedPadded.buf.destroy();
      const compRGBA = convertToRGBA(gpu, compCanvas);
      compCanvas.buf.destroy();

      const absDiff = calculateAbsDiff(gpu, refRGBA, compRGBA);
      const mismatch = calculateMismatch(gpu, absDiff, rms, nTiles);
      absDiff.buf.destroy();

      // port of Swift's texture_mean(crop_texture(mismatch, shift/tile_size_merge, …)):
      // interior tile mean, then normalize so that mean maps to 0.12
      const cropL = shift.left / TILE_MERGE, cropR = shift.right / TILE_MERGE;
      const cropT = shift.top / TILE_MERGE, cropB = shift.bottom / TILE_MERGE;
      const meanW = nTiles.x - cropL - cropR, meanH = nTiles.y - cropT - cropB;
      let meanMismatch = 0.12; // degenerate interior fallback
      if (meanW > 0 && meanH > 0) {
        const interior = cropRectFn(gpu, mismatch, meanW, meanH, cropL, cropT);
        meanMismatch = await sumToScalar(gpu, interior) / (meanW * meanH);
        interior.buf.destroy();
      }
      normalizeMismatch(gpu, mismatch, meanMismatch);
      accumulate(gpu, totalMismatch, mismatch, nImages); // total += mismatch / n_images

      const compFT = forwardFft(gpu, compRGBA, nTiles);
      compRGBA.buf.destroy();
      mergeFrequencyDomain(gpu, refFT, compFT, finalFT, rms, mismatch, params, nTiles);
      mismatch.buf.destroy(); // consumed by the merge dispatch above
      compFT.buf.destroy();
    }

    // mild deconvolution, then back to the image domain
    deconvolute(gpu, finalFT, totalMismatch, nTiles);
    totalMismatch.buf.destroy();
    rms.buf.destroy();
    refFT.buf.destroy();
    const outRGBA = backwardFft(gpu, finalFT, freq.rgbaW, nTiles, nImages);
    finalFT.buf.destroy();
    reduceArtifactsTileBorder(gpu, outRGBA, refRGBA, frames[0].blackLevel, nTiles);
    refRGBA.buf.destroy();
    const outBayer = convertToBayer(gpu, outRGBA);
    outRGBA.buf.destroy();
    const cropped = cropRectFn(gpu, outBayer, W, H, ox, oy);
    outBayer.buf.destroy();
    accumulate(gpu, finalPlane, cropped, 1); // the four iterations are simply summed
    cropped.buf.destroy();

    await device.queue.onSubmittedWorkDone();
    iterationMs.push(performance.now() - tIter);
    onProgress?.({ stage: `merge ${iter + 1}/4`, frame: nImages - 1, total: nImages - 1 });
  }

  for (const g of gridBuffers.values()) g.buf.destroy();

  // ---- finalize: float → uint16 (identical to the spatial pipeline) ---------------
  const whiteLevel = frames[0].whiteLevel > 0 ? frames[0].whiteLevel : 65535;
  const refBlack4 = gpu.storageFromData('freq-black4', new Float32Array(frames[0].blackLevel));
  const u16Plane = new Plane(gpu.zeros('freq-u16', W * H * 4), W, H);
  gpu.dispatch('convert_to_u16',
    [{ buffer: finalPlane.buf }, { buffer: refBlack4 }, { buffer: u16Plane.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32'], [W, H, whiteLevel, 1, 2]) }], groups(W, H));
  refBlack4.destroy();
  finalPlane.buf.destroy();

  const raw = await gpu.readBack(u16Plane.buf, W * H * 4);
  u16Plane.buf.destroy();
  const u32 = new Uint32Array(raw);
  const pixels = new Uint16Array(W * H);
  for (let i = 0; i < pixels.length; i++) pixels[i] = u32[i];

  const tEnd = performance.now();
  const timings = {
    prepareMs: 0,
    alignMs: tAlignEnd - tAlignStart,
    mergeMs: tEnd - tAlignEnd,
    finalizeMs: tAlignStart - t0,
    totalMs: tEnd - t0,
    iterationsMs: iterationMs,
    frames: [],
  };
  return { width: W, height: H, pixels, timings, params, freq };
}
