// WebGPU port of Burst Photo's spatial-domain align+merge pipeline
// (burstphoto/merge/spatial.swift + burstphoto/align/align.swift).
//
// Scope (proof of concept, Bayer sensors, uniform-exposure bursts):
//  - pyramid tile alignment (generic tile-difference kernel, no N=25 fast path,
//    no correct_upsampling_error pass)
//  - no hot-pixel suppression
//  - robust merge in the spatial domain (color_difference + compute_merge_weight)
//  - no bracketed-exposure handling, no frequency-domain merge
//
// frames: [{ width, height, data: Uint16Array (Bayer mosaic), blackLevel: number[4],
//            whiteLevel: number }]  — all frames must share dimensions.
// options: { tileSize: 16|32|64, searchDistance: 32|64|128, noiseReduction: 0..23 }
//
// returns { width, height, pixels: Uint16Array, timings: {...} }

import { packUniforms } from './gpu.js';

const WORKGROUP = 8;

function groups(d1, d2, d3) {
  return [Math.ceil(d1 / WORKGROUP), Math.max(1, Math.ceil(d2 / WORKGROUP)), d3 || 1];
}

// Layer of the pyramid / intermediate plane: {buf, w, h}
class Plane {
  constructor(buf, w, h) {
    this.buf = buf;
    this.w = w;
    this.h = h;
  }
  get size() {
    return this.w * this.h * 4;
  }
}

export async function alignMergeSpatial(gpu, frames, options, onProgress) {
  const t0 = performance.now();
  const { device } = gpu;
  const nImages = frames.length;
  const mosaic = 2; // Bayer

  const { width: W, height: H } = frames[0];
  const { tileSize, searchDistance, noiseReduction } = options;

  // ---- port of align_merge_spatial_domain parameter derivation -------------
  // robustness: 4 steps in noise_reduction => factor 2 in the robustness norm
  const robustnessRev = 0.5 * (36.0 - Math.round(noiseReduction));
  const robustness = 0.12 * Math.pow(1.3, robustnessRev) - 0.4529822;
  const blurKernelSize = 16; // binomial filter kernel for the merge stage

  const minImageDim = Math.min(W, H);
  const downscaleFactorArray = [mosaic];
  const searchDistArray = [2];
  const tileSizeArray = [tileSize];
  let res = minImageDim / downscaleFactorArray[0];
  while (res > searchDistance) {
    downscaleFactorArray.push(2);
    searchDistArray.push(2);
    tileSizeArray.push(Math.max(tileSizeArray[tileSizeArray.length - 1] / 2, 8));
    res /= 2;
  }
  const nLevels = downscaleFactorArray.length;

  // padding so every pyramid level is a multiple of its tile size
  const tileFactor = tileSizeArray[nLevels - 1] * downscaleFactorArray.reduce((a, b) => a * b, 1);
  const padAlignX = Math.ceil(W / tileFactor);
  const padX = Math.trunc((padAlignX * tileFactor - W) / 2);
  const padAlignY = Math.ceil(H / tileFactor);
  const padY = Math.trunc((padAlignY * tileFactor - H) / 2);
  const W2 = W + 2 * padX;
  const H2 = H + 2 * padY;

  // ---- output accumulator ---------------------------------------------------
  const outPlane = new Plane(gpu.zeros('out-accumulator', W * H * 4), W, H);

  // ---- reference frame ------------------------------------------------------
  onProgress?.({ stage: 'prepare', frame: 0, total: nImages });
  const ref = frames[0]; // caller passes frames already ordered with ref first
  const refBlack4 = gpu.storageFromData('ref-black4', new Float32Array(ref.blackLevel));
  const refBlackMean = Math.max(0, ref.blackLevel.reduce((a, b) => a + b, 0) / ref.blackLevel.length);
  const refPadded = await prepareFrame(gpu, ref, padX, padY);
  const refCropped = crop(gpu, refPadded, padX, padY);

  // reference pyramid (port of build_pyramid; see note on buildPyramid below)
  const refPyramid = buildPyramid(gpu, refPadded, downscaleFactorArray, refBlackMean);

  // noise estimate: mean color difference between ref and ref_blurred
  const refBlurred = blurPlane(gpu, refCropped, blurKernelSize, mosaic);
  const noiseDiff = colorDifference(gpu, refCropped, refBlurred, mosaic);
  const noiseSd = await sumToScalar(gpu, noiseDiff) / (noiseDiff.w * noiseDiff.h);

  // ---- comparison frames ----------------------------------------------------
  const tAlign = performance.now();
  for (let f = 1; f < nImages; f++) {
    onProgress?.({ stage: 'align+merge', frame: f, total: nImages });
    const comp = frames[f];
    const compPadded = await prepareFrame(gpu, comp, padX, padY);

    const alignedPadded = alignTexture(gpu, refPyramid, compPadded, {
      nLevels, downscaleFactorArray, tileSizeArray,
      compBlackMean: Math.max(0, comp.blackLevel.reduce((a, b) => a + b, 0) / comp.blackLevel.length),
    });
    const aligned = crop(gpu, alignedPadded, padX, padY);

    // robust spatial merge (port of robust_merge)
    const compBlurred = blurPlane(gpu, aligned, blurKernelSize, mosaic);
    const texDiff = colorDifference(gpu, refBlurred, compBlurred, mosaic);
    const weight = new Plane(gpu.zeros('merge-weight', texDiff.size), texDiff.w, texDiff.h);
    gpu.dispatch('compute_merge_weight',
      [{ buffer: texDiff.buf }, { buffer: weight.buf }, { buffer: packUniforms(['i32', 'f32', 'f32'], [texDiff.w, noiseSd, robustness]) }], groups(texDiff.w, texDiff.h));

    const weightUp = upsampleBilinear(gpu, weight, W, H);
    const merged = addWeighted(gpu, refCropped, aligned, weightUp);
    accumulate(gpu, outPlane, merged, nImages);

    // free per-frame intermediates
    for (const b of [compPadded.buf, alignedPadded.buf, aligned.buf,
                     compBlurred.buf, texDiff.buf, weight.buf, weightUp.buf, merged.buf]) {
      b.destroy();
    }

    // Wait for the GPU to actually finish frame f before starting frame f+1.
    // dispatch() only enqueues work, and a setTimeout(0) yield (~1-4 ms) is far
    // shorter than a vsync — so without this await the loop outruns both the
    // GPU and the compositor: all onProgress messages fire within a few tens
    // of ms, most updates coalesce away unpainted, and the bar freezes at an
    // early value for the entire multi-second GPU drain. Awaiting
    // onSubmittedWorkDone makes each message mean "frame f is done" and gives
    // the browser hundreds of ms of idle main thread to paint between updates.
    // Cost: the GPU idles a few ms per frame boundary — negligible here.
    await device.queue.onSubmittedWorkDone();
  }

  // add reference frame contribution — the last work unit, so report the
  // frame counter at its end value: caller-side frame/total progress then
  // lands on 1 instead of jumping back to the start of the stage
  onProgress?.({ stage: 'accumulate-ref', frame: nImages, total: nImages });
  accumulate(gpu, outPlane, refCropped, nImages);

  const tMerge = performance.now();

  // ---- finalize: float -> uint16 --------------------------------------------
  const whiteLevel = ref.whiteLevel > 0 ? ref.whiteLevel : 65535;
  const u16Plane = new Plane(gpu.zeros('out-u16', W * H * 4), W, H);
  gpu.dispatch('convert_to_u16',
    [{ buffer: outPlane.buf }, { buffer: refBlack4 }, { buffer: u16Plane.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32'], [W, whiteLevel, 1, mosaic]) }], groups(W, H));

  const raw = await gpu.readBack(u16Plane.buf, W * H * 4);
  const u32 = new Uint32Array(raw);
  const pixels = new Uint16Array(W * H);
  for (let i = 0; i < pixels.length; i++) pixels[i] = u32[i];
  u16Plane.buf.destroy();
  outPlane.buf.destroy();

  const timings = {
    prepareMs: tAlign - t0,
    alignMergeMs: tMerge - tAlign,
    finalizeMs: performance.now() - tMerge,
    totalMs: performance.now() - t0,
  };
  return { width: W, height: H, pixels, timings, noiseSd, robustness };
}

// ---------------------------------------------------------------------------
// stage helpers

async function prepareFrame(gpu, frame, padX, padY) {
  const { width: W, height: H } = frame;
  // pack two 16-bit samples per u32 (zero-copy view over the Uint16Array)
  const packed = new Uint32Array(frame.data.buffer, frame.data.byteOffset, Math.floor((W * H) / 2));
  const rawBuf = gpu.storageFromData('raw-in', packed);
  const out = new Plane(gpu.zeros(`prepared-${frame.name ?? ''}`, (W + 2 * padX) * (H + 2 * padY) * 4), W + 2 * padX, H + 2 * padY);
  gpu.dispatch('prepare_bayer',
    [{ buffer: rawBuf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32'], [W, padX, padY]) }], groups(W, H));
  // rawBuf: queue.writeBuffer already ran; destroy after the work is enqueued is fine,
  // but keep it alive until submit — dispatch submits synchronously, so destroy now.
  rawBuf.destroy();
  return out;
}

// Port of build_pyramid(): level 0 is avg_pool(scale = downscale_list[0], black_level_mean),
// level i is avg_pool(blur(level i-1, kernel 2, mosaic 1), scale = downscale_list[i]).
// Note the pyramid BASE is therefore HALF the padded resolution for Bayer (downscale_list[0]
// = mosaic_pattern_width = 2) — tile grids and the warp are defined on that half-res base.
function buildPyramid(gpu, basePadded, downscaleFactorArray, blackMean) {
  const pyramid = [avgPool(gpu, basePadded, downscaleFactorArray[0], blackMean)];
  let cur = pyramid[0];
  for (let i = 1; i < downscaleFactorArray.length; i++) {
    const scale = downscaleFactorArray[i];
    const b1 = blurPlane(gpu, cur, 2, 1);
    const pooled = avgPool(gpu, b1, scale, 0.0);
    b1.buf.destroy();
    pyramid.push(pooled);
    cur = pooled;
  }
  return pyramid;
}

function blurPlane(gpu, plane, kernelSize, mosaicPatternWidth) {
  const tmp = new Plane(gpu.zeros('blur-tmp', plane.size), plane.w, plane.h);
  const out = new Plane(gpu.zeros('blur-out', plane.size), plane.w, plane.h);
  const mkParams = (size, dir) =>
    packUniforms(['i32', 'i32', 'i32', 'i32', 'i32'], [plane.w, size, kernelSize, mosaicPatternWidth, dir]);
  gpu.dispatch('blur', [{ buffer: plane.buf }, { buffer: tmp.buf }, { buffer: mkParams(plane.w, 0) }], groups(plane.w, plane.h));
  gpu.dispatch('blur', [{ buffer: tmp.buf }, { buffer: out.buf }, { buffer: mkParams(plane.h, 1) }], groups(plane.w, plane.h));
  tmp.buf.destroy();
  return out;
}

function avgPool(gpu, plane, scale, blackLevel) {
  const outW = Math.floor(plane.w / scale);
  const outH = Math.floor(plane.h / scale);
  const out = new Plane(gpu.zeros('pool-out', outW * outH * 4), outW, outH);
  gpu.dispatch('avg_pool',
    [{ buffer: plane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32'], [plane.w, outW, scale, blackLevel]) }], groups(outW, outH));
  return out;
}

function crop(gpu, plane, padX, padY) {
  const outW = plane.w - 2 * padX;
  const outH = plane.h - 2 * padY;
  const out = new Plane(gpu.zeros('crop-out', outW * outH * 4), outW, outH);
  gpu.dispatch('crop',
    [{ buffer: plane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32'], [plane.w, outW, padX, padY]) }], groups(outW, outH));
  return out;
}

function upsampleNearest(gpu, align, outW, outH) {
  const out = new Plane(gpu.zeros('align-up', outW * outH * 8), outW, outH); // vec2<i32>
  gpu.dispatch('upsample_nearest_i32',
    [{ buffer: align.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32', 'f32'], [align.w, align.h, outW, outW / align.w, outH / align.h]) }], groups(outW, outH));
  return out;
}

// Port of align_texture(): coarse-to-fine pyramid alignment.
function alignTexture(gpu, refPyramid, compPadded, { nLevels, downscaleFactorArray, tileSizeArray, compBlackMean }) {
  const compPyramid = buildPyramid(gpu, compPadded, downscaleFactorArray, compBlackMean);

  // 1x1 initial alignment (all zeros)
  let prevAlign = new Plane(gpu.zeros('align-0', 16), 1, 1);

  let tileDiff = null;
  for (let i = nLevels - 1; i >= 0; i--) {
    const tile = tileSizeArray[i];
    const scale = downscaleFactorArray[i];
    const refLayer = refPyramid[i];
    const compLayer = compPyramid[i];
    const nTilesX = Math.floor(refLayer.w / (tile / 2)) - 1;
    const nTilesY = Math.floor(refLayer.h / (tile / 2)) - 1;
    const nPos1d = 2 * 2 + 1; // search_dist is always 2 per level in the spatial pipeline
    const nPos2d = nPos1d * nPos1d;
    const downscale = i < nLevels - 1 ? downscaleFactorArray[i + 1] : 0;

    const prevUp = upsampleNearest(gpu, prevAlign, nTilesX, nTilesY);

    tileDiff = new Plane(gpu.zeros('tile-diff', nTilesX * nTilesY * nPos2d * 4), nTilesX * nPos2d, nTilesY);
    const useSsd = i !== 0 ? 1 : 0; // L2 on coarse levels, L1 on the finest level
    gpu.dispatch('compute_tile_differences',
      [{ buffer: refLayer.buf }, { buffer: compLayer.buf }, { buffer: prevUp.buf }, { buffer: tileDiff.buf },
       { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32'],
         [compLayer.w, compLayer.h, nTilesX, tile, 2, downscale, useSsd]) }], [Math.ceil(nTilesX / 8), Math.ceil(nTilesY / 8), nPos2d]);

    const curAlign = new Plane(gpu.zeros('align-cur', nTilesX * nTilesY * 8), nTilesX, nTilesY);
    gpu.dispatch('find_best_tile_alignment',
      [{ buffer: tileDiff.buf }, { buffer: prevUp.buf }, { buffer: curAlign.buf },
       { buffer: packUniforms(['i32', 'i32', 'i32', 'i32'], [nTilesX, nPos1d, 2, downscale]) }], groups(nTilesX, nTilesY));

    if (prevAlign !== curAlign) prevAlign.buf.destroy();
    prevAlign = curAlign;
    prevUp.buf.destroy();
    tileDiff.buf.destroy();
  }

  // warp the full-resolution comparison frame with the final alignment grid.
  // Metal passes half_tile_size = (downscale==2 ? 1 : downscale) * tile_size,
  // i.e. tile_size for Bayer, and the level-0 tile grid (computed on the
  // half-resolution pyramid base).
  const halfTile = downscaleFactorArray[0] === 2 ? 1 * tileSizeArray[0] : downscaleFactorArray[0] * tileSizeArray[0];
  const refBase = refPyramid[0];
  const nTilesX = Math.floor(refBase.w / (tileSizeArray[0] / 2)) - 1;
  const nTilesY = Math.floor(refBase.h / (tileSizeArray[0] / 2)) - 1;

  const out = new Plane(gpu.zeros('warped', compPadded.size), compPadded.w, compPadded.h);
  gpu.dispatch('warp_bayer',
    [{ buffer: compPadded.buf }, { buffer: out.buf }, { buffer: prevAlign.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'],
       [compPadded.w, compPadded.h, halfTile, nTilesX, nTilesY, downscaleFactorArray[0]]) }], groups(compPadded.w, compPadded.h));

  for (const p of compPyramid.slice(1)) p.buf.destroy();
  prevAlign.buf.destroy();
  return out;
}

function colorDifference(gpu, plane1, plane2, mosaic) {
  const outW = Math.floor(plane1.w / mosaic);
  const outH = Math.floor(plane1.h / mosaic);
  const out = new Plane(gpu.zeros('color-diff', outW * outH * 4), outW, outH);
  gpu.dispatch('color_difference',
    [{ buffer: plane1.buf }, { buffer: plane2.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32'], [plane1.w, mosaic]) }], groups(outW, outH));
  return out;
}

function upsampleBilinear(gpu, plane, outW, outH) {
  const out = new Plane(gpu.zeros('ups-bl', outW * outH * 4), outW, outH);
  gpu.dispatch('upsample_bilinear',
    [{ buffer: plane.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32', 'i32', 'i32', 'f32', 'f32'],
       [plane.w, plane.h, outW, outW / plane.w, outH / plane.h]) }], groups(outW, outH));
  return out;
}

function addWeighted(gpu, plane1, plane2, weight) {
  const out = new Plane(gpu.zeros('add-weighted', plane1.size), plane1.w, plane1.h);
  gpu.dispatch('add_texture_weighted',
    [{ buffer: plane1.buf }, { buffer: plane2.buf }, { buffer: weight.buf }, { buffer: out.buf },
     { buffer: packUniforms(['i32'], [plane1.w]) }], groups(plane1.w, plane1.h));
  return out;
}

function accumulate(gpu, outPlane, plane, nImages) {
  gpu.dispatch('accumulate',
    [{ buffer: plane.buf }, { buffer: outPlane.buf },
     { buffer: packUniforms(['i32', 'f32'], [plane.w, nImages]) }], groups(plane.w, plane.h));
}

async function sumToScalar(gpu, plane) {
  const n = plane.w * plane.h;
  const nPartials = Math.max(1, Math.ceil(n / 8192));
  const partials = new Plane(gpu.zeros('sum-partials', nPartials * 4), nPartials, 1);
  gpu.dispatch('sum_partials',
    [{ buffer: plane.buf }, { buffer: partials.buf }, { buffer: packUniforms(['u32'], [n]) }], [nPartials, 1, 1]);
  const scalar = new Plane(gpu.zeros('sum-final', 16), 1, 1);
  gpu.dispatch('sum_final',
    [{ buffer: partials.buf }, { buffer: scalar.buf }, { buffer: packUniforms(['u32'], [nPartials]) }], [1, 1, 1]);
  partials.buf.destroy();
  const data = await gpu.readBack(scalar.buf, 4);
  scalar.buf.destroy();
  return new Float32Array(data)[0];
}
