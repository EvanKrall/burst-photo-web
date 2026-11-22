// Alignment inspection: aligns every comparison frame of a burst against the reference frame
// and exposes the alignment field and per-frame textures for visualization, without merging.
// Port of the Swift visualize branch's inspect/inspect.swift + viz/visualize.swift
// (GPU-side halves), on top of the existing WebGPU merge pipeline.
//
// Data layout notes:
// - The alignment result of each frame is an Int32Array [dx0, dy0, dx1, dy1, ...] with one
//   (dx, dy) texel per tile, storing the tile shift in *pyramid level-0 pixels* (level 0 =
//   image downscaled by the mosaic pattern width). Multiply by `downscale` (2 for Bayer) to
//   get sensor-pixel shifts.
// - Only small artifacts (alignment texels, stats) are retained for every frame. The
//   full-resolution reference plane is retained on the GPU; warped candidate and difference
//   planes are re-rendered on demand from the raw frames (which stay in CPU memory anyway).

import {
  prepareFrame, buildPyramid, crop, alignTextureEx, avgPool, sumToScalar,
  Plane, groups, deriveAlignmentSetup,
} from './pipeline.js';
import { packUniforms } from './gpu.js';
import {
  alignmentStats, renderAlignmentField, alignmentVariationInRoi, normalizeRoi,
  brightnessMatchValues, robustBrightnessMatchValues, medianOfSorted,
  downsampleValues, grayBitmapFromValues, rgbaFromPackedU32, downsampleU16, downsampleRGBA,
} from './viz.js';
import { developBayer } from './raw.js';

const WORKGROUP = 8;

/**
 * Align every frame of the burst against the reference frame (identical setup to
 * alignMergeSpatial, so the visualization shows exactly the alignment the merge would use)
 * and return an InspectionSession. No merging is performed.
 *
 * frames: [{ name, width, height, data: Uint16Array, blackLevel: number[4], ... }] in capture order.
 * options: { tileSize, searchDistance, refIdx? } — refIdx defaults to the burst center.
 * onProgress: ({ stage, frame, total }) => void
 */
export async function performInspection(gpu, frames, options, onProgress) {
  const { width: W, height: H } = frames[0];
  const { tileSize, searchDistance } = options;
  const refIdx = Number.isInteger(options.refIdx) ? options.refIdx : Math.floor(frames.length / 2);
  const { downscaleFactorArray, tileSizeArray, padX, padY } = deriveAlignmentSetup(W, H, tileSize, searchDistance);
  const nLevels = downscaleFactorArray.length;

  const blackMean = (f) => Math.max(0, f.blackLevel.reduce((a, b) => a + b, 0) / f.blackLevel.length);

  // ---- reference frame: prepared + cropped plane (retained for on-demand rendering) + pyramid
  const refPadded = await prepareFrame(gpu, frames[refIdx], padX, padY);
  const refCropped = crop(gpu, refPadded, padX, padY);
  const refPyramid = buildPyramid(gpu, refPadded, downscaleFactorArray, blackMean(frames[refIdx]));
  refPadded.buf.destroy(); // the pyramid holds its own pooled copies

  const session = new InspectionSession(gpu, frames, refIdx, {
    width: W, height: H, padX, padY, refCropped,
    tileInfo: { w: 0, h: 0, tileSize: tileSizeArray[0], downscale: downscaleFactorArray[0] },
  });

  // ---- align every comparison frame, retaining only the small alignment grids
  const alignStart = performance.now();
  const perFrameMs = []; // prepare + pyramid + coarse-to-fine search + grid readback, per frame
  let alignedCount = 0; // comparison frames aligned so far (the ref frame is skipped)
  for (let i = 0; i < frames.length; i++) {
    if (i === refIdx) {
      session.addFrame({ idx: i, name: frames[i].name, isRef: true, texels: null, stats: null });
      continue;
    }
    onProgress?.({ stage: 'align', frame: i, total: frames.length });
    const t0 = performance.now();
    const compPadded = await prepareFrame(gpu, frames[i], padX, padY);
    const { warped, alignment } = alignTextureEx(gpu, refPyramid, compPadded, {
      nLevels, downscaleFactorArray, tileSizeArray,
      compBlackMean: blackMean(frames[i]),
    });
    session.tileInfo.w = alignment.w;
    session.tileInfo.h = alignment.h;
    const texels = new Int32Array(await gpu.readBack(alignment.buf, alignment.w * alignment.h * 8));
    alignment.buf.destroy();
    compPadded.buf.destroy();
    warped.buf.destroy();
    session.addFrame({ idx: i, name: frames[i].name, isRef: false, texels, stats: alignmentStats(texels) });
    perFrameMs.push(performance.now() - t0); // readBack resolves only after the GPU finished
    alignedCount += 1;
    onProgress?.({ stage: 'align', frame: alignedCount, total: frames.length - 1, ms: perFrameMs[perFrameMs.length - 1] });
    // let the GPU drain so progress updates stay visible (see pipeline.js for the rationale)
    await gpu.device.queue.onSubmittedWorkDone();
  }
  const alignTimings = {
    totalMs: performance.now() - alignStart,
    perFrameMs,
    frames: frames.length - 1,
  };
  for (const p of refPyramid) p.buf.destroy();
  session.timings = alignTimings; // exposed for the UI log
  return session;
}

/**
 * Everything needed to render alignment visualizations of a burst on demand.
 * The heavy per-frame data (warped thumbnails, difference statistics, brightness matches)
 * is built by calibrate() in the background; renders fall back to on-demand computation
 * until then.
 */
export class InspectionSession {
  constructor(gpu, frames, refIdx, { width, height, padX, padY, refCropped, tileInfo }) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.framesIn = frames;
    this.refIdx = refIdx;
    this.width = width;
    this.height = height;
    this.padX = padX;
    this.padY = padY;
    this.refCropped = refCropped;
    this.tileInfo = tileInfo;

    this.frameList = []; // [{idx, name, isRef, texels, stats}]
    this.byIdx = new Map();

    // burst-wide shared alignment scale (saturation point), port of the Swift inspector's
    // init: 2 × median mean |shift|, capped by the burst's largest shift
    this.medianMeanShift = 0;
    this.burstMaxShift = 1;
    this.alignmentScale = 1;

    // background calibration state (see calibrate)
    this.calibrated = false;
    this.calibrationProgress = 0;
    this.medianDiffMean = 0;
    this.frameData = new Map(); // idx -> {warped48, diff48, diff192, diffMean, warpedMax}
    this.matchCache = new Map(); // idx -> {gain, offset}
    this.refPooled = null; // {values: Float32Array, width, height}
    this.refMax = 0;
    this.refPooledMean = 0;

    this.gridBuffers = new Map(); // idx -> GPUBuffer with the alignment grid (array<vec2<i32>>)
    this.zeroGrid = null;
    this._rgbCache = new Map();   // developed RGB photo views, key (view, frame, shift, size)
    this._rgbCacheSize = null;    // display size the cache was built for
  }

  addFrame(entry) {
    this.frameList.push(entry);
    this.byIdx.set(entry.idx, entry);
    if (!entry.isRef) {
      const magnitudes = this.frameList.filter((f) => f.stats).map((f) => f.stats.meanMagnitude).sort((a, b) => a - b);
      this.medianMeanShift = medianOfSorted(magnitudes);
      this.burstMaxShift = Math.max(1, ...this.frameList.filter((f) => f.stats).map((f) => f.stats.maxAbsShift));
      this.alignmentScale = Math.max(1, Math.min(Math.round(this.medianMeanShift * 2), this.burstMaxShift));
    }
  }

  frame(idx) { return this.byIdx.get(idx); }
  comparisonFrames() { return this.frameList.filter((f) => !f.isRef); }

  destroy() {
    this.destroyed = true;
    this.refCropped?.buf.destroy();
    for (const b of this.gridBuffers.values()) b.destroy();
    this.zeroGrid?.destroy();
  }

  // ------------------------------------------------------------- alignment access

  /** (dx, dy) shift of one frame's tile at the given grid position (level-0 pixels). */
  texelOffset(idx, tileX, tileY) {
    const f = this.frame(idx);
    if (!f || f.isRef) return [0, 0];
    const x = Math.min(Math.max(tileX, 0), this.tileInfo.w - 1);
    const y = Math.min(Math.max(tileY, 0), this.tileInfo.h - 1);
    const i = (y * this.tileInfo.w + x) * 2;
    return [f.texels[i], f.texels[i + 1]];
  }

  /** ROI alignment-variation score in level-0 px (lower = more rigid mapping); null if unscoreable. */
  alignmentVariation(idx, roi) {
    const f = this.frame(idx);
    if (!f || f.isRef) return null;
    return alignmentVariationInRoi(f.texels, this.tileInfo.w, this.tileInfo.h, this.tileInfo.tileSize,
      this.tileInfo.downscale, roi, this.width, this.height, this.padX, this.padY);
  }

  /**
   * Mean brightness-matched absolute difference inside the ROI, from the calibration pass's
   * 192-px difference downsample (lower = more similar to the reference); null until calibrated.
   */
  roiDiffScore(idx, roi) {
    const data = this.frameData.get(idx);
    if (!data) return null;
    const d = data.diff192;
    const n = normalizeRoi(roi);
    const cx0 = Math.min(d.width - 1, Math.max(0, Math.floor(n.x0 * d.width)));
    const cx1 = Math.max(cx0 + 1, Math.min(d.width, Math.ceil(n.x1 * d.width)));
    const cy0 = Math.min(d.height - 1, Math.max(0, Math.floor(n.y0 * d.height)));
    const cy1 = Math.max(cy0 + 1, Math.min(d.height, Math.ceil(n.y1 * d.height)));
    let sum = 0, count = 0;
    for (let y = cy0; y < cy1; y++) {
      const row = y * d.width;
      for (let x = cx0; x < cx1; x++) { sum += d.values[row + x]; count++; }
    }
    return count ? sum / count : null;
  }

  // ------------------------------------------------------------------ GPU plumbing

  gridBufferFor(idx) {
    const f = this.frame(idx);
    if (f.isRef) {
      if (!this.zeroGrid) {
        const n = this.tileInfo.w * this.tileInfo.h;
        this.zeroGrid = this.gpu.storageFromData('align-zero', new Int32Array(n * 2));
      }
      return this.zeroGrid;
    }
    let buf = this.gridBuffers.get(idx);
    if (!buf) {
      buf = this.gpu.storageFromData(`align-grid-${idx}`, f.texels);
      this.gridBuffers.set(idx, buf);
    }
    return buf;
  }

  constantGridBuffer(dx, dy) {
    const n = this.tileInfo.w * this.tileInfo.h;
    const data = new Int32Array(n * 2);
    for (let i = 0; i < n; i++) { data[2 * i] = dx; data[2 * i + 1] = dy; }
    return this.gpu.storageFromData('align-constant', data);
  }

  /**
   * The warped (aligned) comparison texture of one frame, cropped to the original frame —
   * port of InspectionSession.aligned_texture(for:). The caller owns the returned plane
   * (the reference frame returns the retained refCropped, which must NOT be destroyed).
   */
  async alignedPlane(idx) {
    const f = this.frame(idx);
    if (f.isRef) return this.refCropped;
    const compPadded = await prepareFrame(this.gpu, this.framesIn[idx], this.padX, this.padY);
    const out = new Plane(this.gpu.zeros(`warped-${idx}`, compPadded.size), compPadded.w, compPadded.h);
    const halfTile = this.tileInfo.downscale === 2 ? this.tileInfo.tileSize : this.tileInfo.downscale * this.tileInfo.tileSize;
    this.gpu.dispatch('warp_bayer',
      [{ buffer: compPadded.buf }, { buffer: out.buf }, { buffer: this.gridBufferFor(idx) },
       { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'],
         [compPadded.w, compPadded.h, halfTile, this.tileInfo.w, this.tileInfo.h, this.tileInfo.downscale]) }],
      groups(compPadded.w, compPadded.h));
    const aligned = crop(this.gpu, out, this.padX, this.padY);
    compPadded.buf.destroy();
    out.buf.destroy();
    return aligned;
  }

  /** Integer avg-pool + readback of a plane (pool must divide both dimensions; Swift's pooled_values). */
  async pooledPlane(plane, pool) {
    const outW = plane.w / pool, outH = plane.h / pool;
    const pooled = avgPool(this.gpu, new Plane(plane.buf, plane.w, plane.h), pool, 0.0);
    const raw = await this.gpu.readBack(pooled.buf, outW * outH * 4);
    pooled.buf.destroy();
    return { values: new Float32Array(raw), width: outW, height: outH };
  }

  /** Largest pool size ≤ preferred that divides both dimensions (Swift's pooled_values choice). */
  static pickPool(plane, preferred = 4) {
    for (let c = preferred; c >= 1; c--) {
      if (plane.w % c === 0 && plane.h % c === 0) return c;
    }
    return 1;
  }

  async ensureRefPooled() {
    if (this.refPooled) return this.refPooled;
    const pool = InspectionSession.pickPool(this.refCropped, 4);
    const pooled = await this.pooledPlane(this.refCropped, pool);
    this.refPooled = pooled;
    let max = 0, sum = 0;
    for (const v of pooled.values) { if (v > max) max = v; sum += v; }
    this.refMax = max;
    this.refPooledMean = sum / pooled.values.length;
    return pooled;
  }

  /** Brightness match of frame idx to the reference (global least squares, cached). */
  async globalMatch(idx) {
    const cached = this.matchCache.get(idx);
    if (cached) return cached;
    await this.ensureRefPooled();
    const aligned = await this.alignedPlane(idx);
    const pool = InspectionSession.pickPool(this.refCropped, 4);
    const pooled = await this.pooledPlane(aligned, pool);
    if (!this.frame(idx).isRef) aligned.buf.destroy(); // the ref returns the retained refCropped
    const match = brightnessMatchValues(this.refPooled.values, pooled.values);
    this.matchCache.set(idx, match);
    return match;
  }

  /**
   * Robust brightness match from a window of pooled pixels around the anchor point, so that
   * the fit describes the neighborhood the user chose instead of the global image (whose
   * moving content and occlusions would otherwise bias a global fit).
   * Port of InspectorModel.local_brightness_match.
   */
  async localMatch(idx, anchor) {
    await this.ensureRefPooled();
    const aligned = await this.alignedPlane(idx);
    const pool = InspectionSession.pickPool(this.refCropped, 4);
    const pooled = await this.pooledPlane(aligned, pool);
    if (!this.frame(idx).isRef) aligned.buf.destroy(); // the ref returns the retained refCropped
    const { values: comp, width, height } = pooled;
    const ref = this.refPooled.values;
    const centerX = Math.min(Math.max(Math.round(anchor.u * width), 0), width - 1);
    const centerY = Math.min(Math.max(Math.round(anchor.v * height), 0), height - 1);
    const radius = 48;
    let refWindow = [];
    let compWindow = [];
    for (let y = Math.max(0, centerY - radius); y <= Math.min(height - 1, centerY + radius); y++) {
      for (let x = Math.max(0, centerX - radius); x <= Math.min(width - 1, centerX + radius); x++) {
        const i = y * width + x;
        refWindow.push(ref[i]);
        compWindow.push(comp[i]);
      }
    }
    // cap the number of samples for the (gain × offset) inlier search
    if (refWindow.length > 4096) {
      const step = (refWindow.length / 4096 | 0) + 1;
      refWindow = refWindow.filter((_, i) => i % step === 0);
      compWindow = compWindow.filter((_, i) => i % step === 0);
    }
    let meanRef = 0;
    for (const v of refWindow) meanRef += v;
    meanRef /= refWindow.length;
    const tolerance = Math.max(0.01, 0.04 * meanRef);
    const fit = robustBrightnessMatchValues(refWindow, compWindow, tolerance);
    return fit;
  }

  /** Mean of an f32 plane via the two-stage sum kernels (array length must be ≤ ~2^31). */
  async planeMean(plane) {
    const value = await sumToScalar(this.gpu, plane);
    return value / (plane.w * plane.h);
  }

  /** Brightest integer-pooled pixel of a plane (fallback white point before calibration). */
  async planeMaxPooled(plane, preferredPool = 4) {
    const pool = InspectionSession.pickPool(plane, preferredPool);
    const pooled = await this.pooledPlane(plane, pool);
    let max = 0;
    for (const v of pooled.values) if (v > max) max = v;
    return max;
  }

  // ------------------------------------------------------------------- calibration

  /**
   * Build the downsampled per-frame data (warped + difference), warm the brightness-match
   * cache and measure the shared difference scale (median frame mean difference). Port of
   * InspectorModel.start_background_work. Idempotent; yields to the GPU between frames.
   * The optional isCancelled callback (returning true) stops the pass early.
   */
  async calibrate(onProgress, isCancelled = null) {
    if (this.calibrating) return;
    this.calibrating = true;
    try {
      if (this.destroyed) return;
      await this.ensureRefPooled();
      const refValues = this.refPooled.values;
      const pool = InspectionSession.pickPool(this.refCropped, 4);
      const ref48 = downsampleValues(refValues, this.refPooled.width, this.refPooled.height);
      this.refThumbnailData = { ...ref48, max: this.refMax };

      const comparisons = this.comparisonFrames();
      const means = [];
      for (let n = 0; n < comparisons.length; n++) {
        const idx = comparisons[n].idx;
        const aligned = await this.alignedPlane(idx);
        const pooled = await this.pooledPlane(aligned, pool);
        aligned.buf.destroy();

        let warpedMax = 0;
        for (const v of pooled.values) if (v > warpedMax) warpedMax = v;

        // brightness match and per-pixel difference, computed on the pooled values
        const match = brightnessMatchValues(refValues, pooled.values);
        this.matchCache.set(idx, match);
        const diffValues = new Float32Array(pooled.values.length);
        let diffSum = 0;
        for (let i = 0; i < pooled.values.length; i++) {
          const d = Math.abs(refValues[i] - (match.gain * pooled.values[i] + match.offset));
          diffValues[i] = d;
          diffSum += d;
        }
        const data = {
          warped48: downsampleValues(pooled.values, pooled.width, pooled.height),
          diff48: downsampleValues(diffValues, pooled.width, pooled.height),
          diff192: downsampleValues(diffValues, pooled.width, pooled.height, 192),
          diffMean: diffSum / pooled.values.length,
          warpedMax,
        };
        this.frameData.set(idx, data);
        means.push(data.diffMean);

        this.calibrationProgress = n + 1;
        const cancelled = onProgress?.({ frame: n + 1, total: comparisons.length }) === true;
        if (cancelled || this.destroyed) return;
        // let the GPU + UI breathe between frames
        await this.device.queue.onSubmittedWorkDone();
      }

      this.medianDiffMean = medianOfSorted(means.slice().sort((a, b) => a - b));
      this.calibrated = true;
    } finally {
      this.calibrating = false;
    }
  }

  // ---------------------------------------------------------------------- rendering

  /**
   * Render one frame in the given mode. Returns { bitmap: {rgba, width, height}, info }.
   * params: {
   *   mode: 'alignment' | 'flow' | 'difference' | 'original' | 'warped' | 'reference',
   *   alignmentScale,               // level-0 px mapped to full saturation (alignment/flow)
   *   offsetX, offsetY,             // anchor offset subtracted from the field (level-0 px)
   *   outWidth, outHeight,          // pooled render size for the photo modes
   *   diffGain,                     // difference gain multiplier
   *   diffWhiteOverride,            // shared white point (gain × median), or null = per-frame
   *   matchExposure,                // brightness-match before differencing
   *   anchor,                       // {u, v} fractions — enables the anchor-local robust fit
   * }
   * Port of InspectorModel.render (see AlignmentInspectorView.swift for the Swift originals
   * of the per-mode info strings).
   */
  /**
   * Developed-RGB photo view (Original / Warped / Reference tabs): the frame's Bayer data
   * — or the warped plane, converted back to u16 Bayer — is developed with the whole
   * flashraw chain (src/raw.js developBayer: demosaic, as-shot WB, color matrix, sRGB)
   * and box-averaged to the display size. Cached per (view, frame, anchor shift, size);
   * returns null when the develop fails so the caller can fall back to the gray view.
   * Returns { bitmap, dx, dy } (the anchor shift applied, for the info line).
   */
  async developedPhotoView(mode, idx, outW, outH, anchor) {
    const size = `${outW}x${outH}`;
    if (this._rgbCacheSize !== size) {
      this._rgbCache.clear();
      this._rgbCacheSize = size;
    }
    let dx = 0, dy = 0;
    const f = this.frame(idx);
    if (mode === 'original' && anchor && !f.isRef) {
      [dx, dy] = this.texelOffset(idx, anchor.tileX, anchor.tileY);
    }
    const key = (mode === 'warped' ? `w:${idx}` : `p:${idx}`) + (dx || dy ? `:${dx},${dy}` : '');
    const hit = this._rgbCache.get(key);
    if (hit) return hit;
    try {
      let frame;
      if (mode === 'warped') {
        frame = await this.warpedBayerU16(idx);
      } else if (dx || dy) {
        frame = await this.shiftedBayerU16(idx, dx, dy);
      } else {
        // Reference (and unshifted Original of the reference frame — same content)
        frame = this.framesIn[mode === 'reference' ? this.refIdx : idx];
      }
      const dev = await developBayer(frame);
      const entry = { bitmap: downsampleRGBA(dev.rgba, dev.width, dev.height, outW, outH), dx, dy };
      this._rgbCache.set(key, entry);
      return entry;
    } catch (err) {
      console.warn(`RGB develop failed (${mode} frame ${idx}); using the gray view`, err);
      return null;
    }
  }

  /**
   * The warped candidate plane as u16 Bayer: prepare_bayer is black-level-neutral
   * (black is only subtracted at the final convert_to_u16), so the plane holds original
   * sensor values — a clamp to [0, whiteLevel] reconstructs a developable frame.
   */
  async warpedBayerU16(idx) {
    const plane = await this.alignedPlane(idx);
    const owned = !this.frame(idx).isRef; // the reference frame's plane is the retained ref plane
    const data = await this.planeToBayerU16(plane, this.framesIn[idx]);
    if (owned) plane.buf.destroy();
    return { ...this.framesIn[idx], data, width: plane.w, height: plane.h };
  }

  /** Original (unwarped) frame shifted by a constant (dx, dy) to match the anchor. */
  async shiftedBayerU16(idx, dx, dy) {
    const compPadded = await prepareFrame(this.gpu, this.framesIn[idx], this.padX, this.padY);
    const grid = this.constantGridBuffer(dx, dy);
    const out = new Plane(this.gpu.zeros('shifted', compPadded.size), compPadded.w, compPadded.h);
    const halfTile = this.tileInfo.downscale === 2 ? this.tileInfo.tileSize : this.tileInfo.downscale * this.tileInfo.tileSize;
    this.gpu.dispatch('warp_bayer',
      [{ buffer: compPadded.buf }, { buffer: out.buf }, { buffer: grid },
       { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'],
         [compPadded.w, compPadded.h, halfTile, this.tileInfo.w, this.tileInfo.h, this.tileInfo.downscale]) }],
      groups(compPadded.w, compPadded.h));
    grid.destroy();
    const plane = crop(this.gpu, out, this.padX, this.padY);
    out.buf.destroy();
    const data = await this.planeToBayerU16(plane, this.framesIn[idx]);
    plane.buf.destroy();
    return { ...this.framesIn[idx], data };
  }

  async planeToBayerU16(plane, src) {
    const f32 = new Float32Array(await this.gpu.readBack(plane.buf, plane.size));
    const white = src.whiteLevel > 0 ? src.whiteLevel : 65535;
    const u16 = new Uint16Array(plane.w * plane.h);
    for (let i = 0; i < u16.length; i++) {
      const v = f32[i];
      u16[i] = v <= 0 ? 0 : v >= white ? white : Math.round(v);
    }
    return u16;
  }

  async renderFrame(idx, params) {
    const f = this.frame(idx);
    switch (params.mode) {
      case 'merged': {
        // the merged result (set by the UI after an inspector merge), developed to RGB
        // with the whole flashraw chain — the same image the old preview pane showed.
        // Mode is burst-level, so `idx` is ignored. Cached per output size; falls back
        // to the gray block view when flashraw cannot develop our DNG.
        const m = this.mergedResult;
        if (!m) {
          return { bitmap: grayBitmapFromValues(new Float32Array(1), 1, 1, 1), info: 'no merged result yet' };
        }
        const outW = params.outWidth, outH = params.outHeight;
        let bitmap = this._mergedCache && this._mergedCache.w === outW && this._mergedCache.h === outH
          ? this._mergedCache.bitmap : null;
        let developed = false;
        if (!bitmap) {
          try {
            const dev = await developBayer({ ...m, data: m.pixels });
            bitmap = downsampleRGBA(dev.rgba, dev.width, dev.height, outW, outH);
            developed = true;
          } catch (err) {
            console.warn('merged RGB develop failed; using the gray view', err);
          }
          if (!bitmap) {
            const values = downsampleU16(m.pixels, m.width, m.height, outW, outH);
            const black = m.blackLevel.reduce((a, b) => a + b, 0) / m.blackLevel.length;
            let max = 0;
            for (let i = 0; i < values.length; i++) {
              values[i] -= black;
              if (values[i] > max) max = values[i];
            }
            bitmap = grayBitmapFromValues(values, outW, outH, Math.max(max, 1e-6));
          }
          this._mergedCache = { w: outW, h: outH, bitmap };
        }
        return {
          bitmap,
          info: developed
            ? `merged result of ${m.count} frames · developed RGB (demosaic, WB, color matrix, sRGB) · "Download DNG" below`
            : `merged result of ${m.count} frames · gray tonemap (flashraw develop failed) · "Download DNG" below`,
        };
      }

      case 'alignment': {
        // per-tile shift field as optical-flow colors: hue = shift direction,
        // brightness = shift magnitude (the reference frame's zero field is uniform black)
        const texels = f.isRef
          ? new Int32Array(this.tileInfo.w * this.tileInfo.h * 2)
          : f.texels;
        const bitmap = renderAlignmentField(texels, this.tileInfo.w, this.tileInfo.h, {
          scale: params.alignmentScale,
          offsetX: params.offsetX || 0,
          offsetY: params.offsetY || 0,
        });
        if (f.isRef) {
          return { bitmap, info: 'reference frame · alignment is zero by definition (uniform black)' };
        }
        let info = `alignment colors saturate at ${params.alignmentScale} level-0 px (${params.alignmentScale * this.tileInfo.downscale} sensor px), shared by all frames · hue = direction, brightness = magnitude`;
        if ((params.offsetX || 0) !== 0 || (params.offsetY || 0) !== 0) info += ' · RELATIVE to anchor';
        return { bitmap, info };
      }

      case 'difference': {
        const ref = this.refCropped;
        const aligned = await this.alignedPlane(idx);
        const alignedOwned = !f.isRef; // the reference frame's "aligned" plane is the retained ref plane
        let gain = 1, offset = 0, fitNote = '';
        if (params.matchExposure) {
          if (params.anchor && this.calibrated) {
            const local = await this.localMatch(idx, params.anchor);
            gain = local.gain; offset = local.offset;
            fitNote = `local robust fit around anchor, ${(local.inlierFraction * 100).toFixed(0)}% inliers`;
          } else {
            const global = await this.globalMatch(idx);
            gain = global.gain; offset = global.offset;
            fitNote = params.anchor && !this.calibrated ? 'global fit (local needs calibration)' : 'global fit';
          }
        }
        // render-time statistics (mean |diff|), exactly like Swift's texture_stats on the
        // freshly built abs-difference texture
        const diffPlane = new Plane(this.gpu.zeros('diff-f32', ref.size), ref.w, ref.h);
        this.gpu.dispatch('diff_f32',
          [{ buffer: ref.buf }, { buffer: aligned.buf }, { buffer: diffPlane.buf },
           { buffer: packUniforms(['i32', 'i32', 'f32', 'f32'], [ref.w, ref.h, gain, offset]) }],
          groups(ref.w, ref.h));
        const meanAbsDiff = await this.planeMean(diffPlane);
        diffPlane.buf.destroy();

        const white = params.diffWhiteOverride ?? params.diffGain * meanAbsDiff;
        const outW = params.outWidth, outH = params.outHeight;
        const out = this.gpu.zeros('diff-rgba8', outW * outH * 4);
        this.gpu.dispatch('diff_rgba8',
          [{ buffer: ref.buf }, { buffer: aligned.buf }, { buffer: out },
           { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32', 'f32', 'f32'],
             [ref.w, ref.h, outW, outH, gain, offset, white]) }],
          groups(outW, outH));
        const packed = await this.gpu.readBack(out, outW * outH * 4);
        out.destroy();
        if (alignedOwned) aligned.buf.destroy();

        let info = 'red = warped brighter than reference, blue = warped darker, black = equal';
        info += params.diffWhiteOverride != null
          ? ` · white point ${white.toFixed(0)} (shared)`
          : ` · white point ${white.toFixed(0)} (per-frame)`;
        if (params.matchExposure) {
          info += ` · ΔEV ${(-Math.log2(gain)).toFixed(3)} (${fitNote}: gain ${gain.toFixed(3)}, offset ${offset >= 0 ? '+' : ''}${offset.toFixed(1)}) · mean |diff| ${meanAbsDiff.toFixed(1)}`;
        } else {
          info += ' · raw difference (no brightness match)';
        }
        return { bitmap: { rgba: rgbaFromPackedU32(packed), width: outW, height: outH }, info };
      }

      case 'original':
      case 'warped':
      case 'reference': {
        // preferred path: true RGB — the frame's Bayer data (or the warped plane,
        // converted back to u16 Bayer) is developed with the whole flashraw chain and
        // box-averaged to the display size, cached per (frame, view, shift, size).
        const rgb = await this.developedPhotoView(params.mode, idx, params.outWidth, params.outHeight, params.anchor);
        if (rgb) {
          const DEV = 'developed RGB (demosaic, as-shot WB, color matrix, sRGB)';
          const info = params.mode === 'reference'
            ? `reference frame · ${DEV}`
            : params.mode === 'warped'
              ? `warped candidate frame · ${DEV}`
              : (params.anchor && !f.isRef
                  ? `original frame, shifted by (${rgb.dx}, ${rgb.dy}) l0-px to match the anchor · ${DEV}`
                  : `original frame, exposure-equalized · ${DEV}`);
          return { bitmap: rgb.bitmap, info };
        }
        // fallback: the original gray tonemap (flashraw unavailable / develop failed)
        let plane, owned, info;
        if (params.mode === 'reference') {
          plane = this.refCropped; owned = false;
          info = 'reference frame';
        } else if (params.mode === 'warped') {
          plane = await this.alignedPlane(idx); owned = !this.frame(idx).isRef;
          info = 'warped candidate frame';
        } else {
          const fSel = this.frame(idx);
          if (fSel.isRef) {
            plane = this.refCropped; owned = false;
            info = 'reference frame · unwarped by definition';
          } else {
            const compPadded = await prepareFrame(this.gpu, this.framesIn[idx], this.padX, this.padY);
            if (params.anchor) {
              // shift the unwarped frame by this frame's alignment at the anchor tile, so it
              // stays registered with the reference around the anchor (like the alignment
              // views, which treat the anchor as zero)
              const [dx, dy] = this.texelOffset(idx, params.anchor.tileX, params.anchor.tileY);
              const grid = this.constantGridBuffer(dx, dy);
              const out = new Plane(this.gpu.zeros('shifted', compPadded.size), compPadded.w, compPadded.h);
              const halfTile = this.tileInfo.downscale === 2 ? this.tileInfo.tileSize : this.tileInfo.downscale * this.tileInfo.tileSize;
              this.gpu.dispatch('warp_bayer',
                [{ buffer: compPadded.buf }, { buffer: out.buf }, { buffer: grid },
                 { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'],
                   [compPadded.w, compPadded.h, halfTile, this.tileInfo.w, this.tileInfo.h, this.tileInfo.downscale]) }],
                groups(compPadded.w, compPadded.h));
              grid.destroy();
              plane = crop(this.gpu, out, this.padX, this.padY);
              out.buf.destroy();
              owned = true;
              info = `original (unwarped) frame, shifted by (${dx}, ${dy}) l0-px to match the anchor`;
            } else {
              plane = crop(this.gpu, compPadded, this.padX, this.padY);
              compPadded.buf.destroy();
              owned = true;
              info = 'original (unwarped) frame, exposure-equalized';
            }
          }
        }

        // white point: brightest pooled pixel (calibration data when available)
        let white;
        if (idx === this.refIdx) {
          white = this.refMax || await this.planeMaxPooled(plane);
        } else {
          white = this.frameData.get(idx)?.warpedMax || await this.planeMaxPooled(plane);
        }
        white = Math.max(white, 1e-6);

        const outW = params.outWidth, outH = params.outHeight;
        const out = this.gpu.zeros('gray-rgba8', outW * outH * 4);
        this.gpu.dispatch('gray_rgba8',
          [{ buffer: plane.buf }, { buffer: out },
           { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32'], [plane.w, plane.h, outW, outH, white]) }],
          groups(outW, outH));
        const packed = await this.gpu.readBack(out, outW * outH * 4);
        out.destroy();
        if (owned) plane.buf.destroy();
        info += ` · white = brightest pooled pixel ${white.toFixed(0)}`;
        return { bitmap: { rgba: rgbaFromPackedU32(packed), width: outW, height: outH }, info };
      }

      default:
        throw new Error(`unknown inspection mode ${params.mode}`);
    }
  }
}
