// CPU-side visualization + scoring math for the alignment inspector.
//
// Ports of Burst Photo's visualize branch:
//  - viz/visualize.swift: alignment stats, CPU alignment-field rendering
//    (render_alignment_color kernel logic), brightness matching (least-squares
//    + robust inlier-maximizing), gray tonemapping, box downsampling.
//  - inspect/inspect.swift: alignment_variation_in_roi (the score used both by
//    the inspector's ROI column and for automatic frame rejection).
//
// All functions are DOM-free and run in Node too (unit-testable).

// ---------------------------------------------------------------- alignment field

/**
 * Summary of the per-tile shifts of one alignment grid (all values in pyramid
 * level-0 pixels; multiply by the grid's downscale factor for sensor pixels).
 * texels: Int32Array [dx0, dy0, dx1, dy1, ...], one pair per tile (row-major).
 */
export function alignmentStats(texels) {
  const nTiles = texels.length / 2;
  let minDx = Infinity, maxDx = -Infinity, minDy = Infinity, maxDy = -Infinity;
  let magnitudeSum = 0;
  for (let i = 0; i < nTiles; i++) {
    const dx = texels[2 * i], dy = texels[2 * i + 1];
    if (dx < minDx) minDx = dx;
    if (dx > maxDx) maxDx = dx;
    if (dy < minDy) minDy = dy;
    if (dy > maxDy) maxDy = dy;
    magnitudeSum += Math.sqrt(dx * dx + dy * dy);
  }
  return {
    minDx, maxDx, minDy, maxDy,
    meanMagnitude: magnitudeSum / nTiles,
    // largest absolute shift component in the field
    maxAbsShift: Math.max(Math.abs(minDx), Math.abs(maxDx), Math.abs(minDy), Math.abs(maxDy)),
  };
}

/**
 * Render an alignment grid to RGBA (one pixel per tile — the blocky per-tile look;
 * the UI upscales with nearest-neighbor so the tile grid stays honest).
 * `offset` (level-0 px) is subtracted from every tile shift BEFORE the color mapping,
 * so the anchored tile becomes zero (port of the kernel's offset parameter).
 *
 * Color modes:
 *  - 'rgb':  R = +x shift, G = +y shift, zero shift = 50% gray
 *  - 'flow': hue = shift direction, brightness = shift magnitude (optical-flow wheel)
 *
 * scale: level-0 px shift mapped to full saturation. Returns {rgba: Uint8ClampedArray, width, height}.
 */
export function renderAlignmentField(texels, width, height, { mode = 'rgb', scale = 8, offsetX = 0, offsetY = 0 }) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const s = Math.max(scale, 1e-6);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 2;
      const dx = texels[i] - offsetX;
      const dy = texels[i + 1] - offsetY;
      const o = (y * width + x) * 4;
      if (mode === 'rgb') {
        rgba[o] = 255 * clamp01(0.5 + 0.5 * dx / s);
        rgba[o + 1] = 255 * clamp01(0.5 + 0.5 * dy / s);
        rgba[o + 2] = 0;
      } else {
        // HSV flow: hue = shift direction, brightness = shift magnitude
        const hue = (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI);
        const mag = clamp01(Math.hypot(dx, dy) / s);
        const k = Math.floor(hue * 6);
        const f = hue * 6 - k;
        let r, g, b;
        if (k < 0.5) { r = 1; g = f; b = 0; }
        else if (k < 1.5) { r = 1 - f; g = 1; b = 0; }
        else if (k < 2.5) { r = 0; g = 1; b = f; }
        else if (k < 3.5) { r = 0; g = 1 - f; b = 1; }
        else if (k < 4.5) { r = f; g = 0; b = 1; }
        else { r = 1; g = 0; b = 1 - f; }
        rgba[o] = 255 * r * mag;
        rgba[o + 1] = 255 * g * mag;
        rgba[o + 2] = 255 * b * mag;
      }
      rgba[o + 3] = 255;
    }
  }
  return { rgba, width, height };
}

// ------------------------------------------------------------------ ROI scoring

/**
 * Variation of the tile alignment field inside a rectangle (fractions of the frame,
 * origin top-left): each tile's shift is compared to the per-axis median shift of the
 * region, combined to a magnitude, and averaged over all tiles in the region (no
 * outlier exclusion — the worst-matching tiles are the signal). Only tiles fully
 * inside the rectangle are used when there are at least four of them (boundary tiles
 * mix subject and background motion); otherwise overlapping tiles are used. Returns
 * the score in level-0 pixels (lower = more rigid mapping to the reference), or null
 * if the rectangle is smaller than a few tiles.
 *
 * Port of alignment_variation_in_roi() in inspect/inspect.swift.
 */
export function alignmentVariationInRoi(texels, alignmentWidth, alignmentHeight, tileSize, downscaleFactor,
                                        roi, frameWidth, frameHeight, padX, padY) {
  if (texels.length < alignmentWidth * alignmentHeight * 2) return null;
  const halfTile = tileSize / 2;
  const n = normalizeRoi(roi);
  const x0 = (n.x0 * frameWidth + padX) / downscaleFactor;
  const x1 = (n.x1 * frameWidth + padX) / downscaleFactor;
  const y0 = (n.y0 * frameHeight + padY) / downscaleFactor;
  const y1 = (n.y1 * frameHeight + padY) / downscaleFactor;

  const inner = [];
  const overlap = [];
  for (let ty = 0; ty < alignmentHeight; ty++) {
    const cy = (ty + 1) * halfTile;
    for (let tx = 0; tx < alignmentWidth; tx++) {
      const cx = (tx + 1) * halfTile;
      const i = (ty * alignmentWidth + tx) * 2;
      const shift = [texels[i], texels[i + 1]];
      if (cx >= x0 + halfTile && cx <= x1 - halfTile && cy >= y0 + halfTile && cy <= y1 - halfTile) {
        inner.push(shift);
      } else if (cx >= x0 - halfTile && cx <= x1 + halfTile && cy >= y0 - halfTile && cy <= y1 + halfTile) {
        overlap.push(shift);
      }
    }
  }
  const tiles = inner.length >= 4 ? inner : overlap;
  if (tiles.length < 4) return null;

  // robust center: per-axis median shift, then mean per-tile deviation magnitude (the
  // median only makes the measure translation-invariant — no tiles are excluded)
  const medX = medianOfSorted(tiles.map((t) => t[0]).sort((a, b) => a - b));
  const medY = medianOfSorted(tiles.map((t) => t[1]).sort((a, b) => a - b));
  let sum = 0;
  for (const [dx, dy] of tiles) sum += Math.hypot(dx - medX, dy - medY);
  return sum / tiles.length;
}

/** Normalize + validate a ROI rect given as fractions; null when too small to score. */
export function normalizeRoi(roi) {
  const x0 = Math.min(roi.x0, roi.x1), x1 = Math.max(roi.x0, roi.x1);
  const y0 = Math.min(roi.y0, roi.y1), y1 = Math.max(roi.y0, roi.y1);
  return { x0, y0, x1, y1 };
}

export function roiIsDegenerate(roi) {
  const n = normalizeRoi(roi);
  return Math.abs(n.x1 - n.x0) * Math.abs(n.y1 - n.y0) < 0.0005;
}

// ----------------------------------------------------------- brightness matching

/**
 * Least-squares brightness match (comp_matched = gain * comp + offset ≈ ref) between two
 * equally sized arrays by linear regression. Returns (1, 0) for degenerate inputs.
 * Port of brightness_match_values() in viz/visualize.swift.
 */
export function brightnessMatchValues(refValues, compValues) {
  let n = 0, sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (let i = 0; i < refValues.length; i++) {
    const x = compValues[i], y = refValues[i];
    n += 1;
    sumX += x; sumY += y;
    sumXX += x * x; sumXY += x * y;
  }
  const denominator = n * sumXX - sumX * sumX;
  if (!(denominator > 1e-9)) return { gain: 1, offset: 0 };
  const gain = (n * sumXY - sumX * sumY) / denominator;
  const offset = (sumY - gain * sumX) / n;
  // guard against pathological fits (e.g. mostly-flat frames)
  if (!Number.isFinite(gain) || !Number.isFinite(offset) || gain <= 0.2 || gain >= 5.0) {
    return { gain: 1, offset: 0 };
  }
  return { gain, offset };
}

/**
 * Robust brightness match that maximizes the number of pixels with approximately zero
 * difference (inliers, within `tolerance`) instead of minimizing least squares, so that
 * moving content and occlusions do not bias the fit. For every candidate gain, the
 * optimal offset is found exactly with an interval-covering sweep; the gain with the
 * most inliers wins and is then refined. Returns {gain, offset, inlierFraction}. Falls
 * back to a least-squares fit if fewer than 25% of the values can be described by a
 * single (gain, offset) pair.
 * Port of robust_brightness_match_values() in viz/visualize.swift.
 */
export function robustBrightnessMatchValues(refValues, compValues, tolerance) {
  const n = refValues.length;
  if (n < 16) return { gain: 1, offset: 0, inlierFraction: 0 };
  const tol = Math.max(tolerance, 1e-4);

  // For a fixed gain g, inliers are the residuals r_i = y_i - g*x_i with |r_i - o| <= tol,
  // i.e. the offset o must lie in [r_i - tol, r_i + tol]. The region covered by the most
  // intervals is found exactly by a sweep over the 2n interval endpoints; the midpoint of
  // that region is used as the offset.
  const bestOffsetForGain = (gain) => {
    const events = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const residual = refValues[i] - gain * compValues[i];
      events[2 * i] = residual - tol;
      events[2 * i + 1] = residual + tol + 1e-6;
    }
    const order = Array.from({ length: 2 * n }, (_, i) => i).sort((a, b) => events[a] - events[b]);
    let count = 0, bestInliers = 0, bestLeft = 0, bestRight = 0;
    for (const idx of order) {
      const delta = (idx & 1) === 0 ? 1 : -1; // even index = interval start
      if (count === bestInliers && bestInliers > 0) bestRight = events[idx];
      count += delta;
      if (count > bestInliers) {
        bestInliers = count;
        bestLeft = events[idx];
      }
    }
    return { offset: 0.5 * (bestLeft + bestRight), inliers: bestInliers };
  };

  let best = { inliers: -1, gain: 1, offset: 0 };
  for (let gain = 0.6; gain <= 1.7; gain += 0.01) {
    const { offset, inliers } = bestOffsetForGain(gain);
    if (inliers > best.inliers) best = { inliers, gain, offset };
  }
  for (let fine = best.gain - 0.01; fine <= best.gain + 0.01; fine += 0.002) {
    const { offset, inliers } = bestOffsetForGain(fine);
    if (inliers > best.inliers) best = { inliers, gain: fine, offset };
  }

  if (best.inliers * 4 < n) {
    const leastSquares = brightnessMatchValues(refValues, compValues);
    return { gain: leastSquares.gain, offset: leastSquares.offset, inlierFraction: best.inliers / n };
  }
  return { gain: best.gain, offset: best.offset, inlierFraction: best.inliers / n };
}

// --------------------------------------------------------------- small utilities

export function medianOfSorted(values) {
  if (!values.length) return 1.0;
  const mid = values.length >> 1;
  return values.length % 2 === 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

/**
 * Box-average a flat float array (row-major) to roughly `targetLongSide` samples on the
 * long side, keeping the aspect ratio. CPU counterpart of the area_pool_f32 kernel.
 */
export function downsampleValues(values, width, height, targetLongSide = 48) {
  let targetWidth = targetLongSide, targetHeight = targetLongSide;
  if (width >= height) {
    targetHeight = Math.max((height * targetLongSide / width) | 0, 1);
  } else {
    targetWidth = Math.max((width * targetLongSide / height) | 0, 1);
  }
  const output = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    const y0 = (y * height / targetHeight) | 0;
    const y1 = Math.max(((y + 1) * height / targetHeight) | 0, y0 + 1);
    for (let x = 0; x < targetWidth; x++) {
      const x0 = (x * width / targetWidth) | 0;
      const x1 = Math.max(((x + 1) * width / targetWidth) | 0, x0 + 1);
      let sum = 0;
      for (let yy = y0; yy < y1; yy++) {
        const row = yy * width;
        for (let xx = x0; xx < x1; xx++) sum += values[row + xx];
      }
      output[y * targetWidth + x] = sum / ((y1 - y0) * (x1 - x0));
    }
  }
  return { values: output, width: targetWidth, height: targetHeight };
}

/**
 * Tonemap flat float values to grayscale RGBA (CPU counterpart of the gray_rgba8
 * kernel; used for frame-list thumbnails built from downsampled values).
 */
export function grayBitmapFromValues(values, width, height, whitePoint) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < values.length; i++) {
    const u = clamp01(values[i] / Math.max(whitePoint, 1e-9)) * 255;
    const o = 4 * i;
    rgba[o] = u;
    rgba[o + 1] = u;
    rgba[o + 2] = u;
    rgba[o + 3] = 255;
  }
  return { rgba, width, height };
}

/**
 * Interpret a GPU readback of packed RGBA8 u32s (r | g<<8 | b<<16 | a<<24, as written by
 * the diff_rgba8/gray_rgba8 kernels) as an ImageData-ready Uint8ClampedArray. Relies on
 * little-endian u32 layout in GPU/CPU memory, like the existing convert_to_u16 readback.
 */
export function rgbaFromPackedU32(arrayBuffer) {
  return new Uint8ClampedArray(arrayBuffer);
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
