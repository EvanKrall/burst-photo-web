// Fast CPU preview of a merged Bayer mosaic: 2×2-block CFA→RGB (nearest-neighbour,
// "half-size" style demosaic — no interpolation), plus the two corrections the raw
// numbers need to be viewable at all:
//   1. white balance — flashraw's as-shot multipliers [R, G, B, G2] (G = 1) undo the
//      uneven sensitivity of the camera's color channels; without them R and B sit
//      far below G and the image reads as dim, greenish monochrome.
//   2. gamma — RAW samples are linear; displays expect ~2.2-encoded sRGB. Without
//      the curve, everything lands in the dark bottom of the display range.
//
// This is for preview only: the downloaded DNG remains undeveloped RAW
// (black-offset Bayer), which is what the user's RAW converter expects.
//
// DOM-free (no canvas), so it is unit-testable in Node.

export function renderPreview(pixels, meta, targetWidth = 1600) {
  const { width: W, height: H } = meta;
  const scale = Math.max(1, Math.ceil(W / targetWidth));
  const pw = Math.max(1, Math.floor(W / (2 * scale)));
  const ph = Math.max(1, Math.floor(H / (2 * scale)));

  const cfa = meta.cfa && meta.cfa.length === 4 ? meta.cfa : [0, 1, 1, 2];
  const rawBlacks = Array.isArray(meta.blackLevel)
    ? meta.blackLevel
    : [meta.blackLevel, meta.blackLevel, meta.blackLevel, meta.blackLevel];
  // slots are the row-major 2×2 CFA cells, so slot i ↔ blackLevel[i]
  const slotBlack = [0, 1, 2, 3].map((i) => {
    const b = rawBlacks[i] ?? rawBlacks[0];
    return Number.isFinite(b) ? b : 0;
  });
  const white = Math.max(1, meta.whiteLevel || 65535);

  // per-slot gain from as-shot white balance (G channels stay at 1).
  // Samples that are ALREADY white-balanced (sNEF, Sony lossless YCbCr) must not
  // be balanced again — treat as unit gains.
  const wb = !meta.whiteBalanceApplied && Array.isArray(meta.whiteBalance) && meta.whiteBalance.length === 4
    ? meta.whiteBalance : null;
  const slotGain = cfa.map((color) => (wb ? (color === 0 ? wb[0] : color === 2 ? wb[2] : 1) : 1));

  const OFF = [[0, 0], [1, 0], [0, 1], [1, 1]];
  const gSlots = [0, 1, 2, 3].filter((i) => cfa[i] === 1);
  const rSlot = cfa.indexOf(0) >= 0 ? cfa.indexOf(0) : 0;
  const bSlot = cfa.indexOf(2) >= 0 ? cfa.indexOf(2) : 3;
  const GAMMA = 1 / 2.2;

  const tone = (sx, sy, slot) => {
    const v = pixels[(sy + OFF[slot][1] * scale) * W + sx + OFF[slot][0] * scale];
    const t = ((v - slotBlack[slot]) * slotGain[slot]) / (white - slotBlack[slot]);
    return Math.round(Math.pow(Math.max(0, Math.min(1, t)), GAMMA) * 255);
  };

  const rgba = new Uint8ClampedArray(pw * ph * 4);
  for (let py = 0; py < ph; py++) {
    for (let px = 0; px < pw; px++) {
      const sx = px * 2 * scale;
      const sy = py * 2 * scale;
      let g = 0;
      for (const s of gSlots) g += tone(sx, sy, s);
      g /= gSlots.length || 1;
      const o = (py * pw + px) * 4;
      rgba[o] = tone(sx, sy, rSlot);
      rgba[o + 1] = Math.round(g);
      rgba[o + 2] = tone(sx, sy, bSlot);
      rgba[o + 3] = 255;
    }
  }
  return { width: pw, height: ph, rgba };
}
