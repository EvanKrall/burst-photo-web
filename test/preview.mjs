// Preview renderer checks: CFA→RGB correctness, white-balance gain, gamma lift.
// Run: node test/preview.mjs

import { renderPreview } from '../src/preview.js';

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

const W = 64, H = 64, BLACK = [64, 64, 64, 64], WHITE = 1023;

// Build a CFA frame from scene signals in *scene RGB* (rSig/gSig/bSig), simulating
// a camera whose color channels have the given sensitivity ratios (sig per channel).
const mkScene = (cfa, rSig, gSig, bSig) => {
  const data = new Uint16Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const cell = (y % 2) * 2 + (x % 2);
      const color = cfa[cell]; // 0=R, 1=G, 2=B
      const sig = color === 0 ? rSig : color === 2 ? bSig : gSig;
      data[y * W + x] = BLACK[cell] + sig;
    }
  }
  return data;
};

const meta = {
  width: W, height: H, blackLevel: BLACK, whiteLevel: WHITE,
  cfa: [0, 1, 1, 2], whiteBalance: [2.1, 1, 1.9, 1],
};

// ---- red scene, RGGB: output must be strongly red-dominant after WB ----
const p1 = renderPreview(mkScene(meta.cfa, 400, 100, 60), meta);
check('preview dimensions', p1.width === W / 2 && p1.height === H / 2, `${p1.width}x${p1.height}`);
const [R1, G1, B1] = [p1.rgba[0], p1.rgba[1], p1.rgba[2]];
check('red scene renders red-dominant', R1 > G1 * 1.8 && R1 > B1 * 1.8, `rgb(${R1},${G1},${B1})`);
check('red scene is bright after WB+gamma', R1 > 180, `R=${R1}`);

// ---- linear mid-gray without gamma would be ~46; with gamma ~119 ----
const gray = renderPreview(mkScene(meta.cfa, 180, 180, 180), { ...meta, whiteBalance: [1, 1, 1, 1] });
const [Rg, Gg, Bg] = [gray.rgba[0], gray.rgba[1], gray.rgba[2]];
check('gamma lifts mid-gray into display range', Gg > 95, `gray=${Gg} (linear would be ≈ 46)`);
check('neutral gray stays neutral under unit WB', Math.abs(Rg - Gg) <= 2 && Math.abs(Bg - Gg) <= 2, `rgb(${Rg},${Gg},${Bg})`);

// ---- neutral scene with uneven camera sensitivity: the reason WB exists ----
// White surface; camera sensitivity implied by WB (2.1, 1, 1.9): R = 64+s/2.1, B = 64+s/1.9.
const whiteSurface = mkScene(meta.cfa, Math.round(64 + 236 / 2.1), 300, Math.round(64 + 236 / 1.9));
const noWb = renderPreview(whiteSurface, { ...meta, whiteBalance: null });
const [Rn, Gn, Bn] = [noWb.rgba[0], noWb.rgba[1], noWb.rgba[2]];
check('neutral scene without WB: G dominates (camera-native, looks green)', Gn > Rn * 1.3 && Gn > Bn * 1.3,
  `rgb(${Rn},${Gn},${Bn})`);
const withWb = renderPreview(whiteSurface, meta);
const [Rw, Gw, Bw] = [withWb.rgba[0], withWb.rgba[1], withWb.rgba[2]];
check('same scene with as-shot WB: near-neutral', Math.abs(Rw - Gw) <= 12 && Math.abs(Bw - Gw) <= 12,
  `rgb(${Rw},${Gw},${Bw})`);

// ---- BGGR CFA: red signal lives in slot 3; renderer must follow the CFA, not assume RGGB ----
const bggr = renderPreview(mkScene([2, 1, 1, 0], 400, 100, 60), { ...meta, cfa: [2, 1, 1, 0] });
const [R4, G4, B4] = [bggr.rgba[0], bggr.rgba[1], bggr.rgba[2]];
check('BGGR CFA mapped correctly (slot 3 is R, slot 0 is B)', R4 > 180 && B4 < R4 / 1.8, `rgb(${R4},${G4},${B4})`);

// ---- alpha opaque ----
check('alpha = 255', p1.rgba[3] === 255);

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
