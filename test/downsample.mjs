// downsampleRGBA/downsampleU16 partition checks: the cells must cover the WHOLE
// source. Integer cell sizes once dropped the right/bottom remainder (a 6064-wide
// develop pooled to a 1552-wide target covered only 77% — the photo views showed
// the top-left ~2/3 of the image, scaled up).
import { downsampleRGBA, downsampleU16 } from '../src/viz.js';

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

// wide source, fractional ratio (like a full-res develop → pane-sized target)
const W = 155, H = 103; // ratio 155/59 ≈ 2.63 — floor() would cover 82%
const rgba = new Uint8ClampedArray(W * H * 4);
for (let i = 0; i < W * H; i++) {
  rgba[i * 4] = 255; // all pixels lit in red
}

// one-hot at the very last pixel (bottom-right) — must land in the last output cell
const last = new Uint8ClampedArray(W * H * 4);
last[(W * H - 1) * 4] = 255;
const outLast = downsampleRGBA(last, W, H, 59, 41);
check('downsampleRGBA: last source pixel reaches the last output cell',
  outLast.rgba[(outLast.rgba.length - 4)] > 0,
  `last-cell red=${outLast.rgba[outLast.rgba.length - 4]}`);

// one-hot at the very first pixel — must land in the first output cell
const first = new Uint8ClampedArray(W * H * 4);
first[0] = 255; // pixel (0,0), red channel
const outFirst = downsampleRGBA(first, W, H, 59, 41);
check('downsampleRGBA: first source pixel reaches the first output cell',
  outFirst.rgba[0] > 0, `first-cell red=${outFirst.rgba[0]}`);

// mean is exact: every-lit source pools to fully-lit output (each cell's mean of 255s)
const outAll = downsampleRGBA(rgba, W, H, 59, 41);
let allLit = true;
for (let i = 0; i < outAll.rgba.length; i += 4) {
  if (outAll.rgba[i] !== 255) { allLit = false; break; }
}
check('downsampleRGBA: constant source pools to the same constant', allLit);

// alpha stays opaque, dims echo the request
check('downsampleRGBA: output dims + opaque alpha',
  outAll.width === 59 && outAll.height === 41 && outAll.rgba[3] === 255);

// u16 twin: same one-hot checks on the merged-view gray path
const u16Last = new Uint16Array(W * H);
u16Last[W * H - 1] = 1000;
const outU16 = downsampleU16(u16Last, W, H, 59, 41);
check('downsampleU16: last source pixel reaches the last output cell',
  outU16[outU16.length - 1] > 0, `last-cell=${outU16[outU16.length - 1].toFixed(1)}`);

// upsampling degenerates to nearest (out > in: 4x4 → 8x8 keeps pixel corners)
const small = new Uint8ClampedArray(4 * 4 * 4);
small[0] = 10; small[1] = 20; small[2] = 30; // pixel (0,0) teal-ish
const up = downsampleRGBA(small, 4, 4, 8, 8);
check('downsampleRGBA: upsampling is nearest-neighbor',
  up.rgba[0] === 10 && up.rgba[1] === 20 && up.rgba[2] === 30,
  `corner=[${up.rgba[0]},${up.rgba[1]},${up.rgba[2]}]`);

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
