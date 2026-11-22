// Headless round-trip test: synthetic Bayer frame → writeDNG → flashraw (Node) → verify.
// Validates the minimal DNG writer and the flashraw integration without a browser.
// Run: node test/roundtrip.mjs

import { FlashRaw } from 'flashraw';
import { writeDNG } from '../src/dng_writer.js';
import { extractBlack4, cfaToNumbers, pickColorMatrix } from '../src/raw.js';

const W = 64;
const H = 48;
const BLACK = [200, 204, 200, 206]; // R, G, G, B
const WHITE = 4000;

// synthetic Bayer gradient: value depends on channel and position, + black offset
const pixels = new Uint16Array(W * H);
const cfa = [0, 1, 1, 2]; // RGGB row-major
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const cell = (y % 2) * 2 + (x % 2);
    const base = BLACK[cell];
    const signal = Math.min(WHITE - 1 - base, Math.round((x / W) * 1500 + (y / H) * 800));
    pixels[y * W + x] = base + signal;
  }
}

const dng = writeDNG(pixels, {
  width: W,
  height: H,
  blackLevel: BLACK,
  whiteLevel: WHITE,
  cfa,
  colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622],
  orientation: 1,
  make: 'TestCam',
  model: 'PoC-1',
});

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

// ---- structure sanity: parse the TIFF IFD back ----
const dv = new DataView(dng.buffer, dng.byteOffset, dng.byteLength);
check('TIFF magic II*', dv.getUint16(0, true) === 0x4949 && dv.getUint16(2, true) === 42);
const ifdOff = dv.getUint32(4, true);
const nEntries = dv.getUint16(ifdOff, true);
check('IFD entry count', nEntries >= 20, `${nEntries} entries`);
const tags = new Map();
for (let i = 0; i < nEntries; i++) {
  const e = ifdOff + 2 + i * 12;
  tags.set(dv.getUint16(e, true), { type: dv.getUint16(e + 2, true), count: dv.getUint32(e + 4, true), value: dv.getUint32(e + 8, true) });
}
check('CFA photometric (262=32803)', tags.get(262)?.value === 32803);
check('bits per sample (258=16)', tags.get(258)?.value === 16);
check('orientation (274=1)', tags.get(274)?.value === 1);
check('CFAPattern (33422)', tags.get(33422)?.count === 4);
check('DNGVersion (50706)', tags.get(50706)?.count === 4);

// ---- flashraw round trip ----
const raw = await FlashRaw.open(dng);
const meta = raw.metadata;
check('layout is bayer', meta.layout === 'bayer', meta.layout);
check('dimensions', meta.width === W && meta.height === H, `${meta.width}x${meta.height} (raw ${meta.rawWidth}x${meta.rawHeight})`);
check('camera model', meta.camera?.make === 'TestCam' && meta.camera?.model === 'PoC-1');

const frame = await raw.decode({ backend: 'cpu', output: 'raw-u16', crop: 'active' });
check('frame dimensions', frame.width === W && frame.height === H, `${frame.width}x${frame.height}`);
check('frame channels', frame.channels === 1, `${frame.channels}`);
check('cfa pattern RGGB', frame.cfa?.pattern === 'RGGB', frame.cfa?.pattern);
check('black level values', JSON.stringify(frame.blackLevel?.values) === JSON.stringify(BLACK),
  JSON.stringify(frame.blackLevel?.values));
check('white level', JSON.stringify(frame.whiteLevel) === JSON.stringify([WHITE]), JSON.stringify(frame.whiteLevel));

const data = frame.zeroCopy ? frame.detach() : frame.data;
let maxDiff = 0;
let diffCount = 0;
for (let i = 0; i < W * H; i++) {
  const d = Math.abs(data[i] - pixels[i]);
  if (d > maxDiff) maxDiff = d;
  if (d !== 0) diffCount++;
}
check('pixels round-trip exactly', diffCount === 0, `maxDiff=${maxDiff}, diffs=${diffCount}/${W * H}`);

// ---- helper functions used by raw.js ----
const black4 = extractBlack4(frame.blackLevel, frame.origin, 'test');
check('extractBlack4 (origin 0,0)', JSON.stringify(black4) === JSON.stringify(BLACK), JSON.stringify(black4));
const black4Rot = extractBlack4(frame.blackLevel, { x: 1, y: 1 }, 'test');
check('extractBlack4 rotation (origin 1,1)', JSON.stringify(black4Rot) === JSON.stringify([206, 200, 204, 200]),
  JSON.stringify(black4Rot));
check('cfaToNumbers', JSON.stringify(cfaToNumbers(frame.cfa, 'test')) === JSON.stringify(cfa),
  JSON.stringify(cfaToNumbers(frame.cfa, 'test')));
const cm = pickColorMatrix(frame.colorMatrices, 'test');
check('colorMatrix extracted (9 values)', Array.isArray(cm) && cm.length === 9, cm ? `illuminant-based, [${cm.slice(0, 3).map((v) => v.toFixed(2))}…]` : 'null');

raw.close();

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
