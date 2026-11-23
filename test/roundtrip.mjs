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

const dng = await writeDNG(pixels, {
  width: W,
  height: H,
  blackLevel: BLACK,
  whiteLevel: WHITE,
  cfa,
  colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622],
  orientation: 1,
  make: 'TestCam',
  model: 'PoC-1',
  whiteBalance: [2.1, 1, 1.9, 1],
  // burst-adjusted capture parameters (what main.js computes: Σ exposure, ISO/n)
  exif: {
    exposureTime: 0.032, // e.g. 8 × 1/250 s
    iso: 25,             // e.g. 200 / 8
    fNumber: 4,
    focalLength: 50,
    captureTime: '2024-06-01T12:00:00Z',
  },
});

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

// ---- structure sanity: parse the TIFF IFDs back ----
const dv = new DataView(dng.buffer, dng.byteOffset, dng.byteLength);
const parseIfd = (off) => {
  const tags = new Map();
  const n = dv.getUint16(off, true);
  for (let i = 0; i < n; i++) {
    const e = off + 2 + i * 12;
    tags.set(dv.getUint16(e, true), { type: dv.getUint16(e + 2, true), count: dv.getUint32(e + 4, true), value: dv.getUint32(e + 8, true) });
  }
  return tags;
};
check('TIFF magic II*', dv.getUint16(0, true) === 0x4949 && dv.getUint16(2, true) === 42);
const ifdOff = dv.getUint32(4, true);
const tags = parseIfd(ifdOff);
check('IFD0 entry count', tags.size >= 15, `${tags.size} entries`);

// IFD0 is the thumbnail (DNG spec): reduced-resolution uncompressed RGB
check('thumbnail NewSubfileType (254=1)', tags.get(254)?.value === 1);
check('thumbnail photometric RGB (262=2)', tags.get(262)?.value === 2);
check('thumbnail samples (277=3)', tags.get(277)?.value === 3);
const thumbW = tags.get(256).value, thumbH = tags.get(257).value;
check('thumbnail ≤ 256 px', thumbW <= 256 && thumbH <= 256, `${thumbW}x${thumbH}`);
check('thumbnail byte count = W*H*3', tags.get(279)?.value === thumbW * thumbH * 3,
  `${tags.get(279)?.value} vs ${thumbW * thumbH * 3}`);
const thumbOff = tags.get(273).value;
let thumbNonBlack = 0;
for (let i = 0; i < thumbW * thumbH; i++) {
  if (dng[thumbOff + i * 3] > 8 || dng[thumbOff + i * 3 + 1] > 8 || dng[thumbOff + i * 3 + 2] > 8) thumbNonBlack++;
}
check('thumbnail shows the gradient (non-black pixels)', thumbNonBlack > thumbW * thumbH * 0.3,
  `${thumbNonBlack}/${thumbW * thumbH} non-black`);

check('SubIFDs pointer present (330)', tags.has(330));
const rawOff = tags.get(330).value;
const rawTags = parseIfd(rawOff);
check('raw SubIFD: CFA photometric (262=32803)', rawTags.get(262)?.value === 32803);
check('raw SubIFD: bits per sample (258=16)', rawTags.get(258)?.value === 16);
check('raw SubIFD: full resolution (254=0)', rawTags.get(254)?.value === 0);
check('raw SubIFD: CFAPattern (33422)', rawTags.get(33422)?.count === 4);
check('raw dimensions match', rawTags.get(256)?.value === W && rawTags.get(257)?.value === H,
  `${rawTags.get(256)?.value}x${rawTags.get(257)?.value}`);
check('orientation (274=1) in both IFDs', tags.get(274)?.value === 1 && rawTags.get(274)?.value === 1);
check('DNGVersion (50706)', tags.get(50706)?.count === 4);
check('Make/Model tags present', tags.has(271) && tags.has(272));
check('Software tag (305)', tags.has(305));
check('DateTime tag (306)', tags.has(306));

// ---- EXIF sub-IFD: parse it back and verify burst-adjusted values ----
check('ExifIFDPointer present (34665)', tags.has(34665));
const exifOff = tags.get(34665).value;
const exifTags = parseIfd(exifOff);
check('EXIF IFD entry count', exifTags.size >= 5, `${exifTags.size} entries`);
const readRational = (tag) => {
  const t = exifTags.get(tag);
  if (!t || t.type !== 5) return null;
  return dv.getUint32(t.value, true) / dv.getUint32(t.value + 4, true);
};
const exp = readRational(33434);
check('ExposureTime = total integration (0.032 s)', exp !== null && Math.abs(exp - 0.032) < 1e-9, `${exp} s`);
const fn = readRational(33437);
check('FNumber = 4.0', fn !== null && Math.abs(fn - 4) < 1e-9, `f/${fn}`);
const iso = exifTags.get(34855);
const isoVal = iso ? iso.value & 0xffff : null; // inline SHORT lives in the low half of the value field
check('ISOSpeedRatings = 25 (200/8)', isoVal === 25, String(isoVal));
const focal = readRational(37386);
check('FocalLength = 50 mm', focal !== null && Math.abs(focal - 50) < 1e-9, `${focal} mm`);
const dto = exifTags.get(36867);
const dtoStr = dto ? String.fromCharCode(...dng.subarray(dto.value, dto.value + dto.count)).replace(/\0$/, '') : null;
check('DateTimeOriginal = 2024:06:01 12:00:00', dtoStr === '2024:06:01 12:00:00', dtoStr);

// ---- flashraw round trip ----
const raw = await FlashRaw.open(dng);
const meta = raw.metadata;
check('layout is bayer', meta.layout === 'bayer', meta.layout);
check('dimensions', meta.width === W && meta.height === H, `${meta.width}x${meta.height} (raw ${meta.rawWidth}x${meta.rawHeight})`);
check('camera model', meta.camera?.make === 'TestCam' && meta.camera?.model === 'PoC-1');

// flashraw should surface the burst-adjusted exposure metadata it reads from our EXIF IFD
const ex = meta.exposure;
check('flashraw reads ExposureTime (total)', ex && Math.abs(ex.exposureTime - 0.032) < 1e-6,
  ex ? `${ex.exposureTime} s` : 'missing');
check('flashraw reads ISO (effective)', ex && ex.iso === 25, ex ? String(ex.iso) : 'missing');
check('flashraw reads FNumber', ex && Math.abs(ex.fNumber - 4) < 1e-6, ex ? `f/${ex.fNumber}` : 'missing');

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
