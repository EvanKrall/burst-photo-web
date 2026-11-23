// Non-DNG format support checks: flashraw decodes Canon CR3 natively, and the
// truncated NEF fixture exercises the clear-error path for undecodable data.
// Fixtures come from the flashraw repository (test/fixtures/).
// Run: node test/formats.mjs

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import { decodeRaw } from '../src/raw.js';
import { writeDNG } from '../src/dng_writer.js';

const fixtures = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

// ---- Canon CR3: full decode through the format-agnostic wrapper ----
const cr3 = readFileSync(join(fixtures, 'canon-eos-r5-mark-ii.cr3'));
const frame = await decodeRaw(cr3.buffer.slice(cr3.byteOffset, cr3.byteOffset + cr3.byteLength), 'canon.cr3');
check('CR3 decodes to Bayer mosaic', frame.cfa.join(',') === '0,1,1,2', `cfa=${frame.cfa}`);
check('CR3 active-area dimensions', frame.width === 5118 && frame.height === 3412, `${frame.width}x${frame.height}`);
check('CR3 pixel buffer size', frame.data.length === frame.width * frame.height);
check('CR3 black level (measured from masked areas)',
  frame.blackLevel.length === 4 && frame.blackLevel.every((b) => Math.abs(b - 512) < 1),
  JSON.stringify(frame.blackLevel));
check('CR3 white level', frame.whiteLevel === 16383, String(frame.whiteLevel));
check('CR3 exposure metadata (ISO 125, 1/15 s, f/3.2, 135 mm)',
  frame.exif.iso === 125 && Math.abs(frame.exif.exposureTime - 1 / 15) < 1e-3 && frame.exif.fNumber === 3.2 && frame.exif.focalLength === 135,
  JSON.stringify(frame.exif));
check('CR3 camera make/model', frame.make === 'Canon' && /Canon/.test(frame.model), `${frame.make} ${frame.model}`);
check('CR3 white balance not camera-applied', frame.whiteBalanceApplied === false);

// merged output from CR3 pixels: writer + flashraw develop round-trip on a crop
const crop = { w: 256, h: 192 };
const small = new Uint16Array(crop.w * crop.h);
for (let y = 0; y < crop.h; y++) {
  for (let x = 0; x < crop.w; x++) small[y * crop.w + x] = frame.data[y * frame.width + x];
}
const dngBytes = await writeDNG(small, {
  width: crop.w, height: crop.h,
  blackLevel: frame.blackLevel, whiteLevel: frame.whiteLevel,
  cfa: frame.cfa, colorMatrix: frame.colorMatrix,
  whiteBalance: frame.whiteBalance, whiteBalanceApplied: frame.whiteBalanceApplied,
  orientation: frame.orientation, make: frame.make, model: frame.model, exif: frame.exif,
});
const { FlashRaw } = await import('flashraw');
const dev = await FlashRaw.open(dngBytes);
try {
  const rgb = await dev.decode({ backend: 'cpu', output: 'rgba8', demosaic: 'half-size' });
  const data = rgb.zeroCopy ? rgb.detach() : rgb.data;
  let nonzero = 0;
  for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 30) nonzero++;
  check('CR3-derived merged DNG develops to a real image', rgb.width === crop.w / 2 && nonzero > (crop.w / 2 * crop.h / 2) * 0.5,
    `${rgb.width}x${rgb.height}, ${((nonzero / (rgb.width * rgb.height)) * 100).toFixed(0)}% non-black`);
} finally {
  dev.close();
}

// ---- Nikon HE NEF fixture: metadata opens fine, pixels are intentionally
// truncated in the fixture — must produce a CLEAR error, not garbage ----
const nef = readFileSync(join(fixtures, 'nikon-he-synthetic.nef'));
let nefError = null;
try {
  await decodeRaw(nef.buffer.slice(nef.byteOffset, nef.byteOffset + nef.byteLength), 'nikon.nef');
} catch (err) {
  nefError = err;
}
check('truncated NEF produces a clear error (no silent garbage)',
  nefError !== null && /compress|truncat/i.test(nefError.message),
  nefError ? nefError.message.slice(0, 80) : 'no error thrown');

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
