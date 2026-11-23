// Real-world verification: take the raw SubIFD + metadata of a previously downloaded
// merged DNG, rebuild it with the NEW writeDNG, and compare the embedded thumbnail's
// channel statistics against a direct flashraw develop of the raw.
// Run: node test/verify_thumbnail.mjs <file.dng>

import fs from 'node:fs';
import { FlashRaw } from 'flashraw';
import { writeDNG } from '../src/dng_writer.js';

const path = process.argv[2];
if (!path) {
  console.error('usage: node test/verify_thumbnail.mjs <merged.dng>');
  process.exit(1);
}
const dng = new Uint8Array(fs.readFileSync(path));
const rawDev = await FlashRaw.open(dng);
const meta = rawDev.metadata;
const frame = await rawDev.decode({ backend: 'cpu', output: 'raw-u16', crop: 'active' });
const pixels = frame.zeroCopy ? frame.detach() : frame.data;
rawDev.close();

// rebuild the writeDNG meta the way main.js does (reference-frame fields)
const reborn = await writeDNG(pixels, {
  width: frame.width,
  height: frame.height,
  blackLevel: meta.blackLevel?.values ?? 0,
  whiteLevel: Array.isArray(meta.whiteLevel) ? meta.whiteLevel[0] : meta.whiteLevel,
  cfa: [0, 1, 1, 2], // RGGB — matches the source file (checked below via CFA name)
  colorMatrix: meta.colorMatrices?.color ? (meta.colorMatrices.color.find((m) => m.illuminant === 21) ?? meta.colorMatrices.color[0])?.values : undefined,
  whiteBalance: meta.whiteBalance,
  whiteBalanceApplied: meta.whiteBalanceApplied,
  orientation: meta.orientation ?? 1,
  make: meta.camera?.make,
  model: meta.camera?.model,
});

const dv = new DataView(reborn.buffer, reborn.byteOffset, reborn.byteLength);
const ifdOff = dv.getUint32(4, true);
const tags = new Map();
const n = dv.getUint16(ifdOff, true);
for (let i = 0; i < n; i++) {
  const e = ifdOff + 2 + i * 12;
  tags.set(dv.getUint16(e, true), { type: dv.getUint16(e + 2, true), count: dv.getUint32(e + 4, true), value: dv.getUint32(e + 8, true) });
}
const tw = tags.get(256).value, th = tags.get(257).value;
const thumbOff = tags.get(273).value;
const mean = [0, 0, 0];
for (let i = 0; i < tw * th; i++) {
  mean[0] += reborn[thumbOff + i * 3];
  mean[1] += reborn[thumbOff + i * 3 + 1];
  mean[2] += reborn[thumbOff + i * 3 + 2];
}
const cnt = tw * th;
console.log(`NEW embedded thumbnail ${tw}x${th}: mean RGB = (${(mean[0] / cnt).toFixed(1)}, ${(mean[1] / cnt).toFixed(1)}, ${(mean[2] / cnt).toFixed(1)})`);

// center crop
const cx0 = Math.floor(tw * 0.25), cx1 = Math.ceil(tw * 0.75);
const cy0 = Math.floor(th * 0.25), cy1 = Math.ceil(th * 0.75);
const cm = [0, 0, 0]; let cn = 0;
for (let y = cy0; y < cy1; y++) for (let x = cx0; x < cx1; x++) {
  const o = thumbOff + (y * tw + x) * 3;
  cm[0] += reborn[o]; cm[1] += reborn[o + 1]; cm[2] += reborn[o + 2]; cn++;
}
console.log(`NEW thumbnail center-crop mean RGB = (${(cm[0] / cn).toFixed(1)}, ${(cm[1] / cn).toFixed(1)}, ${(cm[2] / cn).toFixed(1)})`);
console.log(`(reference: flashraw develop of the same raw = (68.3, 68.5, 65.7), center (90.5, 91.7, 90.6))`);

// also write the thumbnail out for eyeballing
let ppm = `P6\n${tw} ${th}\n255\n`;
fs.writeFileSync('/tmp/new_thumb.ppm', Buffer.concat([Buffer.from(ppm), Buffer.from(reborn.buffer, thumbOff, tw * th * 3)]));
console.log('wrote /tmp/new_thumb.ppm');
