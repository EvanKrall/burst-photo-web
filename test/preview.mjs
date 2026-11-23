// Thumbnail + flashraw develop checks: our merged DNG's embedded thumbnail (what
// macOS Finder uses for file icons) is developed through the same flashraw chain
// as the in-app views, and the DNG itself develops to a correct image.
// Run: node test/preview.mjs

import { writeDNG } from '../src/dng_writer.js';
import { FlashRaw } from 'flashraw';

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

// ---- end-to-end: red scene merged Bayer → DNG (with embedded thumbnail) ----
const dngBytes = await writeDNG(mkScene(meta.cfa, 400, 100, 60), meta);

// ---- embedded thumbnail: IFD0 is a reduced-resolution RGB image (DNG spec),
// developed through the flashraw chain — this is what macOS Finder shows ----
const dv = new DataView(dngBytes.buffer, dngBytes.byteOffset, dngBytes.byteLength);
const ifdOff = dv.getUint32(4, true);
const tags = new Map();
const nEntries = dv.getUint16(ifdOff, true);
for (let i = 0; i < nEntries; i++) {
  const e = ifdOff + 2 + i * 12;
  tags.set(dv.getUint16(e, true), { type: dv.getUint16(e + 2, true), count: dv.getUint32(e + 4, true), value: dv.getUint32(e + 8, true) });
}
const tw = tags.get(256).value, th = tags.get(257).value;
check('thumbnail NewSubfileType (254=1)', tags.get(254)?.value === 1);
check('thumbnail photometric RGB (262=2)', tags.get(262)?.value === 2);
check('thumbnail dims ≤ 256 px', tw <= 256 && th <= 256, `${tw}x${th}`);
check('thumbnail byte count = W*H*3', tags.get(279)?.value === tw * th * 3,
  `${tags.get(279)?.value} vs ${tw * th * 3}`);
const thumbOff = tags.get(273).value;
const tm = [0, 0, 0];
for (let i = 0; i < tw * th; i++) {
  tm[0] += dngBytes[thumbOff + i * 3];
  tm[1] += dngBytes[thumbOff + i * 3 + 1];
  tm[2] += dngBytes[thumbOff + i * 3 + 2];
}
const tn = tw * th;
const [tr, tg, tb] = [tm[0] / tn, tm[1] / tn, tm[2] / tn];
check('embedded thumbnail red-dominant (flashraw develop: WB+matrix)', tr > tg * 1.5 && tr > tb * 1.5,
  `mean rgb(${tr.toFixed(0)},${tg.toFixed(0)},${tb.toFixed(0)})`);
check('embedded thumbnail bright (sRGB-encoded)', tr > 170, `R=${tr.toFixed(0)}`);

// ---- end-to-end: flashraw develops OUR merged DNG (the browser preview path) ----
const rawDev = await FlashRaw.open(dngBytes);
try {
  const asShot = rawDev.metadata.dng?.asShotNeutral;
  check('AsShotNeutral tag present in written DNG',
    Array.isArray(asShot) && asShot.length === 3,
    asShot ? JSON.stringify(asShot.map((v) => +v.toFixed(3))) : String(asShot));
  check('AsShotNeutral ≈ [0.48, 1, 0.53]',
    asShot && Math.abs(asShot[0] - 1 / 2.1) < 1e-3 && Math.abs(asShot[1] - 1) < 1e-3 && Math.abs(asShot[2] - 1 / 1.9) < 1e-3,
    asShot ? JSON.stringify(asShot.map((v) => +v.toFixed(3))) : String(asShot));

  const dev = await rawDev.decode({ backend: 'cpu', output: 'rgba8', demosaic: 'half-size', whiteBalance: 'as-shot' });
  check('developed dimensions (half-size)', dev.width === W / 2 && dev.height === H / 2, `${dev.width}x${dev.height}`);
  const [dr, dg, db] = [dev.data[0], dev.data[1], dev.data[2]];
  check('developed red scene is red-dominant (demosaic+WB+matrix)', dr > dg * 1.5 && dr > db * 1.5,
    `rgb(${dr},${dg},${db})`);
  check('developed red scene is bright (sRGB-encoded)', dr > 170, `R=${dr}`);
} finally {
  rawDev.close();
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
