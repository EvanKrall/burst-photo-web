import { writeDNG } from './dng_writer.js';

// flashraw wrapper: DNG file → raw Bayer frame + interpretation metadata.
//
// API verified against flashraw v0.4.0 source (npm/src/index.d.ts, npm/README.md):
//   import { FlashRaw } from "flashraw";
//   const raw = await FlashRaw.open(arrayBuffer);        // RawInput accepts ArrayBuffer
//   raw.metadata                                         // RawMetadata (camera.{make,model}, orientation, …)
//   const frame = await raw.decode({ backend: "cpu", output: "raw-u16", crop: "active" });
//   frame.data                                           // Uint16Array for raw-u16 (FrameData['raw-u16'])
//   frame.{width, height, cfa, blackLevel, whiteLevel, colorMatrices, origin, incomplete, warnings}
//   frame.zeroCopy / frame.detach() / raw.close()
//
// Key semantics (DecodeOptions.crop docs):
//   - `crop: "active"` → frame is the DNG ActiveArea; `frame.origin` is its position in the
//     sensor buffer; `cfa` is anchored at the frame origin, while `blackLevel` keeps buffer
//     anchoring (pattern anchored at sensor (0,0)) — we pre-rotate the black pattern here so
//     the GPU kernels can assume plane-anchored 2×2 black levels.
//   - "raw-u16" output never falls back to the embedded preview and retains raw black-offset
//     sensor values — exactly what prepare_bayer expects (it subtracts black level on GPU).

const FLASHRAW_SOURCES = [
  'flashraw',                                      // bare specifier: Node.js / bundlers
  // vendored copy for static hosting (works at any subpath):
  //   cp -r node_modules/flashraw/dist  <site>/vendor/flashraw/dist
  new URL('../vendor/flashraw/dist/browser.js', import.meta.url).href,
  '/node_modules/flashraw/dist/browser.js',        // npm install + server.mjs dev flow
  'https://cdn.jsdelivr.net/npm/flashraw/+esm',    // jsDelivr ESM build (best-effort)
  'https://esm.sh/flashraw',                       // esm.sh fallback (best-effort)
];

let modulePromise = null;

async function loadFlashraw() {
  if (!modulePromise) {
    modulePromise = (async () => {
      let lastErr = null;
      for (const url of FLASHRAW_SOURCES) {
        try {
          return await import(/* @vite-ignore */ url);
        } catch (err) {
          lastErr = err;
          console.warn(`[raw.js] could not load flashraw from ${url}: ${err.message}`);
        }
      }
      throw new Error(
        `Failed to load flashraw (${lastErr?.message}). Run \`npm install flashraw\` in burst-photo-web/ — the WASM binaries are import.meta.url assets and are served reliably from a local install.`
      );
    })();
  }
  return modulePromise;
}

export async function decodeRaw(arrayBuffer, name = 'frame') {
  // Works for every format flashraw supports (DNG, NEF, ARW, CR3, RAF, …) —
  // decoding is format-agnostic; flashraw normalizes metadata into the same shape.
  const { FlashRaw } = await loadFlashraw();

  const raw = await FlashRaw.open(arrayBuffer);
  try {
    const meta = raw.metadata;

    if (meta.layout !== 'bayer' && meta.layout !== 'quad-bayer') {
      throw new Error(
        `${name}: sensor layout "${meta.layout}" is out of scope for this PoC (bayer only)` +
        (meta.layout === 'x-trans' ? ' — X-Trans is not ported yet.' : '')
      );
    }

    // backend "cpu" keeps decode on flashraw's SIMD WASM path; our WebGPU pipeline stays
    // the only GPU user. crop:'active' → photographic active area, not the masked sensor.
    const frame = await raw.decode({ backend: 'cpu', output: 'raw-u16', crop: 'active' });

    const width = frame.width;
    const height = frame.height;
    const channels = frame.channels;
    if (channels !== 1) throw new Error(`${name}: expected 1-channel CFA frame, got ${channels}`);

    // zeroCopy frames view WASM memory directly; copy out before closing the file.
    const data = frame.zeroCopy ? frame.detach() : frame.data;
    if (!(data instanceof Uint16Array) || data.length < width * height) {
      throw new Error(`${name}: unexpected pixel buffer (${data?.constructor?.name}, ${data?.length} samples for ${width}x${height})`);
    }
    if (frame.incomplete) {
      console.warn(`[raw.js] ${name}: incomplete decode:`, frame.warnings?.map((w) => w.message).join('; '));
    }

    const blackLevel = extractBlack4(frame.blackLevel, frame.origin, name);
    const whiteLevel = Math.max(...(frame.whiteLevel?.length ? frame.whiteLevel : [65535]));
    const whiteBalance = frame.whiteBalance ?? meta.whiteBalance ?? null;
    const whiteBalanceApplied = Boolean(frame.whiteBalanceApplied ?? meta.whiteBalanceApplied ?? false);
    const cfa = cfaToNumbers(frame.cfa, name);
    const colorMatrix = pickColorMatrix(frame.colorMatrices ?? meta.colorMatrices, name);

    return {
      name,
      width,
      height,
      data,                 // Uint16Array, W*H Bayer mosaic (active area, raw black-offset values)
      blackLevel,           // number[4], plane-anchored (pre-rotated by frame.origin)
      whiteLevel,
      cfa,                  // [c0,c1,c2,c3] row-major, 0=R 1=G 2=B
      colorMatrix,          // number[9] | null  (XYZ→camera, illuminant D65 preferred)
      whiteBalance,         // [R, G, B, G2] as-shot multipliers normalized to G=1, or null
      whiteBalanceApplied,  // true when samples are ALREADY white-balanced (sNEF, Sony
                            // lossless YCbCr): never apply WB gains to these
      orientation: meta.orientation ?? 1,
      make: meta.camera?.make ?? null,
      model: meta.camera?.model ?? null,
      // EXIF-relevant capture parameters of this frame (null when unknown)
      exif: {
        iso: meta.exposure?.iso ?? null,
        exposureTime: meta.exposure?.exposureTime ?? null,   // seconds
        fNumber: meta.exposure?.fNumber ?? null,
        focalLength: meta.exposure?.focalLength ?? null,     // mm
        captureTime: meta.captureTime ?? null,               // ISO-8601-ish string
      },
    };
  } finally {
    raw.close();
  }
}

/**
 * Fully develop raw Bayer data with the whole flashraw chain (demosaic, as-shot white
 * balance, color matrix, sRGB encoding) and return display-ready RGBA. `frame` is a
 * decodeRaw-style record whose fields double as the writeDNG metadata (width, height,
 * data, blackLevel, whiteLevel, cfa, colorMatrix, whiteBalance, …). This is how the
 * inspector renders its photo views (Original/Warped/Reference/Merged) as true RGB.
 */
export async function developBayer(frame, { demosaic = 'bilinear' } = {}) {
  const dng = writeDNG(frame.data, frame);
  return developPreview(dng, { demosaic });
}

// Fully develop a DNG (e.g. our own merged output) with flashraw and return display-ready
// RGBA. This gives the preview the whole development chain — demosaic, white balance,
// color matrix, sRGB encoding, orientation — instead of the crude block render in
// preview.js. `demosaic: 'half-size'` (2×2 block per channel) keeps it fast.
export async function developPreview(dngBytes, { demosaic = 'half-size' } = {}) {
  const { FlashRaw } = await loadFlashraw();
  const raw = await FlashRaw.open(dngBytes);
  try {
    const frame = await raw.decode({
      backend: 'cpu',
      output: 'rgba8',
      demosaic,
      whiteBalance: 'as-shot',
    });
    const data = frame.zeroCopy ? frame.detach() : frame.data;
    return { width: frame.width, height: frame.height, rgba: data };
  } finally {
    raw.close();
  }
}

// ---- extraction helpers (also used by test/roundtrip.mjs) ------------------

// BlackLevel { rows, cols, samples, values[], mean } is anchored at sensor-buffer (0,0);
// our GPU kernels expect a plane-anchored 2×2 array indexed [(y%2)*2 + (x%2)]. Rotate the
// pattern by the frame origin so both agree.
export function extractBlack4(blackLevel, origin, name) {
  const oy = origin?.y ?? 0;
  const ox = origin?.x ?? 0;
  const { rows = 2, cols = 2, samples = 1, values = [], mean } = blackLevel ?? {};
  if (rows === 2 && cols === 2 && samples === 1 && values.length === 4) {
    const at = (y, x) => values[((y % rows) * cols) + (x % cols)];
    return [at(oy, ox), at(oy, ox + 1), at(oy + 1, ox), at(oy + 1, ox + 1)];
  }
  console.warn(`[raw.js] ${name}: black level pattern ${rows}x${cols}x${samples} not supported, using mean (${mean})`);
  const m = Number.isFinite(mean) ? mean : 0;
  return [m, m, m, m];
}

// CfaInfo { pattern: "RGGB", width, height, colors } anchored at the frame origin.
export function cfaToNumbers(cfa, name) {
  const map = { R: 0, G: 1, B: 2 };
  if (!cfa || typeof cfa.pattern !== 'string') {
    console.warn(`[raw.js] ${name}: no CFA info; assuming RGGB`);
    return [0, 1, 1, 2];
  }
  const letters = cfa.pattern.toUpperCase().slice(0, cfa.width * cfa.height || 4);
  if (cfa.width !== 2 || cfa.height !== 2) {
    throw new Error(`${name}: CFA pattern is ${cfa.width}x${cfa.height} ("${cfa.pattern}") — PoC supports 2x2 Bayer only`);
  }
  return [...letters].map((ch) => map[ch] ?? 1);
}

// ColorMatrices.color: ColorMatrix[] { illuminant, temperature, rows, values } (XYZ→camera).
// Prefer DNG ColorMatrix1-equivalent: illuminant 21 (D65), else Standard A (17), else first.
export function pickColorMatrix(matrices, name) {
  const color = matrices?.color;
  if (!Array.isArray(color) || color.length === 0) return null;
  const pick =
    color.find((m) => m.illuminant === 21) ??
    color.find((m) => m.illuminant === 17) ??
    color[0];
  if (pick.rows !== 3 || !Array.isArray(pick.values) || pick.values.length < 9) {
    console.warn(`[raw.js] ${name}: color matrix has ${pick.rows} rows, expected 3 — omitting from output DNG`);
    return null;
  }
  return pick.values.slice(0, 9).map(Number);
}

// Deprecated alias — decoding is not DNG-specific.
export const decodeDNG = decodeRaw;
