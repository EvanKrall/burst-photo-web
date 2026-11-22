// AF-area detection for the alignment inspector — reads the autofocus area from the
// reference frame's RAW maker notes and returns it as frame fractions, so the default
// subject area matches what the camera actually focused on (the Swift visualize branch
// read the same metadata from Nikon maker notes; flashraw does not expose it).
//
// Powered by @uswriting/exiftool: a full exiftool (13.42) running under zeroperl, a
// WASM Perl interpreter. The stack is big (~25 MB wasm + ~140 KB script) and slow to
// start (~0.5-3 s after the interpreter is compiled), so it is loaded lazily on the
// first inspection and run on the reference frame only. Requires the vendored copy in
// vendor/exiftool/ (scripts/vendor.mjs, run on postinstall/build).
//
// Verified against Nikon Z-series NEFs (AFInfo2 v0401: AFAreaXPosition/YPosition/
// Width/Height + AFImageWidth/Height). Other makes either lack these tags under the
// same names or need extra work — the caller falls back to the default subject area.

let modPromise = null;
function loadExiftool() {
  if (!modPromise) modPromise = import('../vendor/exiftool/exiftool.js');
  return modPromise;
}

// zeroperl resolves its wasm relative to the *document* URL, not the module — override
// so it always loads the vendored copy regardless of where the page lives
const WASM_URL = new URL('../vendor/exiftool/zeroperl.wasm', import.meta.url);

const num = (tags, name) => {
  const v = tags[`Nikon:${name}`] ?? tags[name];
  return typeof v === 'number' ? v : null;
};

/**
 * Detect the AF area of one RAW file.
 * @param {string} name file name (extension drives exiftool's format detection)
 * @param {Blob|File} file the RAW file
 * @returns {Promise<{x0:number,y0:number,x1:number,y1:number}|null>} AF area as
 *   fractions of the frame (top-left origin), or null when unavailable/undecodable.
 */
export async function detectAfAreaRoi(name, file) {
  const { parseMetadata } = await loadExiftool();
  const result = await parseMetadata(
    { name, data: new Uint8Array(await file.arrayBuffer()) },
    {
      args: ['-json', '-n', '-G1',
             '-Nikon:AFImageWidth', '-Nikon:AFImageHeight',
             '-Nikon:AFAreaXPosition', '-Nikon:AFAreaYPosition',
             '-Nikon:AFAreaWidth', '-Nikon:AFAreaHeight'],
      transform: (data) => JSON.parse(data),
      fetch: () => fetch(WASM_URL),
    },
  );
  if (!result.success || !Array.isArray(result.data) || !result.data.length) return null;
  const tags = result.data[0];
  const x = num(tags, 'AFAreaXPosition');
  const y = num(tags, 'AFAreaYPosition');
  const w = num(tags, 'AFAreaWidth');
  const h = num(tags, 'AFAreaHeight');
  if (x == null || y == null || w == null || h == null) return null;
  // AF coordinates are relative to the AF image (the active sensor area, which is what
  // flashraw decodes); fall back to the frame size only when exiftool omits the dims
  const fw = num(tags, 'AFImageWidth');
  const fh = num(tags, 'AFImageHeight');
  if (!fw || !fh) return null;
  const x0 = x / fw, y0 = y / fh, x1 = (x + w) / fw, y1 = (y + h) / fh;
  const ok = [x0, y0, x1, y1].every((v) => Number.isFinite(v))
    && x1 > x0 && y1 > y0 && x0 >= 0 && y0 >= 0 && x1 <= 1.001 && y1 <= 1.001;
  return ok ? { x0, y0, x1, y1 } : null;
}
