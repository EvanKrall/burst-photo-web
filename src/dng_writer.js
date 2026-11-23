// Minimal DNG writer: produces an uncompressed 16-bit Bayer DNG (the same raw
// layout Android phones emit via the Camera2 API) with CFA / black / white
// level / color matrix tags derived from the reference frame, plus an EXIF
// sub-IFD carrying burst-adjusted capture parameters.
//
// File structure follows the DNG spec: IFD0 holds a reduced-resolution RGB
// thumbnail (≤256 px, uncompressed TIFF — what Adobe DNG Converter embeds and
// what macOS Finder uses for file icons; without one, Finder falls back to
// decoding the raw itself and shows black icons for our minimal DNGs), and the
// raw image lives in a SubIFD referenced by the SubIFDs tag (330). The
// thumbnail is developed with the SAME flashraw chain as the in-app views and
// Quick Look (src/raw.js developPreview: demosaic, as-shot WB, color matrix,
// sRGB) on a parity-preserving pooled-down copy of the Bayer mosaic — so the
// icon matches what the user sees everywhere else.
//
// This is a PoC subset of Burst Photo's texture_to_dng() (Adobe DNG SDK based):
// it does not copy lens-correction opcodes, profiles, or the full EXIF/maker-note
// blocks. Metadata semantics follow Google's HDR+ convention:
//   - ExposureTime = TOTAL integration time (sum over the burst)
//   - ISOSpeedRatings = reference ISO / number of shots
//
// @param {Uint16Array} pixels Bayer mosaic, width*height
// @param {object} meta {
//   width, height,
//   blackLevel: number|number[], whiteLevel,
//   cfa: number[4] (0=R,1=G,2=B row-major),
//   colorMatrix: number[9]|null, whiteBalance: [r,g,b,g2]|null,
//   orientation: 1..8, make, model,
//   exif: { exposureTime (s, TOTAL), iso (effective), fNumber, focalLength, captureTime }
// }
// @returns {Promise<Uint8Array>} DNG file bytes

import { developPreview } from './raw.js'; // lazy cycle: raw.js needs writeDNG for developBayer

// TIFF field types
const BYTE = 1, ASCII = 2, SHORT = 3, LONG = 4, RATIONAL = 5, UNDEFINED = 7, SRATIONAL = 10;

// EXIF tags (subset)
const TAG_EXIF_IFD_POINTER = 34665;
const TAG_EXPOSURE_TIME = 33434;
const TAG_F_NUMBER = 33437;
const TAG_ISO_SPEED_RATINGS = 34855;
const TAG_EXIF_VERSION = 0x9000;      // 36864
const TAG_DATE_TIME_ORIGINAL = 0x9003; // 36867
const TAG_FOCAL_LENGTH = 0x920A;      // 37386

class IFDEntry {
  constructor(tag, type, values) {
    this.tag = tag;
    this.type = type;
    this.values = values;
    this.outOfLineOffset = null; // assigned during layout for values > 4 bytes
  }
  get count() {
    return Array.isArray(this.values) ? this.values.length : 1;
  }
  get byteLength() {
    const n = this.count;
    switch (this.type) {
      case BYTE: case ASCII: case UNDEFINED: return n;
      case SHORT: return n * 2;
      case LONG: return n * 4;
      case RATIONAL: case SRATIONAL: return n * 8;
      default: throw new Error(`unsupported type ${this.type}`);
    }
  }
}

function rational(num, den = 1) {
  return [num, den];
}

// DNG spec: IFD0 thumbnail ≤ 256 px per side.
const THUMB_MAX_PX = 256;
// Placeholder thumbnail for DNGs that don't need a real one (developBayer's
// temp files, or the rare case where the thumbnail develop itself fails): keeps
// the IFD0 structure identical (always a thumbnail image) at 3 bytes' cost.
const ONE_PIXEL_THUMB = { width: 1, height: 1, bytes: new Uint8Array([0, 0, 0]) };

function strBytes(s) {
  return [...new TextEncoder().encode(s), 0]; // NUL-terminated
}

const align2 = (n) => n + (n % 2);

function tiffDateTime(iso) {
  // "2024-06-01T12:00:00.123+02:00" -> "2024:06:01 12:00:00"
  if (typeof iso !== 'string' || iso.length < 19) return null;
  const s = iso.slice(0, 19).replace(/-/g, ':').replace('T', ' ');
  return /^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$/.test(s) ? s : null;
}

export async function writeDNG(pixels, meta, { thumbnail = 'auto' } = {}) {
  let thumb = ONE_PIXEL_THUMB;
  if (thumbnail === 'auto') {
    try {
      thumb = await developThumbnail(pixels, meta);
    } catch (err) {
      console.warn(`[dng_writer] thumbnail develop failed (${err.message}) — embedding a placeholder`);
    }
  } else if (thumbnail) {
    thumb = thumbnail;
  }
  return buildDng(pixels, meta, thumb);
}

// Pool the Bayer mosaic down by an EVEN factor f: output site (x, y) averages
// the input sites of the SAME color class inside its f×f block. f even is what
// preserves parity — input site (x·f + a, y·f + b) has (a % 2, b % 2) parity,
// so starting each scan at a = x % 2, b = y % 2 lands on matching CFA sites.
function poolBayer(pixels, W, H, outW, outH, f) {
  const out = new Uint16Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    const offY = y % 2;
    for (let x = 0; x < outW; x++) {
      let sum = 0, n = 0;
      for (let b = offY; b < f; b += 2) {
        const sy = y * f + b;
        if (sy >= H) break;
        for (let a = x % 2; a < f; a += 2) {
          const sx = x * f + a;
          if (sx >= W) break;
          sum += pixels[sy * W + sx];
          n++;
        }
      }
      out[y * outW + x] = n ? Math.round(sum / n) : 0;
    }
  }
  return out;
}

// Develop the ≤256 px thumbnail through the flashraw chain on a pooled-down
// copy of the mosaic (developing the full ~24 MP frame would be 100× the work
// for the same 252×168 result). Uses buildDng on the small mosaic — NOT
// writeDNG — to avoid infinite recursion.
async function developThumbnail(pixels, meta) {
  const f = Math.max(2, Math.ceil(meta.width / (2 * THUMB_MAX_PX)));
  const factor = f + (f % 2); // round up to even
  const sw = Math.max(2, Math.floor(meta.width / factor));
  const sh = Math.max(2, Math.floor(meta.height / factor));
  const small = poolBayer(pixels, meta.width, meta.height, sw, sh, factor);
  const temp = buildDng(small, { ...meta, width: sw, height: sh }, ONE_PIXEL_THUMB);
  const dev = await developPreview(temp, { demosaic: 'half-size' });
  const bytes = new Uint8Array(dev.width * dev.height * 3);
  for (let i = 0, j = 0; i < dev.rgba.length; i += 4) {
    bytes[j++] = dev.rgba[i];
    bytes[j++] = dev.rgba[i + 1];
    bytes[j++] = dev.rgba[i + 2];
  }
  return { width: dev.width, height: dev.height, bytes };
}

function buildDng(pixels, meta, thumb) {
  const thumbW = thumb.width;
  const thumbH = thumb.height;
  const thumbBytes = thumb.bytes;
  const { width, height } = meta;
  const orientation =
    Number.isFinite(meta.orientation) && meta.orientation >= 1 && meta.orientation <= 8
      ? meta.orientation
      : 1;

  const blackLevels = Array.isArray(meta.blackLevel) ? meta.blackLevel : [meta.blackLevel];
  const perCell = blackLevels.length === 4 && blackLevels.every((b) => Number.isFinite(b));
  const white = Number.isFinite(meta.whiteLevel) && meta.whiteLevel > 0 ? meta.whiteLevel : 65535;
  const cfa = meta.cfa && meta.cfa.length === 4 ? meta.cfa : [0, 1, 1, 2]; // RGGB fallback
  const cameraName = meta.model ? `${meta.make || ''} ${meta.model}`.trim() : 'burst-photo-web';
  const dt = tiffDateTime(meta.exif?.captureTime);

  // ---- IFD0: the thumbnail IFD (DNG spec: reduced-resolution image) + camera-level tags ----
  const entries = [
    new IFDEntry(254, LONG, [1]),                        // NewSubfileType: reduced resolution
    new IFDEntry(256, LONG, [thumbW]),                   // ImageWidth
    new IFDEntry(257, LONG, [thumbH]),                   // ImageLength
    new IFDEntry(258, SHORT, [8, 8, 8]),                 // BitsPerSample
    new IFDEntry(259, SHORT, [1]),                       // Compression: none (TIFF RGB)
    new IFDEntry(262, SHORT, [2]),                       // PhotometricInterpretation: RGB
    new IFDEntry(274, SHORT, [1]),                       // Orientation: 1 — developPreview applies
                                                         // the orientation, thumbnail is display-oriented
    new IFDEntry(273, LONG, [0]),                        // StripOffsets (patched)
    new IFDEntry(277, SHORT, [3]),                       // SamplesPerPixel
    new IFDEntry(278, LONG, [thumbH]),                   // RowsPerStrip
    new IFDEntry(279, LONG, [thumbW * thumbH * 3]),      // StripByteCounts
    new IFDEntry(282, RATIONAL, [rational(72)]),         // XResolution
    new IFDEntry(283, RATIONAL, [rational(72)]),         // YResolution
    new IFDEntry(296, SHORT, [2]),                       // ResolutionUnit: inch
    new IFDEntry(305, ASCII, strBytes('burst-photo-web (Burst Photo PoC)')), // Software
    new IFDEntry(330, LONG, [0]),                        // SubIFDs (patched → raw IFD)
    new IFDEntry(50706, BYTE, [1, 4, 0, 0]),             // DNGVersion
    new IFDEntry(50707, BYTE, [1, 1, 0, 0]),             // DNGBackwardVersion
    new IFDEntry(50708, ASCII, strBytes(cameraName)),    // UniqueCameraModel
  ];

  // ---- raw SubIFD: the full-resolution Bayer image ----
  const rawEntries = [
    new IFDEntry(254, LONG, [0]),                        // NewSubfileType: full resolution
    new IFDEntry(256, LONG, [width]),                    // ImageWidth
    new IFDEntry(257, LONG, [height]),                   // ImageLength
    new IFDEntry(258, SHORT, [16]),                      // BitsPerSample
    new IFDEntry(259, SHORT, [1]),                       // Compression: none
    new IFDEntry(262, SHORT, [32803]),                   // PhotometricInterpretation: CFA
    new IFDEntry(274, SHORT, [orientation]),             // Orientation
    new IFDEntry(273, LONG, [0]),                        // StripOffsets (patched)
    new IFDEntry(277, SHORT, [1]),                       // SamplesPerPixel
    new IFDEntry(278, LONG, [height]),                   // RowsPerStrip
    new IFDEntry(279, LONG, [width * height * 2]),       // StripByteCounts
    new IFDEntry(284, SHORT, [1]),                       // PlanarConfiguration: chunky
    new IFDEntry(33421, SHORT, [2, 2]),                  // CFARepeatPatternDim
    new IFDEntry(33422, BYTE, cfa),                      // CFAPattern
    blackTag(perCell, blackLevels),
    new IFDEntry(50713, SHORT, perCell ? [2, 2] : [1, 1]),  // BlackLevelRepeatDim (0xC619)
    new IFDEntry(50717, LONG, [white]),                  // WhiteLevel
  ];

  if (meta.make) entries.push(new IFDEntry(271, ASCII, strBytes(meta.make)));   // Make
  if (meta.model) entries.push(new IFDEntry(272, ASCII, strBytes(meta.model))); // Model
  if (dt) entries.push(new IFDEntry(306, ASCII, strBytes(dt)));                 // DateTime
  if (meta.colorMatrix && meta.colorMatrix.length === 9) {
    entries.push(new IFDEntry(50721, SRATIONAL, meta.colorMatrix.map((v) => rational(Math.round(v * 10000), 10000)))); // ColorMatrix1
    entries.push(new IFDEntry(50778, SHORT, [21]));      // CalibrationIlluminant1: D65
  }
  if (Array.isArray(meta.whiteBalance) && meta.whiteBalance.length === 4 && !meta.whiteBalanceApplied) {
    // AsShotNeutral: invert as-shot multipliers, normalize so max = 1.
    // Skipped when the camera already applied WB to the samples (sNEF, Sony
    // lossless YCbCr): we then write an explicit [1, 1, 1] below so developers
    // do not balance the samples a second time.
    const [wr, , wb] = meta.whiteBalance;
    const neutral = [1 / wr, 1, 1 / wb];
    const max = Math.max(...neutral);
    entries.push(new IFDEntry(50728, RATIONAL, neutral.map((v) => rational(Math.round((v / max) * 100000), 100000))));
  } else if (meta.whiteBalanceApplied) {
    entries.push(new IFDEntry(50728, RATIONAL, [rational(100000, 100000), rational(100000, 100000), rational(100000, 100000)]));
  }

  // ---- EXIF sub-IFD ----
  const exif = meta.exif ?? {};
  const exifEntries = [new IFDEntry(TAG_EXIF_VERSION, UNDEFINED, [0x30, 0x32, 0x33, 0x00])]; // "0230"
  if (Number.isFinite(exif.exposureTime) && exif.exposureTime > 0) {
    // TOTAL integration time across the burst (Google HDR+ convention)
    exifEntries.push(new IFDEntry(TAG_EXPOSURE_TIME, RATIONAL, [rational(Math.round(exif.exposureTime * 1e6), 1e6)]));
  }
  if (Number.isFinite(exif.fNumber) && exif.fNumber > 0) {
    exifEntries.push(new IFDEntry(TAG_F_NUMBER, RATIONAL, [rational(Math.round(exif.fNumber * 100), 100)]));
  }
  if (Number.isFinite(exif.iso) && exif.iso >= 1) {
    exifEntries.push(new IFDEntry(TAG_ISO_SPEED_RATINGS, SHORT, [Math.round(exif.iso)]));
  }
  if (dt) exifEntries.push(new IFDEntry(TAG_DATE_TIME_ORIGINAL, ASCII, strBytes(dt)));
  if (Number.isFinite(exif.focalLength) && exif.focalLength > 0) {
    exifEntries.push(new IFDEntry(TAG_FOCAL_LENGTH, RATIONAL, [rational(Math.round(exif.focalLength * 100), 100)]));
  }
  exifEntries.sort((a, b) => a.tag - b.tag);
  const hasExif = exifEntries.length > 1; // version alone isn't worth an IFD
  if (hasExif) entries.push(new IFDEntry(TAG_EXIF_IFD_POINTER, LONG, [0])); // patched
  entries.sort((a, b) => a.tag - b.tag);
  rawEntries.sort((a, b) => a.tag - b.tag);

  // ---- layout: header | IFD0 | IFD0 values | EXIF IFD | EXIF values |
  //              raw IFD | raw IFD values | thumbnail pixels | raw pixels ----
  const ifd0Off = 8;
  const ifd0Size = 2 + entries.length * 12 + 4;
  const ifd0ValOff = ifd0Off + ifd0Size;
  // region chain flows through every region unconditionally (exifSize / exif
  // values size are simply 0 when there is no EXIF IFD)
  const exifIfdOff = align2(ifd0ValOff + valuesSize(entries));
  const exifSize = hasExif ? 2 + exifEntries.length * 12 + 4 : 0;
  const exifValOff = align2(exifIfdOff + exifSize);
  const rawIfdOff = align2(exifValOff + valuesSize(exifEntries));
  const rawIfdSize = 2 + rawEntries.length * 12 + 4;
  const rawIfdValOff = rawIfdOff + rawIfdSize;
  const thumbOffset = align2(rawIfdValOff + valuesSize(rawEntries));
  const pixelOffset = align2(thumbOffset + thumbBytes.length);
  const fileLength = pixelOffset + width * height * 2;

  const buf = new ArrayBuffer(fileLength);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);

  dv.setUint16(0, 0x4949, true); // 'II'
  dv.setUint16(2, 42, true);
  dv.setUint32(4, ifd0Off, true);

  // assign out-of-line value offsets
  let c = ifd0ValOff;
  for (const e of entries) if (e.byteLength > 4) { e.outOfLineOffset = c; c += align2(e.byteLength); }
  c = exifValOff;
  for (const e of exifEntries) if (e.byteLength > 4) { e.outOfLineOffset = c; c += align2(e.byteLength); }
  c = rawIfdValOff;
  for (const e of rawEntries) if (e.byteLength > 4) { e.outOfLineOffset = c; c += align2(e.byteLength); }

  const writeIfd = (off, list) => {
    dv.setUint16(off, list.length, true);
    let cur = off + 2;
    for (const e of list) {
      dv.setUint16(cur, e.tag, true);
      dv.setUint16(cur + 2, e.type, true);
      dv.setUint32(cur + 4, e.count, true);
      if (e.byteLength <= 4) writeValues(dv, u8, cur + 8, e);
      else {
        dv.setUint32(cur + 8, e.outOfLineOffset, true);
        writeValues(dv, u8, e.outOfLineOffset, e);
      }
      cur += 12;
    }
    dv.setUint32(cur, 0, true); // no next IFD
  };

  writeIfd(ifd0Off, entries);
  if (hasExif) writeIfd(exifIfdOff, exifEntries);
  writeIfd(rawIfdOff, rawEntries);

  // patches (IFD0: thumbnail strip, raw SubIFD pointer, EXIF pointer; raw IFD: strip)
  const patchLong = (list, ifdOff, tag, value) => {
    const e = list.find((x) => x.tag === tag);
    dv.setUint32(ifdOff + 2 + list.indexOf(e) * 12 + 8, value, true);
  };
  patchLong(entries, ifd0Off, 273, thumbOffset);      // thumbnail StripOffsets
  patchLong(entries, ifd0Off, 330, rawIfdOff);        // SubIFDs → raw IFD
  if (hasExif) patchLong(entries, ifd0Off, TAG_EXIF_IFD_POINTER, exifIfdOff);
  patchLong(rawEntries, rawIfdOff, 273, pixelOffset); // raw StripOffsets

  // ---- thumbnail pixels: RGB8, display-oriented (flashraw applied the orientation) ----
  u8.set(thumbBytes, thumbOffset);

  // ---- pixel data: little-endian 16-bit samples ----
  let p = pixelOffset;
  for (let i = 0; i < pixels.length; i++) {
    dv.setUint16(p, pixels[i], true);
    p += 2;
  }
  return new Uint8Array(buf);
}

function blackTag(perCell, blackLevels) {
  if (perCell) {
    return new IFDEntry(50714, RATIONAL, blackLevels.map((b) => rational(b))); // BlackLevel
  }
  const b = Math.max(...blackLevels.map((v) => (Number.isFinite(v) ? v : 0)));
  return new IFDEntry(50714, RATIONAL, [rational(b)]);
}

function valuesSize(list) {
  return list.reduce((sum, e) => (e.byteLength > 4 ? sum + align2(e.byteLength) : sum), 0);
}

function writeValues(dv, u8, offset, entry) {
  const { type, values } = entry;
  const arr = Array.isArray(values) ? values : [values];
  let o = offset;
  for (const v of arr) {
    switch (type) {
      case BYTE: case ASCII: case UNDEFINED: u8[o++] = v; break;
      case SHORT: dv.setUint16(o, v, true); o += 2; break;
      case LONG: dv.setUint32(o, v, true); o += 4; break;
      case RATIONAL: case SRATIONAL: {
        const [num, den] = Array.isArray(v) ? v : [v, 1];
        if (type === RATIONAL) { dv.setUint32(o, num >>> 0, true); dv.setUint32(o + 4, den >>> 0, true); }
        else { dv.setInt32(o, Math.round(num), true); dv.setInt32(o + 4, Math.round(den), true); }
        o += 8;
        break;
      }
      default: throw new Error(`unsupported type ${type}`);
    }
  }
}
