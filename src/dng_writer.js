// Minimal DNG writer: produces a single-IFD, uncompressed 16-bit Bayer DNG
// (the same layout Android phones emit via the Camera2 API), with CFA /
// black / white level / color matrix tags derived from the reference frame.
//
// This is a PoC subset of Burst Photo's texture_to_dng() (Adobe DNG SDK based):
// it does not copy lens-correction opcodes, profiles, or EXIF blocks beyond
// Make/Model.

// TIFF field types
const BYTE = 1, ASCII = 2, SHORT = 3, LONG = 4, RATIONAL = 5, SRATIONAL = 10;

class IFDEntry {
  constructor(tag, type, values) {
    this.tag = tag;
    this.type = type;
    this.values = values;
  }
  get count() {
    return Array.isArray(this.values) ? this.values.length : 1;
  }
  get byteLength() {
    const n = this.count;
    switch (this.type) {
      case BYTE: case ASCII: return n;
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

/**
 * @param {Uint16Array} pixels Bayer mosaic, width*height
 * @param {object} meta { width, height, blackLevel: number|number[], whiteLevel,
 *                        cfa: number[4] (0=R,1=G,2=B row-major), colorMatrix: number[9]|null,
 *                        make: string|null, model: string|null }
 * @returns {Uint8Array} DNG file bytes
 */
export function writeDNG(pixels, meta) {
  const { width, height } = meta;

  const blackLevels = Array.isArray(meta.blackLevel) ? meta.blackLevel : [meta.blackLevel];
  // DNG allows one BlackLevel value with repeat dim [1,1]; use the max of per-cell values
  const black = Math.max(...blackLevels.map((b) => (Number.isFinite(b) ? b : 0)));
  const white = Number.isFinite(meta.whiteLevel) && meta.whiteLevel > 0 ? meta.whiteLevel : 65535;
  const cfa = meta.cfa && meta.cfa.length === 4 ? meta.cfa : [0, 1, 1, 2]; // RGGB fallback

  const entries = [
    new IFDEntry(254, LONG, [0]),                        // NewSubfileType: full resolution
    new IFDEntry(256, LONG, [width]),                    // ImageWidth
    new IFDEntry(257, LONG, [height]),                   // ImageLength
    new IFDEntry(258, SHORT, [16]),                      // BitsPerSample
    new IFDEntry(259, SHORT, [1]),                       // Compression: none
    new IFDEntry(262, SHORT, [32803]),                   // PhotometricInterpretation: CFA
    new IFDEntry(273, LONG, [0]),                        // StripOffsets (patched)
    new IFDEntry(277, SHORT, [1]),                       // SamplesPerPixel
    new IFDEntry(278, LONG, [height]),                   // RowsPerStrip
    new IFDEntry(279, LONG, [width * height * 2]),       // StripByteCounts
    new IFDEntry(284, SHORT, [1]),                       // PlanarConfiguration: chunky
    new IFDEntry(33421, SHORT, [2, 2]),                  // CFARepeatPatternDim
    new IFDEntry(33422, BYTE, cfa),                      // CFAPattern
    new IFDEntry(50706, BYTE, [1, 4, 0, 0]),             // DNGVersion
    new IFDEntry(50707, BYTE, [1, 1, 0, 0]),             // DNGBackwardVersion
    new IFDEntry(50708, ASCII, strBytes(meta.model ? `${meta.make || ''} ${meta.model}` : 'hdr-plus-web')),
    new IFDEntry(50714, RATIONAL, [rational(black)]),    // BlackLevel
    new IFDEntry(50715, SHORT, [1, 1]),                  // BlackLevelRepeatDim
    new IFDEntry(50717, LONG, [white]),                  // WhiteLevel
  ];

  if (meta.colorMatrix && meta.colorMatrix.length === 9) {
    entries.push(new IFDEntry(50721, SRATIONAL, meta.colorMatrix.map((v) => rational(v, 10000)))); // ColorMatrix1
    entries.push(new IFDEntry(50778, SHORT, [21]));      // CalibrationIlluminant1: D65
  }
  entries.sort((a, b) => a.tag - b.tag);

  // ---- layout ----
  const headerSize = 8;
  const ifdOffset = headerSize;
  const ifdSize = 2 + entries.length * 12 + 4;
  const dataOffset = ifdOffset + ifdSize;

  // place out-of-line values after the IFD
  let dataCursor = dataOffset;
  const valueOffsets = new Map();
  for (const e of entries) {
    if (e.byteLength > 4) {
      valueOffsets.set(e.tag, dataCursor);
      dataCursor += e.byteLength + (e.byteLength % 2); // word-align
    }
  }
  // pixel data starts on a word boundary
  const pixelOffset = dataCursor + (dataCursor % 2);
  const fileLength = pixelOffset + width * height * 2;

  const buf = new ArrayBuffer(fileLength);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // ---- TIFF header (little-endian) ----
  u8[0] = 0x49; u8[1] = 0x49; // 'II'
  dv.setUint16(2, 42, true);
  dv.setUint32(4, ifdOffset, true);

  // ---- IFD ----
  dv.setUint16(ifdOffset, entries.length, true);
  let ifdCursor = ifdOffset + 2;
  for (const e of entries) {
    dv.setUint16(ifdCursor, e.tag, true);
    dv.setUint16(ifdCursor + 2, e.type, true);
    dv.setUint32(ifdCursor + 4, e.count, true);
    const inline = e.byteLength <= 4;
    if (inline) {
      writeValues(dv, u8, ifdCursor + 8, e, ifdCursor + 8);
    } else {
      const off = valueOffsets.get(e.tag);
      dv.setUint32(ifdCursor + 8, off, true);
      writeValues(dv, u8, off, e, off);
    }
    ifdCursor += 12;
  }
  dv.setUint32(ifdCursor, 0, true); // no next IFD

  // strip offset patch
  const stripEntry = entries.find((e) => e.tag === 273);
  dv.setUint32(ifdOffset + 2 + entries.indexOf(stripEntry) * 12 + 8, pixelOffset, true);

  // ---- pixel data: little-endian 16-bit samples ----
  let p = pixelOffset;
  for (let i = 0; i < pixels.length; i++) {
    dv.setUint16(p, pixels[i], true);
    p += 2;
  }
  return new Uint8Array(buf);
}

function strBytes(s) {
  const b = new TextEncoder().encode(s);
  return [...b, 0]; // NUL-terminated
}

function writeValues(dv, u8, offset, entry, baseOffset) {
  const { type, values } = entry;
  const arr = Array.isArray(values) ? values : [values];
  let o = offset;
  for (const v of arr) {
    switch (type) {
      case BYTE: case ASCII: u8[o++] = v; break;
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
