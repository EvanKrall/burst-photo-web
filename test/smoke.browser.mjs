// Browser smoke test: compiles the WGSL module and runs a numerical end-to-end
// pipeline check in a WebGPU-capable Chrome (headed — headless hides navigator.gpu).
//
// Usage: node test/smoke.browser.mjs        (starts its own server on :8123)
// Requires: Google Chrome at the default macOS path (override via CHROME_BIN).

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const root = fileURLToPath(new URL('..', import.meta.url));
const port = 8123;
const chromePath = process.env.CHROME_BIN ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
};

// static server (same semantics as server.mjs)
const server = createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p === '/') p = '/index.html';
    const file = normalize(join(root, p));
    if (!file.startsWith(root)) throw new Error('traversal');
    const data = await readFile(file);
    if (!res.headersSent) {
      res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
      res.end(data);
    }
  } catch {
    if (!res.headersSent) res.writeHead(404);
    res.end();
  }
}).listen(port);

await new Promise((r) => server.on('listening', r));

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: false,
  args: ['--window-size=900,700', '--enable-unsafe-webgpu'],
});

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('  [page error]', m.text().slice(0, 300)); });
  page.on('pageerror', (e) => console.log('  [pageexception]', String(e).slice(0, 300)));
  await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });

  const out = await page.evaluate(async () => {
    const { initGPU, GPUContext, packUniforms } = await import('/src/gpu.js');
    const { WGSL } = await import('/src/kernels.js');
    const { alignMergeSpatial } = await import('/src/pipeline.js');
    const { performInspection } = await import('/src/inspect.js');
    const { alignmentVariationInRoi } = await import('/src/viz.js');
    const { writeDNG } = await import('/src/dng_writer.js');

    const { adapter, device } = await initGPU();
    const gpu = new GPUContext(device);
    gpu.setShaderModule(WGSL);
    const info = await gpu.module.getCompilationInfo();
    const errors = info.messages
      .filter((m) => m.type === 'error')
      .map((e) => `L${e.lineNum}:${e.linePos} ${e.message}`);
    if (errors.length) return { stage: 'wgsl-compile', errors };

    // ---- synthetic RGGB burst, 64x64, black level 64, white 1023 ----
    const W = 64, H = 64, BLACK = [64, 64, 64, 64], WHITE = 1023;
    const mkFrame = (dx = 0, dy = 0) => {
      const data = new Uint16Array(W * H);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const cell = (y % 2) * 2 + (x % 2);
          // smooth gradient + slow 2D variation; samples are far from clipping
          const xs = Math.min(Math.max(x - dx, 0), W - 1);
          const ys = Math.min(Math.max(y - dy, 0), H - 1);
          const signal = 20 + 3.0 * xs + 2.0 * ys + 10 * Math.sin(xs * 0.35) * Math.cos(ys * 0.3);
          data[y * W + x] = Math.max(0, Math.min(WHITE - 1, Math.round(BLACK[cell] + signal)));
        }
      }
      return { width: W, height: H, data, blackLevel: BLACK, whiteLevel: WHITE, cfa: [0, 1, 1, 2], colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622], whiteBalance: [2.1, 1, 1.9, 1], orientation: 1, make: 'TestCam', model: 'PoC-1', name: `synth(${dx},${dy})` };
    };

    const opts = { tileSize: 16, searchDistance: 32, noiseReduction: 13 };

    // test 1: identical frames → output ≈ reference (weights saturate at 1, alignment is identity)
    const ref = mkFrame();
    const r1 = await alignMergeSpatial(gpu, [ref, mkFrame()], opts, null);
    let finite = true, maxDiffInterior = 0;
    const interior = { x0: 12, x1: W - 12, y0: 12, y1: H - 12 };
    for (let y = interior.y0; y < interior.y1; y++) {
      for (let x = interior.x0; x < interior.x1; x++) {
        const v = r1.pixels[y * W + x];
        if (!Number.isFinite(v)) finite = false;
        maxDiffInterior = Math.max(maxDiffInterior, Math.abs(v - ref.data[y * W + x]));
      }
    }

    // test 2: comp shifted by (2,1) → alignment should mostly undo it; output stays
    // close to ref (loose threshold: integer half-res alignment + warp blending)
    const r2 = await alignMergeSpatial(gpu, [ref, mkFrame(2, 1)], opts, null);
    let maxDiffShifted = 0, finite2 = true;
    for (let y = interior.y0; y < interior.y1; y++) {
      for (let x = interior.x0; x < interior.x1; x++) {
        const v = r2.pixels[y * W + x];
        if (!Number.isFinite(v)) finite2 = false;
        maxDiffShifted = Math.max(maxDiffShifted, Math.abs(v - ref.data[y * W + x]));
      }
    }
    // what an unaligned average would give on this gradient (shift error 2px * ~3/px)
    const unalignedBaseline = 2 * 3.0 + 1 * 2.0;

    // ---- tests 3-5: alignment inspection --------------------------------
    // Synthetic 128x128 burst with a textured subject region; comparison frames are
    // (a) identical, (b) shifted by a known (4, 2) sensor px, (c) wildly different noise.
    const WI = 128, HI = 128;
    // smooth gradient + multi-scale sines + deterministic high-frequency detail (like real
    // image texture — without the detail term, tile costs tie across shifts on a pure gradient)
    const texture = (x, y) => {
      const detail = (Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
      return 20 + 3.0 * x + 2.0 * y + 10 * Math.sin(x * 0.35) * Math.cos(y * 0.3) + 4 * Math.sin(x * 0.11 + y * 0.07) + 22 * detail;
    };
    const mkFrameI = (dx = 0, dy = 0, seed = 0) => {
      const data = new Uint16Array(WI * HI);
      for (let y = 0; y < HI; y++) {
        for (let x = 0; x < WI; x++) {
          const xs = Math.min(Math.max(x - dx, 0), WI - 1);
          const ys = Math.min(Math.max(y - dy, 0), HI - 1);
          const noise = seed ? ((Math.sin(x * 12.9898 + y * 78.233 + seed) * 43758.5453) % 1) * 12 : 0;
          data[y * WI + x] = Math.max(0, Math.min(WHITE - 1, Math.round(BLACK[(y % 2) * 2 + (x % 2)] + texture(xs, ys) + noise)));
        }
      }
      return { width: WI, height: HI, data, blackLevel: BLACK, whiteLevel: WHITE, cfa: [0, 1, 1, 2], colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622], whiteBalance: [2.1, 1, 1.9, 1], orientation: 1, make: 'TestCam', model: 'PoC-1', name: `synth(${dx},${dy},${seed})` };
    };
    // a frame with a locally displaced subject block (moving subject), aligned background:
    // the ROI alignment-variation score must distinguish it from rigidly shifted frames
    const mkSubject = () => {
      const f = mkFrameI();
      for (let y = 44; y < 84; y++) {
        for (let x = 44; x < 84; x++) {
          const cell = (y % 2) * 2 + (x % 2);
          f.data[y * WI + x] = Math.max(0, Math.min(WHITE - 1, Math.round(BLACK[cell] + texture(x - 6, y - 4))));
        }
      }
      f.name = 'subject';
      return f;
    };

    // order: ref (explicit refIdx 0), shifted by a known (4, 2) sensor px, identical, subject motion
    const burst = [mkFrameI(), mkFrameI(4, 2), mkFrameI(), mkSubject()];
    const session = await performInspection(gpu, burst, { tileSize: 16, searchDistance: 32, refIdx: 0 }, null);

    // T3: the shifted frame's alignment field should report ~(2, 1) level-0 px (= (4, 2)
    // sensor px) in the interior; border tiles legitimately differ (the synthetic shift is
    // clamped at the edges, so border tiles see unmoved content)
    const shifted = session.frame(1);
    const gw = session.tileInfo.w, gh = session.tileInfo.h;
    const interiorDx = [], interiorDy = [];
    for (let ty = 1; ty < gh - 1; ty++) {
      for (let tx = 1; tx < gw - 1; tx++) {
        const i = (ty * gw + tx) * 2;
        interiorDx.push(shifted.texels[i]);
        interiorDy.push(shifted.texels[i + 1]);
      }
    }
    interiorDx.sort((a, b) => a - b);
    interiorDy.sort((a, b) => a - b);
    const median = (arr) => arr.length % 2 ? arr[arr.length >> 1] : 0.5 * (arr[(arr.length >> 1) - 1] + arr[arr.length >> 1]);
    const meanDx = median(interiorDx);
    const meanDy = median(interiorDy);

    // T4: ROI alignment-variation score — near-rigid frames score low, noisy/shift-mismatched score high
    const roi = { x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75 };
    const vIdentical = session.alignmentVariation(2, roi);
    const vShifted = session.alignmentVariation(1, roi);
    const vSubject = session.alignmentVariation(3, roi);

    // T5: calibration — brightness match of the noisy frame should be ~unit (same exposure),
    // shared scale computed, warped thumbnail data present
    await session.calibrate(null);
    const matchNoisy = session.matchCache.get(3);
    const matchShifted = session.matchCache.get(1);

    // T6: render the shifted frame in each mode end-to-end (warped / difference / original)
    // — frame 1, whose alignment field is non-trivial (the identical frame's field is legitimately all-black)
    const renders = {};
    for (const mode of ['alignment', 'difference', 'original', 'warped', 'reference']) {
      const r = await session.renderFrame(1, {
        mode, alignmentScale: session.alignmentScale, outWidth: 64, outHeight: 64,
        diffGain: 4, diffWhiteOverride: null, matchExposure: true,
        anchor: { u: 0.5, v: 0.5, tileX: session.tileInfo.w >> 1, tileY: session.tileInfo.h >> 1 },
      });
      const px = r.bitmap.rgba;
      let nonBlack = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] > 12) nonBlack++;
      renders[mode] = { nonBlackRatio: nonBlack / (px.length / 4), info: r.info };
    }

    // T7: rendering the REFERENCE frame in difference mode (with brightness match + anchor,
    // so localMatch/globalMatch run) must not clobber the retained reference plane — the
    // match helpers used to destroy the caller-owned refCropped buffer, blacking out every
    // subsequent render that reads it (reference/original modes, all further diff renders).
    const t7 = {};
    {
      const common = {
        alignmentScale: session.alignmentScale, outWidth: 64, outHeight: 64,
        diffGain: 4, diffWhiteOverride: null, matchExposure: true,
        anchor: { u: 0.5, v: 0.5, tileX: session.tileInfo.w >> 1, tileY: session.tileInfo.h >> 1 },
      };
      const nb = (r) => {
        const px = r.bitmap.rgba;
        let n = 0;
        for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] > 12) n++;
        return n / (px.length / 4);
      };
      t7.refDiff = nb(await session.renderFrame(0, { mode: 'difference', ...common }));
      t7.refAfter = nb(await session.renderFrame(0, { mode: 'reference', ...common }));
      t7.origAfter = nb(await session.renderFrame(0, { mode: 'original', ...common }));
      t7.diffAfter = nb(await session.renderFrame(1, { mode: 'difference', ...common }));
      t7.warpedAfter = nb(await session.renderFrame(1, { mode: 'warped', ...common }));
    }

    // T8: merging with an inspection session's retained alignment grids must produce
    // pixel-identical output to a full align+merge — the grids are the exact texels
    // alignTextureEx computes, so reuse must only skip the pyramid search
    const t8 = {};
    let t9 = null;
    {
      const frames8 = [mkFrame(0, 0), mkFrame(2, 1), mkFrame(-1, 2)];
      const r8full = await alignMergeSpatial(gpu, frames8, opts, null);
      const session8 = await performInspection(gpu, frames8,
        { tileSize: opts.tileSize, searchDistance: opts.searchDistance, refIdx: 0 }, null);
      const prealigned = frames8.map((_, i) => {
        const f = session8.frameList.find((fr) => fr.idx === i);
        return f && !f.isRef ? { texels: f.texels, w: session8.tileInfo.w, h: session8.tileInfo.h } : null;
      });
      const t0 = performance.now();
      const r8reuse = await alignMergeSpatial(gpu, frames8, { ...opts, prealigned }, null);
      t8.reuseMs = performance.now() - t0;
      let maxDelta = 0;
      for (let i = 0; i < r8full.pixels.length; i++) {
        maxDelta = Math.max(maxDelta, Math.abs(r8full.pixels[i] - r8reuse.pixels[i]));
      }
      t8.identical = r8full.pixels.length === r8reuse.pixels.length && maxDelta === 0;
      t8.maxDelta = maxDelta;
      // grid dims must line up with what the merge derives (otherwise warpWithGrid throws)
      t8.gridDims = `${session8.tileInfo.w}×${session8.tileInfo.h}`;

      // T9: the merged result can be rendered as a session view (the inspector's "Merged"
      // tab): flashraw-developed RGB (the fixture's warm as-shot WB must show as channel
      // spread), cached per output size and across selected frames
      session8.mergedResult = {
        pixels: r8reuse.pixels, width: r8reuse.width, height: r8reuse.height,
        dngBytes: writeDNG(r8reuse.pixels, {
          width: r8reuse.width, height: r8reuse.height, blackLevel: BLACK, whiteLevel: WHITE,
          cfa: [0, 1, 1, 2], colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622],
          whiteBalance: [2.1, 1, 1.9, 1], orientation: 1, make: 'TestCam', model: 'PoC-1',
        }),
        blackLevel: BLACK, whiteLevel: WHITE,
        cfa: [0, 1, 1, 2], colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622],
        whiteBalance: [2.1, 1, 1.9, 1], orientation: 1, make: 'TestCam', model: 'PoC-1',
        count: frames8.length, name: 'synth',
      };
      const m1 = await session8.renderFrame(0, { mode: 'merged', outWidth: 32, outHeight: 32 });
      const m2 = await session8.renderFrame(1, { mode: 'merged', outWidth: 32, outHeight: 32 }); // cache hit
      const stats = (b) => {
        let litN = 0, colored = 0;
        for (let i = 0; i < b.rgba.length; i += 4) {
          if (b.rgba[i] + b.rgba[i + 1] + b.rgba[i + 2] > 12) litN++;
          const mx = Math.max(b.rgba[i], b.rgba[i + 1], b.rgba[i + 2]), mn = Math.min(b.rgba[i], b.rgba[i + 1], b.rgba[i + 2]);
          if (mx - mn > 10) colored++;
        }
        return { lit: litN / (b.rgba.length / 4), colored: colored / (b.rgba.length / 4) };
      };
      const st = stats(m1.bitmap);
      t9 = { lit: st.lit, colored: st.colored, cached: m1.bitmap === m2.bitmap, info: m1.info };
      session8.destroy();
    }
    session.destroy();

    // ---- T10-T14: frequency-domain merge (Burst Photo "Higher quality" port) ----
    const t10 = {};
    {
      const { alignMergeFrequencyDomain, deriveFrequencySetup, frequencyMergeParams } =
        await import('/src/merge_frequency.js');
      const { Plane } = await import('/src/pipeline.js');

      // T10a: geometry derivation matches the Swift padding math
      // (W=256, tile 32 → tileFactor 64 → pad_align 28 → pad_merge 12 → canvas 288 = 18 tiles)
      const fa = { tileSizeArray: [32, 16], nLevels: 2, downscaleFactorArray: [2, 2] };
      const f256 = deriveFrequencySetup(256, 256, fa);
      t10.geometry = f256;
      t10.geometryOk = f256.canvasW === 288 && f256.canvasH === 288 && f256.nTilesX === 18 && f256.padMergeX === 12;
      t10.paramsOk = (() => {
        const p = frequencyMergeParams(13);
        // robustness_rev = 0.5·(26.5-13) = 6.75 → 2^0.75, read = (2^3.25)^1.6, mmn = 1.3^4.25
        const eps = (a, b) => Math.abs(a - b) < 1e-9 * Math.max(1, b);
        return eps(p.robustnessNorm, Math.pow(2, 0.75))
          && eps(p.readNoise, Math.pow(Math.pow(2, 3.25), 1.6))
          && eps(p.maxMotionNorm, Math.pow(1.3, 4.25));
      })();

      // T10b: FFT pair — backward(forward(x)) == window·x on a 2×2-tile random plane
      {
        const TW = 16, TH = 16;
        const rgba = new Float32Array(TW * TH * 4);
        for (let i = 0; i < TW * TH * 4; i++) rgba[i] = 500 + 3000 * ((Math.sin(i * 12.9898) * 43758.5453) % 1);
        const inBuf = gpu.storageFromData('fft-in', rgba);
        const ftBuf = gpu.zeros('fft-ft', TW * 2 * TH * 16);
        const outBuf = gpu.zeros('fft-out', TW * TH * 16);
        gpu.dispatch('forward_fft',
          [{ buffer: inBuf }, { buffer: ftBuf }, { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32'], [TW, TW * 2, 2, 2, 0]) }],
          [1, 1, 1]);
        gpu.dispatch('backward_fft',
          [{ buffer: ftBuf }, { buffer: outBuf }, { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'f32'], [TW, TW * 2, 2, 2, 64]) }],
          [1, 1, 1]);
        await device.queue.onSubmittedWorkDone();
        const round = new Float32Array(await gpu.readBack(outBuf, TW * TH * 16));
        let maxErr = 0;
        const wv = (d) => 0.5 - 0.5 * Math.cos(Math.PI * (d + 0.5) / 4);
        for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
          const w = wv(x % 8) * wv(y % 8);
          for (let ch = 0; ch < 4; ch++) {
            const i = (y * TW + x) * 4 + ch;
            maxErr = Math.max(maxErr, Math.abs(round[i] - w * rgba[i]));
          }
        }
        t10.fftRoundTrip = maxErr;
        inBuf.destroy(); ftBuf.destroy(); outBuf.destroy();
      }

      // T10c: place → rgba → bayer round trip preserves values
      {
        const SW = 64, SH = 64, CW = 96, CH = 96, OX = 12, OY = 20;
        const src = new Float32Array(SW * SH);
        for (let i = 0; i < src.length; i++) src[i] = 100 + ((i * 7919) % 900);
        const srcBuf = gpu.storageFromData('place-in', src);
        const canvas = new Plane(gpu.zeros('place-canvas', CW * CH * 4), CW, CH);
        gpu.dispatch('place_f32',
          [{ buffer: srcBuf }, { buffer: canvas.buf },
           { buffer: packUniforms(['i32', 'i32', 'i32', 'i32', 'i32', 'i32'], [SW, SH, CW, CH, OX, OY]) }],
          [Math.ceil(CW / 8), Math.ceil(CH / 8), 1]);
        const rgbaP = new Plane(gpu.zeros('place-rgba', CW * CH * 4), CW / 2, CH / 2);
        gpu.dispatch('convert_to_rgba',
          [{ buffer: canvas.buf }, { buffer: rgbaP.buf },
           { buffer: packUniforms(['i32', 'i32', 'i32'], [CW, CW / 2, CH / 2]) }],
          [Math.ceil(CW / 16), Math.ceil(CH / 16), 1]);
        const back = new Plane(gpu.zeros('place-back', CW * CH * 4), CW, CH);
        gpu.dispatch('convert_to_bayer',
          [{ buffer: rgbaP.buf }, { buffer: back.buf },
           { buffer: packUniforms(['i32'], [CW]) }],
          [Math.ceil(CW / 16), Math.ceil(CH / 16), 1]);
        await device.queue.onSubmittedWorkDone();
        const out = new Float32Array(await gpu.readBack(back.buf, CW * CH * 4));
        let maxErr = 0;
        for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
          maxErr = Math.max(maxErr, Math.abs(out[(y + OY) * CW + (x + OX)] - src[y * SW + x]));
        }
        let zeroOk = true;
        for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
          const inSrc = x >= OX && x < OX + SW && y >= OY && y < OY + SH;
          if (!inSrc && out[y * CW + x] !== 0) zeroOk = false;
        }
        t10.placeRgbaBayer = { maxErr, zeroOk };
        srcBuf.destroy(); canvas.buf.destroy(); rgbaP.buf.destroy(); back.buf.destroy();
      }

      // T11: full frequency merge — identical frames reproduce the reference
      // (tight on a constant image: only plumbing noise; the textured burst additionally
      // exercises Burst Photo's deconvolution sharpening, so it gets loose bounds)
      const flat = (v) => {
        const data = new Uint16Array(W * H);
        for (let i = 0; i < data.length; i++) data[i] = v;
        return { width: W, height: H, data, blackLevel: BLACK, whiteLevel: WHITE, name: `flat${v}` };
      };
      const rf = flat(800);
      const rFlat = await alignMergeFrequencyDomain(gpu, [rf, flat(800)], opts, null);
      let flatMax = 0;
      for (let y = 8; y < H - 8; y++) for (let x = 8; x < W - 8; x++) {
        flatMax = Math.max(flatMax, Math.abs(rFlat.pixels[y * W + x] - 800));
      }
      t10.flatIdentical = flatMax;

      const refF = mkFrame();
      const rIdent = await alignMergeFrequencyDomain(gpu, [refF, mkFrame()], opts, null);
      let identMax = 0, identSum = 0, identN = 0;
      for (let y = 8; y < H - 8; y++) for (let x = 8; x < W - 8; x++) {
        const d = Math.abs(rIdent.pixels[y * W + x] - refF.data[y * W + x]);
        identMax = Math.max(identMax, d); identSum += d; identN++;
      }
      t10.identical = { max: identMax, mean: identSum / identN };

      // T12: shifted comparison frame — the sub-pixel alignment must track it much
      // better than a no-alignment merge would (interior mean |diff|)
      const noAlign = (dx, dy) => {
        let sum = 0, n = 0;
        const f = mkFrame(dx, dy);
        for (let y = 8; y < H - 8; y++) for (let x = 8; x < W - 8; x++) {
          sum += Math.abs(f.data[y * W + x] - refF.data[y * W + x]); n++;
        }
        return sum / n;
      };
      const rShift = await alignMergeFrequencyDomain(gpu, [refF, mkFrame(2, 1)], opts, null);
      let shiftSum = 0, shiftN = 0, shiftMax = 0;
      for (let y = 8; y < H - 8; y++) for (let x = 8; x < W - 8; x++) {
        const d = Math.abs(rShift.pixels[y * W + x] - refF.data[y * W + x]);
        shiftSum += d; shiftN++; shiftMax = Math.max(shiftMax, d);
      }
      t10.shifted = { mean: shiftSum / shiftN, max: shiftMax, unalignedMean: noAlign(2, 1) };

      // T13: merging twice must be deterministic (read-modify-write kernels must not race)
      const rAgain = await alignMergeFrequencyDomain(gpu, [refF, mkFrame(2, 1)], opts, null);
      let drift = 0;
      for (let i = 0; i < rShift.pixels.length; i++) drift = Math.max(drift, Math.abs(rShift.pixels[i] - rAgain.pixels[i]));
      t10.deterministic = drift === 0;

      // T14: fresh alignment vs inspection-session grids → bit-identical output
      const framesF = [mkFrame(0, 0), mkFrame(2, 1), mkFrame(-1, 2)];
      const sessionF = await performInspection(gpu, framesF,
        { tileSize: opts.tileSize, searchDistance: opts.searchDistance, refIdx: 0 }, null);
      const grids = framesF.map((_, i) => {
        const f = sessionF.frameList.find((fr) => fr.idx === i);
        return f && !f.isRef ? { texels: f.texels, w: sessionF.tileInfo.w, h: sessionF.tileInfo.h } : null;
      });
      const rFresh = await alignMergeFrequencyDomain(gpu, framesF, opts, null);
      const rReuse = await alignMergeFrequencyDomain(gpu, framesF, { ...opts, prealigned: grids }, null);
      let reuseDelta = 0;
      for (let i = 0; i < rFresh.pixels.length; i++) reuseDelta = Math.max(reuseDelta, Math.abs(rFresh.pixels[i] - rReuse.pixels[i]));
      t10.reuseIdentical = reuseDelta === 0;
      sessionF.destroy();
    }

    return {
      stage: 'done',
      adapter: `${adapter.info?.vendor || '?'} ${adapter.info?.architecture || ''}`,
      t1: { finite, maxDiffInterior },
      t2: { finite: finite2, maxDiffShifted, unalignedBaseline },
      timings: r1.timings,
      t3: { meanDx, meanDy, tileInfo: session.tileInfo },
      t4: { vIdentical, vShifted, vSubject },
      t5: { matchNoisy, matchShifted, medianDiffMean: session.medianDiffMean, calibrated: session.calibrated,
            progress: session.calibrationProgress },
      t6: renders,
      t8,
      t9,
      t7,
      t10,
    };
  });

  if (out.stage === 'wgsl-compile') {
    console.log('WGSL COMPILE ERRORS:');
    for (const e of out.errors) console.log('  ' + e);
    failures++;
  } else {
    check('WebGPU adapter', true, out.adapter);
    check('T1 identical frames: all interior samples finite', out.t1.finite);
    check('T1 output ≈ reference in interior (< 6 levels)', out.t1.maxDiffInterior < 6,
      `maxDiff=${out.t1.maxDiffInterior}`);
    check('T2 shifted comp: all interior samples finite', out.t2.finite);
    check(`T2 aligned output closer to ref than unaligned (${out.t2.unalignedBaseline} levels)`,
      out.t2.maxDiffShifted < out.t2.unalignedBaseline,
      `maxDiff=${out.t2.maxDiffShifted.toFixed(0)} vs baseline ${out.t2.unalignedBaseline}`);
    check('T2 aligned output tight (< 25 levels)', out.t2.maxDiffShifted < 25,
      `maxDiff=${out.t2.maxDiffShifted.toFixed(0)}`);
    console.log('timings:', JSON.stringify(out.timings));

    // ---- inspection ----
    check('T3 shifted frame interior alignment ≈ (2,1) l0-px',
      Math.abs(out.t3.meanDx - 2) <= 0.6 && Math.abs(out.t3.meanDy - 1) <= 0.6,
      `median interior shift (${out.t3.meanDx}, ${out.t3.meanDy}) l0-px, grid ${out.t3.tileInfo.w}x${out.t3.tileInfo.h}`);
    check('T3 alignment grid has tiles', out.t3.tileInfo.w >= 4 && out.t3.tileInfo.h >= 4,
      `${out.t3.tileInfo.w}x${out.t3.tileInfo.h}`);
    check('T4 ROI variation: identical < shifted', out.t4.vIdentical < out.t4.vShifted,
      `identical=${out.t4.vIdentical.toFixed(2)} shifted=${out.t4.vShifted.toFixed(2)}`);
    check('T4 ROI variation: rigid < subject motion', out.t4.vIdentical < out.t4.vSubject && out.t4.vShifted < out.t4.vSubject,
      `identical=${out.t4.vIdentical.toFixed(2)} shifted=${out.t4.vShifted.toFixed(2)} subject=${out.t4.vSubject.toFixed(2)}`);
    check('T5 calibration completed', out.t5.calibrated && out.t5.progress === 3,
      `progress=${out.t5.progress} medianDiffMean=${out.t5.medianDiffMean?.toFixed(2)}`);
    check('T5 brightness match ≈ unit gain for uniform exposure',
      Math.abs(out.t5.matchNoisy.gain - 1) < 0.15,
      `noisy gain=${out.t5.matchNoisy.gain.toFixed(3)} offset=${out.t5.matchNoisy.offset.toFixed(1)}`);
    console.log('T5 shifted match:', JSON.stringify(out.t5.matchShifted));
    for (const [mode, r] of Object.entries(out.t6)) {
      check(`T6 render ${mode}: non-trivial output`, r.nonBlackRatio > 0.1,
        `nonBlack=${(r.nonBlackRatio * 100).toFixed(0)}% — ${r.info.slice(0, 80)}`);
    }
    check('T8 reuse-merge output is pixel-identical to a full align+merge', out.t8.identical,
      `maxDelta=${out.t8.maxDelta} grid=${out.t8.gridDims} reuseMs=${out.t8.reuseMs?.toFixed(0)}`);
    check('T9 merged view renders developed RGB (cached across frames)',
      out.t9 && out.t9.lit > 0.1 && out.t9.colored > 0.02 && out.t9.cached,
      `lit=${((out.t9?.lit || 0) * 100).toFixed(0)}% colored=${((out.t9?.colored || 0) * 100).toFixed(0)}% cached=${out.t9?.cached} — ${(out.t9?.info || '').slice(0, 60)}`);
    check('T7 ref difference render produces output', out.t7.refDiff > 0.1,
      `nonBlack=${(out.t7.refDiff * 100).toFixed(0)}%`);
    for (const k of ['refAfter', 'origAfter', 'diffAfter', 'warpedAfter']) {
      check(`T7 renders after a ref difference render stay non-black (${k})`, out.t7[k] > 0.1,
        `nonBlack=${(out.t7[k] * 100).toFixed(0)}%`);
    }

    // ---- frequency-domain merge (T10) ----
    check('T10 geometry derivation matches the Swift padding math', out.t10.geometryOk,
      `canvas=${out.t10.geometry?.canvasW}×${out.t10.geometry?.canvasH} tiles=${out.t10.geometry?.nTilesX} padMerge=${out.t10.geometry?.padMergeX}`);
    check('T10 robustness parameter derivation matches the Swift formulas', out.t10.paramsOk,
      JSON.stringify(out.t10));
    check('T10 FFT pair round trip (backward∘forward == window·x) < 0.05', out.t10.fftRoundTrip < 0.05,
      `maxErr=${out.t10.fftRoundTrip?.toExponential(2)}`);
    check('T10 place/rgba/bayer round trip preserves values, zero fill intact',
      out.t10.placeRgbaBayer && out.t10.placeRgbaBayer.maxErr < 1e-2 && out.t10.placeRgbaBayer.zeroOk,
      `maxErr=${out.t10.placeRgbaBayer?.maxErr} zeroOk=${out.t10.placeRgbaBayer?.zeroOk}`);
    check('T11 frequency merge, constant identical frames reproduce the reference (< 3)',
      out.t10.flatIdentical < 3, `maxDiff=${out.t10.flatIdentical}`);
    check('T11 frequency merge, textured identical frames stay close (mean < 4, max < 40)',
      out.t10.identical.mean < 4 && out.t10.identical.max < 40,
      `mean=${out.t10.identical.mean?.toFixed(2)} max=${out.t10.identical.max}`);
    check('T12 frequency merge tracks a (2,1)-shifted frame much better than no alignment',
      out.t10.shifted.mean < out.t10.shifted.unalignedMean * 0.5 && out.t10.shifted.mean < 12,
      `merged mean=${out.t10.shifted.mean?.toFixed(1)} vs unaligned ${out.t10.shifted.unalignedMean?.toFixed(1)} (max ${out.t10.shifted.max})`);
    check('T13 frequency merge is deterministic across runs', out.t10.deterministic);
    check('T14 frequency merge: fresh alignment == inspection-grid reuse (bit-identical)',
      out.t10.reuseIdentical);
    console.log('T10 details:', JSON.stringify(out.t10, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
  }
} catch (err) {
  console.log('FAIL  smoke test crashed —', String(err?.stack || err).slice(0, 1200));
  failures++;
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\nSMOKE TEST PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
