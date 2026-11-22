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
    const { initGPU, GPUContext } = await import('/src/gpu.js');
    const { WGSL } = await import('/src/kernels.js');
    const { alignMergeSpatial } = await import('/src/pipeline.js');
    const { performInspection } = await import('/src/inspect.js');
    const { alignmentVariationInRoi } = await import('/src/viz.js');

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
      return { width: W, height: H, data, blackLevel: BLACK, whiteLevel: WHITE, name: `synth(${dx},${dy})` };
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
      return { width: WI, height: HI, data, blackLevel: BLACK, whiteLevel: WHITE, name: `synth(${dx},${dy},${seed})` };
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
    // — frame 1, whose alignment field is non-trivial (the identical frame's flow field is legitimately all-black)
    const renders = {};
    for (const mode of ['alignment', 'flow', 'difference', 'original', 'warped', 'reference']) {
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
    session.destroy();

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
