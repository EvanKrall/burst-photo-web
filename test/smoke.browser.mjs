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

    return {
      stage: 'done',
      adapter: `${adapter.info?.vendor || '?'} ${adapter.info?.architecture || ''}`,
      t1: { finite, maxDiffInterior },
      t2: { finite: finite2, maxDiffShifted, unalignedBaseline },
      timings: r1.timings,
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
  }
} catch (err) {
  console.log('FAIL  smoke test crashed —', String(err).slice(0, 500));
  failures++;
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\nSMOKE TEST PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
