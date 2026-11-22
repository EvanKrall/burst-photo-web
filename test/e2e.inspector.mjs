// End-to-end inspector test: generates a synthetic DNG burst, loads it through the real UI
// (file input → decode → alignment inspection → frame selection → merge → preview) in a
// WebGPU-capable Chrome, and verifies the inspector behavior end to end.
//
// Usage: node test/e2e.inspector.mjs   (starts its own server on :8124, opens a Chrome window)
// Requires: Google Chrome at the default macOS path (override via CHROME_BIN).

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { writeDNG } from '../src/dng_writer.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const port = 8124;
const chromePath = process.env.CHROME_BIN ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
};

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
  args: ['--window-size=1280,900', '--enable-unsafe-webgpu'],
});

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
};

// ---- synthetic burst: 256x256 RGGB DNGs (a 7x7 tile grid at the default tile size 32);
// ref, shifted by (4,2), identical, subject motion ----
const W = 256, H = 256, BLACK = [200, 204, 200, 206], WHITE = 4000;
const texture = (x, y) => {
  // classic hash coordinates (127.1, 311.7 — no spurious small-period aliasing, unlike
  // 12.9898/78.233 whose ~6 px ratio creates shift ties); amplitude near full range to make
  // tile costs discriminate shifts sharply
  const detail = (Math.sin(x * 127.1 + y * 311.7) * 43758.5453) % 1;
  return Math.min(WHITE - 260, Math.round(80 + 3.0 * x + 2.0 * y + 10 * Math.sin(x * 0.35) * Math.cos(y * 0.3) + 150 * detail));
};
const mkPixels = (dx = 0, dy = 0) => {
  const pixels = new Uint16Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const cell = (y % 2) * 2 + (x % 2);
      const xs = Math.min(Math.max(x - dx, 0), W - 1), ys = Math.min(Math.max(y - dy, 0), H - 1);
      pixels[y * W + x] = BLACK[cell] + texture(xs, ys);
    }
  }
  return pixels;
};
// a frame with a locally displaced subject block (moving subject): the block spans several
// alignment tiles and is displaced by a few tiles' worth of sensor pixels
// a frame with a locally displaced subject block (moving subject): the block spans a full
// tile region (tile = 64 sensor px at level 0) so its tiles must disagree with the rest
const mkSubjectDNG = () => {
  const pixels = mkPixels();
  for (let y = 64; y < 192; y++) {
    for (let x = 64; x < 192; x++) {
      const cell = (y % 2) * 2 + (x % 2);
      pixels[y * W + x] = BLACK[cell] + texture(x - 7, y - 5);
    }
  }
  return pixels;
};
const dngOptions = {
  width: W, height: H, blackLevel: BLACK, whiteLevel: WHITE, cfa: [0, 1, 1, 2],
  colorMatrix: [0.9505, 0, -0.0006, 0.3181, 0.7288, 0.0248, 0.0447, -0.0178, 0.9622],
  orientation: 1, make: 'TestCam', model: 'PoC-1', whiteBalance: [2.1, 1, 1.9, 1],
  exif: { exposureTime: 0.032, iso: 25, fNumber: 4, focalLength: 50, captureTime: '2024-06-01T12:00:00Z' },
};
const bursts = {
  'a_ref.dng': mkPixels(),
  'b_shifted.dng': mkPixels(4, 2),
  'c_identical.dng': mkPixels(),
  'd_subject.dng': mkSubjectDNG(),
};
const dngBase64 = Object.fromEntries(Object.entries(bursts).map(([name, pixels]) => [
  name, Buffer.from(writeDNG(pixels, dngOptions)).toString('base64'),
]));

const pageErrors = [];
try {
  const page = await browser.newPage();
  // uncaught page exceptions only — console resource 404s (vendored flashraw fallback) are expected in dev
  page.on('pageerror', (e) => pageErrors.push(String(e).slice(0, 300)));
  await page.setViewport({ width: 1280, height: 900 });
  await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });

  // inject the synthetic burst through the file input
  await page.evaluate((files) => {
    const toBytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const dt = new DataTransfer();
    for (const [name, b64] of Object.entries(files)) {
      dt.items.add(new File([toBytes(b64)], name, { type: 'image/x-adobe-dng' }));
    }
    const input = document.getElementById('fileInput');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, dngBase64);

  // wait for the inspector pane to become visible
  await page.waitForSelector('#inspectorPane:not([hidden])', { timeout: 60000 });

  const insp = await page.evaluate(() => ({
    rows: document.querySelectorAll('.insp-row').length,
    refBadge: document.querySelectorAll('.insp-refbadge').length,
    activeMode: document.querySelector('#inspModes button.active')?.dataset.mode,
    mergeText: document.getElementById('inspMergeText').textContent,
    mergeDisabled: document.getElementById('inspMerge').disabled,
    stats: document.getElementById('inspStatsLine').textContent,
  }));
  check('inspector shown with one row per frame', insp.rows === 4, `${insp.rows} rows`);
  check('reference row marked', insp.refBadge === 1);
  check('default mode is Alignment', insp.activeMode === 'alignment', insp.activeMode);
  check('merge enabled for all frames', !insp.mergeDisabled, insp.mergeText);
  check('stats line names the selected frame', /frame 0/.test(insp.stats), insp.stats.slice(0, 60));

  // AF-area detection (lazy WASM exiftool on the reference frame) must resolve and report
  // a definite outcome — the synthetic DNGs carry no AF metadata, so the fallback must be
  // logged (a fetch/path misconfiguration would show up as the failure branch instead)
  await page.waitForFunction(() => {
    const text = document.getElementById('log').textContent;
    return text.includes('No AF area metadata') || text.includes('AF area from') || text.includes('AF-area detection failed');
  }, { timeout: 120000 });
  const afLog = await page.evaluate(() => {
    const lines = document.getElementById('log').textContent.split('\n');
    return lines.find((l) => l.includes('AF area') || l.includes('AF-area'));
  });
  check('AF-area detection reports a definite outcome', /No AF area metadata/.test(afLog), afLog?.trim());

  // wait for calibration to finish: every comparison row's diff score must become a number
  // (the reference row shows "–" by design — it has no alignment field to score)
  await page.waitForFunction(() => {
    const rows = [...document.querySelectorAll('.insp-row')].filter((r) => !r.querySelector('.insp-refbadge'));
    return rows.every((r) => r.children[3].textContent !== '–');
  }, { timeout: 60000 });

  const scores = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.insp-row')];
    return rows.map((r) => ({
      name: r.querySelector('.insp-file').textContent,
      diff: parseFloat(r.children[3].textContent),
      align: parseFloat(r.children[4].textContent),
      included: r.querySelector('.insp-check').classList.contains('included'),
    }));
  });
  const byName = Object.fromEntries(scores.map((s) => [s.name, s]));
  check('all rows included initially', scores.every((s) => s.included));
  // the burst center (c_identical) is the reference frame — a_ref.dng has identical content
  check('identical-content frame has the lowest diff score',
    byName['a_ref.dng'].diff <= byName['b_shifted.dng'].diff,
    JSON.stringify(byName, null, 0).slice(0, 220));
  check('subject-motion frame scores worst on alignment variation',
    byName['d_subject.dng'].align > byName['a_ref.dng'].align &&
    byName['d_subject.dng'].align > byName['b_shifted.dng'].align,
    `align a_ref=${byName['a_ref.dng'].align.toFixed(2)} shifted=${byName['b_shifted.dng'].align.toFixed(2)} subject=${byName['d_subject.dng'].align.toFixed(2)}`);
  check('subject-motion frame also scores worst on diff',
    byName['d_subject.dng'].diff > byName['a_ref.dng'].diff,
    `diff a_ref=${byName['a_ref.dng'].diff.toFixed(1)} subject=${byName['d_subject.dng'].diff.toFixed(1)}`);

  // sorting by alignment score (table header) must move the reference row to the top and the
  // subject frame last, and mark the header cell active
  await page.click('#inspColHead button[data-sort="2"]');
  const sortedNames = await page.evaluate(() =>
    [...document.querySelectorAll('.insp-row .insp-file')].map((e) => e.textContent));
  check('align-sort: reference pinned first', sortedNames[0] === 'c_identical.dng', sortedNames.join(', '));
  check('align-sort: subject-motion frame ranked last', sortedNames[sortedNames.length - 1] === 'd_subject.dng');
  const activeHeader = await page.evaluate(() =>
    document.querySelector('#inspColHead button.active')?.dataset.sort);
  check('sort header marks the active column', activeHeader === '2', `active=${activeHeader}`);

  // shift-click range selection must follow the display order, not the capture order: with
  // a_ref (display pos 1) as anchor, shift-clicking the last row selects exactly the three
  // rows seen between them — the old capture-order range (idx 0..3) would include the
  // pinned reference row too
  await page.evaluate(() => {
    [...document.querySelectorAll('.insp-row')]
      .find((r) => r.querySelector('.insp-file').textContent === 'a_ref.dng')
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await new Promise((r) => setTimeout(r, 300));
  await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.insp-row')];
    rows[rows.length - 1].dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
  });
  await new Promise((r) => setTimeout(r, 300));
  const selNames = await page.evaluate(() =>
    [...document.querySelectorAll('.insp-row.selected .insp-file')].map((e) => e.textContent));
  check('shift-click selects the display-order range',
    JSON.stringify(selNames) === JSON.stringify(['a_ref.dng', 'b_shifted.dng', 'd_subject.dng']),
    selNames.join(','));
  // reset to a single selection so the following exclusion step is unaffected
  await page.evaluate(() => {
    [...document.querySelectorAll('.insp-row')]
      .find((r) => r.querySelector('.insp-file').textContent === 'd_subject.dng')
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await new Promise((r) => setTimeout(r, 300));

  // exclude the subject frame via its checkbox
  await page.evaluate(() => {
    const row = [...document.querySelectorAll('.insp-row')]
      .find((r) => r.querySelector('.insp-file').textContent === 'd_subject.dng');
    row.querySelector('.insp-check').click();
  });
  const mergeAfterExclude = await page.evaluate(() => document.getElementById('inspMerge').textContent);
  check('merge counts excluded frame', /Merge 3 of 4 images/.test(mergeAfterExclude), mergeAfterExclude);

  // mode switch to Difference and back renders without errors
  await page.click('#inspModes button[data-mode="difference"]');
  await new Promise((r) => setTimeout(r, 1500));
  await page.click('#inspModes button[data-mode="alignment"]');
  await new Promise((r) => setTimeout(r, 800));
  const infoAfterModes = await page.evaluate(() => document.getElementById('inspInfoLine').textContent);
  check('render info present after mode switches', infoAfterModes.length > 20, infoAfterModes.slice(0, 90));

  // arrow-key scrub moves the selection (in frame order)
  await page.click('#inspColHead button[data-sort="0"]');
  await page.evaluate(() => {
    const row = [...document.querySelectorAll('.insp-row')].find((r) => r.querySelector('.insp-file').textContent === 'b_shifted.dng');
    row.click();
  });
  await new Promise((r) => setTimeout(r, 400));
  await page.keyboard.press('ArrowDown');
  await new Promise((r) => setTimeout(r, 400));
  const statsAfterDown = await page.evaluate(() => document.getElementById('inspStatsLine').textContent);
  check('arrow-down selects the next frame', /frame 2/.test(statsAfterDown), statsAfterDown.slice(0, 60));

  // image-area interactions: wheel zoom, drag = subject area, click = anchor, esc = clear ROI
  const overlayBox = await (await page.$('#inspOverlay')).boundingBox();
  const cx = overlayBox.x + overlayBox.width / 2, cy = overlayBox.y + overlayBox.height / 2;

  await page.mouse.move(cx, cy);
  await page.mouse.wheel({ deltaY: -600 });
  await new Promise((r) => setTimeout(r, 300));
  const zoomed = await page.evaluate(() => ({
    badge: document.getElementById('inspZoomBadge').hidden === false,
    scale: parseFloat(document.getElementById('inspCanvas').style.width) / document.getElementById('inspCanvas').width,
  }));
  check('wheel zooms the image', zoomed.badge && zoomed.scale > 1.3,
    `badge=${zoomed.badge} scale=${zoomed.scale.toFixed(2)}`);
  await page.keyboard.press('Escape'); // no ROI yet — must be a no-op
  await page.evaluate(() => { window.__inspector.state.zoom = 1; window.__inspector.applyCanvasSize(); });
  await new Promise((r) => setTimeout(r, 200));

  // drag a subject-area rectangle on the left part of the image
  await page.mouse.move(cx - overlayBox.width * 0.3, cy - overlayBox.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(cx - overlayBox.width * 0.1, cy + overlayBox.height * 0.2, { steps: 8 });
  await page.mouse.up();
  await new Promise((r) => setTimeout(r, 500));
  const roiState = await page.evaluate(() => ({
    roi: window.__inspector.state.roi,
    clearBtn: document.getElementById('inspClearRoi').hidden === false,
    anchorFollows: window.__inspector.state.anchorManual === false,
  }));
  check('drag sets the subject area', roiState.roi != null && roiState.clearBtn,
    `roi=${JSON.stringify(roiState.roi)}`);
  check('anchor follows the ROI until taken manually', roiState.anchorFollows);

  // a plain click sets the anchor and takes manual control
  await page.mouse.move(cx + overlayBox.width * 0.25, cy + overlayBox.height * 0.1);
  await page.mouse.down();
  await page.mouse.up();
  await new Promise((r) => setTimeout(r, 600));
  const anchorState = await page.evaluate(() => ({
    anchor: window.__inspector.state.anchor,
    manual: window.__inspector.state.anchorManual,
    clearBtn: document.getElementById('inspClearAnchor').hidden === false,
  }));
  check('click sets the anchor point', anchorState.anchor != null && anchorState.manual && anchorState.clearBtn,
    `anchor=${JSON.stringify(anchorState.anchor)}`);

  // escape clears the subject area
  await page.keyboard.press('Escape');
  await new Promise((r) => setTimeout(r, 400));
  const escState = await page.evaluate(() => window.__inspector.state.roi);
  check('escape clears the subject area', escState === null);

  // "Set subject from autofocus point": hidden until an AF area exists (the synthetic
  // DNGs have none — applyAfAreaRoi(null) has run by now), revealed when one is supplied,
  // and clicking it applies the area as a user edit (roiTouched, Clear subject appears)
  const afHidden = await page.evaluate(() => document.getElementById('inspSetAfRoi').hidden);
  check('AF-subject button hidden without AF metadata', afHidden === true);
  await page.evaluate(() => window.__inspector.applyAfAreaRoi({ x0: 0.3, y0: 0.3, x1: 0.6, y1: 0.6 }));
  const afShown = await page.evaluate(() => ({
    hidden: document.getElementById('inspSetAfRoi').hidden,
    anchorStillManual: window.__inspector.state.anchorManual,
  }));
  check('AF-subject button appears once an AF area exists', afShown.hidden === false);
  await page.click('#inspSetAfRoi');
  await new Promise((r) => setTimeout(r, 500));
  const afApplied = await page.evaluate(() => {
    const s = window.__inspector.state;
    return { roi: s.roi, touched: s.roiTouched, clearBtn: !document.getElementById('inspClearRoi').hidden,
             anchor: s.anchor, anchorManual: s.anchorManual };
  });
  check('AF-subject button applies the detected area',
    afApplied.touched && afApplied.clearBtn
    && Math.abs(afApplied.roi.x0 - 0.3) < 1e-9 && Math.abs(afApplied.roi.y0 - 0.3) < 1e-9
    && Math.abs(afApplied.roi.x1 - 0.6) < 1e-9 && Math.abs(afApplied.roi.y1 - 0.6) < 1e-9,
    `roi=${JSON.stringify(afApplied.roi)}`);
  // the anchor follows the AF area (center of 0.3..0.6) even though a manual anchor was
  // set earlier in the test: the button hands anchor control back to the automatic placement
  check('AF-subject button re-places the anchor at the AF-area center',
    afApplied.anchor != null && afApplied.anchorManual === false
    && Math.abs(afApplied.anchor.u - 0.45) < 1e-9 && Math.abs(afApplied.anchor.v - 0.45) < 1e-9
    && Number.isInteger(afApplied.anchor.tileX) && Number.isInteger(afApplied.anchor.tileY),
    `anchor=${JSON.stringify(afApplied.anchor)} manual=${afApplied.anchorManual}`);

  // merge the remaining 3 frames → back into the inspector with a "Merged" view
  await page.click('#inspMerge');
  // the inspection session's alignment grids must actually be reused: the per-frame
  // merge stages must report 'warp+merge' (the intent log line alone once passed an
  // e2e run while a wiring bug still re-ran the full pyramid search per frame)
  await page.waitForFunction(() => document.getElementById('log').textContent.includes('warp+merge 1/'), { timeout: 120000 });
  // the inspection view must come back with the merged result as an extra mode
  await page.waitForFunction(() => {
    const b = document.getElementById('inspMergedMode');
    return b && !b.hidden && !document.getElementById('inspectorPane').hidden;
  }, { timeout: 30000 });
  const reuseLogged = await page.evaluate(() => {
    const text = document.getElementById('log').textContent;
    return {
      intent: text.includes('reusing the inspection alignment'),
      warpStages: (text.match(/warp\+merge \d+\//g) || []).length,
      alignStages: (text.match(/align\+merge \d+\//g) || []).length,
      mergedActive: document.querySelector('#inspModes button[data-mode="merged"]')?.classList.contains('active'),
      openPreviewShown: !document.getElementById('inspOpenPreview').hidden,
    };
  });
  check('merge reuses the inspection alignment (warp stages, no align stages)',
    reuseLogged.intent && reuseLogged.warpStages >= 1 && reuseLogged.alignStages === 0,
    JSON.stringify(reuseLogged));
  check('after the merge, the inspector shows the Merged view + preview button',
    reuseLogged.mergedActive && reuseLogged.openPreviewShown, JSON.stringify(reuseLogged));

  // the merged render must show real content (tonemapped, not black)
  await new Promise((r) => setTimeout(r, 1000)); // let the render pump finish
  const mergedLit = await page.evaluate(() => {
    const c = document.getElementById('inspCanvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let lit = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 24) lit++;
    return { ratio: lit / (d.length / 4), info: document.getElementById('inspInfoLine').textContent };
  });
  check('Merged view renders non-black output', mergedLit.ratio > 0.1,
    `lit=${(mergedLit.ratio * 100).toFixed(0)}% — ${mergedLit.info.slice(0, 80)}`);

  // the merge bar's "Preview & download" opens the preview pane with the merged DNG
  await page.click('#inspOpenPreview');
  await page.waitForSelector('#previewPane:not([hidden])', { timeout: 30000 });
  const merged = await page.evaluate(() => ({
    canvasW: document.getElementById('canvas').width,
    href: document.getElementById('download').href.startsWith('blob:'),
    disabled: document.getElementById('download').classList.contains('disabled'),
  }));
  check('merge produced a preview', merged.canvasW > 0, `canvas ${merged.canvasW}px`);
  check('download link carries the merged DNG', merged.href && !merged.disabled);

  check('no page errors during e2e run', pageErrors.length === 0,
    pageErrors.slice(0, 3).join(' | '));
} catch (err) {
  console.log('FAIL  e2e test crashed —', String(err?.stack || err).slice(0, 1200));
  if (pageErrors.length) console.log('page errors:', pageErrors.slice(0, 6).join('\n  '));
  try {
    const state = await page.evaluate(() => ({
      stats: document.getElementById('inspStatsLine')?.textContent,
      info: document.getElementById('inspInfoLine')?.textContent,
      diffCells: [...document.querySelectorAll('.insp-row')].map((r) => r.children[3].textContent),
    }));
    console.log('page state:', JSON.stringify(state, null, 1).slice(0, 800));
  } catch {}
  failures++;
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\nE2E TEST PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
