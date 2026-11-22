// UI orchestration: file loading, flashraw decode, WebGPU pipeline, preview + DNG output.

import { initGPU, GPUContext } from './gpu.js';
import { WGSL } from './kernels.js';
import { alignMergeSpatial } from './pipeline.js';
import { decodeRaw, developPreview } from './raw.js';
import { writeDNG } from './dng_writer.js';
import { renderPreview } from './preview.js';

const logEl = document.getElementById('log');
const drop = document.getElementById('drop');
const stage = document.getElementById('stage');
const introPane = document.getElementById('intro');
const progressPane = document.getElementById('progressPane');
const previewPane = document.getElementById('previewPane');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('process');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const canvas = document.getElementById('canvas');
const downloadLink = document.getElementById('download');

function log(msg) {
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

function setProgress(fraction, text) {
  progressFill.style.width = `${Math.round(fraction * 100)}%`;
  if (text) progressText.textContent = text;
}

// ---- progress-bar stage allocation ----
// Each named stage maps its inner 0..1 progress linearly onto the
// [start, end] fraction of the overall bar (see stageProgress).
//
// Widths are re-planned per burst (planStageWidths) from duration estimates:
// decode and align+merge scale with the shot count, the DNG write and the
// preview develop are treated as constant-cost stages. Estimates are EWMA
// averages of measured durations persisted in localStorage, so the bar
// converges to this machine's real speeds after a few runs.
const TIMINGS_KEY = 'burst-photo-web-timings-v1';
const EWMA_ALPHA = 0.3; // weight of a new measurement in the running average
const DEFAULT_TIMINGS = { // ms, used until localStorage has real measurements
  decodePerShot: 120,
  alignMergePerShot: 900,
  dng: 500,
  preview: 500,
};

let PROGRESS_STAGES = {
  decode:     [0.00, 0.50], // RAW decode of the dropped files
  alignMerge: [0.50, 0.90], // GPU pipeline (prepare / align+merge / accumulate-ref callbacks)
  dng:        [0.90, 0.95], // merged-DNG serialization
  preview:    [0.95, 1.00], // flashraw develop + canvas render
};

const stageProgress = (stage, inner) => {
  const [start, end] = PROGRESS_STAGES[stage];
  return start + (end - start) * Math.min(1, Math.max(0, inner));
};

function loadTimingEstimates() {
  try {
    return { ...DEFAULT_TIMINGS, ...JSON.parse(localStorage.getItem(TIMINGS_KEY) ?? '{}') };
  } catch {
    return { ...DEFAULT_TIMINGS }; // corrupted/absent entry — fall back to defaults
  }
}

// Fold a fresh measurement into the persisted EWMA.
function recordTiming(key, ms) {
  if (!Number.isFinite(ms) || ms <= 0) return;
  try {
    const est = loadTimingEstimates();
    est[key] = est[key] * (1 - EWMA_ALPHA) + ms * EWMA_ALPHA;
    localStorage.setItem(TIMINGS_KEY, JSON.stringify(est));
  } catch {
    // storage unavailable (e.g. private mode) — estimates just stay at defaults
  }
}

// Re-weight the bar for a burst of nShots: scale the per-shot stages by shot
// count, keep the constant stages fixed, normalize the widths to [0, 1].
function planStageWidths(nShots) {
  const est = loadTimingEstimates();
  const weight = {
    decode: est.decodePerShot * nShots,
    alignMerge: est.alignMergePerShot * nShots,
    dng: est.dng,
    preview: est.preview,
  };
  const total = weight.decode + weight.alignMerge + weight.dng + weight.preview || 1;
  let acc = 0;
  for (const stage of Object.keys(PROGRESS_STAGES)) {
    const end = acc + weight[stage] / total;
    PROGRESS_STAGES[stage] = [acc, end];
    acc = end;
  }
}

let loadedFrames = []; // [{ name, width, height, data, blackLevel, whiteLevel, cfa, colorMatrix, make, model }]
let decoding = false; // a dropped burst is currently being decoded
let processing = false; // the GPU pipeline is currently running
let settingsDirty = false; // settings changed since the last completed run

// ---- drag & drop + right-pane switching ----
drop.addEventListener('click', () => fileInput.click());
// The whole right pane is a drop target in every state: the intro pane has
// the visible drop zone, and dropping onto the preview starts a new burst
// without leaving the current result first.
stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.classList.add('dropping'); });
stage.addEventListener('dragleave', (e) => { if (!stage.contains(e.relatedTarget)) stage.classList.remove('dropping'); });
stage.addEventListener('drop', (e) => {
  e.preventDefault();
  stage.classList.remove('dropping');
  if (e.dataTransfer.files.length) loadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => loadFiles(fileInput.files));

// The right pane shows exactly one of: intro / progress / preview.
const panes = { intro: introPane, progress: progressPane, preview: previewPane };
function showPane(name) {
  for (const key of Object.keys(panes)) panes[key].hidden = key !== name;
}

// Extensions flashraw is documented to decode (docs/formats.md); DNG subsumes
// most phone RAWs. Decoding itself is format-agnostic — see src/raw.js.
const RAW_EXTENSIONS = new Set([
  'dng', 'nef', 'nrw', 'arw', 'srf', 'sr2', 'arq', 'cr2', 'cr3', 'crw', 'raf',
  'rwl', 'rw2', 'orf', 'ori', 'pef', 'srw', 'mrw', 'iiq', 'mos', '3fr', 'fff',
  'kdc', 'dcr', 'dcs', 'erf', 'mef', 'x3f', 'ari',
]);

async function loadFiles(fileList) {
  const files = [...fileList].filter((f) => RAW_EXTENSIONS.has(f.name.split('.').pop().toLowerCase()));
  const rejected = [...fileList].filter((f) => !RAW_EXTENSIONS.has(f.name.split('.').pop().toLowerCase()));
  if (rejected.length) log(`Skipped ${rejected.length} unsupported file(s): ${rejected.map((f) => f.name).join(', ')}`);
  if (files.length < 2) {
    log('Please provide at least 2 RAW files from the same burst (e.g. .dng, .nef, .arw, .cr3).');
    return;
  }
  const exts = new Set(files.map((f) => f.name.split('.').pop().toLowerCase()));
  if (exts.size > 1) {
    log(`Mixed formats in one burst are not supported (${[...exts].join(', ')}). Use files from a single camera/format.`);
    return;
  }
  if (decoding || processing) {
    log('Still working on the previous burst — wait for it to finish first.');
    return;
  }
  decoding = true;
  logEl.textContent = '';
  loadedFrames = [];
  processBtn.disabled = true;
  downloadLink.classList.add('disabled');
  downloadLink.removeAttribute('href');
  showPane('progress');

  planStageWidths(files.length); // size the bar for this burst

  let decodeMsTotal = 0;
  try {
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      setProgress(stageProgress('decode', i / files.length), `decoding ${i + 1}/${files.length}: ${f.name}`);
      const t = performance.now();
      const frame = await decodeRaw(await f.arrayBuffer(), f.name);
      const ms = Math.round(performance.now() - t);
      decodeMsTotal += ms;

      if (loadedFrames.length > 0) {
        const ref = loadedFrames[0];
        if (frame.width !== ref.width || frame.height !== ref.height) {
          throw new Error(`${f.name}: resolution ${frame.width}x${frame.height} differs from ${ref.width}x${ref.height}`);
        }
      }
      loadedFrames.push({ name: f.name, ...frame });
      log(`${f.name}: ${frame.width}x${frame.height}, black=${JSON.stringify(frame.blackLevel)}, white=${frame.whiteLevel} (${ms} ms)`);
      await yieldToUI();
    }
    log(`Loaded ${loadedFrames.length} frames. Reference frame: ${loadedFrames[Math.floor(loadedFrames.length / 2)].name} (burst center)`);
    recordTiming('decodePerShot', decodeMsTotal / files.length);
    setProgress(stageProgress('decode', 1), 'decoding complete');
  } catch (err) {
    log(`ERROR: ${err.message}`);
    console.error(err);
    setProgress(0, 'decode failed');
    return;
  } finally {
    decoding = false;
  }
  // No click needed: run the pipeline right away. The "Process burst" button
  // stays enabled afterwards for reprocessing with different settings.
  await processBurst();
}

const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// ---- processing ----
async function processBurst() {
  if (decoding || processing || loadedFrames.length < 2) return;
  settingsDirty = false;
  processing = true;
  processBtn.disabled = true;
  try {
    log('Initializing WebGPU…');
    const { adapter, device } = await initGPU();
    log(`Adapter: ${adapter.info?.vendor || '?'} ${adapter.info?.architecture || ''} — max buffer ${(device.limits.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB`);

    const gpu = new GPUContext(device);
    gpu.setShaderModule(WGSL);

    // reference = center frame of the burst (matches Burst Photo default)
    const refIdx = Math.floor(loadedFrames.length / 2);
    const frames = [loadedFrames[refIdx], ...loadedFrames.slice(0, refIdx), ...loadedFrames.slice(refIdx + 1)];

    const options = {
      tileSize: Number(document.getElementById('tileSize').value),
      searchDistance: Number(document.getElementById('searchDistance').value),
      noiseReduction: Number(document.getElementById('noiseReduction').value),
    };
    log(`Aligning & merging ${frames.length} frames (spatial domain, tile=${options.tileSize}, search=${options.searchDistance}, noise=${options.noiseReduction})…`);

    const result = await alignMergeSpatial(gpu, frames, options, (p) => {
      const inner = p.total > 1 ? p.frame / p.total : 1;
      setProgress(stageProgress('alignMerge', inner), `${p.stage} ${p.frame}/${p.total}`);
      log(`${p.stage} ${p.frame}/${p.total}`);
    });

    log(`align+merge: ${result.timings.alignMergeMs.toFixed(0)} ms ` +
        `(prepare ${result.timings.prepareMs.toFixed(0)} ms, finalize ${result.timings.finalizeMs.toFixed(0)} ms) — noise_sd=${result.noiseSd.toFixed(2)}`);
    // Per-shot pipeline cost for the bar's width estimate. Deliberately
    // totalMs (prepare + enqueue + finalize) / frames, not alignMergeMs: the
    // CPU-side loop finishes enqueueing long before the GPU drains, and that
    // drain lands in finalizeMs via the final readBack — totalMs is the wall
    // time the alignMerge bar segment actually spans, per shot.
    recordTiming('alignMergePerShot', result.timings.totalMs / frames.length);

    // ---- DNG output (metadata follows Google's HDR+ convention) ----
    setProgress(stageProgress('dng', 0), 'writing DNG');
    const refMeta = frames[0];
    const n = frames.length;
    const exposureTimes = frames.map((f) => f.exif?.exposureTime).filter((t) => Number.isFinite(t) && t > 0);
    const totalExposure = exposureTimes.length === n
      ? exposureTimes.reduce((a, b) => a + b, 0)
      : null;
    const refIso = refMeta.exif?.iso;
    const exif = {
      // total integration time across the burst (matches Night Sight/HDR+ output)
      exposureTime: totalExposure,
      // effective ISO: reference ISO / number of shots
      iso: Number.isFinite(refIso) && refIso > 0 ? refIso / n : null,
      fNumber: refMeta.exif?.fNumber ?? null,
      focalLength: refMeta.exif?.focalLength ?? null,
      captureTime: refMeta.exif?.captureTime ?? null,
    };
    if (totalExposure) log(`metadata: exposure ${totalExposure.toFixed(3)} s (Σ ${n} frames), ISO ≈ ${Math.round(exif.iso)} (ref ${refIso} / ${n})`);
    const tDng = performance.now();
    const dngBytes = writeDNG(result.pixels, {
      width: result.width,
      height: result.height,
      blackLevel: refMeta.blackLevel,
      whiteLevel: refMeta.whiteLevel,
      cfa: refMeta.cfa,
      colorMatrix: refMeta.colorMatrix,
      whiteBalance: refMeta.whiteBalance,
      whiteBalanceApplied: refMeta.whiteBalanceApplied,
      orientation: refMeta.orientation,
      make: refMeta.make,
      model: refMeta.model,
      exif,
    });
    recordTiming('dng', performance.now() - tDng);
    const blob = new Blob([dngBytes], { type: 'image/x-adobe-dng' });
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.download = refMeta.name.replace(/\.dng$/i, '') + '_merged_web.dng';
    downloadLink.classList.remove('disabled');

    // ---- quick preview: 2x2 CFA blocks -> RGB ----
    // ---- preview: let flashraw fully develop our merged DNG (demosaic + WB + color
    // matrix + sRGB). Falls back to the fast block render if that fails. ----
    setProgress(stageProgress('preview', 0), 'developing preview');
    try {
      const tPrev = performance.now();
      const dev = await developPreview(dngBytes);
      recordTiming('preview', performance.now() - tPrev);
      canvas.width = dev.width;
      canvas.height = dev.height;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(new ImageData(dev.rgba, dev.width, dev.height), 0, 0);
      log('Preview: flashraw-developed from the merged DNG (demosaic, WB, color matrix, sRGB).');
    } catch (err) {
      log(`flashraw preview failed (${err.message}) — using fast block preview instead.`);
      const preview = renderPreview(result.pixels, refMeta);
      canvas.width = preview.width;
      canvas.height = preview.height;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(new ImageData(preview.rgba, preview.width, preview.height), 0, 0);
    }

    showPane('preview');
    setProgress(stageProgress('preview', 1), 'done');
    log('Done. Preview: flashraw-developed; download the DNG for full quality in your RAW converter.');
  } catch (err) {
    log(`ERROR: ${err.message}`);
    console.error(err);
    setProgress(0, 'failed');
  } finally {
    processing = false;
    updateProcessBtn();
  }
}

processBtn.addEventListener('click', processBurst);

// "Process burst" is only clickable when a processed result is out of date:
// greyed before any burst, during decode/processing, and right after a run
// finishes. Changing any setting marks the result stale and re-enables it.
function updateProcessBtn() {
  processBtn.disabled = decoding || processing || loadedFrames.length < 2 || !settingsDirty;
}
for (const id of ['noiseReduction', 'tileSize', 'searchDistance']) {
  document.getElementById(id).addEventListener('change', () => {
    settingsDirty = true;
    updateProcessBtn();
  });
}
