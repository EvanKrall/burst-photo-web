// UI orchestration: file loading, flashraw decode, WebGPU pipeline, preview + DNG output.

import { initGPU, GPUContext } from './gpu.js';
import { WGSL } from './kernels.js';
import { alignMergeSpatial } from './pipeline.js';
import { decodeDNG } from './raw.js';
import { writeDNG } from './dng_writer.js';

const logEl = document.getElementById('log');
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('process');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultEl = document.getElementById('result');
const canvas = document.getElementById('canvas');
const downloadLink = document.getElementById('download');

function log(msg) {
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

function setProgress(fraction, text) {
  progressWrap.style.display = 'block';
  progressFill.style.width = `${Math.round(fraction * 100)}%`;
  if (text) progressText.textContent = text;
}

let loadedFrames = []; // [{ name, width, height, data, blackLevel, whiteLevel, cfa, colorMatrix, make, model }]

// ---- drag & drop ----
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('hover'); loadFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => loadFiles(fileInput.files));

async function loadFiles(fileList) {
  const files = [...fileList].filter((f) => /\.(dng|DNG)$/.test(f.name));
  if (files.length < 2) {
    log('Please provide at least 2 DNG files from the same burst.');
    return;
  }
  logEl.textContent = '';
  loadedFrames = [];
  processBtn.disabled = true;
  resultEl.style.display = 'none';

  try {
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      setProgress((i / files.length) * 0.5, `decoding ${i + 1}/${files.length}: ${f.name}`);
      const t = performance.now();
      const frame = await decodeDNG(await f.arrayBuffer(), f.name);
      const ms = Math.round(performance.now() - t);

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
    processBtn.disabled = false;
    setProgress(0.5, 'decoding complete');
  } catch (err) {
    log(`ERROR: ${err.message}`);
    console.error(err);
    setProgress(0, 'decode failed');
  }
}

const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// ---- processing ----
processBtn.addEventListener('click', async () => {
  if (loadedFrames.length < 2) return;
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
      setProgress(0.5 + inner * 0.4, `${p.stage} ${p.frame}/${p.total}`);
      log(`${p.stage} ${p.frame}/${p.total}`);
    });

    log(`align+merge: ${result.timings.alignMergeMs.toFixed(0)} ms ` +
        `(prepare ${result.timings.prepareMs.toFixed(0)} ms, finalize ${result.timings.finalizeMs.toFixed(0)} ms) — noise_sd=${result.noiseSd.toFixed(2)}`);

    // ---- DNG output ----
    setProgress(0.95, 'writing DNG');
    const refMeta = frames[0];
    const dngBytes = writeDNG(result.pixels, {
      width: result.width,
      height: result.height,
      blackLevel: refMeta.blackLevel,
      whiteLevel: refMeta.whiteLevel,
      cfa: refMeta.cfa,
      colorMatrix: refMeta.colorMatrix,
      make: refMeta.make,
      model: refMeta.model,
    });
    const blob = new Blob([dngBytes], { type: 'image/x-adobe-dng' });
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.download = refMeta.name.replace(/\.dng$/i, '') + '_merged_web.dng';

    // ---- quick preview: 2x2 CFA blocks -> RGB ----
    const scale = Math.max(1, Math.ceil(result.width / 1600)); // cap preview width
    const pw = Math.floor(result.width / (2 * scale));
    const ph = Math.floor(result.height / (2 * scale));
    canvas.width = pw;
    canvas.height = ph;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(pw, ph);
    const px = result.pixels;
    const white = Math.max(1, refMeta.whiteLevel || 65535);
    const black = Math.max(0, Math.min(...(Array.isArray(refMeta.blackLevel) ? refMeta.blackLevel : [refMeta.blackLevel])));
    const norm = (v) => Math.max(0, Math.min(255, Math.round(((v - black) / (white - black)) * 255)));
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        const sx = x * 2 * scale;
        const sy = y * 2 * scale;
        const r = px[sy * result.width + sx];
        const g1 = px[sy * result.width + sx + 1];
        const g2 = px[(sy + 1) * result.width + sx];
        const b = px[(sy + 1) * result.width + sx + 1];
        const o = (y * pw + x) * 4;
        img.data[o] = norm(r);
        img.data[o + 1] = norm((g1 + g2) / 2);
        img.data[o + 2] = norm(b);
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    resultEl.style.display = 'block';
    setProgress(1, 'done');
    log('Done. Preview shows a fast 2x2-block render; download the DNG for full quality.');
  } catch (err) {
    log(`ERROR: ${err.message}`);
    console.error(err);
    setProgress(0, 'failed');
  } finally {
    processBtn.disabled = false;
  }
});
