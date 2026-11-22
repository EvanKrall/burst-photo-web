#!/usr/bin/env node
// Build a deployable static site into dist/ (cross-platform, no shell required).
//   npm install && npm run build   →  dist/
//
// dist layout:
//   dist/index.html
//   dist/src/*.js                       (app modules, ES modules — relative imports)
//   dist/vendor/flashraw/dist/**        (flashraw JS glue + .wasm binaries;
//                                        resolved relative to dist/src/raw.js)
//
// Serve dist/ over HTTPS with .wasm mapped to application/wasm. For a local
// check of exactly this layout: ROOT=dist npm start

import { cp, mkdir, rm, readdir, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');

async function sizeKB(path) {
  const s = await stat(path);
  return s.isDirectory() ? 'dir' : `${(s.size / 1024).toFixed(0)} KB`;
}

async function main() {
  // fresh build
  await rm(dist, { recursive: true, force: true });
  await mkdir(dist, { recursive: true });

  // app files
  await cp(join(root, 'index.html'), join(dist, 'index.html'));
  await cp(join(root, 'src'), join(dist, 'src'), { recursive: true });

  // flashraw (vendored: JS glue + .wasm binaries are import.meta.url assets and
  // must sit in the same directory relative to dist/src/raw.js as deployed)
  const flashrawDist = join(root, 'node_modules', 'flashraw', 'dist');
  try {
    await readdir(flashrawDist);
  } catch {
    console.error('node_modules/flashraw not found — run `npm install` first.');
    process.exit(1);
  }
  await cp(flashrawDist, join(dist, 'vendor', 'flashraw', 'dist'), { recursive: true });

  // exiftool (vendored: the bare '@6over3/zeroperl-ts' specifier is rewritten to a
  // relative path by scripts/vendor.mjs; the 25 MB WASM Perl interpreter is a lazy
  // dynamic import from src/af_area.js and must sit next to the rewritten module)
  const exiftoolVendor = join(root, 'vendor', 'exiftool');
  try {
    await stat(join(exiftoolVendor, 'zeroperl.wasm'));
  } catch {
    console.error('vendor/exiftool not found — run `npm run vendor` (npm install does this automatically).');
    process.exit(1);
  }
  await cp(exiftoolVendor, join(dist, 'vendor', 'exiftool'), { recursive: true });

  // summary
  console.log('dist/ built:');
  for (const entry of ['index.html', 'src', 'vendor/flashraw/dist', 'vendor/exiftool']) {
    console.log(`  ${entry.padEnd(28)} ${await sizeKB(join(dist, entry))}`);
  }
  const wasm = (await readdir(join(dist, 'vendor', 'flashraw', 'dist'))).filter((f) => f.endsWith('.wasm'));
  console.log(`  wasm binaries: ${wasm.join(', ')}`);
  console.log('done. serve over HTTPS (WebGPU requires a secure context); .wasm must map to application/wasm.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
