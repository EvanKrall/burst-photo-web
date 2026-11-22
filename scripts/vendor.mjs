#!/usr/bin/env node
// Materialize vendor/exiftool/ from node_modules (gitignored, like vendor/flashraw).
//
// The published @uswriting/exiftool ESM build keeps its only dependency as a bare
// specifier (`import ... from "@6over3/zeroperl-ts"`), which browsers cannot resolve.
// We copy both ESM builds side by side and rewrite the bare specifier to a relative
// path. The 25 MB zeroperl.wasm (WASM Perl interpreter with exiftool's library
// embedded) is copied next to it; src/af_area.js overrides the wasm fetch URL so it
// resolves regardless of the page path.

import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'vendor', 'exiftool');
const exiftoolEsm = join(root, 'node_modules', '@uswriting', 'exiftool', 'dist', 'esm', 'index.js');
const zeroperlEsm = join(root, 'node_modules', '@6over3', 'zeroperl-ts', 'dist', 'esm', 'index.js');
const zeroperlWasm = join(root, 'node_modules', '@6over3', 'zeroperl-ts', 'dist', 'esm', 'zeroperl.wasm');

let js = await readFile(exiftoolEsm, 'utf8');
const patched = js.replace(/from\s*"@6over3\/zeroperl-ts"/g, 'from "./zeroperl.js"');
if (patched === js) {
  console.error('vendor: failed to patch the @6over3/zeroperl-ts specifier — upstream layout changed?');
  process.exit(1);
}
await mkdir(out, { recursive: true });
await writeFile(join(out, 'exiftool.js'), patched);
await copyFile(zeroperlEsm, join(out, 'zeroperl.js'));
await copyFile(zeroperlWasm, join(out, 'zeroperl.wasm'));
console.log('vendor/exiftool ready (exiftool.js, zeroperl.js, zeroperl.wasm)');
