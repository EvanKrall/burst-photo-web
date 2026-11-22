// Minimal zero-dependency static file server for the PoC.
// Usage: node server.mjs [port]          (default 8080, serves repo root)
//        ROOT=dist node server.mjs 8080  (serve the built dist/ output)
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = process.env.ROOT
  ? resolve(process.env.ROOT)
  : fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.argv[2]) || 8080;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.dng': 'image/x-adobe-dng',
};

createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (urlPath === '/') urlPath = '/index.html';
    const filePath = normalize(join(root, urlPath));
    if (!filePath.startsWith(root)) throw new Error('path traversal');
    const data = await readFile(filePath);
    res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
}).listen(port, () => {
  console.log(`burst-photo-web running at http://localhost:${port}`);
});
