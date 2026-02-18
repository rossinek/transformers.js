import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const PKG_ROOT = join(__dirname, "../..");

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".wasm": "application/wasm",
  ".mov": "video/quicktime",
  ".mp4": "video/mp4",
  ".json": "application/json",
  ".css": "text/css",
};

const ROUTES = [
  { prefix: "/dist/", dir: join(PKG_ROOT, "dist") },
  { prefix: "/wasm/", dir: join(PKG_ROOT, "node_modules/onnxruntime-web/dist") },
  { prefix: "/fixtures/", dir: join(PKG_ROOT, "tests/e2e/fixtures") },
];

function parseArgs() {
  const args = process.argv.slice(2);
  let port = 8484;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port" && args[i + 1]) {
      port = parseInt(args[i + 1], 10);
    }
  }
  return { port };
}

async function serveFile(res, filePath) {
  try {
    const data = await readFile(filePath);
    const ext = extname(filePath);
    const mime = MIME_TYPES[ext] || "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": mime,
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}

const { port } = parseArgs();

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${port}`);
  const pathname = url.pathname;

  // Root → index.html
  if (pathname === "/") {
    return serveFile(res, join(PKG_ROOT, "tests/e2e/fixtures/index.html"));
  }

  // Matched routes
  for (const route of ROUTES) {
    if (pathname.startsWith(route.prefix)) {
      const relativePath = pathname.slice(route.prefix.length);
      return serveFile(res, join(route.dir, relativePath));
    }
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(port, () => {
  console.log(`Static server listening on http://localhost:${port}`);
});
