import { createServer } from "node:http";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import { randomBytes } from "node:crypto";
import { existsSync, readFileSync as readFileSyncFS } from "node:fs";

const execFileAsync = promisify(execFile);

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "../../../..");

// Load .env from repo root
const envPath = join(REPO_ROOT, ".env");
if (existsSync(envPath)) {
  for (const line of readFileSyncFS(envPath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!process.env[key]) process.env[key] = val;
  }
}
const PKG_ROOT = join(__dirname, "../..");
const FIXTURES_DIR = join(PKG_ROOT, "tests/e2e/fixtures");
const GROUND_TRUTH_DIR = join(PKG_ROOT, "tests/e2e/ground-truth");

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".wasm": "application/wasm",
  ".mov": "video/quicktime",
  ".mp4": "video/mp4",
  ".wav": "audio/wav",
  ".json": "application/json",
  ".css": "text/css",
};

const ROUTES = [
  { prefix: "/dist/", dir: join(PKG_ROOT, "dist") },
  { prefix: "/wasm/", dir: join(PKG_ROOT, "node_modules/onnxruntime-web/dist") },
  { prefix: "/fixtures/", dir: FIXTURES_DIR },
  { prefix: "/ground-truth/", dir: GROUND_TRUTH_DIR },
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

async function serveFile(req, res, filePath) {
  try {
    const data = await readFile(filePath);
    const ext = extname(filePath);
    const mime = MIME_TYPES[ext] || "application/octet-stream";
    const total = data.length;

    const headers = {
      "Content-Type": mime,
      "Accept-Ranges": "bytes",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    };

    // Handle range requests (needed for audio seeking)
    const range = req.headers.range;
    if (range) {
      const match = range.match(/bytes=(\d+)-(\d*)/);
      if (match) {
        const start = parseInt(match[1], 10);
        const end = match[2] ? parseInt(match[2], 10) : total - 1;
        headers["Content-Range"] = `bytes ${start}-${end}/${total}`;
        headers["Content-Length"] = end - start + 1;
        res.writeHead(206, headers);
        res.end(data.subarray(start, end + 1));
        return;
      }
    }

    headers["Content-Length"] = total;
    res.writeHead(200, headers);
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

// POST /api/add-test-case — multipart upload, ffmpeg convert, ElevenLabs transcribe
async function handleAddTestCase(req, res) {
  try {
    const contentType = req.headers["content-type"] || "";
    if (!contentType.includes("multipart/form-data")) {
      res.writeHead(400, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ error: "Expected multipart/form-data" }));
    }

    const boundary = contentType.split("boundary=")[1];
    const body = await readBody(req);
    const { fields, files } = parseMultipart(body, boundary);

    const name = fields.name?.trim();
    const lang = fields.lang?.trim() || "en";
    if (!name) {
      res.writeHead(400, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ error: "Missing 'name' field" }));
    }
    if (!files.file) {
      res.writeHead(400, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ error: "Missing 'file' upload" }));
    }

    // Write uploaded file to temp
    const tmpId = randomBytes(6).toString("hex");
    const tmpInput = join(tmpdir(), `bench-input-${tmpId}${extname(files.file.filename) || ".bin"}`);
    const wavOutput = join(FIXTURES_DIR, `${name}.wav`);

    await writeFile(tmpInput, files.file.data);

    // Convert with ffmpeg
    await execFileAsync("ffmpeg", [
      "-y", "-i", tmpInput,
      "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le",
      wavOutput,
    ]);

    // Transcribe with ElevenLabs Scribe
    const gtResult = await transcribeWithElevenLabs(wavOutput, lang);

    // Save ground truth
    await mkdir(GROUND_TRUTH_DIR, { recursive: true });
    await writeFile(
      join(GROUND_TRUTH_DIR, `${name}.json`),
      JSON.stringify(gtResult, null, 2),
    );

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, name, wordCount: gtResult.words.length }));
  } catch (e) {
    console.error("add-test-case error:", e);
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: e.message }));
  }
}

// ElevenLabs Scribe API
async function transcribeWithElevenLabs(wavPath, lang) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) throw new Error("ELEVENLABS_API_KEY env var not set");

  const fileData = await readFile(wavPath);
  const fileName = basename(wavPath);

  // Build multipart body manually
  const boundary = "----ElevenLabsBoundary" + randomBytes(8).toString("hex");
  const parts = [];

  // File part
  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="file"; filename="${fileName}"\r\n` +
    `Content-Type: audio/wav\r\n\r\n`
  );
  parts.push(fileData);
  parts.push("\r\n");

  // Model ID (required)
  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="model_id"\r\n\r\n` +
    `scribe_v2\r\n`
  );

  // Language part
  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="language_code"\r\n\r\n` +
    `${lang}\r\n`
  );

  // Timestamps granularity
  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="timestamps_granularity"\r\n\r\n` +
    `word\r\n`
  );

  parts.push(`--${boundary}--\r\n`);

  const bodyParts = parts.map(p => typeof p === "string" ? Buffer.from(p) : p);
  const bodyBuffer = Buffer.concat(bodyParts);

  const resp = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
    method: "POST",
    headers: {
      "xi-api-key": apiKey,
      "Content-Type": `multipart/form-data; boundary=${boundary}`,
    },
    body: bodyBuffer,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`ElevenLabs API error ${resp.status}: ${text}`);
  }

  const result = await resp.json();

  return normalizeGroundTruth(result, lang);
}

function normalizeGroundTruth(result, lang) {
  const words = (result.words || [])
    .filter(w => w.text && w.text.trim().length > 0)
    .map(w => ({
      text: w.text.trim(),
      start: w.start,
      end: w.end,
    }));

  return {
    text: result.text || "",
    language_code: lang,
    words,
  };
}

// Minimal multipart parser
function parseMultipart(buffer, boundary) {
  const fields = {};
  const files = {};
  const boundaryBuf = Buffer.from(`--${boundary}`);

  // Split by boundary
  let pos = 0;
  const parts = [];
  while (true) {
    const idx = buffer.indexOf(boundaryBuf, pos);
    if (idx === -1) break;
    if (pos > 0) parts.push(buffer.subarray(pos, idx - 2)); // -2 for \r\n before boundary
    pos = idx + boundaryBuf.length;
    if (buffer[pos] === 0x2d && buffer[pos + 1] === 0x2d) break; // --
    pos += 2; // skip \r\n
  }

  for (const part of parts) {
    const headerEnd = part.indexOf("\r\n\r\n");
    if (headerEnd === -1) continue;
    const headerStr = part.subarray(0, headerEnd).toString("utf-8");
    const data = part.subarray(headerEnd + 4);

    const nameMatch = headerStr.match(/name="([^"]+)"/);
    const filenameMatch = headerStr.match(/filename="([^"]+)"/);

    if (!nameMatch) continue;
    const name = nameMatch[1];

    if (filenameMatch) {
      files[name] = { filename: filenameMatch[1], data };
    } else {
      fields[name] = data.toString("utf-8");
    }
  }
  return { fields, files };
}

// List available test cases
async function handleListCases(req, res) {
  const { readdir } = await import("node:fs/promises");
  try {
    const wavFiles = (await readdir(FIXTURES_DIR)).filter(f => f.endsWith(".wav"));
    const cases = [];
    for (const wav of wavFiles) {
      const name = wav.replace(".wav", "");
      const gtPath = join(GROUND_TRUTH_DIR, `${name}.json`);
      let hasGT = false;
      try { await readFile(gtPath); hasGT = true; } catch {}
      cases.push({ name, file: wav, hasGroundTruth: hasGT });
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(cases));
  } catch (e) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: e.message }));
  }
}

const { port } = parseArgs();

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${port}`);
  const pathname = url.pathname;

  // API routes
  if (req.method === "POST" && pathname === "/api/add-test-case") {
    return handleAddTestCase(req, res);
  }
  if (req.method === "GET" && pathname === "/api/cases") {
    return handleListCases(req, res);
  }

  // Root → benchmark page
  if (pathname === "/") {
    return serveFile(req, res, join(FIXTURES_DIR, "benchmark.html"));
  }

  // Static routes
  for (const route of ROUTES) {
    if (pathname.startsWith(route.prefix)) {
      const relativePath = pathname.slice(route.prefix.length);
      return serveFile(req, res, join(route.dir, relativePath));
    }
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(port, () => {
  console.log(`Benchmark server listening on http://localhost:${port}`);
});
