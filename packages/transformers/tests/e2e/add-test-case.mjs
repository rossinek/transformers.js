#!/usr/bin/env node

/**
 * Add a test case to the Whisper benchmark.
 *
 * Usage:
 *   node add-test-case.mjs <input-file> <name> [--lang <code>]
 *
 * Example:
 *   node add-test-case.mjs ~/Downloads/interview.mp4 interview --lang en
 *
 * This will:
 *   1. Convert the input to 16kHz mono f32le WAV with ffmpeg
 *   2. Transcribe with ElevenLabs Scribe API (requires ELEVENLABS_API_KEY env var)
 *   3. Save the WAV to fixtures/<name>.wav
 *   4. Save ground truth to ground-truth/<name>.json
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { join, basename, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomBytes } from "node:crypto";

const execFileAsync = promisify(execFile);
const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "../../../..");
const FIXTURES_DIR = join(__dirname, "fixtures");
const GROUND_TRUTH_DIR = join(__dirname, "ground-truth");

// Load .env from repo root
const envPath = join(REPO_ROOT, ".env");
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!process.env[key]) process.env[key] = val;
  }
}

function parseArgs() {
  const args = process.argv.slice(2);
  let inputFile = null;
  let name = null;
  let lang = "en";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--lang" && args[i + 1]) {
      lang = args[i + 1];
      i++;
    } else if (!inputFile) {
      inputFile = args[i];
    } else if (!name) {
      name = args[i];
    }
  }

  if (!inputFile || !name) {
    console.error("Usage: node add-test-case.mjs <input-file> <name> [--lang <code>]");
    console.error("  Requires ELEVENLABS_API_KEY env var.");
    process.exit(1);
  }

  return { inputFile, name, lang };
}

async function convertToWav(inputFile, outputPath) {
  console.log(`Converting to WAV: ${inputFile} -> ${outputPath}`);
  await execFileAsync("ffmpeg", [
    "-y", "-i", inputFile,
    "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le",
    outputPath,
  ]);
  console.log("  Done.");
}

async function transcribeWithElevenLabs(wavPath, lang) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    console.error("Error: ELEVENLABS_API_KEY env var not set.");
    process.exit(1);
  }

  console.log(`Transcribing with ElevenLabs Scribe (lang=${lang})...`);

  const fileData = await readFile(wavPath);
  const fileName = basename(wavPath);

  const boundary = "----Boundary" + randomBytes(8).toString("hex");
  const parts = [];

  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="file"; filename="${fileName}"\r\n` +
    `Content-Type: audio/wav\r\n\r\n`
  );
  parts.push(fileData);
  parts.push("\r\n");

  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="model_id"\r\n\r\n` +
    `scribe_v2\r\n`
  );

  parts.push(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="language_code"\r\n\r\n` +
    `${lang}\r\n`
  );

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
    console.error(`ElevenLabs API error ${resp.status}: ${text}`);
    process.exit(1);
  }

  const result = await resp.json();
  console.log(`  Got ${(result.words || []).length} raw chunks.`);

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

async function main() {
  const { inputFile, name, lang } = parseArgs();
  const wavPath = join(FIXTURES_DIR, `${name}.wav`);
  const gtPath = join(GROUND_TRUTH_DIR, `${name}.json`);

  // Ensure directories exist
  await mkdir(FIXTURES_DIR, { recursive: true });
  await mkdir(GROUND_TRUTH_DIR, { recursive: true });

  // Step 1: Convert
  await convertToWav(inputFile, wavPath);

  // Step 2: Transcribe
  const gt = await transcribeWithElevenLabs(wavPath, lang);

  // Step 3: Save ground truth
  await writeFile(gtPath, JSON.stringify(gt, null, 2));
  console.log(`\nSaved:`);
  console.log(`  WAV:          ${wavPath}`);
  console.log(`  Ground truth: ${gtPath}`);
  console.log(`  Text preview: ${gt.text.slice(0, 100)}...`);
  console.log(`  Words:        ${gt.words.length}`);
}

main();
