import { test as base, expect } from "@playwright/test";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync, existsSync, readFileSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROFILE_DIR = join(__dirname, ".cache", "chromium-profile");
const REPO_ROOT = join(__dirname, "../../../..");
const SNAPSHOT_PATH = join(REPO_ROOT, "whisper-test-snapshots.json");
const TIMING_PATH = join(REPO_ROOT, "whisper-test-timing.json");

const test = base.extend({
  context: async ({ playwright }, use) => {
    const context = await playwright.chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
      ],
    });
    await use(context);
    await context.close();
  },
  page: async ({ context, baseURL }, use) => {
    const page = context.pages()[0] || await context.newPage();
    page._baseURL = baseURL;
    await use(page);
  },
});

const TEST_FILES = [
  { name: "office", file: "office.mp4", description: "short clip (~3.5s)" },
  { name: "lucy", file: "lucy.mp4", description: "medium clip (~52s)" },
  { name: "regan-joke", file: "regan-joke.mp4", description: "longer clip (~61s)" },
];

// Mode: "record" to generate snapshots, "verify" to check against them
const MODE = existsSync(SNAPSHOT_PATH) ? "verify" : "record";

test("whisper word-level transcription snapshot test", async ({ page, baseURL }) => {
  await page.goto(baseURL + "/");
  await page.waitForFunction(() => window.__READY__ === true, null, { timeout: 30_000 });

  // Load the model once
  await page.evaluate(async () => {
    const { pipeline } = window.transformers;
    window._transcriber = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-large-v3-turbo_timestamped",
      {
        dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
        device: "webgpu",
      },
    );
  }, { timeout: 300_000 });

  const results = {};
  const timings = {};

  for (const testFile of TEST_FILES) {
    console.log(`\n=== Processing: ${testFile.name} (${testFile.description}) ===`);

    const result = await page.evaluate(async ({ file }) => {
      const processor = new window.AudioProcessor();
      const audio = await processor.processMediaUrl(`/fixtures/${file}`);
      const audioLengthS = (audio.length / 16000).toFixed(1);

      const start = performance.now();
      const output = await window._transcriber(audio, {
        language: "en",
        return_timestamps: "word",
        chunk_length_s: 30,
        stride_length_s: 5,
      });
      const elapsed = performance.now() - start;

      return {
        audioLengthS: parseFloat(audioLengthS),
        transcriptionTimeMs: Math.round(elapsed),
        text: output.text,
        chunks: output.chunks.map(c => ({
          text: c.text,
          timestamp: [
            Math.round(c.timestamp[0] * 100) / 100,
            c.timestamp[1] != null ? Math.round(c.timestamp[1] * 100) / 100 : null,
          ],
        })),
      };
    }, { file: testFile.file });

    console.log(`  Audio: ${result.audioLengthS}s`);
    console.log(`  Transcription time: ${result.transcriptionTimeMs}ms`);
    console.log(`  Words: ${result.chunks.length}`);
    console.log(`  Text: ${result.text.slice(0, 100)}...`);

    results[testFile.name] = {
      text: result.text,
      chunks: result.chunks,
    };

    timings[testFile.name] = {
      audioLengthS: result.audioLengthS,
      transcriptionTimeMs: result.transcriptionTimeMs,
      wordCount: result.chunks.length,
    };
  }

  if (MODE === "record") {
    console.log("\n=== RECORDING SNAPSHOTS ===");
    writeFileSync(SNAPSHOT_PATH, JSON.stringify(results, null, 2));
    writeFileSync(TIMING_PATH, JSON.stringify(timings, null, 2));
    console.log(`Snapshots saved to: ${SNAPSHOT_PATH}`);
    console.log(`Timings saved to: ${TIMING_PATH}`);
  } else {
    console.log("\n=== VERIFYING AGAINST SNAPSHOTS ===");
    const snapshots = JSON.parse(readFileSync(SNAPSHOT_PATH, "utf-8"));

    // Log timings to console only — don't write files during verify
    console.log("\nTimings (this run):");
    for (const [name, t] of Object.entries(timings)) {
      console.log(`  ${name}: ${t.transcriptionTimeMs}ms (${t.audioLengthS}s audio, ${t.wordCount} words)`);
    }

    for (const testFile of TEST_FILES) {
      const snapshot = snapshots[testFile.name];
      const actual = results[testFile.name];

      console.log(`\n--- ${testFile.name} ---`);

      // Check full text equality
      expect(actual.text, `${testFile.name}: full text mismatch`).toBe(snapshot.text);

      // Check chunk count
      expect(
        actual.chunks.length,
        `${testFile.name}: word count changed (${actual.chunks.length} vs ${snapshot.chunks.length})`,
      ).toBe(snapshot.chunks.length);

      // Check each chunk
      for (let i = 0; i < snapshot.chunks.length; i++) {
        const sc = snapshot.chunks[i];
        const ac = actual.chunks[i];
        expect(ac.text, `${testFile.name} word ${i}: text mismatch`).toBe(sc.text);
        expect(ac.timestamp[0], `${testFile.name} word ${i}: start time`).toBe(sc.timestamp[0]);
        expect(ac.timestamp[1], `${testFile.name} word ${i}: end time`).toBe(sc.timestamp[1]);
      }

      console.log(`  PASS: text matches (${actual.chunks.length} words)`);
    }
  }
});
