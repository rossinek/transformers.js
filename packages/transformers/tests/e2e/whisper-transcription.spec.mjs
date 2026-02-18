import { test as base, expect } from "@playwright/test";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROFILE_DIR = join(__dirname, ".cache", "chromium-profile");

// Use a persistent browser context so Cache API / IndexedDB survive between runs.
// This avoids re-downloading the ~1.7GB model on every test invocation.
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
    // Apply baseURL from config — persistent context doesn't inherit it
    page._baseURL = baseURL;
    await use(page);
  },
});

test("whisper large model transcribes video with timestamps", async ({ page, baseURL }) => {
  // 1. Navigate to test page and wait for it to be ready
  await page.goto(baseURL + "/");
  await page.waitForFunction(() => window.__READY__ === true, null, { timeout: 30_000 });

  // 2. In browser: extract audio from example.mov
  await page.evaluate(() => window.setStatus("Extracting audio from video..."));
  const audioLength = await page.evaluate(async () => {
    const processor = new window.AudioProcessor();
    window._audio = await processor.processMediaUrl("/fixtures/example.mov");
    return window._audio.length;
  });
  await page.evaluate((len) => {
    const secs = (len / 16000).toFixed(1);
    window.setDone(`Audio extracted: ${(len).toLocaleString()} samples (${secs}s)`);
  }, audioLength);
  expect(audioLength).toBeGreaterThan(0);

  // 3. In browser: load model and run transcription
  const results = await page.evaluate(async () => {
    window.setStatus("Loading model (onnx-community/whisper-large-v3-turbo_timestamped)...");

    const { pipeline } = window.transformers;
    const transcriber = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-large-v3-turbo_timestamped",
      {
        dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
        device: "webgpu",
      },
    );

    window.setStatus("Transcribing audio to segments...");

    const outputSegments = await transcriber(window._audio, {
      language: "en",
      return_timestamps: true,
      chunk_length_s: 30,
      stride_length_s: 5,
    });

    window.setStatus("Transcribing audio to words...");

    const outputWords = await transcriber(window._audio, {
      language: "en",
      return_timestamps: "word",
      chunk_length_s: 30,
      stride_length_s: 5,
    });

    window.setDone(`Transcription complete — ${outputSegments.chunks.length} segments and ${outputWords.chunks.length} words`);
    window.showResult(outputSegments.text);

    return { segments: outputSegments, words: outputWords };
  }, { timeout: 600_000 });

  // 4. Assert output format
  await page.evaluate(() => window.setStatus("Validating output..."));

  const { segments, words } = results;

  for (const result of [segments, words]) {
    expect(result).toHaveProperty("text");
    expect(typeof result.text).toBe("string");
    expect(result.text.length).toBeGreaterThan(0);

    expect(result).toHaveProperty("chunks");
    expect(Array.isArray(result.chunks)).toBe(true);
    expect(result.chunks.length).toBeGreaterThan(0);

    for (const chunk of result.chunks) {
      expect(chunk).toHaveProperty("text");
      expect(typeof chunk.text).toBe("string");
      expect(chunk).toHaveProperty("timestamp");
      expect(Array.isArray(chunk.timestamp)).toBe(true);
      expect(chunk.timestamp).toHaveLength(2);
      expect(typeof chunk.timestamp[0]).toBe("number");
      // timestamp[1] can be number or null (last chunk)
    }
  }

  // expect words to be more or less within the segments
  const TIME_PRECISION = 0.25;
  for (const word of words.chunks) {
    const segment = segments.chunks.find(segment =>
      segment.timestamp[0] <= word.timestamp[0] + TIME_PRECISION
      && segment.timestamp[1] >= word.timestamp[1] - TIME_PRECISION
    );
    expect(segment, `Word "${JSON.stringify(word)}" is not within a segment`).toBeDefined();
  }

  const COUNT_PRECISION_PERCENT = 0.2;
  // expect segments to have more or less the same number of word chunks that it has words
  for (const segment of segments.chunks) {
    const wordChunks = words.chunks.filter(word =>
      segment.timestamp[0] <= word.timestamp[0] + TIME_PRECISION
      && segment.timestamp[1] >= word.timestamp[1] - TIME_PRECISION);
    const segmentWordCount = segment.text.split(/\s+/).length;
    const wordChunkCount = wordChunks.length;
    expect(
      Math.abs(segmentWordCount - wordChunkCount) / (segmentWordCount || 1) < COUNT_PRECISION_PERCENT,
      `Segment "${JSON.stringify(segment)}" has ${wordChunkCount} overlapping word chunks, but should have ${segmentWordCount}`
    ).toBe(true);
  }

  await page.evaluate(() => window.setDone("All assertions passed"));
});
