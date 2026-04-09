/**
 * Whisper Benchmark — automated quality validation.
 *
 * Usage:
 *   pnpm build
 *   npx playwright test benchmark                        # run all clips
 *   npx playwright test benchmark --grep "single"        # run one clip (set CLIP env var)
 *   CLIP=en-space npx playwright test benchmark --grep "single"
 *
 * The test loads the benchmark page in autorun mode, which:
 *   - Loads the Whisper model (WebGPU)
 *   - Transcribes test clips in word + sentence modes
 *   - Computes quality metrics against ElevenLabs ground truth
 *   - Exposes full diff (aligned words with match/mismatch status)
 *
 * Thresholds below define what counts as a regression.
 * Adjust per-clip overrides in CLIP_THRESHOLDS as baselines improve.
 */

import { test, expect } from "@playwright/test";

// Default thresholds — any clip not in CLIP_THRESHOLDS uses these
const DEFAULT_THRESHOLDS = {
  maxWER: 0.25,           // 25% word error rate
  maxTimestampMAE: 1.5,   // 1.5 seconds mean absolute error
  maxMonotonicity: 2,     // at most 2 monotonicity violations
  textMatch: true,        // word and sentence mode text must match
};

// Per-clip threshold overrides (merge with defaults)
const CLIP_THRESHOLDS = {
  // Example:
  // "en-gem": { maxWER: 0.10 },
};

function getThresholds(clipName) {
  return { ...DEFAULT_THRESHOLDS, ...(CLIP_THRESHOLDS[clipName] || {}) };
}

// Helper: load page, wait for autorun to finish, return { metrics, results }
async function runBenchmark(page, clipName) {
  const url = clipName
    ? `/?autorun=true&clip=${encodeURIComponent(clipName)}`
    : "/?autorun=true";

  await page.goto(url);
  await page.waitForFunction(() => window.__READY__ === true, null, { timeout: 30_000 });
  await page.waitForFunction(() => window.__DONE__ === true, null, { timeout: 600_000 });

  const error = await page.evaluate(() => window.__ERROR__);
  if (error) throw new Error(error);

  const metrics = await page.evaluate(() => window.__METRICS__);
  const results = await page.evaluate(() => window.__RESULTS__);
  return { metrics, results };
}

function printMetrics(name, m) {
  const werPct = (m.wer * 100).toFixed(1);
  const maeStr = m.mae != null ? m.mae.toFixed(3) + "s" : "N/A";
  console.log(`  ${name}: WER=${werPct}% MAE=${maeStr} Mono=${m.monotonicity} W/S=${m.textMatch}`);
}

function printDiff(diff) {
  if (!diff) return;
  const mismatches = diff.filter(d => d.status !== "match");
  if (mismatches.length === 0) {
    console.log("  No mismatches.");
    return;
  }
  console.log(`  ${mismatches.length} mismatch(es):`);
  for (const d of mismatches) {
    if (d.status === "missing") {
      console.log(`    MISSING: "${d.gt.text}" (GT ${d.gt.start?.toFixed(2)}s-${d.gt.end?.toFixed(2)}s) — not in model output`);
    } else if (d.status === "extra") {
      console.log(`    EXTRA:   "${d.model.text}" (Model ${d.model.start?.toFixed(2)}s-${d.model.end?.toFixed(2)}s) — not in ground truth`);
    } else if (d.status === "substitution") {
      console.log(`    SUBST:   GT "${d.gt.text}" (${d.gt.start?.toFixed(2)}s) → Model "${d.model.text}" (${d.model.start?.toFixed(2)}s)`);
    }
  }
}

// ── Test: all clips ──
test("whisper benchmark — all clips pass quality thresholds", async ({ page }) => {
  const { metrics, results } = await runBenchmark(page);
  expect(metrics, "No metrics returned").toBeTruthy();

  const clipNames = Object.keys(metrics);
  expect(clipNames.length, "No clips were tested").toBeGreaterThan(0);

  console.log(`\nBenchmark results (${clipNames.length} clips):`);
  console.log("─".repeat(70));

  const failures = [];

  for (const name of clipNames) {
    const m = metrics[name];
    const t = getThresholds(name);
    printMetrics(name, m);

    const clipFailures = [];
    if (m.wer > t.maxWER) clipFailures.push(`WER ${(m.wer * 100).toFixed(1)}% > ${(t.maxWER * 100).toFixed(0)}%`);
    if (m.mae != null && m.mae > t.maxTimestampMAE) clipFailures.push(`MAE ${m.mae.toFixed(3)}s > ${t.maxTimestampMAE}s`);
    if (m.monotonicity > t.maxMonotonicity) clipFailures.push(`monotonicity ${m.monotonicity} > ${t.maxMonotonicity}`);
    if (t.textMatch && !m.textMatch) clipFailures.push("word/sentence text mismatch");

    if (clipFailures.length > 0) {
      console.log(`    FAIL: ${clipFailures.join(", ")}`);
      // Print diff for failed clips
      printDiff(results?.[name]?.diff);
      failures.push(`${name}: ${clipFailures.join(", ")}`);
    }
  }

  console.log("─".repeat(70));
  expect(failures, `Quality regression:\n${failures.join("\n")}`).toHaveLength(0);
});

// ── Test: single clip (use CLIP env var) ──
test("whisper benchmark — single clip analysis", async ({ page }) => {
  const clipName = process.env.CLIP;
  test.skip(!clipName, "Set CLIP env var to run, e.g.: CLIP=en-space npx playwright test benchmark --grep single");

  console.log(`\nRunning single clip: ${clipName}`);
  console.log("─".repeat(70));

  const { metrics, results } = await runBenchmark(page, clipName);

  const m = metrics?.[clipName];
  expect(m, `No metrics for clip "${clipName}"`).toBeTruthy();

  printMetrics(clipName, m);

  // Print full diff
  const r = results?.[clipName];
  if (r?.diff) {
    console.log(`\n  Full alignment (${r.diff.length} pairs):`);
    for (let i = 0; i < r.diff.length; i++) {
      const d = r.diff[i];
      if (d.status === "match") {
        console.log(`    [${i}] ✓ "${d.gt.text}" GT=${d.gt.start?.toFixed(2)}s Model=${d.model.start?.toFixed(2)}s (Δ${Math.abs((d.model.start || 0) - (d.gt.start || 0)).toFixed(2)}s)`);
      } else if (d.status === "substitution") {
        console.log(`    [${i}] ≠ GT="${d.gt.text}" → Model="${d.model.text}" GT=${d.gt.start?.toFixed(2)}s Model=${d.model.start?.toFixed(2)}s`);
      } else if (d.status === "missing") {
        console.log(`    [${i}] - "${d.gt.text}" (GT ${d.gt.start?.toFixed(2)}s-${d.gt.end?.toFixed(2)}s) MISSING from model`);
      } else if (d.status === "extra") {
        console.log(`    [${i}] + "${d.model.text}" (Model ${d.model.start?.toFixed(2)}s-${d.model.end?.toFixed(2)}s) EXTRA in model`);
      }
    }
  }

  // Print model sentence output too
  if (r?.model?.sentence?.chunks) {
    console.log(`\n  Sentence segments:`);
    for (const c of r.model.sentence.chunks) {
      console.log(`    [${c.start?.toFixed(2)}s-${c.end?.toFixed(2)}s] ${c.text}`);
    }
  }

  console.log("─".repeat(70));

  // Still assert thresholds
  const t = getThresholds(clipName);
  const failures = [];
  if (m.wer > t.maxWER) failures.push(`WER ${(m.wer * 100).toFixed(1)}% > ${(t.maxWER * 100).toFixed(0)}%`);
  if (m.mae != null && m.mae > t.maxTimestampMAE) failures.push(`MAE ${m.mae.toFixed(3)}s > ${t.maxTimestampMAE}s`);
  if (m.monotonicity > t.maxMonotonicity) failures.push(`monotonicity ${m.monotonicity} > ${t.maxMonotonicity}`);
  if (t.textMatch && !m.textMatch) failures.push("word/sentence text mismatch");

  expect(failures, `Quality regression:\n${failures.join("\n")}`).toHaveLength(0);
});
