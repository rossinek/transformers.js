# Whisper Benchmark & E2E Test Setup

Benchmark tooling for measuring Whisper transcription quality (text accuracy, word-level timestamp accuracy) against ElevenLabs Scribe ground truth.

## Files

- `packages/transformers/tests/e2e/serve.mjs` — Dev server with API endpoints for the benchmark page
- `packages/transformers/tests/e2e/fixtures/benchmark.html` — Interactive comparison page (ground truth vs model output)
- `packages/transformers/tests/e2e/add-test-case.mjs` — CLI script to add new test cases
- `packages/transformers/tests/e2e/benchmark.spec.mjs` — Playwright test for automated validation
- `packages/transformers/tests/e2e/ground-truth/` — Cached ElevenLabs transcriptions (JSON)
- `packages/transformers/tests/e2e/fixtures/` — Audio fixture files (WAV, 16kHz mono)
- `packages/transformers/playwright.config.mjs` — Playwright configuration

## Environment variables

Create a `.env` file in the repo root (it's gitignored):

```bash
ELEVENLABS_API_KEY=your_key_here
```

| Variable | Required for | Description |
|----------|-------------|-------------|
| `ELEVENLABS_API_KEY` | Adding test cases (CLI and upload) | API key from [ElevenLabs](https://elevenlabs.io/). Used to call the Scribe speech-to-text API for generating ground truth transcriptions. |

Both `serve.mjs` and `add-test-case.mjs` auto-load `.env` from the repo root. Shell-exported env vars take precedence.

## Prerequisites

- **pnpm** — package manager
- **ffmpeg** — must be installed and on PATH (used to convert input files to 16kHz mono WAV)
- **Chrome with WebGPU** — required for running Whisper in the browser

```bash
pnpm install                    # from repo root
npx playwright install chromium # download Playwright browser binaries
cd packages/transformers
pnpm build                      # build the transformers library
```

## Adding test cases

### Via CLI

```bash
cd packages/transformers
node tests/e2e/add-test-case.mjs <input-file> <name> [--lang <code>]
```

Example:

```bash
node tests/e2e/add-test-case.mjs ~/Downloads/interview.mp4 interview --lang en
node tests/e2e/add-test-case.mjs ~/Downloads/discurso.mp4 discurso --lang pt
```

This will:
1. Convert the file to 16kHz mono WAV with ffmpeg
2. Transcribe with ElevenLabs Scribe API (word-level timestamps)
3. Save `fixtures/<name>.wav` and `ground-truth/<name>.json`

### Via the benchmark page

Drag & drop a file onto the upload area, enter a name and language, click "Add Test Case". Same conversion and transcription happens server-side.

## Running the benchmark page

```bash
cd packages/transformers
node tests/e2e/serve.mjs
```

Open http://localhost:8484 in Chrome (WebGPU required).

1. Click **Load Model** — downloads and compiles `whisper-large-v3-turbo_timestamped`
2. Click **Run All Clips** — transcribes each test case in both word and sentence modes
3. Review results:
   - **Words view** — side-by-side word-level comparison with timestamps
   - **Text view** — full text comparison
   - **Sentences view** — sentence-level segments from the model
   - **Metrics** — WER, Timestamp MAE, monotonicity violations, zero-start anomalies, word/sentence text consistency

## Automated validation (for agents / CI)

After modifying Whisper transcription code, validate that quality hasn't regressed:

```bash
cd packages/transformers
pnpm build

# Run all clips
npx playwright test benchmark

# Run a single clip (faster — for investigating specific issues)
CLIP=en-space npx playwright test benchmark --grep "single"
```

Running all clips will:
1. Start the dev server automatically
2. Open Chrome with WebGPU
3. Load the Whisper model
4. Transcribe all test clips (word + sentence modes)
5. Compare against ElevenLabs ground truth
6. Assert quality thresholds per clip (WER, timestamp MAE, monotonicity, zero-starts, text consistency)

### Thresholds

Default thresholds are defined in `tests/e2e/benchmark.spec.mjs`:

| Metric | Default max | Description |
|--------|------------|-------------|
| WER | 25% | Word error rate vs ground truth |
| Timestamp MAE | 1.5s | Mean absolute error of word timestamps |
| Monotonicity | 2 | Timestamps going backwards |
| W/S match | true | Word mode and sentence mode text must be identical |

Per-clip overrides can be set in the `CLIP_THRESHOLDS` object in the test file.

### Available clips

```bash
ls packages/transformers/tests/e2e/ground-truth/ | sed 's/.json//'
```

### Single clip analysis

The single-clip mode prints the full word-by-word alignment, showing exactly what matched, what's missing, what's extra, and timestamp deltas. This is useful for investigating specific issues.

Output looks like:
```
[0] ✓ "Hello" GT=0.50s Model=0.48s (Δ0.02s)
[1] ≠ GT="world" → Model="word" GT=0.80s Model=0.79s
[2] - "the" (GT 1.20s-1.35s) MISSING from model
[3] + "a" (Model 1.40s-1.55s) EXTRA in model
```

### Programmatic API

The benchmark page exposes globals for automation:
- `window.__READY__` — `true` when page is initialized
- `window.__DONE__` — `true` when all transcriptions complete
- `window.__METRICS__` — `{ clipName: { wer, mae, monotonicity, zeroStarts, textMatch } }`
- `window.__RESULTS__` — `{ clipName: { model, groundTruth, diff } }` — full aligned word diff
- `window.__ERROR__` — error message if autorun failed

URL parameters:
- `?autorun=true` — load model, run all clips
- `?autorun=true&clip=en-space` — load model, run only that clip

## Supported languages

en, es, pt, fr, de, ja, zh, ar, ko, ru, it, nl, pl, sk, he (selectable in upload form; pass any ISO code via CLI `--lang`).
