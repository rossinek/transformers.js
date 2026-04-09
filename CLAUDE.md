# Whisper-focused Transformers.js Fork

This is a fork of [Transformers.js](https://github.com/huggingface/transformers.js) used exclusively for running the **Whisper speech-to-text model in the browser** (WebGPU/WASM). Ignore everything unrelated to Whisper and browser execution.

## Key Constraints

- **Browser only** — no Node.js. The relevant build target is `transformers.web.js` / `transformers.web.min.js`.
- **Whisper only** — this repo supports many models but we only care about Whisper (and LiteWhisper).
- When making changes, think about the Whisper pipeline end-to-end: audio in → mel spectrogram → ONNX inference → token generation → text + timestamps out.

## Core Source Files

All paths relative to `packages/transformers/src/`.

### Whisper Model Implementation (`models/whisper/`)

| File | Purpose |
|---|---|
| `modeling_whisper.js` | Model architecture: `WhisperForConditionalGeneration`, `LiteWhisperForConditionalGeneration`. Encoder/decoder inference, seek loop for long audio, DTW timestamp alignment. **Most complex file.** |
| `feature_extraction_whisper.js` | `WhisperFeatureExtractor` — converts raw audio waveforms to log-Mel spectrograms (FFT, mel filterbanks, Hann window). |
| `tokenization_whisper.js` | `WhisperTokenizer` — token↔text conversion, timestamp token handling, DTW token alignment, chunk merging with stride. **Second most complex file.** |
| `processing_whisper.js` | `WhisperProcessor` — thin orchestrator combining tokenizer + feature extractor. |
| `generation_whisper.js` | `WhisperGenerationConfig` — generation parameters (language, task, timestamps, alignment heads). |
| `common_whisper.js` | Language code mappings (99+ languages). |

### Pipeline & Generation

| File | Purpose |
|---|---|
| `pipelines/automatic-speech-recognition.js` | `AutomaticSpeechRecognitionPipeline` — high-level entry point. Handles audio chunking (>30s), stride overlap, timestamp modes. |
| `generation/logits_process.js` | `WhisperTimeStampLogitsProcessor` — enforces timestamp constraints during decoding. |
| `generation/streamers.js` | `WhisperTextStreamer` — real-time text output as chunks complete. |

### Infrastructure

| File | Purpose |
|---|---|
| `utils/audio.js` | `read_audio()` via AudioContext, FFT, mel filterbanks, spectrogram math. |
| `backends/onnx.js` | ONNX Runtime backend selection — uses `onnxruntime-web` for browser (WASM/WebGPU). |

### Auto-registration (rarely need to edit)

- `models/auto/modeling_auto.js` — registers `whisper` → model classes
- `models/feature_extractors.js`, `models/tokenizers.js`, `models/processors.js`, `models/models.js` — re-exports
- `pipelines.js` — `pipeline()` factory

## Tests

- `packages/transformers/tests/models/whisper/` — unit tests for modeling, tokenization, feature extraction
- `packages/transformers/tests/pipelines/test_pipelines_automatic_speech_recognition.js` — pipeline integration tests

## Quality Validation (IMPORTANT)

This fork has a dedicated benchmark and E2E test setup for validating Whisper transcription quality. **Before and after any change to the transcription pipeline, run these tests to catch regressions in text accuracy, timestamp precision, and output consistency.**

The setup is also useful for **debugging problems with specific audio inputs** — you can run a single clip in analysis mode to get word-by-word alignment output showing exactly what matched, what's missing, and timestamp deltas.

Full documentation for the setup (how to run, thresholds, adding test cases, CLI commands): **[WITHSUBTITLES.md](WITHSUBTITLES.md)**

## Build

- Build scripts: `packages/transformers/scripts/build/`
- `npm run build` produces `packages/transformers/dist/`
- For browser: use `transformers.web.js` (ESM) or `transformers.web.min.js`
- The web build excludes: `onnxruntime-node`, `sharp`, `fs`, `path`, `url`, `stream`

## Browser Pipeline Flow

```
Raw Audio (Float32Array / URL)
  → read_audio()                          [utils/audio.js]
  → WhisperFeatureExtractor._call()       [models/whisper/feature_extraction_whisper.js]
  → Log-Mel Spectrogram
  → WhisperForConditionalGeneration.generate()  [models/whisper/modeling_whisper.js]
      → ONNX Runtime (WebGPU or WASM)     [backends/onnx.js]
      → WhisperTimeStampLogitsProcessor   [generation/logits_process.js]
  → Token IDs
  → WhisperTokenizer._decode_asr()        [models/whisper/tokenization_whisper.js]
  → Text Output + Timestamps/Chunks
```
