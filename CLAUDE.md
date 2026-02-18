# CLAUDE.md

## Project Overview
Fork of `@huggingface/transformers` (v4.0.0-next.4). Focus: **browser-only ASR (Whisper models)**.
Node.js support is irrelevant — only the browser/web-worker bundle matters.

## Monorepo Layout
- Root uses **pnpm workspaces**; the main package lives at `packages/transformers/`
- All commands below should be run from `packages/transformers/` (or use `pnpm --filter @huggingface/transformers`)

## Key Commands
```bash
pnpm install              # from repo root
cd packages/transformers
pnpm dev                  # watch mode (esbuild + type generation)
pnpm build                # production build (esbuild + tsc declarations)
pnpm test                 # all tests (jest, requires --experimental-vm-modules)
pnpm test -- ./tests/models/whisper                                        # whisper model tests
pnpm test -- ./tests/pipelines/test_pipelines_automatic_speech_recognition.js  # ASR pipeline test
```

## Browser Build
- Build tool: **esbuild** (config in `scripts/build/`)
- Browser entry: `dist/transformers.web.js` (ESM, platform: neutral)
- Ignores node modules: `onnxruntime-node`, `sharp`, `fs`, `path`, `url`, `stream`
- External: `onnxruntime-common`, `onnxruntime-web`
- CDN bundles: `dist/transformers.min.js` (fully bundled, for `<script type="module">`)
- `env.js`: `IS_NODE_ENV` is forced `false` on this branch

## ASR / Whisper Files (the stuff that matters)

### Pipeline
- `src/pipelines/automatic-speech-recognition.js` — `AutomaticSpeechRecognitionPipeline`
  - Supports: whisper, lite-whisper, wav2vec2, hubert, parakeet_ctc, moonshine
  - Key options: `return_timestamps`, `chunk_length_s`, `stride_length_s`, `language`, `task`

### Whisper Model (`src/models/whisper/`)
| File | Purpose |
|------|---------|
| `modeling_whisper.js` | `WhisperForConditionalGeneration`, encoder/decoder |
| `feature_extraction_whisper.js` | Log-Mel spectrogram extraction (80 mel bins x 3000 frames) |
| `tokenization_whisper.js` | `WhisperTokenizer` |
| `processing_whisper.js` | `WhisperProcessor` (combines feature extractor + tokenizer) |
| `generation_whisper.js` | `WhisperGenerationConfig` |
| `common_whisper.js` | Language code mappings |

### Audio Utilities
- `src/utils/audio.js` — `read_audio()` uses browser `AudioContext` for decoding

### ONNX Backend
- `src/backends/onnx.js` — runtime selection; browser uses `onnxruntime-web` (WASM / WebGPU)

### Environment
- `src/env.js` — runtime detection (`IS_BROWSER_ENV`, `IS_WEBWORKER_ENV`, `IS_WEBGPU_AVAILABLE`)

## Tests for ASR/Whisper
- `tests/pipelines/test_pipelines_automatic_speech_recognition.js`
- `tests/models/whisper/test_modeling_whisper.js`
- `tests/models/whisper/test_feature_extraction_whisper.js`
- `tests/models/whisper/test_tokenization_whisper.js`
- Test models: `Xenova/tiny-random-WhisperForConditionalGeneration`, `Xenova/whisper-tiny.en`
- Test init: `tests/init.js` (registers ONNX backend, sets timeouts, custom matchers)

## Code Style
- Formatting: **Prettier** (`pnpm format`)
- No TypeScript source — plain JS with JSDoc annotations, types generated via `tsc --build`
- Strict mode is off in tsconfig

## Browser Runtime Notes
- Audio input: URL string (decoded via `AudioContext`) or `Float32Array` at 16 kHz
- Model caching: browser Cache API + IndexedDB
- Execution providers: WASM (default), WebGPU (if `navigator.gpu` available)
- Long audio (>30s): use `chunk_length_s` + `stride_length_s` for chunked inference
- Timestamps: segment-level (`return_timestamps: true`) or word-level (`return_timestamps: 'word'`)
