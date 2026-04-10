# VAD Usage

Whisper ASR now supports optional voice activity detection preprocessing through `voice_activity_detection`.

## Basic use

```js
const output = await transcriber(audio, {
  voice_activity_detection: {
    pre_speech_pad_ms: 200,
    post_speech_pad_ms: 300,
    min_speech_ms: 150,
    min_silence_ms: 400,
    max_merge_gap_ms: 500,
    preserve_full_audio_if_empty: true,
  },
});
```

## Behavior

- Off by default.
- Applies only to Whisper / lite-whisper in the ASR pipeline.
- Removes inactive regions before transcription.
- Remaps returned timestamps back to original audio time.
- Falls back to normal transcription if VAD fails to load or run.

## Options

```ts
voice_activity_detection?: false | {
  provider?: 'vad-web'
  pre_speech_pad_ms?: number
  post_speech_pad_ms?: number
  min_speech_ms?: number
  min_silence_ms?: number
  max_merge_gap_ms?: number
  preserve_full_audio_if_empty?: boolean
  debug?: boolean
}
```

## Browser asset note

VAD uses `@ricky0123/vad-web` and ONNX WASM in the browser.
If the app sets `env.backends.onnx.wasm.wasmPaths`, that setting is reused by the VAD path as well.

Example:

```js
import { env } from '@huggingface/transformers';

env.backends.onnx.wasm.wasmPaths = {
  mjs: '/wasm/ort-wasm-simd-threaded.asyncify.mjs',
  wasm: '/wasm/ort-wasm-simd-threaded.asyncify.wasm',
};
```

The VAD model file still needs to be reachable by the browser runtime.
