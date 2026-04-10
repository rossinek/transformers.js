# VAD Integration Spec

## Context

This fork is private and exists only as a dependency of one browser-based application.
It is not intended for upstreaming as-is, and Node parity is not a product requirement.

The current Whisper ASR flow already has:

- optional `hallucination_recovery` in [`packages/transformers/src/models/whisper/generation_whisper.js`](/Users/ar/Code/transformers.js/packages/transformers/src/models/whisper/generation_whisper.js)
- Whisper pipeline entrypoint in [`packages/transformers/src/pipelines/automatic-speech-recognition.js`](/Users/ar/Code/transformers.js/packages/transformers/src/pipelines/automatic-speech-recognition.js)
- chunk recovery / candidate scoring in [`packages/transformers/src/pipelines/automatic-speech-recognition.js`](/Users/ar/Code/transformers.js/packages/transformers/src/pipelines/automatic-speech-recognition.js)

The goal is to add optional voice-activity-detection preprocessing using `@ricky0123/vad-web` so the transcription pipeline can:

- remove inactive regions before transcription
- potentially reduce hallucinations around long gaps/silence
- potentially improve throughput on sparse audio
- remap timestamps back to original audio time

This must be optional and must not affect current output unless enabled.

## Product Goal

Add an optional preprocessing step to the Whisper ASR pipeline that:

1. runs browser-side VAD on the raw waveform
2. extracts voiced regions with configurable padding/merge rules
3. concatenates voiced regions into a compact waveform
4. transcribes that compact waveform with the existing Whisper pipeline
5. remaps returned timestamps back to original audio time

This should work alongside the existing generation and recovery logic, but remain isolated enough that it can be disabled cleanly.

## Non-Goals

- do not modify non-Whisper ASR paths
- do not make VAD always-on
- do not entangle VAD logic with Whisper generation internals
- do not require Node support
- do not redesign the current ASR pipeline architecture beyond what is needed

## High-Level Design

Implement VAD as a separate preprocessing module that integrates at the pipeline level, before Whisper chunking/generation.

Recommended flow in `_call_whisper(...)`:

1. `prepareAudios(...)` produces raw mono waveform at the processor sampling rate
2. if VAD is disabled:
   - keep current behavior unchanged
3. if VAD is enabled:
   - run VAD preprocessing on the raw waveform
   - return:
     - `processedAudio`
     - `timelineMap`
     - `segments`
     - optional debug metadata
4. send `processedAudio` through the existing Whisper pipeline
5. if timestamps are returned:
   - remap word/chunk timestamps back to original time using `timelineMap`
6. return normal ASR output shape

## API

Add a new optional pipeline option. Keep it explicit.

Recommended public shape:

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

Behavior:

- default: `false`
- `false`: current behavior must remain unchanged
- object: enables preprocessing

If you want a simpler initial version, support only:

```ts
voice_activity_detection?: boolean | {
  pre_speech_pad_ms?: number
  post_speech_pad_ms?: number
  max_merge_gap_ms?: number
}
```

and internally assume `provider: 'vad-web'`.

## File / Module Layout

Keep the VAD logic extracted from the main pipeline file.

Suggested new files:

- `packages/transformers/src/pipelines/vad/index.js`
- `packages/transformers/src/pipelines/vad/run_vad_web.js`
- `packages/transformers/src/pipelines/vad/build_timeline.js`
- `packages/transformers/src/pipelines/vad/remap_timestamps.js`

Responsibilities:

- `run_vad_web.js`
  - isolate all interaction with `@ricky0123/vad-web`
  - use dynamic import
  - normalize VAD results into a simple internal segment format

- `build_timeline.js`
  - merge/pad voiced regions
  - build compact waveform
  - build mapping from compact-time to original-time

- `remap_timestamps.js`
  - remap chunk timestamps
  - remap word timestamps
  - keep timestamps monotonic and bounded

- `index.js`
  - expose one orchestrator function for the pipeline

## Dependency / Loading Requirements

Use dynamic import for `@ricky0123/vad-web`.

Requirements:

- do not statically import it into the main ASR pipeline file
- keep bundle impact limited to the feature path
- fail gracefully if the dependency cannot load

Preferred behavior when VAD fails:

- warn once
- fall back to normal transcription
- do not throw unless the caller explicitly requests strict failure

## Internal Data Shapes

Use simple internal segment objects in seconds.

Suggested segment shape:

```ts
type SpeechSegment = {
  original_start_s: number
  original_end_s: number
  compact_start_s: number
  compact_end_s: number
}
```

Suggested preprocess result:

```ts
type VADPreprocessResult = {
  processedAudio: Float32Array
  segments: SpeechSegment[]
  originalDurationS: number
  compactDurationS: number
}
```

`segments` should be sufficient to remap any monotonic timestamp from compact time back to original time.

## Timestamp Remapping Rules

This is the critical correctness area.

Rules:

- remap only when timestamps are requested
- compact time is monotonic; original time must remain monotonic after remap
- word timestamps:
  - remap each boundary independently
  - clamp to segment bounds if needed
- chunk timestamps:
  - remap start/end independently
  - keep `end >= start`
- if a timestamp falls exactly on a removed gap boundary:
  - map it to the nearest valid original-time point inside the neighboring segment
- do not introduce negative timestamps

If remap cannot be done safely:

- prefer returning transcription with conservative timestamps over throwing
- if necessary, warn and fall back to no-VAD path

## Audio Preprocessing Rules

Use conservative defaults. The point is to cut obvious inactivity, not aggressively re-segment speech.

Recommended initial defaults:

- `pre_speech_pad_ms = 200`
- `post_speech_pad_ms = 300`
- `min_speech_ms = 150`
- `min_silence_ms = 400`
- `max_merge_gap_ms = 500`
- `preserve_full_audio_if_empty = true`

Rules:

- merge nearby voiced regions if the silent gap is small
- add leading/trailing pad around each voiced segment
- clamp padded regions to audio bounds
- if VAD returns no speech and `preserve_full_audio_if_empty` is true:
  - fall back to original audio

## Integration Requirements

Integrate only in `_call_whisper(...)`.

Do not:

- add VAD behavior to `wav2vec2`, `moonshine`, or other ASR paths
- change Whisper generation internals for this feature

Interaction with existing `hallucination_recovery`:

- VAD should run before chunking / transcription
- `hallucination_recovery` should still operate on the compacted audio if enabled
- both features must be independently toggleable

## Expected Edge Cases

Implementation must handle:

- fully silent audio
- audio with a few short speech bursts
- long audio with large silent gaps
- speech that starts immediately at the beginning
- speech that ends immediately at the end
- word timestamps near VAD boundaries
- chunk timestamps across concatenated segments

## Suggested Development Order

1. add the new pipeline option type/docs
2. create VAD helper module with mockable interfaces
3. integrate preprocessing into `_call_whisper(...)`
4. remap timestamps for:
   - `return_timestamps: true`
   - `return_timestamps: 'word'`
5. add tests
6. add a small debug mode if useful

## Testing Requirements

Add focused tests that do not depend on the real VAD model unless necessary.

Test levels:

- unit tests for timeline build/remap logic
- pipeline-level tests with mocked VAD results
- optional e2e validation with local fixtures

Must-have tests:

1. VAD disabled:
   - output is unchanged from current behavior
2. no speech detected:
   - falls back to original audio
3. two voiced regions separated by silence:
   - compact audio is shorter
   - remapped timestamps return to original time correctly
4. word timestamps:
   - remain monotonic after remap
5. chunk timestamps:
   - remain monotonic and non-inverted
6. interaction with `hallucination_recovery`:
   - both toggles can be enabled together without breaking output shape

## Acceptance Criteria

The work is complete when:

- `voice_activity_detection` is optional and off by default
- enabling it uses `@ricky0123/vad-web` via an extracted module
- current output remains unchanged when disabled
- timestamps are remapped back to original audio time when enabled
- the implementation is isolated enough that VAD logic is not baked into Whisper generation code
- failures in VAD degrade gracefully to normal transcription

## Implementation Notes For The Agent

- prefer touching only the Whisper ASR pipeline plus new helper modules
- keep code readable and explicit rather than over-generalized
- preserve the existing browser build assumptions
- avoid introducing broad public API churn beyond the new option
- do not remove or weaken current `hallucination_recovery` behavior
- if a tradeoff is needed, prefer timestamp correctness over aggressive silence removal
