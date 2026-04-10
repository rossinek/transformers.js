/**
 * @param {boolean | Partial<typeof DEFAULT_VAD_OPTIONS> | undefined | null} voice_activity_detection
 * @returns {typeof DEFAULT_VAD_OPTIONS | null}
 */
export function normalizeVoiceActivityDetectionOptions(voice_activity_detection: boolean | Partial<typeof DEFAULT_VAD_OPTIONS> | undefined | null): typeof DEFAULT_VAD_OPTIONS | null;
/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {{ start_ms: number, end_ms: number }[]} detected_segments
 * @param {ReturnType<typeof normalizeVoiceActivityDetectionOptions>} options
 * @returns {{ processedAudio: Float32Array, segments: SpeechSegment[], originalDurationS: number, compactDurationS: number, debugMetadata?: Record<string, unknown> }}
 */
export function buildVoiceActivityTimeline(audio: Float32Array, sampling_rate: number, detected_segments: {
    start_ms: number;
    end_ms: number;
}[], options: ReturnType<typeof normalizeVoiceActivityDetectionOptions>): {
    processedAudio: Float32Array;
    segments: SpeechSegment[];
    originalDurationS: number;
    compactDurationS: number;
    debugMetadata?: Record<string, unknown>;
};
export type SpeechSegment = {
    original_start_s: number;
    original_end_s: number;
    compact_start_s: number;
    compact_end_s: number;
};
declare namespace DEFAULT_VAD_OPTIONS {
    let provider: string;
    let pre_speech_pad_ms: number;
    let post_speech_pad_ms: number;
    let min_speech_ms: number;
    let min_silence_ms: number;
    let max_merge_gap_ms: number;
    let preserve_full_audio_if_empty: boolean;
    let debug: boolean;
}
export {};
//# sourceMappingURL=build_timeline.d.ts.map