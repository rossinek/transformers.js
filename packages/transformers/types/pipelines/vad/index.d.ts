/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {boolean | Record<string, unknown>} voice_activity_detection
 * @returns {Promise<{ applied: boolean, processedAudio: Float32Array, segments: { original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[], originalDurationS: number, compactDurationS: number, debugMetadata?: Record<string, unknown> }>}
 */
export function preprocessAudioWithVoiceActivityDetection(audio: Float32Array, sampling_rate: number, voice_activity_detection: boolean | Record<string, unknown>): Promise<{
    applied: boolean;
    processedAudio: Float32Array;
    segments: {
        original_start_s: number;
        original_end_s: number;
        compact_start_s: number;
        compact_end_s: number;
    }[];
    originalDurationS: number;
    compactDurationS: number;
    debugMetadata?: Record<string, unknown>;
}>;
export { remapWhisperOutputTimestamps };
import { remapWhisperOutputTimestamps } from './remap_timestamps.js';
//# sourceMappingURL=index.d.ts.map