import { logger } from '../../utils/logger.js';
import { buildVoiceActivityTimeline, normalizeVoiceActivityDetectionOptions } from './build_timeline.js';
import { remapWhisperOutputTimestamps } from './remap_timestamps.js';
import { runVADWeb } from './run_vad_web.js';

let hasWarnedAboutVADFailure = false;

/**
 * @param {unknown} error
 * @returns {string}
 */
function getErrorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}

/**
 * @param {string} message
 * @param {unknown} error
 */
function warnOnce(message, error) {
    if (hasWarnedAboutVADFailure) {
        return;
    }

    hasWarnedAboutVADFailure = true;
    logger.warn(`${message} Falling back to normal Whisper transcription.`, getErrorMessage(error));
}

/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {boolean | Record<string, unknown>} voice_activity_detection
 * @returns {Promise<{ applied: boolean, processedAudio: Float32Array, segments: { original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[], originalDurationS: number, compactDurationS: number, debugMetadata?: Record<string, unknown> }>}
 */
export async function preprocessAudioWithVoiceActivityDetection(audio, sampling_rate, voice_activity_detection) {
    const options = normalizeVoiceActivityDetectionOptions(voice_activity_detection);
    const fallback = {
        applied: false,
        processedAudio: audio,
        segments: [],
        originalDurationS: sampling_rate > 0 ? audio.length / sampling_rate : 0,
        compactDurationS: sampling_rate > 0 ? audio.length / sampling_rate : 0,
    };

    if (!options) {
        return fallback;
    }

    if (options.provider !== 'vad-web') {
        warnOnce(`Unsupported voice activity detection provider "${options.provider}".`, options.provider);
        return fallback;
    }

    try {
        const detectedSegments = await runVADWeb(audio, sampling_rate, options);
        const timeline = buildVoiceActivityTimeline(audio, sampling_rate, detectedSegments, options);

        if (timeline.processedAudio.length === 0 && options.preserve_full_audio_if_empty) {
            return {
                ...fallback,
                ...(timeline.debugMetadata ? { debugMetadata: timeline.debugMetadata } : {}),
            };
        }

        return {
            applied: true,
            ...timeline,
        };
    } catch (error) {
        warnOnce('Voice activity detection preprocessing failed.', error);
        return fallback;
    }
}

export { remapWhisperOutputTimestamps };
