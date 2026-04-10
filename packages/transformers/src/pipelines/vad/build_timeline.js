/**
 * @typedef {Object} SpeechSegment
 * @property {number} original_start_s
 * @property {number} original_end_s
 * @property {number} compact_start_s
 * @property {number} compact_end_s
 */

const DEFAULT_VAD_OPTIONS = {
    provider: 'vad-web',
    pre_speech_pad_ms: 200,
    post_speech_pad_ms: 300,
    min_speech_ms: 150,
    min_silence_ms: 400,
    max_merge_gap_ms: 500,
    preserve_full_audio_if_empty: true,
    debug: false,
};

/**
 * @param {boolean | Partial<typeof DEFAULT_VAD_OPTIONS> | undefined | null} voice_activity_detection
 * @returns {typeof DEFAULT_VAD_OPTIONS | null}
 */
export function normalizeVoiceActivityDetectionOptions(voice_activity_detection) {
    if (!voice_activity_detection) {
        return null;
    }

    if (voice_activity_detection === true) {
        return { ...DEFAULT_VAD_OPTIONS };
    }

    return {
        ...DEFAULT_VAD_OPTIONS,
        ...voice_activity_detection,
        provider: 'vad-web',
    };
}

/**
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

/**
 * @param {number} seconds
 * @param {number} sampling_rate
 * @param {number} max_length
 * @param {'floor' | 'ceil'} mode
 * @returns {number}
 */
function toSampleIndex(seconds, sampling_rate, max_length, mode) {
    const scaled = seconds * sampling_rate;
    const index = mode === 'ceil' ? Math.ceil(scaled) : Math.floor(scaled);
    return clamp(index, 0, max_length);
}

/**
 * @param {{ start_ms: number, end_ms: number }[]} detected_segments
 * @param {number} original_duration_s
 * @param {ReturnType<typeof normalizeVoiceActivityDetectionOptions>} options
 * @returns {{ start_s: number, end_s: number }[]}
 */
function mergeSpeechRegions(detected_segments, original_duration_s, options) {
    if (!options || detected_segments.length === 0) {
        return [];
    }

    const pre_pad_s = options.pre_speech_pad_ms / 1000;
    const post_pad_s = options.post_speech_pad_ms / 1000;
    const max_merge_gap_s = options.max_merge_gap_ms / 1000;

    const padded = detected_segments
        .map(({ start_ms, end_ms }) => ({
            start_s: clamp(start_ms / 1000 - pre_pad_s, 0, original_duration_s),
            end_s: clamp(end_ms / 1000 + post_pad_s, 0, original_duration_s),
        }))
        .filter((segment) => segment.end_s > segment.start_s)
        .sort((a, b) => a.start_s - b.start_s);

    if (padded.length === 0) {
        return [];
    }

    const merged = [padded[0]];
    for (let i = 1; i < padded.length; ++i) {
        const current = padded[i];
        const previous = merged[merged.length - 1];
        if (current.start_s <= previous.end_s + max_merge_gap_s) {
            previous.end_s = Math.max(previous.end_s, current.end_s);
        } else {
            merged.push({ ...current });
        }
    }

    return merged;
}

/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {{ start_ms: number, end_ms: number }[]} detected_segments
 * @param {ReturnType<typeof normalizeVoiceActivityDetectionOptions>} options
 * @returns {{ processedAudio: Float32Array, segments: SpeechSegment[], originalDurationS: number, compactDurationS: number, debugMetadata?: Record<string, unknown> }}
 */
export function buildVoiceActivityTimeline(audio, sampling_rate, detected_segments, options) {
    const originalDurationS = sampling_rate > 0 ? audio.length / sampling_rate : 0;
    const mergedSegments = mergeSpeechRegions(detected_segments, originalDurationS, options);

    if (mergedSegments.length === 0) {
        return {
            processedAudio: new Float32Array(0),
            segments: [],
            originalDurationS,
            compactDurationS: 0,
            ...(options?.debug
                ? {
                      debugMetadata: {
                          detectedSegments: detected_segments,
                          mergedSegments,
                          originalDurationS,
                          compactDurationS: 0,
                      },
                  }
                : {}),
        };
    }

    const slices = [];
    let compactLength = 0;
    for (const segment of mergedSegments) {
        const startIndex = toSampleIndex(segment.start_s, sampling_rate, audio.length, 'floor');
        const endIndex = toSampleIndex(segment.end_s, sampling_rate, audio.length, 'ceil');
        if (endIndex <= startIndex) {
            continue;
        }

        slices.push({ startIndex, endIndex });
        compactLength += endIndex - startIndex;
    }

    const processedAudio = new Float32Array(compactLength);
    /** @type {SpeechSegment[]} */
    const segments = [];

    let writeOffset = 0;
    for (const { startIndex, endIndex } of slices) {
        const slice = audio.subarray(startIndex, endIndex);
        processedAudio.set(slice, writeOffset);

        const compactStart = writeOffset / sampling_rate;
        writeOffset += slice.length;
        const compactEnd = writeOffset / sampling_rate;

        segments.push({
            original_start_s: startIndex / sampling_rate,
            original_end_s: endIndex / sampling_rate,
            compact_start_s: compactStart,
            compact_end_s: compactEnd,
        });
    }

    const compactDurationS = sampling_rate > 0 ? processedAudio.length / sampling_rate : 0;
    return {
        processedAudio,
        segments,
        originalDurationS,
        compactDurationS,
        ...(options?.debug
            ? {
                  debugMetadata: {
                      detectedSegments: detected_segments,
                      mergedSegments,
                      originalDurationS,
                      compactDurationS,
                  },
              }
            : {}),
    };
}
