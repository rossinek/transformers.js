const EPSILON = 1e-6;

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
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @returns {number}
 */
function getOriginalDurationS(segments) {
    return segments.length > 0 ? segments[segments.length - 1].original_end_s : 0;
}

/**
 * @param {number} compact_time_s
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @param {'start' | 'end'} boundary
 * @returns {number}
 */
export function remapCompactTimestamp(compact_time_s, segments, boundary = 'start') {
    if (!Number.isFinite(compact_time_s) || segments.length === 0) {
        return 0;
    }

    if (compact_time_s <= segments[0].compact_start_s + EPSILON) {
        return segments[0].original_start_s;
    }

    const lastSegment = segments[segments.length - 1];
    if (compact_time_s >= lastSegment.compact_end_s - EPSILON) {
        return lastSegment.original_end_s;
    }

    for (let i = 0; i < segments.length; ++i) {
        const segment = segments[i];
        const nextSegment = segments[i + 1];

        if (Math.abs(compact_time_s - segment.compact_start_s) <= EPSILON) {
            return segment.original_start_s;
        }

        if (compact_time_s > segment.compact_start_s && compact_time_s < segment.compact_end_s - EPSILON) {
            return segment.original_start_s + (compact_time_s - segment.compact_start_s);
        }

        if (Math.abs(compact_time_s - segment.compact_end_s) <= EPSILON) {
            if (boundary === 'start' && nextSegment) {
                return nextSegment.original_start_s;
            }
            return segment.original_end_s;
        }

        if (nextSegment && compact_time_s > segment.compact_end_s && compact_time_s < nextSegment.compact_start_s) {
            return boundary === 'start' ? nextSegment.original_start_s : segment.original_end_s;
        }
    }

    return lastSegment.original_end_s;
}

/**
 * @param {[number | null, number | null]} timestamp
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @param {number} original_duration_s
 * @param {number} previous_end_s
 * @returns {{ timestamp: [number | null, number | null], previousEndS: number }}
 */
function remapTimestampPair(timestamp, segments, original_duration_s, previous_end_s) {
    let [start, end] = timestamp;

    if (typeof start === 'number') {
        start = clamp(remapCompactTimestamp(start, segments, 'start'), 0, original_duration_s);
        start = Math.max(start, previous_end_s);
    }

    if (typeof end === 'number') {
        end = clamp(remapCompactTimestamp(end, segments, 'end'), 0, original_duration_s);
        if (typeof start === 'number') {
            end = Math.max(end, start);
        }
        end = Math.max(end, previous_end_s);
    }

    return {
        timestamp: [start, end],
        previousEndS: typeof end === 'number' ? end : typeof start === 'number' ? start : previous_end_s,
    };
}

/**
 * @param {{ text: string, chunks?: { text: string, timestamp: [number | null, number | null] }[] }} output
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @param {number} [original_duration_s]
 * @returns {{ text: string, chunks?: { text: string, timestamp: [number | null, number | null] }[] }}
 */
export function remapWhisperOutputTimestamps(output, segments, original_duration_s = getOriginalDurationS(segments)) {
    if (!output?.chunks?.length || segments.length === 0) {
        return output;
    }

    let previousEndS = 0;
    for (const chunk of output.chunks) {
        if (!Array.isArray(chunk.timestamp) || chunk.timestamp.length !== 2) {
            continue;
        }

        const remapped = remapTimestampPair(chunk.timestamp, segments, original_duration_s, previousEndS);
        chunk.timestamp = remapped.timestamp;
        previousEndS = remapped.previousEndS;
    }

    return output;
}
