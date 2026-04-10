/**
 * @param {number} compact_time_s
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @param {'start' | 'end'} boundary
 * @returns {number}
 */
export function remapCompactTimestamp(compact_time_s: number, segments: {
    original_start_s: number;
    original_end_s: number;
    compact_start_s: number;
    compact_end_s: number;
}[], boundary?: "start" | "end"): number;
/**
 * @param {{ text: string, chunks?: { text: string, timestamp: [number | null, number | null] }[] }} output
 * @param {{ original_start_s: number, original_end_s: number, compact_start_s: number, compact_end_s: number }[]} segments
 * @param {number} [original_duration_s]
 * @returns {{ text: string, chunks?: { text: string, timestamp: [number | null, number | null] }[] }}
 */
export function remapWhisperOutputTimestamps(output: {
    text: string;
    chunks?: {
        text: string;
        timestamp: [number | null, number | null];
    }[];
}, segments: {
    original_start_s: number;
    original_end_s: number;
    compact_start_s: number;
    compact_end_s: number;
}[], original_duration_s?: number): {
    text: string;
    chunks?: {
        text: string;
        timestamp: [number | null, number | null];
    }[];
};
//# sourceMappingURL=remap_timestamps.d.ts.map