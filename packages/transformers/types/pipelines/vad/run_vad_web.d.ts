/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {{ min_speech_ms: number, min_silence_ms: number }} options
 * @returns {Promise<{ start_ms: number, end_ms: number }[]>}
 */
export function runVADWeb(audio: Float32Array, sampling_rate: number, options: {
    min_speech_ms: number;
    min_silence_ms: number;
}): Promise<{
    start_ms: number;
    end_ms: number;
}[]>;
//# sourceMappingURL=run_vad_web.d.ts.map