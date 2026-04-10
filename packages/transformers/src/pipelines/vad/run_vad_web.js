import { env } from '../../env.js';

const VAD_WEB_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/silero_vad_legacy.onnx';

/**
 * @param {unknown} moduleNamespace
 * @returns {{ NonRealTimeVAD?: { new?: Function, run?: Function } & { new(options?: Record<string, unknown>): Promise<{ run(inputAudio: Float32Array, sampleRate: number): AsyncGenerator<{ audio: Float32Array, start: number, end: number }> }> } }}
 */
function unwrapVadModule(moduleNamespace) {
    if (moduleNamespace && typeof moduleNamespace === 'object' && 'default' in moduleNamespace) {
        return /** @type {Record<string, any>} */ (moduleNamespace.default);
    }

    return /** @type {Record<string, any>} */ (moduleNamespace);
}

/**
 * @returns {(ort: any) => void}
 */
function createOrtConfig() {
    const sharedOnnxEnv = env.backends.onnx ?? {};
    const sharedWasmEnv = sharedOnnxEnv.wasm ?? {};

    return (ort) => {
        if (!ort?.env) return;

        if (ort.env.wasm && sharedWasmEnv) {
            if (sharedWasmEnv.wasmPaths) {
                ort.env.wasm.wasmPaths = sharedWasmEnv.wasmPaths;
            }
            if (typeof sharedWasmEnv.proxy === 'boolean') {
                ort.env.wasm.proxy = sharedWasmEnv.proxy;
            }
            if (typeof sharedWasmEnv.numThreads === 'number') {
                ort.env.wasm.numThreads = sharedWasmEnv.numThreads;
            }
            if (typeof sharedWasmEnv.simd === 'boolean') {
                ort.env.wasm.simd = sharedWasmEnv.simd;
            }
        }

        if (sharedOnnxEnv.logLevel) {
            ort.env.logLevel = sharedOnnxEnv.logLevel;
        }
    };
}

/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {{ min_speech_ms: number, min_silence_ms: number }} options
 * @returns {Promise<{ start_ms: number, end_ms: number }[]>}
 */
export async function runVADWeb(audio, sampling_rate, options) {
    const imported = await import('@ricky0123/vad-web/dist/index.js');
    const vadModule = unwrapVadModule(imported);
    const NonRealTimeVAD = vadModule.NonRealTimeVAD ?? imported.NonRealTimeVAD;

    if (!NonRealTimeVAD?.new) {
        throw new Error('Failed to load NonRealTimeVAD from @ricky0123/vad-web.');
    }

    const vad = await NonRealTimeVAD.new({
        modelURL: VAD_WEB_MODEL_URL,
        minSpeechMs: options.min_speech_ms,
        redemptionMs: options.min_silence_ms,
        ortConfig: createOrtConfig(),
    });

    /** @type {{ start_ms: number, end_ms: number }[]} */
    const segments = [];
    for await (const { start, end } of vad.run(audio, sampling_rate)) {
        segments.push({
            start_ms: start,
            end_ms: end,
        });
    }

    return segments;
}
