import { createInferenceSession } from '../../backends/onnx.js';
import { Tensor } from 'onnxruntime-common';

const VAD_WEB_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/silero_vad_legacy.onnx';
const FRAME_SAMPLES = 1536;
const SAMPLE_RATE = 16000;
const MS_PER_FRAME = FRAME_SAMPLES / (SAMPLE_RATE / 1000);

// --- Inlined from @ricky0123/vad-web (pure JS, no ORT dependency) ---

const Message = /** @type {const} */ ({
    SpeechStart: 'SPEECH_START',
    SpeechEnd: 'SPEECH_END',
    VADMisfire: 'VAD_MISFIRE',
});

/**
 * @param {{ redemptionMs: number, preSpeechPadMs: number, minSpeechMs: number }} options
 * @param {number} msPerFrame
 */
function calculateFrameParams(options, msPerFrame) {
    return {
        redemptionFrames: Math.floor(options.redemptionMs / msPerFrame),
        preSpeechPadFrames: Math.floor(options.preSpeechPadMs / msPerFrame),
        minSpeechFrames: Math.floor(options.minSpeechMs / msPerFrame),
    };
}

/** @param {Float32Array[]} arrays */
function concatFloat32Arrays(arrays) {
    const sizes = arrays.reduce((out, next) => {
        out.push(/** @type {number} */ (out.at(-1)) + next.length);
        return out;
    }, [0]);
    const outArray = new Float32Array(/** @type {number} */ (sizes.at(-1)));
    arrays.forEach((arr, index) => {
        outArray.set(arr, sizes[index]);
    });
    return outArray;
}

class FrameProcessor {
    /**
     * @param {(frame: Float32Array) => Promise<{ isSpeech: number, notSpeech: number }>} modelProcessFunc
     * @param {() => void} modelResetFunc
     * @param {{ positiveSpeechThreshold: number, negativeSpeechThreshold: number, redemptionMs: number, preSpeechPadMs: number, minSpeechMs: number }} options
     * @param {number} msPerFrame
     */
    constructor(modelProcessFunc, modelResetFunc, options, msPerFrame) {
        this.modelProcessFunc = modelProcessFunc;
        this.modelResetFunc = modelResetFunc;
        this.options = options;
        this.msPerFrame = msPerFrame;
        this.speaking = false;
        this.redemptionCounter = 0;
        this.speechFrameCount = 0;
        this.active = false;
        /** @type {{ frame: Float32Array, isSpeech: boolean }[]} */
        this.audioBuffer = [];
        const { redemptionFrames, preSpeechPadFrames, minSpeechFrames } = calculateFrameParams(options, msPerFrame);
        this.redemptionFrames = redemptionFrames;
        this.preSpeechPadFrames = preSpeechPadFrames;
        this.minSpeechFrames = minSpeechFrames;
        this.modelResetFunc();
    }

    resume() {
        this.active = true;
    }

    /** @param {(event: any) => void} handleEvent */
    endSegment(handleEvent) {
        const audioBuffer = this.audioBuffer;
        this.audioBuffer = [];
        const speaking = this.speaking;
        this.speaking = false;
        this.audioBuffer = [];
        this.modelResetFunc();
        this.redemptionCounter = 0;
        this.speechFrameCount = 0;

        if (speaking) {
            const speechFrameCount = audioBuffer.reduce((acc, item) => (item.isSpeech ? acc + 1 : acc), 0);
            if (speechFrameCount >= this.minSpeechFrames) {
                handleEvent({ msg: Message.SpeechEnd, audio: concatFloat32Arrays(audioBuffer.map((item) => item.frame)) });
            }
        }
    }

    /**
     * @param {Float32Array} frame
     * @param {(event: any) => void} handleEvent
     */
    async process(frame, handleEvent) {
        if (!this.active) return;

        const probs = await this.modelProcessFunc(frame);
        const isSpeech = probs.isSpeech >= this.options.positiveSpeechThreshold;

        this.audioBuffer.push({ frame, isSpeech });

        if (isSpeech) {
            this.speechFrameCount++;
            this.redemptionCounter = 0;
        }

        if (isSpeech && !this.speaking) {
            this.speaking = true;
            handleEvent({ msg: Message.SpeechStart });
        }

        if (
            probs.isSpeech < this.options.negativeSpeechThreshold &&
            this.speaking &&
            ++this.redemptionCounter >= this.redemptionFrames
        ) {
            this.redemptionCounter = 0;
            this.speechFrameCount = 0;
            this.speaking = false;
            const audioBuffer = this.audioBuffer;
            this.audioBuffer = [];
            const speechFrameCount = audioBuffer.reduce((acc, item) => (item.isSpeech ? acc + 1 : acc), 0);
            if (speechFrameCount >= this.minSpeechFrames) {
                handleEvent({ msg: Message.SpeechEnd, audio: concatFloat32Arrays(audioBuffer.map((item) => item.frame)) });
            }
        }

        if (!this.speaking) {
            while (this.audioBuffer.length > this.preSpeechPadFrames) {
                this.audioBuffer.shift();
            }
            this.speechFrameCount = 0;
        }
    }
}

class Resampler {
    /**
     * @param {{ nativeSampleRate: number, targetSampleRate: number, targetFrameSize: number }} options
     */
    constructor(options) {
        this.options = options;
        /** @type {number[]} */
        this.inputBuffer = [];
    }

    /** @param {Float32Array} audioInput */
    async *stream(audioInput) {
        for (const sample of audioInput) {
            this.inputBuffer.push(sample);
            while (this.hasEnoughDataForFrame()) {
                yield this.generateOutputFrame();
            }
        }
    }

    hasEnoughDataForFrame() {
        return (
            (this.inputBuffer.length * this.options.targetSampleRate) / this.options.nativeSampleRate >=
            this.options.targetFrameSize
        );
    }

    generateOutputFrame() {
        const outputFrame = new Float32Array(this.options.targetFrameSize);
        let outputIndex = 0;
        let inputIndex = 0;
        while (outputIndex < this.options.targetFrameSize) {
            let sum = 0;
            let num = 0;
            while (
                inputIndex <
                Math.min(
                    this.inputBuffer.length,
                    ((outputIndex + 1) * this.options.nativeSampleRate) / this.options.targetSampleRate,
                )
            ) {
                const value = this.inputBuffer[inputIndex];
                if (value !== undefined) {
                    sum += value;
                    num++;
                }
                inputIndex++;
            }
            outputFrame[outputIndex] = sum / num;
            outputIndex++;
        }
        this.inputBuffer = this.inputBuffer.slice(inputIndex);
        return outputFrame;
    }
}

// --- End inlined vad-web code ---

/**
 * Create a Silero VAD model using transformers.js's own ORT backend.
 * @param {string} modelURL
 */
async function createSileroModel(modelURL) {
    const response = await fetch(modelURL);
    const modelBuffer = new Uint8Array(await response.arrayBuffer());
    const session = await createInferenceSession(modelBuffer, {}, {});

    /** @type {import('onnxruntime-common').Tensor} */
    let h = new Tensor('float32', new Float32Array(2 * 64), [2, 1, 64]);
    /** @type {import('onnxruntime-common').Tensor} */
    let c = new Tensor('float32', new Float32Array(2 * 64), [2, 1, 64]);
    const sr = new Tensor('int64', BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);

    return {
        /** @param {Float32Array} audioFrame */
        process: async (audioFrame) => {
            const input = new Tensor('float32', audioFrame, [1, audioFrame.length]);
            const out = await session.run({ input, h, c, sr });
            h = out['hn'];
            c = out['cn'];
            const isSpeech = /** @type {number} */ (out['output'].data[0]);
            return { isSpeech, notSpeech: 1 - isSpeech };
        },
        reset_state: () => {
            h = new Tensor('float32', new Float32Array(2 * 64), [2, 1, 64]);
            c = new Tensor('float32', new Float32Array(2 * 64), [2, 1, 64]);
        },
    };
}

const DEFAULT_FRAME_PROCESSOR_OPTIONS = {
    positiveSpeechThreshold: 0.3,
    negativeSpeechThreshold: 0.25,
    preSpeechPadMs: 800,
    redemptionMs: 1400,
    minSpeechMs: 400,
};

/**
 * @param {Float32Array} audio
 * @param {number} sampling_rate
 * @param {{ min_speech_ms: number, min_silence_ms: number }} options
 * @returns {Promise<{ start_ms: number, end_ms: number }[]>}
 */
export async function runVADWeb(audio, sampling_rate, options) {
    const model = await createSileroModel(VAD_WEB_MODEL_URL);

    const frameProcessorOptions = {
        ...DEFAULT_FRAME_PROCESSOR_OPTIONS,
        minSpeechMs: options.min_speech_ms,
        redemptionMs: options.min_silence_ms,
    };

    const frameProcessor = new FrameProcessor(model.process, model.reset_state, frameProcessorOptions, MS_PER_FRAME);
    frameProcessor.resume();

    const resampler = new Resampler({
        nativeSampleRate: sampling_rate,
        targetSampleRate: SAMPLE_RATE,
        targetFrameSize: FRAME_SAMPLES,
    });

    /** @type {{ start_ms: number, end_ms: number }[]} */
    const segments = [];
    let start = 0;
    let frameIndex = 0;

    for await (const frame of resampler.stream(audio)) {
        /** @type {any[]} */
        const events = [];
        await frameProcessor.process(frame, (event) => events.push(event));

        for (const event of events) {
            if (event.msg === Message.SpeechStart) {
                start = (frameIndex * FRAME_SAMPLES) / (SAMPLE_RATE / 1000);
            } else if (event.msg === Message.SpeechEnd) {
                const end = ((frameIndex + 1) * FRAME_SAMPLES) / (SAMPLE_RATE / 1000);
                segments.push({ start_ms: start, end_ms: end });
            }
        }
        frameIndex++;
    }

    // Flush any remaining speech segment
    /** @type {any[]} */
    const finalEvents = [];
    frameProcessor.endSegment((event) => finalEvents.push(event));
    for (const event of finalEvents) {
        if (event.msg === Message.SpeechEnd) {
            segments.push({
                start_ms: start,
                end_ms: (frameIndex * FRAME_SAMPLES) / (SAMPLE_RATE / 1000),
            });
        }
    }

    return segments;
}
