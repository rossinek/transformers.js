declare const AutomaticSpeechRecognitionPipeline_base: new (options: TextAudioPipelineConstructorArgs) => AutomaticSpeechRecognitionPipelineType;
/**
 * @typedef {import('./_base.js').TextAudioPipelineConstructorArgs} TextAudioPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').AudioInput} AudioInput
 */
/**
 * @typedef {Object} Chunk
 * @property {[number, number]} timestamp The start and end timestamp of the chunk in seconds.
 * @property {string} text The recognized text.
 */
/**
 * @typedef {Object} AutomaticSpeechRecognitionOutput
 * @property {string} text The recognized text.
 * @property {Chunk[]} [chunks] When using `return_timestamps`, the `chunks` will become a list
 * containing all the various text chunks identified by the model.
 *
 * @typedef {Object} AutomaticSpeechRecognitionSpecificParams Parameters specific to automatic-speech-recognition pipelines.
 * @property {boolean|'word'} [return_timestamps] Whether to return timestamps or not. Default is `false`.
 * @property {number} [chunk_length_s] The length of audio chunks to process in seconds. Default is 0 (no chunking).
 * @property {number} [stride_length_s] The length of overlap between consecutive audio chunks in seconds. If not provided, defaults to `chunk_length_s / 6`.
 * @property {boolean} [force_full_sequences] Whether to force outputting full sequences or not. Default is `false`.
 * @property {boolean} [hallucination_recovery] Whether to enable best-effort Whisper hallucination recovery. Default is `true`.
 * @property {string} [initial_prompt] Optional prompt text used to bias Whisper toward expected names or terms.
 * @property {number[]} [prompt_ids] Whisper prompt token ids, typically produced from `initial_prompt`.
 * @property {boolean} [carry_initial_prompt] Whether prompt text should be reapplied to every sequential Whisper segment. Default is `false`.
 * @property {number} [compression_ratio_threshold] Treat overly repetitive decodes as failed and retry with fallback.
 * @property {number} [no_speech_threshold] Treat a low-confidence segment with high `<|nospeech|>` probability as silence.
 * @property {string} [language] The source language. Default is `null`, meaning it should be auto-detected. Use this to potentially improve performance if the source language is known.
 * @property {string} [task] The task to perform. Default is `null`, meaning it should be auto-detected.
 * @property {number} [num_frames] The number of frames in the input audio.
 * @property {boolean|{provider?: 'vad-web', pre_speech_pad_ms?: number, post_speech_pad_ms?: number, min_speech_ms?: number, min_silence_ms?: number, max_merge_gap_ms?: number, preserve_full_audio_if_empty?: boolean, debug?: boolean}} [voice_activity_detection]
 * Optional browser-side voice activity detection preprocessing. Disabled by default.
 * @typedef {import('../generation/parameters.js').GenerationFunctionParameters & AutomaticSpeechRecognitionSpecificParams} AutomaticSpeechRecognitionConfig
 *
 * @callback AutomaticSpeechRecognitionPipelineCallbackSingle Transcribe the audio sequence given as inputs to text.
 * @param {AudioInput} audio The input audio file(s) to be transcribed. The input is either:
 * - `string` or `URL` that is the filename/URL of the audio file, the file will be read at the processor's sampling rate
 * to get the waveform using the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) API.
 * If `AudioContext` is not available, you should pass the raw waveform in as a Float32Array of shape `(n, )`.
 * - `Float32Array` or `Float64Array` of shape `(n, )`, representing the raw audio at the correct sampling rate (no further check will be done).
 * @param {Partial<AutomaticSpeechRecognitionConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<AutomaticSpeechRecognitionOutput>} An object containing the transcription text and optionally timestamps if `return_timestamps` is `true`.
 *
 * @callback AutomaticSpeechRecognitionPipelineCallbackBatch Transcribe the audio sequences given as inputs to text.
 * @param {AudioInput[]} audio The input audio file(s) to be transcribed. Each entry is either:
 * - `string` or `URL` that is the filename/URL of the audio file, the file will be read at the processor's sampling rate
 * to get the waveform using the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) API.
 * If `AudioContext` is not available, you should pass the raw waveform in as a Float32Array of shape `(n, )`.
 * - `Float32Array` or `Float64Array` of shape `(n, )`, representing the raw audio at the correct sampling rate (no further check will be done).
 * @param {Partial<AutomaticSpeechRecognitionConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<AutomaticSpeechRecognitionOutput[]>} An object containing the transcription text and optionally timestamps if `return_timestamps` is `true`.
 *
 * @typedef {AutomaticSpeechRecognitionPipelineCallbackSingle & AutomaticSpeechRecognitionPipelineCallbackBatch} AutomaticSpeechRecognitionPipelineCallback
 *
 * @typedef {TextAudioPipelineConstructorArgs & AutomaticSpeechRecognitionPipelineCallback & Disposable} AutomaticSpeechRecognitionPipelineType
 */
/**
 * Pipeline that aims at extracting spoken text contained within some audio.
 *
 * **Example:** Transcribe English.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url);
 * // { text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country." }
 * ```
 *
 * **Example:** Transcribe English w/ timestamps.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url, { return_timestamps: true });
 * // {
 * //   text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
 * //   chunks: [
 * //     { timestamp: [0, 8],  text: " And so my fellow Americans ask not what your country can do for you" }
 * //     { timestamp: [8, 11], text: " ask what you can do for your country." }
 * //   ]
 * // }
 * ```
 *
 * **Example:** Transcribe English w/ word-level timestamps.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await transcriber(url, { return_timestamps: 'word' });
 * // {
 * //   "text": " And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
 * //   "chunks": [
 * //     { "text": " And", "timestamp": [0, 0.78] },
 * //     { "text": " so", "timestamp": [0.78, 1.06] },
 * //     { "text": " my", "timestamp": [1.06, 1.46] },
 * //     ...
 * //     { "text": " for", "timestamp": [9.72, 9.92] },
 * //     { "text": " your", "timestamp": [9.92, 10.22] },
 * //     { "text": " country.", "timestamp": [10.22, 13.5] }
 * //   ]
 * // }
 * ```
 *
 * **Example:** Transcribe French.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
 * const output = await transcriber(url, { language: 'french', task: 'transcribe' });
 * // { text: " J'adore, j'aime, je n'aime pas, je déteste." }
 * ```
 *
 * **Example:** Translate French to English.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
 * const output = await transcriber(url, { language: 'french', task: 'translate' });
 * // { text: " I love, I like, I don't like, I hate." }
 * ```
 *
 * **Example:** Transcribe/translate audio longer than 30 seconds.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/ted_60.wav';
 * const output = await transcriber(url, { chunk_length_s: 30, stride_length_s: 5 });
 * // { text: " So in college, I was a government major, which means [...] So I'd start off light and I'd bump it up" }
 * ```
 */
export class AutomaticSpeechRecognitionPipeline extends AutomaticSpeechRecognitionPipeline_base {
    _call(audio: any, kwargs?: {}): Promise<any>;
    _call_wav2vec2(audio: any, kwargs: any): Promise<{
        text: string;
    } | {
        text: string;
    }[]>;
    _call_whisper(audio: any, kwargs: any): Promise<any>;
    _transcribeWhisperVadSegments({ audio, vadResult, generation_config, return_timestamps, time_precision, force_full_sequences, timestamp_begin, hop_length, sampling_rate, chunk_length_s, stride_length_s, }: {
        audio: any;
        vadResult: any;
        generation_config: any;
        return_timestamps: any;
        time_precision: any;
        force_full_sequences: any;
        timestamp_begin: any;
        hop_length: any;
        sampling_rate: any;
        chunk_length_s: any;
        stride_length_s: any;
    }): Promise<{
        output: {
            text: any;
            chunks?: undefined;
        } | {
            text: any;
            chunks: any;
        };
        analyses: {
            start_s: any;
            end_s: any;
            duration_s: number;
            text_token_count: any;
            avg_logprob: any;
            tokens_per_second: number;
            suspicious: boolean;
        }[];
    }>;
    _shouldUseSegmentedVadTranscription(vadResult: any): boolean;
    _transcribeWhisperAudio({ audio, generation_config, return_timestamps, time_precision, force_full_sequences, timestamp_begin, hop_length, sampling_rate, chunk_length_s, stride_length_s, }: {
        audio: any;
        generation_config: any;
        return_timestamps: any;
        time_precision: any;
        force_full_sequences: any;
        timestamp_begin: any;
        hop_length: any;
        sampling_rate: any;
        chunk_length_s: any;
        stride_length_s: any;
    }): Promise<{
        output: any;
        analyses: {
            start_s: any;
            end_s: any;
            duration_s: number;
            text_token_count: any;
            avg_logprob: any;
            tokens_per_second: number;
            suspicious: boolean;
        }[];
    }>;
    _normalizeWordTimestampOutput(output: any): any;
    _shiftOutputTimestamps(output: any, offset_s: any): any;
    _combineTranscriptionOutputs(outputs: any, return_timestamps: any): {
        text: any;
        chunks?: undefined;
    } | {
        text: any;
        chunks: any;
    };
    _filterWordOutputToSentenceText(output: any, sentence_text: any): any;
    _getStrictRecoveryGenerationConfig(generation_config: any): any;
    _createWhisperChunks(audio: any, chunk_length_s: any, stride_length_s: any, sampling_rate: any): Promise<{
        stride: number[];
        input_features: Tensor;
        is_last: boolean;
    }[]>;
    _buildChunkAnalysis(result: any, start_s: any, end_s: any, logprob_threshold: any): {
        start_s: any;
        end_s: any;
        duration_s: number;
        text_token_count: any;
        avg_logprob: any;
        tokens_per_second: number;
        suspicious: boolean;
    };
    _remapChunkAnalyses(analyses: any, vadResult: any): any;
    _isSuspiciousChunkResult(result: any, chunk_duration_s: any, logprob_threshold: any): boolean;
    _collectFallbackWindows(analyses: any, vadResult: any): any[];
    _collectRemovedVadGapWindows(vadResult: any): {
        start_s: number;
        end_s: any;
        remapped: boolean;
    }[];
    _shouldPreferFallbackAnalysis(baseAnalysis: any, fallbackAnalysis: any): boolean;
    _scoreAnalysisWindow(analyses: any, start_s: any, end_s: any): {
        text_token_count: number;
        tokens_per_second: number;
        avg_logprob: number;
        suspicious: boolean;
    };
    /**
     * Processes a single audio chunk, detecting hallucination (very low token density)
     * and recursively splitting into smaller sub-chunks when needed.
     * @private
     */
    private _processChunkWithRetry;
    _generateChunkResult(chunk: any, generation_config: any, return_timestamps: any, timestamp_begin: any, hop_length: any, sampling_rate: any): Promise<{
        chunks: {
            stride: any;
            is_last: any;
        }[];
        total_logprob: any;
        text_token_count: any;
        avg_logprob: number;
        has_valid_timestamps: boolean;
    }>;
    _buildChunkResult({ chunk, data, return_timestamps, timestamp_begin, sampling_rate, timestamp_shift_s }: {
        chunk: any;
        data: any;
        return_timestamps: any;
        timestamp_begin: any;
        sampling_rate: any;
        timestamp_shift_s?: number;
    }): {
        chunks: {
            stride: any;
            is_last: any;
        }[];
        total_logprob: any;
        text_token_count: any;
        avg_logprob: number;
        has_valid_timestamps: boolean;
    };
    _countTextTokens(tokens: any, timestamp_begin: any): any;
    _hasValidTokenTimestamps(token_timestamps: any): boolean;
    _shouldRetryChunk(result: any, chunk_duration_s: any, logprob_threshold: any): boolean;
    _chooseBetterChunkResult(a: any, b: any): any;
    _combineChunkResults(results: any): {
        chunks: any[];
        total_logprob: number;
        text_token_count: number;
        avg_logprob: number;
        has_valid_timestamps: boolean;
    };
    _splitChunkWithRetry(chunk: any, fullAudio: any, generation_config: any, return_timestamps: any, timestamp_begin: any, hop_length: any, sampling_rate: any, audioOffset: any, depth: any): Promise<{
        chunks: any[];
        total_logprob: number;
        text_token_count: number;
        avg_logprob: number;
        has_valid_timestamps: boolean;
    }>;
    _generatePaddedChunkResult(chunk: any, fullAudio: any, generation_config: any, return_timestamps: any, timestamp_begin: any, hop_length: any, sampling_rate: any, audioOffset: any): Promise<{
        chunks: {
            stride: any;
            is_last: any;
        }[];
        total_logprob: any;
        text_token_count: any;
        avg_logprob: number;
        has_valid_timestamps: boolean;
    }>;
    _call_moonshine(audio: any, kwargs: any): Promise<{
        text: string;
    } | {
        text: string;
    }[]>;
    _call_cohere_asr(audio: any, kwargs: any): Promise<{
        text: any;
    } | {
        text: any;
    }[]>;
}
export type TextAudioPipelineConstructorArgs = import("./_base.js").TextAudioPipelineConstructorArgs;
export type Disposable = import("./_base.js").Disposable;
export type AudioInput = import("./_base.js").AudioInput;
export type Chunk = {
    /**
     * The start and end timestamp of the chunk in seconds.
     */
    timestamp: [number, number];
    /**
     * The recognized text.
     */
    text: string;
};
export type AutomaticSpeechRecognitionOutput = {
    /**
     * The recognized text.
     */
    text: string;
    /**
     * When using `return_timestamps`, the `chunks` will become a list
     * containing all the various text chunks identified by the model.
     */
    chunks?: Chunk[];
};
/**
 * Parameters specific to automatic-speech-recognition pipelines.
 */
export type AutomaticSpeechRecognitionSpecificParams = {
    /**
     * Whether to return timestamps or not. Default is `false`.
     */
    return_timestamps?: boolean | "word";
    /**
     * The length of audio chunks to process in seconds. Default is 0 (no chunking).
     */
    chunk_length_s?: number;
    /**
     * The length of overlap between consecutive audio chunks in seconds. If not provided, defaults to `chunk_length_s / 6`.
     */
    stride_length_s?: number;
    /**
     * Whether to force outputting full sequences or not. Default is `false`.
     */
    force_full_sequences?: boolean;
    /**
     * Whether to enable best-effort Whisper hallucination recovery. Default is `true`.
     */
    hallucination_recovery?: boolean;
    /**
     * Optional prompt text used to bias Whisper toward expected names or terms.
     */
    initial_prompt?: string;
    /**
     * Whisper prompt token ids, typically produced from `initial_prompt`.
     */
    prompt_ids?: number[];
    /**
     * Whether prompt text should be reapplied to every sequential Whisper segment. Default is `false`.
     */
    carry_initial_prompt?: boolean;
    /**
     * Treat overly repetitive decodes as failed and retry with fallback.
     */
    compression_ratio_threshold?: number;
    /**
     * Treat a low-confidence segment with high `<|nospeech|>` probability as silence.
     */
    no_speech_threshold?: number;
    /**
     * The source language. Default is `null`, meaning it should be auto-detected. Use this to potentially improve performance if the source language is known.
     */
    language?: string;
    /**
     * The task to perform. Default is `null`, meaning it should be auto-detected.
     */
    task?: string;
    /**
     * The number of frames in the input audio.
     */
    num_frames?: number;
    /**
     * Optional browser-side voice activity detection preprocessing. Disabled by default.
     */
    voice_activity_detection?: boolean | {
        provider?: "vad-web";
        pre_speech_pad_ms?: number;
        post_speech_pad_ms?: number;
        min_speech_ms?: number;
        min_silence_ms?: number;
        max_merge_gap_ms?: number;
        preserve_full_audio_if_empty?: boolean;
        debug?: boolean;
    };
};
export type AutomaticSpeechRecognitionConfig = import("../generation/parameters.js").GenerationFunctionParameters & AutomaticSpeechRecognitionSpecificParams;
/**
 * Transcribe the audio sequence given as inputs to text.
 */
export type AutomaticSpeechRecognitionPipelineCallbackSingle = (audio: AudioInput, options?: Partial<AutomaticSpeechRecognitionConfig>) => Promise<AutomaticSpeechRecognitionOutput>;
/**
 * Transcribe the audio sequences given as inputs to text.
 */
export type AutomaticSpeechRecognitionPipelineCallbackBatch = (audio: AudioInput[], options?: Partial<AutomaticSpeechRecognitionConfig>) => Promise<AutomaticSpeechRecognitionOutput[]>;
export type AutomaticSpeechRecognitionPipelineCallback = AutomaticSpeechRecognitionPipelineCallbackSingle & AutomaticSpeechRecognitionPipelineCallbackBatch;
export type AutomaticSpeechRecognitionPipelineType = TextAudioPipelineConstructorArgs & AutomaticSpeechRecognitionPipelineCallback & Disposable;
import { Tensor } from '../utils/tensor.js';
export {};
//# sourceMappingURL=automatic-speech-recognition.d.ts.map