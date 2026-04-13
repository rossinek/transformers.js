import { Pipeline, prepareAudios } from './_base.js';
import {
    preprocessAudioWithVoiceActivityDetection,
    remapCompactTimestamp,
    remapWhisperOutputTimestamps,
} from './vad/index.js';

import { Tensor } from '../utils/tensor.js';
import { max, round } from '../utils/maths.js';
import { logger } from '../utils/logger.js';

const MIN_HALLUCINATION_CHUNK_S = 2.0;
const MAX_HALLUCINATION_DEPTH = 3;
const MIN_HALLUCINATION_TOKENS_PER_SECOND = 1.0;
const HARD_HALLUCINATION_TOKENS_PER_SECOND = 0.6;
const MIN_SUSPICIOUS_LOGPROB_THRESHOLD = -0.45;
const LOW_TEXT_TOKEN_COUNT_TRIGGER = 12;
const TRUNCATION_GUARD_MIN_RATIO = 0.5;
const TRUNCATION_GUARD_SCORE_MARGIN = 0.5;
const VAD_FALLBACK_WINDOW_PADDING_S = 0.75;
const MIN_VAD_REMOVED_GAP_S = 8.0;
const MIN_VAD_REMOVED_GAP_TOTAL_S = 12.0;
const MIN_VAD_REMOVED_GAP_RATIO = 0.2;
const MAX_SEGMENTED_VAD_SEGMENTS = 2;

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
export class AutomaticSpeechRecognitionPipeline
    extends /** @type {new (options: TextAudioPipelineConstructorArgs) => AutomaticSpeechRecognitionPipelineType} */ (
        Pipeline
    )
{
    async _call(audio, kwargs = {}) {
        switch (this.model.config.model_type) {
            case 'whisper':
            case 'lite-whisper':
                return this._call_whisper(audio, kwargs);
            case 'wav2vec2':
            case 'wav2vec2-bert':
            case 'unispeech':
            case 'unispeech-sat':
            case 'hubert':
            case 'parakeet_ctc':
                return this._call_wav2vec2(audio, kwargs);
            case 'moonshine':
                return this._call_moonshine(audio, kwargs);
            case 'cohere_asr':
                return this._call_cohere_asr(audio, kwargs);
            default:
                throw new Error(
                    `AutomaticSpeechRecognitionPipeline does not support model type '${this.model.config.model_type}'.`,
                );
        }
    }

    async _call_wav2vec2(audio, kwargs) {
        // TODO use kwargs

        if (kwargs.language) {
            logger.warn('`language` parameter is not yet supported for `wav2vec2` models, defaulting to "English".');
        }
        if (kwargs.task) {
            logger.warn('`task` parameter is not yet supported for `wav2vec2` models, defaulting to "transcribe".');
        }

        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;

        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);
            const output = await this.model(inputs);
            const logits = output.logits[0];

            const predicted_ids = [];
            for (const item of logits) {
                predicted_ids.push(max(item.data)[1]);
            }
            const predicted_sentences = this.tokenizer.decode(predicted_ids, { skip_special_tokens: true }).trim();
            toReturn.push({ text: predicted_sentences });
        }
        return single ? toReturn[0] : toReturn;
    }

    async _call_whisper(audio, kwargs) {
        const return_timestamps = kwargs.return_timestamps ?? false;
        const chunk_length_s = kwargs.chunk_length_s ?? 0;
        const force_full_sequences = kwargs.force_full_sequences ?? false;
        const hallucination_recovery = kwargs.hallucination_recovery ?? false;
        const voice_activity_detection = kwargs.voice_activity_detection ?? false;
        let stride_length_s = kwargs.stride_length_s ?? null;

        const generation_config = { ...kwargs };
        generation_config['hallucination_recovery'] = hallucination_recovery;
        delete generation_config['voice_activity_detection'];
        if (generation_config.initial_prompt && generation_config.prompt_ids == null) {
            generation_config.prompt_ids = /** @type {{ get_prompt_ids?: (text: string) => number[] }} */ (this.processor)
                .get_prompt_ids?.(generation_config.initial_prompt) ?? null;
        }
        delete generation_config['initial_prompt'];

        if (generation_config.no_speech_threshold != null) {
            const no_speech_token_id = this.tokenizer?._tokenizer?.token_to_id?.('<|nospeech|>');
            if (no_speech_token_id != null) {
                generation_config['no_speech_token_id'] = no_speech_token_id;
            }
        }
        if (return_timestamps === 'word') {
            generation_config['return_token_timestamps'] = true;
            generation_config['return_timestamps'] = true;
        }
        if (hallucination_recovery) {
            generation_config['return_dict_in_generate'] = true;
        }

        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;
        const feature_extractor_config = this.processor.feature_extractor.config;

        // @ts-expect-error TS2339
        const time_precision = feature_extractor_config.chunk_length / this.model.config.max_source_positions;
        const hop_length = feature_extractor_config.hop_length;

        const sampling_rate = feature_extractor_config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const toReturn = [];
        for (const aud of preparedAudios) {
            const vadResult = await preprocessAudioWithVoiceActivityDetection(
                aud,
                sampling_rate,
                voice_activity_detection,
            );
            if (vadResult.applied && vadResult.segments.length === 0) {
                const output = return_timestamps ? { text: '', chunks: [] } : { text: '' };
                if (voice_activity_detection && vadResult.debugMetadata) {
                    output.voice_activity_detection = vadResult.debugMetadata;
                }
                toReturn.push(output);
                continue;
            }

            // @ts-expect-error ts(2339)
            const timestamp_begin = this.tokenizer.timestamp_begin;
            const useSegmentedVadPath = this._shouldUseSegmentedVadTranscription(vadResult);
            const activePath = useSegmentedVadPath
                ? await this._transcribeWhisperVadSegments({
                      audio: aud,
                      vadResult,
                      generation_config,
                      return_timestamps,
                      time_precision,
                      force_full_sequences,
                      timestamp_begin,
                      hop_length,
                      sampling_rate,
                      chunk_length_s,
                      stride_length_s,
                  })
                : await this._transcribeWhisperAudio({
                      audio: vadResult.applied ? vadResult.processedAudio : aud,
                      generation_config,
                      return_timestamps,
                      time_precision,
                      force_full_sequences,
                      timestamp_begin,
                      hop_length,
                      sampling_rate,
                      chunk_length_s,
                      stride_length_s,
                  });

            let output = activePath.output;

            if (return_timestamps === 'word') {
                this._normalizeWordTimestampOutput(output);
            }

            if (vadResult.applied && !useSegmentedVadPath) {
                output = remapWhisperOutputTimestamps(output, vadResult.segments, vadResult.originalDurationS);
            }

            if (voice_activity_detection && vadResult.debugMetadata) {
                output.voice_activity_detection = vadResult.debugMetadata;
            }

            toReturn.push(output);
        }
        return single ? toReturn[0] : toReturn;
    }

    async _transcribeWhisperVadSegments({
        audio,
        vadResult,
        generation_config,
        return_timestamps,
        time_precision,
        force_full_sequences,
        timestamp_begin,
        hop_length,
        sampling_rate,
        chunk_length_s,
        stride_length_s,
    }) {
        const outputs = [];
        const analyses = [];

        for (const segment of vadResult.segments) {
            const startIndex = Math.max(0, Math.floor(segment.original_start_s * sampling_rate));
            const endIndex = Math.min(audio.length, Math.ceil(segment.original_end_s * sampling_rate));
            if (endIndex <= startIndex) {
                continue;
            }

            const segmentAudio = audio.subarray(startIndex, endIndex);
            const segmentPath = await this._transcribeWhisperAudio({
                audio: segmentAudio,
                generation_config,
                return_timestamps,
                time_precision,
                force_full_sequences,
                timestamp_begin,
                hop_length,
                sampling_rate,
                chunk_length_s,
                stride_length_s,
            });

            this._shiftOutputTimestamps(segmentPath.output, segment.original_start_s);
            analyses.push(
                ...segmentPath.analyses.map((analysis) => ({
                    ...analysis,
                    start_s: analysis.start_s + segment.original_start_s,
                    end_s: analysis.end_s + segment.original_start_s,
                })),
            );
            outputs.push(segmentPath.output);
        }

        return {
            output: this._combineTranscriptionOutputs(outputs, return_timestamps),
            analyses,
        };
    }

    _shouldUseSegmentedVadTranscription(vadResult) {
        return vadResult.applied && vadResult.segments.length > 1 && vadResult.segments.length <= MAX_SEGMENTED_VAD_SEGMENTS;
    }

    async _transcribeWhisperAudio({
        audio,
        generation_config,
        return_timestamps,
        time_precision,
        force_full_sequences,
        timestamp_begin,
        hop_length,
        sampling_rate,
        chunk_length_s,
        stride_length_s,
    }) {
        const chunks = await this._createWhisperChunks(audio, chunk_length_s, stride_length_s, sampling_rate);
        const processedChunks = [];
        const analyses = [];
        const hallucination_recovery = generation_config.hallucination_recovery !== false;
        const decode_generation_config =
            return_timestamps && generation_config.return_token_timestamps !== true
                ? {
                      ...generation_config,
                      return_token_timestamps: true,
                  }
                : generation_config;
        const chunkJump =
            chunk_length_s > 0
                ? sampling_rate * chunk_length_s - 2 * sampling_rate * (stride_length_s ?? chunk_length_s / 6)
                : 0;

        for (let ci = 0; ci < chunks.length; ++ci) {
            const audioOffset = ci * chunkJump;
            const chunk_generation_config =
                ci === 0 || decode_generation_config.carry_initial_prompt || decode_generation_config.prompt_ids == null
                    ? decode_generation_config
                    : {
                          ...decode_generation_config,
                          prompt_ids: null,
                      };
            let result;

            if (hallucination_recovery) {
                result = await this._processChunkWithRetry(
                    chunks[ci],
                    audio,
                    chunk_generation_config,
                    return_timestamps,
                    timestamp_begin,
                    hop_length,
                    sampling_rate,
                    audioOffset,
                );
            } else {
                result = await this._generateChunkResult(
                    chunks[ci],
                    chunk_generation_config,
                    return_timestamps,
                    timestamp_begin,
                    hop_length,
                    sampling_rate,
                );
            }

            processedChunks.push(...result.chunks);
            analyses.push(
                this._buildChunkAnalysis(
                    result,
                    audioOffset / sampling_rate,
                    (audioOffset + chunks[ci].stride[0]) / sampling_rate,
                    chunk_generation_config.logprob_threshold ?? -1.0,
                ),
            );
        }

        // @ts-ignore
        const [full_text, optional] = this.tokenizer._decode_asr(processedChunks, {
            time_precision,
            return_timestamps,
            force_full_sequences,
        });

        const output = { text: full_text, ...optional };

        return {
            output,
            analyses,
        };
    }

    _normalizeWordTimestampOutput(output) {
        if (!Array.isArray(output?.chunks)) {
            return output;
        }

        output.chunks = output.chunks
            .filter((chunk) => typeof chunk?.text === 'string' && Array.isArray(chunk.timestamp) && chunk.timestamp.length === 2)
            .sort((a, b) => {
                const a_start = typeof a.timestamp[0] === 'number' ? a.timestamp[0] : a.timestamp[1] ?? Infinity;
                const b_start = typeof b.timestamp[0] === 'number' ? b.timestamp[0] : b.timestamp[1] ?? Infinity;
                if (a_start !== b_start) return a_start - b_start;

                const a_end = typeof a.timestamp[1] === 'number' ? a.timestamp[1] : a.timestamp[0] ?? Infinity;
                const b_end = typeof b.timestamp[1] === 'number' ? b.timestamp[1] : b.timestamp[0] ?? Infinity;
                return a_end - b_end;
            });

        const deduped = [];
        for (const chunk of output.chunks) {
            const prev = deduped[deduped.length - 1];
            if (!prev) {
                deduped.push(chunk);
                continue;
            }

            const prev_start = prev.timestamp[0] ?? prev.timestamp[1];
            const prev_end = prev.timestamp[1] ?? prev.timestamp[0];
            const current_start = chunk.timestamp[0] ?? chunk.timestamp[1];
            const current_end = chunk.timestamp[1] ?? chunk.timestamp[0];
            if (
                typeof prev_start === 'number' &&
                typeof prev_end === 'number' &&
                typeof current_start === 'number' &&
                typeof current_end === 'number' &&
                prev.text.trim() === chunk.text.trim() &&
                Math.abs(prev_start - current_start) <= 0.1 &&
                Math.abs(prev_end - current_end) <= 0.1
            ) {
                continue;
            }

            deduped.push(chunk);
        }

        output.chunks = deduped;
        if (typeof output.text !== 'string' || output.text.trim().length === 0) {
            output.text = deduped.map((chunk) => chunk.text).join('').trim();
        }
        return output;
    }

    _shiftOutputTimestamps(output, offset_s) {
        if (!Array.isArray(output?.chunks) || !Number.isFinite(offset_s) || offset_s === 0) {
            return output;
        }

        for (const chunk of output.chunks) {
            if (!Array.isArray(chunk?.timestamp) || chunk.timestamp.length !== 2) {
                continue;
            }

            if (typeof chunk.timestamp[0] === 'number') {
                chunk.timestamp[0] = round(chunk.timestamp[0] + offset_s, 2);
            }
            if (typeof chunk.timestamp[1] === 'number') {
                chunk.timestamp[1] = round(chunk.timestamp[1] + offset_s, 2);
            }
        }
        return output;
    }

    _combineTranscriptionOutputs(outputs, return_timestamps) {
        const filtered = outputs.filter((output) => typeof output?.text === 'string' || Array.isArray(output?.chunks));
        const combinedChunks = return_timestamps ? filtered.flatMap((output) => output.chunks ?? []) : null;
        const text =
            combinedChunks && combinedChunks.length > 0
                ? combinedChunks.map((chunk) => chunk.text ?? '').join('').trim()
                : filtered.map((output) => output.text ?? '').join('').trim();

        if (!return_timestamps) {
            return { text };
        }

        return {
            text,
            chunks: combinedChunks ?? [],
        };
    }

    _getStrictRecoveryGenerationConfig(generation_config) {
        if (generation_config.logprob_threshold != null) {
            return generation_config;
        }
        return {
            ...generation_config,
            logprob_threshold: MIN_SUSPICIOUS_LOGPROB_THRESHOLD,
        };
    }

    async _createWhisperChunks(audio, chunk_length_s, stride_length_s, sampling_rate) {
        /** @type {{stride: number[], input_features: Tensor, is_last: boolean}[]} */
        const chunks = [];
        if (chunk_length_s > 0) {
            if (stride_length_s === null) {
                stride_length_s = chunk_length_s / 6;
            } else if (chunk_length_s <= stride_length_s) {
                throw Error('`chunk_length_s` must be larger than `stride_length_s`.');
            }

            const window = sampling_rate * chunk_length_s;
            const stride = sampling_rate * stride_length_s;
            const jump = window - 2 * stride;
            let offset = 0;

            while (true) {
                const offset_end = offset + window;
                const subarr = audio.subarray(offset, offset_end);
                const feature = await this.processor(subarr);

                const is_first = offset === 0;
                const is_last = offset_end >= audio.length;
                chunks.push({
                    stride: [subarr.length, is_first ? 0 : stride, is_last ? 0 : stride],
                    input_features: feature.input_features,
                    is_last,
                });
                if (is_last) break;
                offset += jump;
            }
        } else {
            chunks.push({
                stride: [audio.length, 0, 0],
                input_features: (await this.processor(audio)).input_features,
                is_last: true,
            });
        }

        return chunks;
    }

    _buildChunkAnalysis(result, start_s, end_s, logprob_threshold) {
        const duration_s = Math.max(end_s - start_s, 0.1);
        const tokens_per_second = result.text_token_count / duration_s;

        return {
            start_s,
            end_s,
            duration_s,
            text_token_count: result.text_token_count,
            avg_logprob: result.avg_logprob,
            tokens_per_second,
            suspicious: this._isSuspiciousChunkResult(result, duration_s, logprob_threshold),
        };
    }

    _remapChunkAnalyses(analyses, vadResult) {
        if (!vadResult.applied || vadResult.segments.length === 0) {
            return analyses;
        }

        return analyses.map((analysis) => ({
            ...analysis,
            start_s: remapCompactTimestamp(analysis.start_s, vadResult.segments, 'start'),
            end_s: remapCompactTimestamp(analysis.end_s, vadResult.segments, 'end'),
            duration_s: Math.max(
                remapCompactTimestamp(analysis.end_s, vadResult.segments, 'end') -
                    remapCompactTimestamp(analysis.start_s, vadResult.segments, 'start'),
                0.1,
            ),
        }));
    }

    _isSuspiciousChunkResult(result, chunk_duration_s, logprob_threshold) {
        const tokens_per_second = result.text_token_count / Math.max(chunk_duration_s, 0.1);
        if (result.text_token_count === 0) {
            return true;
        }
        if (
            tokens_per_second < HARD_HALLUCINATION_TOKENS_PER_SECOND &&
            result.text_token_count <= LOW_TEXT_TOKEN_COUNT_TRIGGER
        ) {
            return true;
        }

        const effective_logprob_threshold = Math.max(logprob_threshold, MIN_SUSPICIOUS_LOGPROB_THRESHOLD);
        const low_text_coverage = tokens_per_second < MIN_HALLUCINATION_TOKENS_PER_SECOND;
        const low_logprob =
            result.avg_logprob === null
                ? result.text_token_count === 0
                : result.avg_logprob < effective_logprob_threshold;
        return low_text_coverage && (low_logprob || result.text_token_count <= LOW_TEXT_TOKEN_COUNT_TRIGGER);
    }

    _collectFallbackWindows(analyses, vadResult) {
        const sourceWindows = analyses.filter((analysis) => {
            if (analysis.suspicious || !vadResult.applied || vadResult.segments.length === 0) {
                return analysis.suspicious;
            }

            const remappedStart = remapCompactTimestamp(analysis.start_s, vadResult.segments, 'start');
            const remappedEnd = remapCompactTimestamp(analysis.end_s, vadResult.segments, 'end');
            const remappedDuration = Math.max(remappedEnd - remappedStart, 0.1);
            return analysis.text_token_count / remappedDuration < HARD_HALLUCINATION_TOKENS_PER_SECOND;
            });
        sourceWindows.push(...this._collectRemovedVadGapWindows(vadResult));
        if (sourceWindows.length === 0) {
            return [];
        }

        const windows = sourceWindows
            .map((analysis) => {
                let start_s = analysis.start_s;
                let end_s = analysis.end_s;

                if (analysis.remapped !== true && vadResult.applied && vadResult.segments.length > 0) {
                    start_s = remapCompactTimestamp(start_s, vadResult.segments, 'start');
                    end_s = remapCompactTimestamp(end_s, vadResult.segments, 'end');
                }

                start_s = Math.max(0, start_s - VAD_FALLBACK_WINDOW_PADDING_S);
                end_s = Math.min(vadResult.originalDurationS, end_s + VAD_FALLBACK_WINDOW_PADDING_S);
                return { start_s, end_s };
            })
            .sort((a, b) => a.start_s - b.start_s);

        const merged = [windows[0]];
        for (let i = 1; i < windows.length; ++i) {
            const current = windows[i];
            const previous = merged[merged.length - 1];
            if (current.start_s <= previous.end_s) {
                previous.end_s = Math.max(previous.end_s, current.end_s);
            } else {
                merged.push({ ...current });
            }
        }

        return merged;
    }

    _collectRemovedVadGapWindows(vadResult) {
        if (!vadResult.applied || vadResult.segments.length === 0) {
            return [];
        }

        const windows = [];
        let cursor = 0;

        for (const segment of vadResult.segments) {
            const gapDuration = segment.original_start_s - cursor;
            if (gapDuration >= MIN_VAD_REMOVED_GAP_S) {
                windows.push({
                    start_s: cursor,
                    end_s: segment.original_start_s,
                    remapped: true,
                });
            }
            cursor = segment.original_end_s;
        }

        const trailingGap = vadResult.originalDurationS - cursor;
        if (trailingGap >= MIN_VAD_REMOVED_GAP_S) {
            windows.push({
                start_s: cursor,
                end_s: vadResult.originalDurationS,
                remapped: true,
            });
        }

        const totalGapDuration = windows.reduce((sum, window) => sum + (window.end_s - window.start_s), 0);
        const minUsefulDuration = Math.max(
            MIN_VAD_REMOVED_GAP_TOTAL_S,
            vadResult.originalDurationS * MIN_VAD_REMOVED_GAP_RATIO,
        );
        if (totalGapDuration < minUsefulDuration) {
            return [];
        }

        return windows;
    }

    _shouldPreferFallbackAnalysis(baseAnalysis, fallbackAnalysis) {
        if (fallbackAnalysis.text_token_count === 0) {
            return false;
        }
        if (baseAnalysis.text_token_count === 0) {
            return true;
        }
        if (baseAnalysis.suspicious && !fallbackAnalysis.suspicious) {
            return true;
        }

        const fallbackLogprob = fallbackAnalysis.avg_logprob ?? -Infinity;
        const baseLogprob = baseAnalysis.avg_logprob ?? -Infinity;
        return (
            fallbackAnalysis.tokens_per_second > baseAnalysis.tokens_per_second + 0.5 &&
            fallbackLogprob >= baseLogprob - 0.1
        );
    }

    _scoreAnalysisWindow(analyses, start_s, end_s) {
        const overlapping = analyses.filter((analysis) => analysis.start_s <= end_s && analysis.end_s >= start_s);
        if (overlapping.length === 0) {
            return {
                text_token_count: 0,
                tokens_per_second: 0,
                avg_logprob: null,
                suspicious: true,
            };
        }

        let text_token_count = 0;
        let total_duration_s = 0;
        let total_logprob = 0;
        let has_any_score = false;

        for (const analysis of overlapping) {
            const overlap_duration = Math.max(0, Math.min(analysis.end_s, end_s) - Math.max(analysis.start_s, start_s));
            if (overlap_duration <= 0) {
                continue;
            }

            text_token_count += analysis.text_token_count;
            total_duration_s += overlap_duration;
            if (analysis.avg_logprob !== null) {
                total_logprob += analysis.avg_logprob * analysis.text_token_count;
                has_any_score = true;
            }
        }

        const duration_s = Math.max(total_duration_s, 0.1);
        const tokens_per_second = text_token_count / duration_s;
        const avg_logprob = has_any_score && text_token_count > 0 ? total_logprob / text_token_count : null;
        const suspicious =
            text_token_count === 0 ||
            (tokens_per_second < HARD_HALLUCINATION_TOKENS_PER_SECOND &&
                text_token_count <= LOW_TEXT_TOKEN_COUNT_TRIGGER) ||
            (tokens_per_second < MIN_HALLUCINATION_TOKENS_PER_SECOND &&
                (text_token_count <= LOW_TEXT_TOKEN_COUNT_TRIGGER ||
                    (avg_logprob !== null && avg_logprob < MIN_SUSPICIOUS_LOGPROB_THRESHOLD)));

        return {
            text_token_count,
            tokens_per_second,
            avg_logprob,
            suspicious,
        };
    }

    /**
     * Processes a single audio chunk, detecting hallucination (very low token density)
     * and recursively splitting into smaller sub-chunks when needed.
     * @private
     */
    async _processChunkWithRetry(
        chunk,
        fullAudio,
        generation_config,
        return_timestamps,
        timestamp_begin,
        hop_length,
        sampling_rate,
        audioOffset = 0,
        depth = 0,
    ) {
        const chunk_duration_s = chunk.stride[0] / sampling_rate;
        const logprob_threshold = generation_config.logprob_threshold ?? -1.0;
        const original = await this._generateChunkResult(
            chunk,
            generation_config,
            return_timestamps,
            timestamp_begin,
            hop_length,
            sampling_rate,
        );

        if (
            chunk_duration_s <= MIN_HALLUCINATION_CHUNK_S ||
            !this._shouldRetryChunk(original, chunk_duration_s, logprob_threshold)
        ) {
            return original;
        }

        const recovery_generation_config = this._getStrictRecoveryGenerationConfig(generation_config);
        let best = original;

        if (recovery_generation_config !== generation_config) {
            const strict = await this._generateChunkResult(
                chunk,
                recovery_generation_config,
                return_timestamps,
                timestamp_begin,
                hop_length,
                sampling_rate,
            );
            best = this._chooseBetterChunkResult(best, strict);
        }

        if (depth < MAX_HALLUCINATION_DEPTH) {
            const split = await this._splitChunkWithRetry(
                chunk,
                fullAudio,
                recovery_generation_config,
                return_timestamps,
                timestamp_begin,
                hop_length,
                sampling_rate,
                audioOffset,
                depth,
            );
            return this._chooseBetterChunkResult(best, split);
        }

        const padded = await this._generatePaddedChunkResult(
            chunk,
            fullAudio,
            recovery_generation_config,
            return_timestamps,
            timestamp_begin,
            hop_length,
            sampling_rate,
            audioOffset,
        );
        return this._chooseBetterChunkResult(best, padded);
    }

    async _generateChunkResult(
        chunk,
        generation_config,
        return_timestamps,
        timestamp_begin,
        hop_length,
        sampling_rate,
    ) {
        generation_config.num_frames = Math.floor(chunk.stride[0] / hop_length);

        const data = await this.model.generate({
            inputs: chunk.input_features,
            ...generation_config,
        });

        return this._buildChunkResult({
            chunk,
            data,
            return_timestamps,
            timestamp_begin,
            sampling_rate,
        });
    }

    _buildChunkResult({ chunk, data, return_timestamps, timestamp_begin, sampling_rate, timestamp_shift_s = 0 }) {
        const outputChunk = {
            stride: chunk.stride.map((x) => x / sampling_rate),
            is_last: chunk.is_last,
        };

        const token_timestamps =
            data?.token_timestamps?.tolist?.()?.[0]?.map((/** @type {number} */ x) =>
                round(Math.max(0, x + timestamp_shift_s), 2),
            ) ?? null;

        if (return_timestamps === 'word') {
            const sequences = data.sequences.tolist()[0];
            const prefixLength = Math.max(
                sequences.findIndex((/** @type {bigint} */ t) => Number(t) >= timestamp_begin),
                0,
            );
            outputChunk.tokens = sequences.slice(prefixLength);
            if (token_timestamps) {
                outputChunk.token_timestamps = token_timestamps.slice(prefixLength);
            }
        } else {
            const sequences = data?.sequences ?? data;
            outputChunk.tokens = /** @type {Tensor} */ (sequences)[0].tolist();
            if (token_timestamps) {
                outputChunk.token_timestamps = token_timestamps;
            }
        }

        const text_token_count = this._countTextTokens(outputChunk.tokens ?? [], timestamp_begin);
        const total_logprob = data?.scores?.[0] ?? null;

        return {
            chunks: [outputChunk],
            total_logprob,
            text_token_count,
            avg_logprob: total_logprob !== null && text_token_count > 0 ? total_logprob / text_token_count : null,
            has_valid_timestamps:
                outputChunk.token_timestamps === undefined ||
                this._hasValidTokenTimestamps(outputChunk.token_timestamps),
        };
    }

    _countTextTokens(tokens, timestamp_begin) {
        const all_special_ids = new Set(this.tokenizer.all_special_ids);
        return tokens.filter((/** @type {bigint|number} */ token) => {
            const value = Number(token);
            return value < timestamp_begin && !all_special_ids.has(value);
        }).length;
    }

    _hasValidTokenTimestamps(token_timestamps) {
        let prev = -Infinity;
        for (const timestamp of token_timestamps) {
            if (!Number.isFinite(timestamp) || timestamp < 0 || timestamp < prev) {
                return false;
            }
            prev = timestamp;
        }
        return true;
    }

    _shouldRetryChunk(result, chunk_duration_s, logprob_threshold) {
        return this._isSuspiciousChunkResult(result, chunk_duration_s, logprob_threshold);
    }

    _chooseBetterChunkResult(a, b) {
        const pickBetterByScore = (first, second) => {
            const first_score = first.avg_logprob ?? -Infinity;
            const second_score = second.avg_logprob ?? -Infinity;
            return first_score >= second_score ? first : second;
        };

        const first_empty = a.text_token_count === 0;
        const second_empty = b.text_token_count === 0;
        if (first_empty !== second_empty) {
            return first_empty ? b : a;
        }

        if (a.has_valid_timestamps !== b.has_valid_timestamps) {
            return a.has_valid_timestamps ? a : b;
        }

        const better = pickBetterByScore(a, b);
        const other = better === a ? b : a;
        if (
            better.text_token_count > 0 &&
            other.text_token_count > 0 &&
            better.text_token_count < other.text_token_count * TRUNCATION_GUARD_MIN_RATIO
        ) {
            const score_margin = (better.avg_logprob ?? -Infinity) - (other.avg_logprob ?? -Infinity);
            if (score_margin < TRUNCATION_GUARD_SCORE_MARGIN) {
                return other;
            }
        }

        return better;
    }

    _combineChunkResults(results) {
        const chunks = [];
        let total_logprob = 0;
        let has_any_score = false;
        let text_token_count = 0;
        let has_valid_timestamps = true;

        for (const result of results) {
            chunks.push(...result.chunks);
            text_token_count += result.text_token_count;
            has_valid_timestamps &&= result.has_valid_timestamps;
            if (result.total_logprob !== null) {
                total_logprob += result.total_logprob;
                has_any_score = true;
            }
        }

        return {
            chunks,
            total_logprob: has_any_score ? total_logprob : null,
            text_token_count,
            avg_logprob: has_any_score && text_token_count > 0 ? total_logprob / text_token_count : null,
            has_valid_timestamps,
        };
    }

    async _splitChunkWithRetry(
        chunk,
        fullAudio,
        generation_config,
        return_timestamps,
        timestamp_begin,
        hop_length,
        sampling_rate,
        audioOffset,
        depth,
    ) {
        const half_s = chunk.stride[0] / sampling_rate / 2;
        const sub_stride_s = Math.min(half_s / 6, 2);
        const sub_window = Math.round(sampling_rate * half_s);
        const sub_stride_samples = Math.round(sampling_rate * sub_stride_s);
        const sub_jump = sub_window - 2 * sub_stride_samples;
        const chunk_audio_len = chunk.stride[0];

        const sub_results = [];
        let sub_offset = 0;
        while (true) {
            const sub_end = sub_offset + sub_window;
            const subarr = fullAudio.subarray(audioOffset + sub_offset, audioOffset + sub_end);
            const feature = await this.processor(subarr);

            const is_first = sub_offset === 0;
            const is_last = sub_end >= chunk_audio_len;

            const sub_chunk = {
                stride: [subarr.length, is_first ? 0 : sub_stride_samples, is_last ? 0 : sub_stride_samples],
                input_features: feature.input_features,
                is_last,
            };

            sub_results.push(
                await this._processChunkWithRetry(
                    sub_chunk,
                    fullAudio,
                    generation_config,
                    return_timestamps,
                    timestamp_begin,
                    hop_length,
                    sampling_rate,
                    audioOffset + sub_offset,
                    depth + 1,
                ),
            );

            if (is_last) break;
            sub_offset += sub_jump;
        }

        const combined = this._combineChunkResults(sub_results);
        if (combined.chunks.length > 0) {
            combined.chunks[0].stride[1] = chunk.stride[1] / sampling_rate;
            combined.chunks[combined.chunks.length - 1].stride[2] = chunk.stride[2] / sampling_rate;
        }
        return combined;
    }

    async _generatePaddedChunkResult(
        chunk,
        fullAudio,
        generation_config,
        return_timestamps,
        timestamp_begin,
        hop_length,
        sampling_rate,
        audioOffset,
    ) {
        const silence_samples = Math.round(sampling_rate * 0.5);
        const padded_audio = new Float32Array(silence_samples + chunk.stride[0]);
        padded_audio.set(fullAudio.subarray(audioOffset, audioOffset + chunk.stride[0]), silence_samples);
        const padded_feature = await this.processor(padded_audio);

        const padded_chunk = {
            stride: [padded_audio.length, 0, 0],
            input_features: padded_feature.input_features,
            is_last: chunk.is_last,
        };

        const result = await this._generateChunkResult(
            padded_chunk,
            generation_config,
            return_timestamps,
            timestamp_begin,
            hop_length,
            sampling_rate,
        );

        if (result.chunks[0].token_timestamps) {
            const silence_s = silence_samples / sampling_rate;
            result.chunks[0].token_timestamps = result.chunks[0].token_timestamps.map((timestamp) =>
                round(Math.max(0, timestamp - silence_s), 2),
            );
            result.has_valid_timestamps = this._hasValidTokenTimestamps(result.chunks[0].token_timestamps);
        }

        result.chunks[0].stride = chunk.stride.map((x) => x / sampling_rate);
        return result;
    }

    async _call_moonshine(audio, kwargs) {
        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;
        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);
        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);

            // According to the [paper](https://huggingface.co/papers/2410.15608):
            // "We use greedy decoding, with a heuristic limit of 6 output tokens
            // per second of audio to avoid repeated output sequences."
            const max_new_tokens = Math.floor(aud.length / sampling_rate) * 6;
            const outputs = await this.model.generate({ max_new_tokens, ...kwargs, ...inputs });

            const text = this.processor.batch_decode(/** @type {Tensor} */ (outputs), { skip_special_tokens: true })[0];
            toReturn.push({ text });
        }
        return single ? toReturn[0] : toReturn;
    }

    async _call_cohere_asr(audio, kwargs) {
        const single = !Array.isArray(audio);
        const batchedAudio = single ? [audio] : audio;

        const feature_extractor = this.processor.feature_extractor;
        const sampling_rate = feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(batchedAudio, sampling_rate);

        const language = kwargs.language ?? 'en';
        // @ts-expect-error TS2339
        const decoder_input_ids = this.processor.get_decoder_prompt_ids(language);

        const toReturn = [];
        for (const aud of preparedAudios) {
            // Split long audio at energy-based boundaries
            // @ts-expect-error TS2339
            const audioChunks = feature_extractor.split_audio(aud);

            const chunk_texts = [];
            for (const chunk of audioChunks) {
                const inputs = await this.processor(chunk);

                const outputs = await this.model.generate({
                    ...inputs,
                    decoder_input_ids,
                    ...kwargs,
                });

                const text = this.tokenizer
                    .decode(/** @type {Tensor} */ (outputs)[0].tolist(), { skip_special_tokens: true })
                    .trim();
                chunk_texts.push(text);
            }

            // @ts-expect-error TS2339
            const full_text = this.processor.constructor.join_chunks(chunk_texts, language);
            toReturn.push({ text: full_text });
        }
        return single ? toReturn[0] : toReturn;
    }
}
