import { Pipeline, prepareAudios } from './_base.js';
import { preprocessAudioWithVoiceActivityDetection, remapWhisperOutputTimestamps } from './vad/index.js';

import { Tensor } from '../utils/tensor.js';
import { max, round } from '../utils/maths.js';
import { logger } from '../utils/logger.js';

const MIN_HALLUCINATION_CHUNK_S = 2.0;
const MAX_HALLUCINATION_DEPTH = 3;
const MIN_HALLUCINATION_TOKENS_PER_SECOND = 1.0;
const TRUNCATION_GUARD_MIN_RATIO = 0.5;
const TRUNCATION_GUARD_SCORE_MARGIN = 0.5;

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
        const hallucination_recovery = kwargs.hallucination_recovery ?? true;
        const voice_activity_detection = kwargs.voice_activity_detection ?? false;
        let stride_length_s = kwargs.stride_length_s ?? null;

        const generation_config = { ...kwargs };
        generation_config['hallucination_recovery'] = hallucination_recovery;
        delete generation_config['voice_activity_detection'];

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
            const transcriptionAudio = vadResult.processedAudio;
            if (vadResult.applied && transcriptionAudio.length === 0) {
                const output = return_timestamps ? { text: '', chunks: [] } : { text: '' };
                if (voice_activity_detection && vadResult.debugMetadata) {
                    output.voice_activity_detection = vadResult.debugMetadata;
                }
                toReturn.push(output);
                continue;
            }

            /** @type {{stride: number[], input_features: Tensor, is_last: boolean, tokens?: bigint[], token_timestamps?: number[]}[]} */
            let chunks = [];
            if (chunk_length_s > 0) {
                if (stride_length_s === null) {
                    stride_length_s = chunk_length_s / 6;
                } else if (chunk_length_s <= stride_length_s) {
                    throw Error('`chunk_length_s` must be larger than `stride_length_s`.');
                }

                // TODO support different stride_length_s (for left and right)

                const window = sampling_rate * chunk_length_s;
                const stride = sampling_rate * stride_length_s;
                const jump = window - 2 * stride;
                let offset = 0;

                // Create subarrays of audio with overlaps
                while (true) {
                    const offset_end = offset + window;
                    const subarr = transcriptionAudio.subarray(offset, offset_end);
                    const feature = await this.processor(subarr);

                    const is_first = offset === 0;
                    const is_last = offset_end >= transcriptionAudio.length;
                    chunks.push({
                        stride: [subarr.length, is_first ? 0 : stride, is_last ? 0 : stride],
                        input_features: feature.input_features,
                        is_last,
                    });
                    if (is_last) break;
                    offset += jump;
                }
            } else {
                chunks = [
                    {
                        stride: [transcriptionAudio.length, 0, 0],
                        input_features: (await this.processor(transcriptionAudio)).input_features,
                        is_last: true,
                    },
                ];
            }

            // @ts-expect-error ts(2339)
            const timestamp_begin = this.tokenizer.timestamp_begin;
            if (hallucination_recovery) {
                const processedChunks = [];
                const chunkJump =
                    chunk_length_s > 0
                        ? sampling_rate * chunk_length_s - 2 * sampling_rate * (stride_length_s ?? chunk_length_s / 6)
                        : 0;
                for (let ci = 0; ci < chunks.length; ++ci) {
                    const audioOffset = ci * chunkJump;
                    const result = await this._processChunkWithRetry(
                        chunks[ci],
                        transcriptionAudio,
                        generation_config,
                        return_timestamps,
                        timestamp_begin,
                        hop_length,
                        sampling_rate,
                        audioOffset,
                    );
                    processedChunks.push(...result.chunks);
                }
                chunks = processedChunks;
            } else {
                // Preserve pre-recovery behavior when the switch is disabled.
                for (const chunk of chunks) {
                    generation_config.num_frames = Math.floor(chunk.stride[0] / hop_length);

                    const data = await this.model.generate({
                        inputs: chunk.input_features,
                        ...generation_config,
                    });

                    if (return_timestamps === 'word') {
                        // @ts-expect-error TS2339
                        const sequences = data.sequences.tolist()[0];
                        // @ts-expect-error TS2339
                        const token_ts = data.token_timestamps.tolist()[0];

                        const prefixLength = Math.max(
                            sequences.findIndex((/** @type {bigint} */ t) => Number(t) >= timestamp_begin),
                            0,
                        );

                        chunk.tokens = sequences.slice(prefixLength);
                        chunk.token_timestamps = token_ts
                            .slice(prefixLength)
                            .map((/** @type {number} */ x) => round(x, 2));
                    } else {
                        const sequences = /** @type {Tensor} */ (/** @type {any} */ (data)?.sequences ?? data);
                        chunk.tokens = /** @type {Tensor} */ (sequences)[0].tolist();
                    }

                    chunk.stride = chunk.stride.map((x) => x / sampling_rate);
                }
            }

            // Merge text chunks
            // @ts-ignore
            const [full_text, optional] = this.tokenizer._decode_asr(chunks, {
                time_precision,
                return_timestamps,
                force_full_sequences,
            });

            const output = { text: full_text, ...optional };
            if (vadResult.applied && return_timestamps) {
                remapWhisperOutputTimestamps(output, vadResult.segments, vadResult.originalDurationS);
            }
            if (voice_activity_detection && vadResult.debugMetadata) {
                output.voice_activity_detection = vadResult.debugMetadata;
            }

            toReturn.push(output);
        }
        return single ? toReturn[0] : toReturn;
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

        if (depth < MAX_HALLUCINATION_DEPTH) {
            const split = await this._splitChunkWithRetry(
                chunk,
                fullAudio,
                generation_config,
                return_timestamps,
                timestamp_begin,
                hop_length,
                sampling_rate,
                audioOffset,
                depth,
            );
            return this._chooseBetterChunkResult(original, split);
        }

        const padded = await this._generatePaddedChunkResult(
            chunk,
            fullAudio,
            generation_config,
            return_timestamps,
            timestamp_begin,
            hop_length,
            sampling_rate,
            audioOffset,
        );
        return this._chooseBetterChunkResult(original, padded);
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

        if (return_timestamps === 'word') {
            const sequences = data.sequences.tolist()[0];
            const token_ts = data.token_timestamps.tolist()[0];
            const prefixLength = Math.max(
                sequences.findIndex((/** @type {bigint} */ t) => Number(t) >= timestamp_begin),
                0,
            );
            outputChunk.tokens = sequences.slice(prefixLength);
            outputChunk.token_timestamps = token_ts
                .slice(prefixLength)
                .map((/** @type {number} */ x) => round(Math.max(0, x + timestamp_shift_s), 2));
        } else {
            const sequences = data?.sequences ?? data;
            outputChunk.tokens = /** @type {Tensor} */ (sequences)[0].tolist();
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
        const tokens_per_second = result.text_token_count / Math.max(chunk_duration_s, 0.1);
        const low_text_coverage = tokens_per_second < MIN_HALLUCINATION_TOKENS_PER_SECOND;
        const low_logprob =
            result.avg_logprob === null ? result.text_token_count === 0 : result.avg_logprob < logprob_threshold;
        return low_text_coverage && low_logprob;
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

        if (return_timestamps === 'word' && result.chunks[0].token_timestamps) {
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
