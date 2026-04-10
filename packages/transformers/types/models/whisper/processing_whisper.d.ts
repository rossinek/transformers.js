/**
 * Represents a WhisperProcessor that extracts features from an audio input.
 */
export class WhisperProcessor extends Processor {
    static tokenizer_class: typeof AutoTokenizer;
    static feature_extractor_class: typeof AutoFeatureExtractor;
    /**
     * Converts prompt text into Whisper prompt token ids.
     * Whisper expects `<|startofprev|>` followed by the prompt text tokens.
     *
     * @param {string} text
     * @returns {number[]}
     */
    get_prompt_ids(text: string): number[];
    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    _call(audio: any): Promise<any>;
}
import { Processor } from '../../processing_utils.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
//# sourceMappingURL=processing_whisper.d.ts.map