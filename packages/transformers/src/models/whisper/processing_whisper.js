import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';

/**
 * Represents a WhisperProcessor that extracts features from an audio input.
 */
export class WhisperProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;

    /**
     * Converts prompt text into Whisper prompt token ids.
     * Whisper expects `<|startofprev|>` followed by the prompt text tokens.
     *
     * @param {string} text
     * @returns {number[]}
     */
    get_prompt_ids(text) {
        const prompt = text?.trim();
        if (!prompt) {
            return [];
        }

        const start_of_prev_id = this.tokenizer?._tokenizer?.token_to_id?.('<|startofprev|>');
        if (start_of_prev_id == null) {
            throw new Error('Whisper tokenizer does not define the <|startofprev|> token.');
        }

        const prompt_ids = this.tokenizer.encode(` ${prompt}`, { add_special_tokens: false });
        return [start_of_prev_id, ...prompt_ids];
    }

    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(audio) {
        return await this.feature_extractor(audio);
    }
}
