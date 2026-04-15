import { WhisperTokenizer, WhisperForConditionalGeneration, Tensor, full } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("WhisperForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-WhisperForConditionalGeneration";

    /** @type {WhisperForConditionalGeneration} */
    let model;
    /** @type {WhisperTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await WhisperForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await WhisperTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    describe("prefix tokens", () => {
      const input_features = full([1, 80, 3000], 0.0);

      describe("English-only", () => {
        it(
          "default",
          async () => {
            const outputs = await model.generate({
              input_features,
              is_multilingual: false,
              max_new_tokens: 1,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "return_timestamps=true",
          async () => {
            const outputs = await model.generate({
              input_features,
              is_multilingual: false,
              max_new_tokens: 1,
              return_timestamps: true,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, /* Generated */ 51682n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      describe("multilingual", () => {
        it(
          "language unset; task unset",
          async () => {
            // language defaults to 'en'
            // task defaults to 'transcribe'

            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "language set; task unset",
          async () => {
            // task defaults to 'transcribe'
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "af",
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50327n, 50359n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "language set; task set",
          async () => {
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "zh",
              task: "translate",
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50260n, 50358n, 50363n, /* Generated */ 45084n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "return_timestamps=true",
          async () => {
            const outputs = await model.generate({
              input_features,
              max_new_tokens: 1,
              language: "en",
              task: "transcribe",
              return_timestamps: true,
            });

            expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, /* Generated */ 51812n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });
    });

    describe("decoder_start_ids", () => {
      const input_features = full([1, 80, 3000], 0.0);

      it(
        "broadcast inputs",
        async () => {
          const { decoder_start_token_id, lang_to_id, task_to_id, no_timestamps_token_id } = model.generation_config;

          const outputs = await model.generate({
            input_features, // batch size 1
            max_new_tokens: 1,
            decoder_input_ids: [
              // batch size 2
              // <|startoftranscript|> <|lang_id|> <|task|> [<|notimestamps|>]
              [decoder_start_token_id, lang_to_id["<|en|>"], task_to_id["translate"], no_timestamps_token_id],
              [decoder_start_token_id, lang_to_id["<|fr|>"], task_to_id["transcribe"], no_timestamps_token_id],
            ],
          });
          expect(outputs.tolist()).toEqual([
            [/* Prefix */ 50258n, 50259n, 50358n, 50363n, /* Generated */ 45084n],
            [/* Prefix */ 50258n, 50265n, 50359n, 50363n, /* Generated */ 45084n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("_extract_token_timestamps", () => {
      it("should refine token timestamps to attention-weighted anchors within the DTW span", () => {
        const extractor = Object.create(WhisperForConditionalGeneration.prototype);
        extractor.config = {
          decoder_layers: 1,
          median_filter_width: 1,
        };

        const makeStep = (values) => [new Tensor("float32", Float32Array.from(values), [1, 1, 1, 4])];

        const outputs = {
          cross_attentions: [makeStep([5, 5, 0, 0]), makeStep([0, 0, 5, 5])],
          sequences: new Tensor("int64", BigInt64Array.from([1n, 2n, 3n]), [1, 3]),
        };

        const { timestamps, rawTimestamps } = extractor._extract_token_timestamps(outputs, [[0, 0]], 4, 0.02, 0);
        // 30th percentile of [exp(5),exp(5)] at frames [0,1] → frame 0; at frames [2,3] → frame 2
        expect(timestamps.tolist()).toBeCloseToNested([[0, 0.04, 0.04]], 5);
        // Raw DTW offsets (last frame per token): token 0 → frame 1, token 1 → frame 3
        expect(rawTimestamps.tolist()).toBeCloseToNested([[0.02, 0.06, 0.06]], 5);
      });
    });

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
