import { jest } from "@jest/globals";

const mockRunVADWeb = jest.fn();
jest.unstable_mockModule("../src/pipelines/vad/run_vad_web.js", () => ({
  runVADWeb: mockRunVADWeb,
}));

const { AutomaticSpeechRecognitionPipeline } = await import("../src/pipelines/automatic-speech-recognition.js");

function createListTensor(values) {
  return {
    tolist() {
      return values;
    },
  };
}

function createSequenceTensor(values) {
  return {
    0: createListTensor(values),
    tolist() {
      return [values];
    },
  };
}

function createFakePipeline(decodeResult = ["text", { chunks: [] }]) {
  const pipeline = Object.create(AutomaticSpeechRecognitionPipeline.prototype);

  const processor = jest.fn(async (audio) => ({
    input_features: audio,
  }));
  processor.feature_extractor = {
    config: {
      chunk_length: 30,
      hop_length: 10,
      sampling_rate: 1000,
    },
  };

  pipeline.processor = processor;
  pipeline.model = {
    config: {
      max_source_positions: 3000,
    },
    generate: jest.fn(async () => ({
      sequences: createSequenceTensor([1n, 2n, 3n]),
      token_timestamps: {
        tolist() {
          return [[0.2, 0.5, 0.8]];
        },
      },
    })),
  };
  pipeline.tokenizer = {
    timestamp_begin: 999,
    all_special_ids: [],
    _decode_asr: jest.fn(() => decodeResult),
  };
  pipeline._processChunkWithRetry = jest.fn(async (chunk, _fullAudio, _generationConfig, _returnTimestamps, _timestampBegin, _hopLength, samplingRate) => ({
    chunks: [
      {
        stride: chunk.stride.map((x) => x / samplingRate),
        is_last: true,
        tokens: [1n, 2n, 3n],
      },
    ],
    total_logprob: null,
    text_token_count: 3,
    avg_logprob: null,
    has_valid_timestamps: true,
  }));

  return pipeline;
}

describe("Whisper VAD pipeline integration", () => {
  beforeEach(() => {
    mockRunVADWeb.mockReset();
  });

  it("leaves Whisper output unchanged when VAD is disabled", async () => {
    const pipe = createFakePipeline(["hello", { chunks: [{ text: "hello", timestamp: [0.2, 0.5] }] }]);
    const audio = new Float32Array(8000).fill(1);

    const output = await pipe._call_whisper(audio, {
      return_timestamps: true,
      hallucination_recovery: false,
      voice_activity_detection: false,
    });

    expect(mockRunVADWeb).not.toHaveBeenCalled();
    expect(output).toEqual({
      text: "hello",
      chunks: [{ text: "hello", timestamp: [0.2, 0.5] }],
    });
    expect(pipe.processor).toHaveBeenCalledWith(audio);
  });

  it("uses compacted audio and remaps timestamps back to the original timeline", async () => {
    mockRunVADWeb.mockResolvedValue([
      { start_ms: 1000, end_ms: 1500 },
      { start_ms: 4000, end_ms: 4500 },
      { start_ms: 7000, end_ms: 7500 },
    ]);

    const pipe = createFakePipeline([
      "first second",
      {
        chunks: [
          { text: "first", timestamp: [0.2, 0.5] },
          { text: "second", timestamp: [0.5, 0.8] },
        ],
      },
    ]);
    const audio = new Float32Array(8000).fill(1);

    const output = await pipe._call_whisper(audio, {
      return_timestamps: "word",
      hallucination_recovery: false,
      voice_activity_detection: {
        pre_speech_pad_ms: 0,
        post_speech_pad_ms: 0,
        max_merge_gap_ms: 0,
      },
    });

    expect(mockRunVADWeb).toHaveBeenCalledTimes(1);
    expect(pipe.processor.mock.calls[0][0]).toHaveLength(1500);
    expect(output).toEqual({
      text: "first second",
      chunks: [
        { text: "first", timestamp: [1.2, 1.5] },
        { text: "second", timestamp: [4, 4.3] },
      ],
    });
  });

  it("works with hallucination recovery enabled without changing the output shape", async () => {
    mockRunVADWeb.mockResolvedValue([{ start_ms: 1000, end_ms: 2000 }]);

    const pipe = createFakePipeline(["hello", { chunks: [{ text: "hello", timestamp: [0.1, 0.6] }] }]);
    const audio = new Float32Array(5000).fill(1);

    const output = await pipe._call_whisper(audio, {
      return_timestamps: true,
      hallucination_recovery: true,
      voice_activity_detection: {
        pre_speech_pad_ms: 0,
        post_speech_pad_ms: 0,
        max_merge_gap_ms: 0,
      },
    });

    expect(pipe._processChunkWithRetry).toHaveBeenCalled();
    expect(output).toEqual({
      text: "hello",
      chunks: [{ text: "hello", timestamp: [1.1, 1.6] }],
    });
  });

  it("preserves split compound word chunks in word mode", async () => {
    const pipe = createFakePipeline();
    pipe.tokenizer._decode_asr = jest
      .fn()
      .mockReturnValueOnce([
        " C++ high-level",
        {
          chunks: [
            { text: " C", timestamp: [0.2, 0.4] },
            { text: "++", timestamp: [0.4, 0.5] },
            { text: " high", timestamp: [0.6, 0.9] },
            { text: "-level", timestamp: [0.9, 1.1] },
          ],
        },
      ])
      .mockReturnValueOnce([
        " C high-level",
        {
          chunks: [],
        },
      ]);

    const audio = new Float32Array(5000).fill(1);
    const output = await pipe._call_whisper(audio, {
      return_timestamps: "word",
      hallucination_recovery: false,
      voice_activity_detection: false,
    });

    expect(pipe.tokenizer._decode_asr).toHaveBeenCalledTimes(1);
    expect(output).toEqual({
      text: " C++ high-level",
      chunks: [
        { text: " C", timestamp: [0.2, 0.4] },
        { text: "++", timestamp: [0.4, 0.5] },
        { text: " high", timestamp: [0.6, 0.9] },
        { text: "-level", timestamp: [0.9, 1.1] },
      ],
    });
  });

  it("returns an empty transcription when VAD finds no speech and preserve_full_audio_if_empty is false", async () => {
    mockRunVADWeb.mockResolvedValue([]);

    const pipe = createFakePipeline(["unused", { chunks: [{ text: "unused", timestamp: [0.1, 0.6] }] }]);
    const audio = new Float32Array(5000).fill(1);

    const output = await pipe._call_whisper(audio, {
      return_timestamps: true,
      hallucination_recovery: false,
      voice_activity_detection: {
        preserve_full_audio_if_empty: false,
      },
    });

    expect(pipe.processor).not.toHaveBeenCalled();
    expect(output).toEqual({
      text: "",
      chunks: [],
    });
  });
});
