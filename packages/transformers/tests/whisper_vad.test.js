import { jest } from "@jest/globals";

const mockRunVADWeb = jest.fn();
jest.unstable_mockModule("../src/pipelines/vad/run_vad_web.js", () => ({
  runVADWeb: mockRunVADWeb,
}));

const { preprocessAudioWithVoiceActivityDetection } = await import("../src/pipelines/vad/index.js");
const { buildVoiceActivityTimeline, normalizeVoiceActivityDetectionOptions } = await import("../src/pipelines/vad/build_timeline.js");
const { remapWhisperOutputTimestamps } = await import("../src/pipelines/vad/remap_timestamps.js");

describe("Whisper VAD helpers", () => {
  beforeEach(() => {
    mockRunVADWeb.mockReset();
  });

  it("falls back to the original audio when no speech is detected", async () => {
    mockRunVADWeb.mockResolvedValue([]);

    const audio = new Float32Array(4000).fill(1);
    const result = await preprocessAudioWithVoiceActivityDetection(audio, 1000, {
      debug: true,
    });

    expect(result.applied).toBe(false);
    expect(result.processedAudio).toBe(audio);
    expect(result.originalDurationS).toBe(4);
    expect(result.compactDurationS).toBe(4);
  });

  it("returns empty compact audio when preserve_full_audio_if_empty is false", async () => {
    mockRunVADWeb.mockResolvedValue([]);

    const audio = new Float32Array(4000).fill(1);
    const result = await preprocessAudioWithVoiceActivityDetection(audio, 1000, {
      preserve_full_audio_if_empty: false,
    });

    expect(result.applied).toBe(true);
    expect(result.processedAudio).toHaveLength(0);
    expect(result.segments).toEqual([]);
    expect(result.compactDurationS).toBe(0);
  });

  it("compacts disjoint voiced regions and remaps word timestamps back to original time", () => {
    const audio = Float32Array.from({ length: 8000 }, (_, i) => i);
    const options = normalizeVoiceActivityDetectionOptions({
      pre_speech_pad_ms: 0,
      post_speech_pad_ms: 0,
      max_merge_gap_ms: 0,
    });

    const timeline = buildVoiceActivityTimeline(
      audio,
      1000,
      [
        { start_ms: 1000, end_ms: 1500 },
        { start_ms: 4000, end_ms: 4500 },
      ],
      options,
    );

    expect(timeline.processedAudio).toHaveLength(1000);
    expect(timeline.compactDurationS).toBe(1);
    expect(timeline.segments).toEqual([
      {
        original_start_s: 1,
        original_end_s: 1.5,
        compact_start_s: 0,
        compact_end_s: 0.5,
      },
      {
        original_start_s: 4,
        original_end_s: 4.5,
        compact_start_s: 0.5,
        compact_end_s: 1,
      },
    ]);

    const output = {
      text: "first second",
      chunks: [
        { text: "first", timestamp: [0.2, 0.5] },
        { text: "second", timestamp: [0.5, 0.8] },
      ],
    };

    remapWhisperOutputTimestamps(output, timeline.segments, timeline.originalDurationS);

    expect(output.chunks).toEqual([
      { text: "first", timestamp: [1.2, 1.5] },
      { text: "second", timestamp: [4, 4.3] },
    ]);
  });

  it("keeps chunk timestamps monotonic and non-inverted after remapping", () => {
    const output = {
      text: "a b c",
      chunks: [
        { text: "a", timestamp: [0.4, 0.7] },
        { text: "b", timestamp: [0.7, 0.7] },
        { text: "c", timestamp: [0.7, 0.65] },
      ],
    };

    remapWhisperOutputTimestamps(
      output,
      [
        {
          original_start_s: 1,
          original_end_s: 1.5,
          compact_start_s: 0,
          compact_end_s: 0.5,
        },
        {
          original_start_s: 4,
          original_end_s: 4.5,
          compact_start_s: 0.5,
          compact_end_s: 1,
        },
      ],
      8,
    );

    expect(output.chunks[0].timestamp).toEqual([1.4, 4.2]);
    expect(output.chunks[1].timestamp).toEqual([4.2, 4.2]);
    expect(output.chunks[2].timestamp).toEqual([4.2, 4.2]);
  });
});
