import { WhisperForConditionalGeneration } from "../src/transformers.js";

describe("Whisper hallucination recovery helpers", () => {
  it("uses the default fallback ladder when temperature is unset", () => {
    const schedule = WhisperForConditionalGeneration.prototype._get_fallback_temperatures({
      temperature: 1.0,
      _temperature_is_explicit: false,
    });
    expect(schedule).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1.0]);
  });

  it("respects an explicit scalar temperature", () => {
    const schedule = WhisperForConditionalGeneration.prototype._get_fallback_temperatures({
      temperature: 0.4,
      _temperature_is_explicit: true,
    });
    expect(schedule).toEqual([0.4]);
  });

  it("respects an explicit fallback schedule", () => {
    const schedule = WhisperForConditionalGeneration.prototype._get_fallback_temperatures({
      temperature: [0, 0.4, 0.8],
      _temperature_is_explicit: true,
    });
    expect(schedule).toEqual([0, 0.4, 0.8]);
  });
});
