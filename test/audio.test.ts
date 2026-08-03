import { describe, expect, it, vi } from 'vitest';
import {
  AUDIO_MAX_SECONDS,
  AUDIO_SAMPLE_RATE,
  downmixToMono,
  warnIfTooLong,
} from '../src/audio.js';

describe('downmixToMono', () => {
  it('returns a mono channel unchanged', () => {
    const mono = new Float32Array([0.1, -0.2, 0.3]);
    expect(downmixToMono([mono])).toBe(mono);
  });

  it('sums stereo channels with a 1/sqrt(2) scale', () => {
    // The same factor transformers.js load_audio uses, so both paths agree.
    const left = new Float32Array([0.5, 0.0]);
    const right = new Float32Array([0.5, 1.0]);
    const result = downmixToMono([left, right]);
    const scale = Math.sqrt(2);
    expect(result[0]).toBeCloseTo((scale * 1.0) / 2, 6);
    expect(result[1]).toBeCloseTo((scale * 1.0) / 2, 6);
  });

  it('preserves length when downmixing', () => {
    const left = new Float32Array(5);
    const right = new Float32Array(5);
    expect(downmixToMono([left, right]).length).toBe(5);
  });

  it('throws when given no channels', () => {
    expect(() => downmixToMono([])).toThrow(/no channels/i);
  });
});

describe('warnIfTooLong', () => {
  it('warns above the clamp length', () => {
    // gemma-4 silently ignores the tail past 30s, so the user must be told.
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnIfTooLong(new Float32Array(AUDIO_SAMPLE_RATE * (AUDIO_MAX_SECONDS + 1)));
    expect(spy).toHaveBeenCalledOnce();
    expect(String(spy.mock.calls[0])).toMatch(/30/);
    spy.mockRestore();
  });

  it('stays silent at exactly the clamp length', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnIfTooLong(new Float32Array(AUDIO_SAMPLE_RATE * AUDIO_MAX_SECONDS));
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it('stays silent for short audio', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnIfTooLong(new Float32Array(AUDIO_SAMPLE_RATE));
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});
