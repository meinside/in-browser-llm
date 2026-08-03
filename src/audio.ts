// Everything that needs AudioContext lives here, because AudioContext exists only
// on the main thread. Keeping it in one file makes that boundary explicit: if
// inference ever moves into a worker, this is the piece that cannot follow.

/** gemma-4's feature_extractor.sampling_rate, from processor_config.json. */
export const AUDIO_SAMPLE_RATE = 16000;

/**
 * gemma-4 clamps audio at audio_seq_length (750) × audio_ms_per_token (40ms).
 *
 * A constant rather than a value read from the model config: other models may
 * differ, but gemma-4 is currently the only one with an audio encoder, and
 * warning with a wrong number would be worse than stating where 30 came from.
 */
export const AUDIO_MAX_SECONDS = 30;

// Matches transformers.js load_audio: sum the channels and scale by 1/sqrt(2) so
// a full-scale stereo signal does not clip when folded to mono. Using a different
// factor would make pre-decoded and library-decoded inputs disagree.
export function downmixToMono(channels: Float32Array[]): Float32Array {
  const first = channels[0];
  if (!first) throw new Error('downmixToMono received no channels');
  if (channels.length === 1) return first;

  const scale = Math.sqrt(2);
  const mono = new Float32Array(first.length);
  for (let i = 0; i < first.length; i += 1) {
    let sum = 0;
    for (const channel of channels) sum += channel[i] ?? 0;
    mono[i] = (scale * sum) / channels.length;
  }
  return mono;
}

// Not unit-tested: AudioContext does not exist in Node, and mocking it would
// verify nothing about resampling while implying coverage. downmixToMono is split
// out so the part that can be tested is.
export async function decodeAudio(blob: Blob): Promise<Float32Array> {
  if (typeof AudioContext === 'undefined') {
    throw new Error(
      'Audio decoding needs AudioContext, which is unavailable here. Decode on ' +
        'the main thread and pass { type: "audio", data } instead.',
    );
  }

  // Constructing with sampleRate makes decodeAudioData resample for us.
  const ctx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    const channels: Float32Array[] = [];
    for (let i = 0; i < decoded.numberOfChannels; i += 1) {
      channels.push(decoded.getChannelData(i));
    }
    return downmixToMono(channels);
  } finally {
    void ctx.close();
  }
}

// Warns rather than truncating: choosing which 30 seconds matter is the caller's
// decision, not the library's.
export function warnIfTooLong(samples: Float32Array): void {
  const seconds = samples.length / AUDIO_SAMPLE_RATE;
  if (seconds > AUDIO_MAX_SECONDS) {
    console.warn(
      `[in-browser-llm] audio is ${seconds.toFixed(1)}s but the model clamps at ` +
        `${AUDIO_MAX_SECONDS}s; everything past that is ignored.`,
    );
  }
}
