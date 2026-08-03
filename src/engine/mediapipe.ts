import { DEFAULT_MAX_TOKENS, type Engine, type GenerateOptions, type LoadOptions } from './types.js';
import { createDeltaStream } from './stream.js';
import {
  collectAudio,
  collectImages,
  createStopTruncator,
  formatPrompt,
  formatPromptParts,
  hasNonTextParts,
  type Message,
} from '../prompt.js';
import { decodeAudio, AUDIO_SAMPLE_RATE, warnIfTooLong } from '../audio.js';
import { lookupCatalog, type TurnFormat } from '../catalog.js';
import { fetchWithProgress } from '../loader.js';
import { CapabilityError, RuntimeError } from '../errors.js';
import type { ModelRef } from '../types.js';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.29/wasm';

// MediaPipe's Prompt parts are structural: it tests for an `imageSource` or
// `audioSource` key rather than a class, so a plain object is enough. Typed here
// rather than imported so prompt.ts stays runtime-agnostic.
type MediaPipePart =
  | string
  | { imageSource: HTMLImageElement }
  | { audioSource: { audioSamples: Float32Array; audioSampleRateHz: number } };

// MediaPipe accepts a URL string for an image, but only fetches it with
// crossOrigin="Anonymous" — which fails for a blob: URL from a file input, the
// common case. Decoding here also surfaces a broken image before generation starts.
async function toImageSource(src: string | Blob): Promise<HTMLImageElement> {
  const url = typeof src === 'string' ? src : URL.createObjectURL(src);
  try {
    const img = new Image();
    if (typeof src === 'string') img.crossOrigin = 'Anonymous';
    img.src = url;
    await img.decode();
    return img;
  } finally {
    if (typeof src !== 'string') URL.revokeObjectURL(url);
  }
}

/**
 * Rewrites MediaPipe's modality failures into one error that names the cause.
 *
 * One cause, more than one message, which is why this matches several patterns:
 *
 * - **Predict time, either modality.** The option is accepted, the graph is built, and
 *   `Llm{Vision,Audio}InferenceCalculator` dies with "… models could not be created"
 *   (`llm_gpu_runner_manager.cc:509`). This is the form reproduced on demand, with
 *   `maxImages` or `supportAudio`, alone or together.
 * - **Load time**: "Audio options should not be null when loading audio runner" from
 *   `createFromOptions` (`:572`), before any prompt exists. Observed once on Firefox and
 *   not reproducible since — including with both options set, which was the suspected
 *   trigger. The condition is unknown, so the pattern stays: it names no calculator and
 *   shares no wording with the predict-time form, so nothing else here would catch it,
 *   and `createFromOptions` remains where a container declaring an unusable encoder would
 *   fail. An earlier note here blamed the timing on `supportAudio`, then on the
 *   combination; both were inferred from that one observation. Do not attribute it to an
 *   option again without reproducing it.
 *
 * Each pattern still requires modality-specific wording, so unrelated failures pass
 * through untouched.
 */
export function explainModalityFailure(cause: unknown): unknown {
  const message = cause instanceof Error ? cause.message : String(cause);
  const isModalityFailure =
    // Predict-time: the graph was built but the model could not be instantiated.
    (/Llm(Vision|Audio)InferenceCalculator/.test(message) &&
      /could not be created/.test(message)) ||
    // Load-time: the runner refuses before a session exists.
    /(Audio|Vision) options should not be null/.test(message) ||
    // The underlying reason, if it ever surfaces directly.
    /(audio_encoder_parameters|vision_encoder_parameters) are not defined/.test(message);
  if (!isModalityFailure) return cause;

  return new CapabilityError(
    'This MediaPipe model has no vision or audio weights, so the modality could ' +
      'not be initialized even though the option was accepted. The published ' +
      'gemma-4 *web* containers carry a text decoder only (model_type ' +
      'tf_lite_artisan_text_decoder); the builds that do include vision and audio ' +
      'encoders declare backend_constraint "cpu" and are not for the browser. Use a ' +
      'transformers.js model for image or audio input.',
    { cause },
  );
}

async function toMediaPipePrompt(
  messages: Message[],
  format: TurnFormat,
): Promise<MediaPipePart[]> {
  const parts = formatPromptParts(messages, format);
  return Promise.all(
    parts.map(async (part): Promise<MediaPipePart> => {
      if (typeof part === 'string') return part;
      if (part.type === 'image') {
        return { imageSource: await toImageSource(part.src) };
      }
      // MediaPipe resamples nothing: it reads audioSampleRateHz and trusts it, so
      // the rate must describe the samples. decodeAudio always yields 16kHz mono.
      const samples = 'data' in part ? part.data : await decodeAudio(part.src);
      warnIfTooLong(samples);
      return {
        audioSource: { audioSamples: samples, audioSampleRateHz: AUDIO_SAMPLE_RATE },
      };
    }),
  );
}

export async function createMediaPipeEngine(
  ref: ModelRef,
  opts: LoadOptions = {},
): Promise<Engine> {
  // MediaPipe caps input+output together and only accepts the value at load time,
  // so it cannot vary per call the way transformers.js max_new_tokens can.
  const maxTokens = opts.maxTokens ?? DEFAULT_MAX_TOKENS;

  if (ref.runtime !== 'mediapipe') {
    throw new RuntimeError(`Expected a mediapipe ref, received ${ref.runtime}`);
  }

  // MediaPipe cannot apply a chat template, so the turn format must be known.
  // A catalog entry may exist only to carry a dtype override, so require the turn
  // format specifically rather than merely the entry.
  const entry = lookupCatalog(ref);
  const turnFormat = entry?.turnFormat;
  if (!turnFormat) {
    throw new CapabilityError(
      'No turn format in the catalog for this model. The MediaPipe path needs ' +
        'turn delimiters up front — run scripts/inspect-chat-template.mjs and add them.',
    );
  }
  const stopSequences = entry.stopSequences ?? [turnFormat.end];

  let mod: typeof import('@mediapipe/tasks-genai');
  try {
    mod = await import('@mediapipe/tasks-genai');
  } catch (cause) {
    throw new RuntimeError(
      'Install @mediapipe/tasks-genai (0.10.29 verified) to use this runtime.',
      { cause },
    );
  }

  const { FilesetResolver, LlmInference } = mod;
  const genai = await FilesetResolver.forGenAiTasks(WASM_BASE);

  // Fetch here rather than passing modelAssetPath: that path has no progress hook.
  // A TFL3 .task streams, so passing the reader avoids buffering ~2GB.
  const { stream, fromCache } = await fetchWithProgress(ref.url, {
    onProgress: opts.onProgress,
  });
  opts.onCacheStatus?.(fromCache);

  // MediaPipe accepts this only if it passes
  // `instanceof ReadableStreamDefaultReader`; loader.ts re-wraps cached bodies in
  // a local TransformStream so the identity holds.
  // Both are omitted unless asked for, since each switches MediaPipe to streaming
  // loading. Opt-in rather than inferred: a .task or .litertlm exposes no config to
  // read the modality from, so only the caller knows.
  //
  // Enabling them is accepted by the runtime but currently fails on the published
  // gemma-4 web builds, whose weights are a text decoder only. See CLAUDE.md trap 13.
  const maxImages = opts.maxImages ?? 0;
  const supportAudio = opts.supportAudio ?? false;

  let llm: Awaited<ReturnType<typeof LlmInference.createFromOptions>>;
  try {
    llm = await LlmInference.createFromOptions(genai, {
      baseOptions: { modelAssetBuffer: stream.getReader() },
      maxTokens,
      topK: 40,
      temperature: 0.8,
      randomSeed: 101,
      ...(maxImages > 0 ? { maxNumImages: maxImages } : {}),
      ...(supportAudio ? { supportAudio: true } : {}),
    });
  } catch (cause) {
    // The current gemma-4 web containers get past this and fail at predict time instead,
    // so in practice the catch below is the one that fires. Kept because the audio runner
    // is built here — see explainModalityFailure on the load-time message.
    throw explainModalityFailure(cause);
  }

  return {
    generate(messages: Message[], genOpts: GenerateOptions = {}) {
      const images = collectImages(messages);
      const audioParts = collectAudio(messages);

      // Named per modality rather than as one "no multimodal input" error: the
      // caller enabling images and forgetting audio should not be told to switch
      // runtimes. Checked here rather than deferred to MediaPipe because its own
      // message for audio ("supportAudio was not enabled") does not say where to
      // set it, and it reports images as a count mismatch against zero.
      if (images.length > 0 && maxImages === 0) {
        throw new CapabilityError(
          'This MediaPipe engine was loaded without image support. Pass ' +
            '{ maxImages: n } to load() — MediaPipe fixes it at load time, so it ' +
            'cannot be enabled per call.',
        );
      }
      if (images.length > maxImages) {
        throw new CapabilityError(
          `${images.length} images supplied but this engine was loaded with ` +
            `maxImages: ${maxImages}. Raise it at load() time.`,
        );
      }
      if (audioParts.length > 0 && !supportAudio) {
        throw new CapabilityError(
          'This MediaPipe engine was loaded without audio support. Pass ' +
            '{ supportAudio: true } to load() — MediaPipe fixes it at load time, ' +
            'so it cannot be enabled per call.',
        );
      }

      const truncate = createStopTruncator(stopSequences);
      const deltas = createDeltaStream();
      let emitted = '';

      const run = async () => {
        // A string when nothing else is present, so the text-only path stays
        // exactly what it was: formatPromptParts would collapse to one string
        // anyway, but passing the string keeps the verified path untouched.
        const prompt = hasNonTextParts(messages)
          ? await toMediaPipePrompt(messages, turnFormat)
          : formatPrompt(messages, turnFormat);

        // Deliberately no setOptions() here. MediaPipe deletes
        // baseOptions.modelAssetBuffer once it consumes the stream, so a later
        // setOptions() call re-validates against an empty asset and throws
        // "No model asset provided." Pass maxTokens to load() instead.
        if (genOpts.maxTokens && genOpts.maxTokens !== maxTokens) {
          console.warn(
            `[in-browser-llm] MediaPipe fixes maxTokens at load time; ignoring ` +
              `${genOpts.maxTokens} and keeping ${maxTokens}. Pass maxTokens to load().`,
          );
        }
        // partial is a DELTA, and completion arrives as ("", true).
        try {
          await llm.generateResponse(prompt, (partial: string) => {
            if (!partial) return;
            const visible = truncate(partial);
            if (visible.length > emitted.length) {
              deltas.push(visible.slice(emitted.length));
              emitted = visible;
            }
          });
        } catch (cause) {
          throw explainModalityFailure(cause);
        }
      };

      run().then(
        () => deltas.finish(),
        (error: unknown) => deltas.fail(error),
      );

      return deltas.iterate();
    },
    close() {
      llm.close();
    },
  };
}
