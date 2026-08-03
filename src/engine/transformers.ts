import { DEFAULT_MAX_TOKENS, type Engine, type GenerateOptions, type LoadOptions } from './types.js';
import { createDeltaStream } from './stream.js';
import {
  collectAudio,
  collectImages,
  toParts,
  toTemplateToolCalls,
  toText,
  type Message,
  type TemplateMessage,
} from '../prompt.js';
import { decodeAudio, warnIfTooLong } from '../audio.js';
import { resolveBaseUrl, type ModelRef } from '../types.js';
import { RuntimeError } from '../errors.js';
import type { Capabilities } from '../capability.js';
import type { Progress } from '../loader.js';

// The library's own dtype type, so a mismatch is a compile error rather than a
// silent fp32 download. A single string or a per-session map.
type Dtype = NonNullable<
  Parameters<typeof import('@huggingface/transformers').AutoModelForCausalLM.from_pretrained>[1]
>['dtype'];

// A single string applies to every session, whatever the export's layout is.
//
// Do NOT pass a per-session object here. selectDtype() resolves an unlisted
// session to null and falls back to fp32 — so hardcoding gemma-4's session names
// (embed_tokens, decoder_model_merged, …) silently downloads fp32 for a model
// that ships a single `model` session, which is 4x larger than intended.
// Per-session dtypes are worth supporting only alongside reading the actual
// session list from config.json.
// Exported for testing: this decision silently changes what gets downloaded.
export function dtypeFor(caps: Capabilities | undefined): Dtype {
  if (!caps) return 'q4f16';
  // Needs the feature AND no known shader bug. The second condition carries the
  // weight: some exports include fp16 overflow-guard Clip nodes that Firefox
  // fails to compile at inference time, and q4 variants omit them.
  return caps.hasF16 && caps.knownF16ShaderIssues.length === 0 ? 'q4f16' : 'q4';
}

type TensorCtor = typeof import('@huggingface/transformers').Tensor;

interface PositionTensor {
  data: BigInt64Array;
  dims: number[];
}

/**
 * Recomputes mrope position deltas with the sign the rest of transformers.js uses.
 *
 * `Qwen2VLForConditionalGeneration.get_rope_index` returns
 * `[position_ids, mrope_position_deltas]` from two branches that disagree about what
 * the delta means:
 *
 * - multimodal: `max(positions) + 1 - seq_len` (`modeling_qwen2_vl.js:257`)
 * - text-only:  `max(positions) + 1 `**`+`**` seq_len` (line 37)
 *
 * A `+` where the other has a `-`. The delta is consumed as an offset —
 * `prepare_inputs_for_generation` places each decode step at `past_length + delta` —
 * so for a contiguous text-only sequence it must be 0. Upstream yields `2 * seq_len`,
 * putting the first generated token near `3 * seq_len`, far outside the trained
 * position range. Output degenerates into a repeated token and EOS never arrives, so
 * generation runs to `max_new_tokens`.
 *
 * Only the delta is wrong; the `position_ids` from that branch are already
 * `[0, 1, 2, …]` and are left alone.
 *
 * Derived from the returned positions rather than recomputed from the mask, so this
 * stays correct if upstream changes how positions are built: the delta is defined in
 * terms of them. Padding is handled the same way the multimodal branch handles it —
 * `seq_len` is the full padded length.
 */
export function correctedMropeDeltas(positionIds: PositionTensor): BigInt64Array {
  // Shape is [3, batch, seq]; the three sections are identical for text, so section 0
  // is enough.
  const [, batch, seqLength] = positionIds.dims as [number, number, number];
  return BigInt64Array.from({ length: batch }, (_, i) => {
    const section = positionIds.data.subarray(i * seqLength, (i + 1) * seqLength);
    let highest = section[0] ?? 0n;
    for (const value of section) if (value > highest) highest = value;
    return highest + 1n - BigInt(seqLength);
  });
}

/**
 * Warns when a model ships no sampling defaults, because the runtime's fallback is
 * greedy and greedy decoding on a small model degenerates into repetition.
 *
 * transformers.js defaults `do_sample = false` (`configuration_utils.js:68`), so a
 * `generation_config.json` carrying only `eos_token_id` — as
 * `Qwen3.5-0.8B-Text-ONNX` does — decodes greedily. Observed: a conversational prompt
 * answered with one phrase repeated to the token cap. Both other models in the
 * lineup do ship `do_sample: true`, which is why only this one shows it.
 *
 * A warning rather than an injected default. Trap 7 in CLAUDE.md is that overriding a
 * model's own sampling defaults made a different model worse, and the library's rule
 * is to apply what the model asks for. Here the model asks for nothing, so the honest
 * options are to say so or to invent values — and inventing them silently is what
 * trap 7 warns against. The caller passes `overrides` if they want different
 * behaviour; this only makes the cause visible.
 */
export function warnIfGreedyByDefault(model: unknown, modelId: string): void {
  const config = (model as { generation_config?: Record<string, unknown> | null })
    ?.generation_config;
  if (!config) return;
  // do_sample explicitly false is a deliberate choice; absent is the silent case.
  if ('do_sample' in config || 'temperature' in config || 'top_k' in config) return;

  console.warn(
    `[in-browser-llm] ${modelId} ships no sampling parameters in ` +
      'generation_config.json, so transformers.js decodes greedily ' +
      '(do_sample: false). Small models often repeat a phrase until the token cap ' +
      'that way. Pass sampling values through GenerateOptions.overrides — check the ' +
      "source model's card for what it recommends.",
  );
}

// Only the qwen-vl lineage carries the sign error. Matched on config.model_type rather
// than the repo name so a re-upload under a different id still gets the fix.
const MROPE_MODEL_TYPES = ['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'];

export function expectsMropePositions(model: unknown): boolean {
  const modelType = (model as { config?: { model_type?: string } })?.config?.model_type;
  return typeof modelType === 'string' && MROPE_MODEL_TYPES.includes(modelType);
}

interface MropeModel {
  _get_text_only_rope_index?: (
    inputIds: unknown,
    attentionMask: unknown,
  ) => [PositionTensor, PositionTensor];
  __mropeDeltaPatched?: boolean;
}

/**
 * Patches the delta at its source, on the loaded model instance.
 *
 * Supplying `position_ids` to `generate()` instead does NOT work, and the reason is
 * worth keeping: `prepare_inputs_for_generation` returns early when they are set, so
 * it fixes the prefill step — but `_update_model_kwargs_for_generation` assigns
 * `position_ids = null` after every step (`modeling_utils.js:687`), forcing a
 * recompute that goes straight back through the buggy branch. The first token comes
 * out right and every one after it does not, which shows up as a repeated word.
 *
 * Wrapping `_get_text_only_rope_index` covers prefill and decode alike, since
 * `get_rope_index` routes to it whenever no `image_grid_thw` is present. Turns
 * carrying an image take the multimodal branch and are untouched.
 *
 * Idempotent, and a no-op if the method is absent — remove the whole thing once
 * upstream fixes the sign.
 */
export function patchMropeDeltas(model: unknown, Tensor: TensorCtor): void {
  if (!expectsMropePositions(model)) return;
  const target = model as MropeModel;
  if (target.__mropeDeltaPatched || typeof target._get_text_only_rope_index !== 'function') {
    return;
  }

  const original = target._get_text_only_rope_index.bind(target);
  target._get_text_only_rope_index = (inputIds, attentionMask) => {
    const [positionIds] = original(inputIds, attentionMask);
    const deltas = correctedMropeDeltas(positionIds);
    return [positionIds, new Tensor('int64', deltas, [deltas.length, 1]) as PositionTensor];
  };
  target.__mropeDeltaPatched = true;
}

/** What the model itself accepts, from inspectModel(). Never what a caller supplied. */
interface Modalities {
  vision: boolean;
  audio: boolean;
}

/**
 * Picks the Auto class for a model's modalities. The three registries do not overlap,
 * so a wrong class throws "Unsupported model type" at load.
 *
 * Vision is tested first because gemma-4 declares both modalities and is registered
 * only under ImageTextToText — reversing the two conditions breaks it silently.
 */
export function selectModelClass<T>(
  classes: { causalLM: T; imageTextToText: T; audioTextToText: T },
  spec: Modalities,
): T {
  if (spec.vision) return classes.imageTextToText;
  if (spec.audio) return classes.audioTextToText;
  return classes.causalLM;
}

/**
 * Builds the positional arguments for a processor call.
 *
 * Arity AND order are fixed per model family, and both differ:
 * gemma-4 is `_call(text, images, audio, options)`, qwen2/3-vl is
 * `_call(text, images, ...args)`, and UltravoxProcessor is `_call(text, audio, kwargs)`
 * — audio where gemma-4 puts images. So the shape comes from the model's modalities,
 * never from what the caller happened to send: a text-only turn on gemma-4 must still
 * pass `null` into the audio slot, or the options object lands there and the feature
 * extractor rejects it with "expects a Float32Array … but got Object".
 *
 * The audio-only shape is UNVERIFIED — no model has reached inference through it. It
 * is written to UltravoxProcessor's signature, which reads `audio.length` as a sample
 * count, hence one clip rather than the array gemma-4 wants.
 */
export function processorCallArgs(
  spec: Modalities,
  prompt: unknown,
  images: unknown[] | null,
  audio: Float32Array[] | null,
  options: object,
): unknown[] {
  if (spec.vision) {
    return spec.audio ? [prompt, images, audio, options] : [prompt, images, options];
  }
  return [prompt, audio?.[0] ?? null, options];
}

/**
 * Shapes message content the way the family's chat template requires. The two
 * requirements are opposite, which is why this cannot be one form.
 *
 * Vision templates index `message['content'][0]['type']` and loop over it, so a plain
 * string fails with "Expected iterable or object type in for loop: got StringValue".
 * Llama-derived templates — every audio-only family here — apply `content | trim`,
 * which fails on an array with "Unknown ArrayValue filter: trim".
 *
 * **Empty content is left empty rather than wrapped.** `toParts('')` would yield
 * `[{type: 'text', text: ''}]`, and a one-element array is *truthy* where `''` is
 * falsy — which matters because templates branch on that. gemma-4's does:
 *
 *     {%- elif not (ns_tr_out.flag and not message.get('content')) -%}
 *         {{- '<turn|>\n' -}}
 *
 * An assistant turn carrying only `tool_calls` has empty content by construction, so
 * wrapping it closed the model's turn after the tool response. `add_generation_prompt`
 * then declines to reopen it (it skips when the previous block was a tool_response), so
 * the third turn handed the model a finished conversation and it answered with EOS —
 * zero chunks, no error. Nothing about vision was involved; the wrap was.
 */
export function normalizeContent(
  messages: TemplateMessage[],
  isVision: boolean,
): TemplateMessage[] {
  return messages.map((m) => {
    if (typeof m.content === 'string' && m.content === '') return m;
    return { ...m, content: isVision ? toParts(m.content) : toText(m.content) };
  });
}

interface ProgressInfo {
  status?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

// progress_total is derived only from 'progress' events, which a cache hit never
// emits — the bytes never stream. Left alone the bar freezes mid-way even though
// loading succeeded, so completion is reported explicitly once loading returns.
function reportComplete(onProgress: ((p: Progress) => void) | undefined, totalBytes: number): void {
  if (!onProgress) return;
  onProgress({ loaded: totalBytes, total: totalBytes, ratio: 1 });
}

export async function createTransformersEngine(
  ref: ModelRef,
  opts: LoadOptions = {},
): Promise<Engine> {
  if (ref.runtime !== 'transformers') {
    throw new RuntimeError(`Expected a transformers ref, received ${ref.runtime}`);
  }

  let mod: typeof import('@huggingface/transformers');
  try {
    mod = await import('@huggingface/transformers');
  } catch (cause) {
    throw new RuntimeError(
      'Install @huggingface/transformers@4.2.0 to use this runtime.',
      { cause },
    );
  }

  const {
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForAudioTextToText,
    TextStreamer,
    Tensor,
    load_image,
    env,
  } = mod;

  // Point the library at a self-hosted directory when a baseUrl was given.
  let modelId: string;
  if ('repo' in ref) {
    modelId = ref.repo;
  } else {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = `${resolveBaseUrl(ref)}/`;
    modelId = '.';
  }

  const onProgress = opts.onProgress;
  let lastTotal = 0;
  const progress_callback = (info: ProgressInfo) => {
    // Only progress_total is aggregated across files; per-file events double-count.
    if (info.status === 'progress_total' && onProgress) {
      const total = info.total ?? 0;
      const loaded = info.loaded ?? 0;
      lastTotal = total;
      onProgress({
        loaded,
        total,
        ratio: total ? loaded / total : (info.progress ?? 0) / 100,
      });
    }
  };

  // Modalities come from the spec, never from what a caller supplied — see
  // processorCallArgs. A vision or audio model needs a processor rather than a
  // tokenizer: only a processor turns pixels or samples into tensors. Both expose
  // apply_chat_template, so prompt construction below is shared.
  const modalities: Modalities = {
    vision: opts.spec?.vision ?? false,
    audio: opts.spec?.audio ?? false,
  };
  const needsProcessor = modalities.vision || modalities.audio;

  const tokenizer = needsProcessor
    ? await AutoProcessor.from_pretrained(modelId, { progress_callback })
    : await AutoTokenizer.from_pretrained(modelId, { progress_callback });

  // AutoModelForImageTextToText rather than AutoModelForVision2Seq: the VISION_2_SEQ
  // registry mapping omits gemma4. The audio-only class is UNVERIFIED — every
  // audio-text model transformers.js registers (granite_speech, ultravox, voxtral,
  // voxtral_realtime) is reachable only through it, but none has been made to work
  // end-to-end here (see CLAUDE.md). It stays because the alternative is a guaranteed
  // "Unsupported model type" throw.
  const ModelClass = selectModelClass(
    {
      causalLM: AutoModelForCausalLM,
      imageTextToText: AutoModelForImageTextToText,
      audioTextToText: AutoModelForAudioTextToText,
    },
    modalities,
  );

  const model = await ModelClass.from_pretrained(modelId, {
    device: 'webgpu',
    dtype: dtypeFor(opts.capabilities),
    progress_callback,
  });

  // Applied once, at load: the delta is recomputed on every decode step, so a
  // per-call override would only fix the first token. See patchMropeDeltas.
  patchMropeDeltas(model, Tensor);

  warnIfGreedyByDefault(model, modelId);

  reportComplete(onProgress, lastTotal);

  return {
    generate(messages: Message[], genOpts: GenerateOptions = {}) {
      const deltas = createDeltaStream();

      const run = async () => {
        const images = collectImages(messages);

        const audioParts = collectAudio(messages);

        if (audioParts.length > 0 && !modalities.audio) {
          throw new RuntimeError(
            'This model has no audio encoder, so it cannot accept audio input. ' +
              'Pick a model whose config.json declares audio_config.',
          );
        }
        if (audioParts.length > 1) {
          // processing_gemma4.js feature-extracts audio_array[0] only, using the
          // rest for token arithmetic — so extra clips are silently ignored.
          console.warn(
            `[in-browser-llm] ${audioParts.length} audio parts supplied, but the ` +
              'processor only extracts features from the first one.',
          );
        }
        if (images.length > 0 && !modalities.vision) {
          throw new RuntimeError(
            'This model was loaded without a vision encoder, so it cannot see ' +
              'images. Pass the inspectModel() result as LoadOptions.spec.',
          );
        }

        // The library interprets the model's Jinja chat template, so the turn
        // format does not need a catalog entry on this path.
        // Cast because the typed options omit template-specific kwargs like
        // enable_thinking, which Jinja templates read at runtime.
        const templateOptions = {
          add_generation_prompt: true,
          return_dict: true,
          // Reasoning models otherwise emit long <think> blocks. Harmless for
          // models whose template ignores the flag.
          enable_thinking: false,
          ...genOpts.templateArgs,
        } as Parameters<typeof tokenizer.apply_chat_template>[1];

        // Templates read OpenAI's nested tool_calls; the library's own shape is flat.
        // gemma-4 rejects the flat one outright, and Qwen accepts either.
        const forTemplate = toTemplateToolCalls(messages);

        let inputs;
        if (needsProcessor) {
          // A processor must build the inputs even for a text-only turn: a vision
          // model expects pixel-related tensors alongside the token ids, and
          // apply_chat_template alone leaves them null (surfacing later as
          // "Cannot read properties of null (reading 'dims')"). So the template
          // renders to a string here and the processor turns it into tensors.
          const prompt = tokenizer.apply_chat_template(
            normalizeContent(forTemplate, modalities.vision) as Parameters<
              typeof tokenizer.apply_chat_template
            >[0],
            {
              ...templateOptions,
              return_dict: false,
              tokenize: false,
            } as Parameters<typeof tokenizer.apply_chat_template>[1],
          );

          const objectUrls: string[] = [];
          try {
            const loaded = await Promise.all(
              images.map((src) => {
                if (typeof src === 'string') return load_image(src);
                const url = URL.createObjectURL(src);
                objectUrls.push(url);
                return load_image(url);
              }),
            );

            // Blob parts are decoded here; pre-decoded ones pass straight through.
            const audio = await Promise.all(
              audioParts.map((part) => ('data' in part ? part.data : decodeAudio(part.src))),
            );
            audio.forEach(warnIfTooLong);

            const call = tokenizer as unknown as {
              (...args: unknown[]): Promise<object>;
            };
            inputs = await call(
              ...processorCallArgs(
                modalities,
                prompt,
                loaded.length > 0 ? loaded : null,
                audio.length > 0 ? audio : null,
                { add_special_tokens: false },
              ),
            );
          } finally {
            objectUrls.forEach((url) => URL.revokeObjectURL(url));
          }
        } else {
          inputs = tokenizer.apply_chat_template(
            forTemplate as Parameters<typeof tokenizer.apply_chat_template>[0],
            templateOptions,
          );
        }

        // A Processor wraps a tokenizer rather than being one, and TextStreamer
        // needs the tokenizer itself to decode. The getter is optional on
        // Processor, so a processor without one cannot stream.
        const decoder = 'tokenizer' in tokenizer ? tokenizer.tokenizer : tokenizer;
        if (!decoder) {
          throw new RuntimeError(
            'This model\'s processor exposes no tokenizer, so generated tokens ' +
              'cannot be decoded.',
          );
        }

        // Collected only when asked for, since holding every id costs memory that a
        // caller who does not need raw text should not pay.
        //
        // token_callback_function receives the raw ids at streamers.js:94, BEFORE the
        // branch that drops special tokens — which is the only place in this pipeline
        // where gemma-4's tool-call delimiters still exist. Verified: decoding an
        // incrementally accumulated array equals decoding the whole thing at once.
        const rawIds: bigint[] = [];
        const streamer = new TextStreamer(decoder, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (text: string) => deltas.push(text),
          ...(genOpts.onRawText
            ? {
                token_callback_function: (tokens: bigint[]) => rawIds.push(...tokens),
              }
            : {}),
        });

        // Sampling parameters are deliberately NOT set here. Each model ships a
        // generation_config.json with its own recommendation — gemma-3-270m asks
        // for do_sample with top_k 64 / top_p 0.95 — and the library applies it.
        // Forcing do_sample: false made that model greedy, which for a 270M model
        // produced rambling, repetitive output. Callers who want determinism can
        // pass sampling options through GenerateOptions instead.
        await model.generate({
          ...(inputs as object),
          max_new_tokens: genOpts.maxTokens ?? opts.maxTokens ?? DEFAULT_MAX_TOKENS,
          ...genOpts.overrides,
          streamer,
        });

        // After generation, not per delta: half a tool call cannot be parsed.
        // Guarded on length because decode() throws "token_ids must be a non-empty
        // array" — reachable when the model emits EOS immediately, which must not turn
        // an empty answer into an exception.
        if (genOpts.onRawText) {
          genOpts.onRawText(
            rawIds.length > 0 ? decoder.decode(rawIds, { skip_special_tokens: false }) : '',
          );
        }
      };

      run().then(
        () => deltas.finish(),
        (error: unknown) => deltas.fail(error),
      );

      return deltas.iterate();
    },
    close() {
      void model.dispose?.();
    },
  };
}
