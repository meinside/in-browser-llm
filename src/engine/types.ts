import type { Message } from '../prompt.js';
import type { Progress } from '../loader.js';
import type { Capabilities } from '../capability.js';
import type { ModelSpec } from '../gate.js';

export interface GenerateOptions {
  /**
   * Token cap for this call. Defaults to DEFAULT_MAX_TOKENS when omitted.
   *
   * Pass a larger value for long answers. Hitting the cap truncates output
   * mid-sentence, which reads like a bug but is not one.
   *
   * Note the two runtimes count differently, and this option is normalized over
   * that difference rather than hiding it:
   *   - transformers.js `max_new_tokens` caps OUTPUT tokens only.
   *   - MediaPipe `maxTokens` caps INPUT + OUTPUT combined.
   * So for a long prompt, MediaPipe reaches the cap sooner than transformers.js.
   */
  maxTokens?: number;
  /**
   * Extra generation options merged in verbatim, e.g.
   * `{ do_sample: false }` for deterministic output or `{ temperature: 0.3 }`.
   *
   * Left unset, each model's own `generation_config.json` applies — which is
   * usually what you want, since the model authors tuned it. Only transformers.js
   * honours this; MediaPipe fixes its sampling parameters at load time.
   */
  overrides?: Record<string, unknown>;
  /**
   * Extra arguments for the model's chat template, merged into the
   * `apply_chat_template` call. Only transformers.js uses this.
   *
   * Needed by reasoning models: Qwen3 emits long `<think>` blocks unless given
   * `{ enable_thinking: false }`, and Gemma 4 accepts the same flag.
   */
  templateArgs?: Record<string, unknown>;
  /**
   * Called once when generation finishes, with the answer decoded **keeping special
   * tokens**. Only transformers.js supplies this.
   *
   * The streamed deltas drop special tokens, which is right for display and wrong for
   * parsing a format built out of them. gemma-4's tool calls are delimited by
   * `<|tool_call>` and quote strings with `<|"|>`, all `special: true` — so in the
   * streamed text `{a: 3}` and `{a: "3"}` are the same string and the argument types
   * are gone. Pass this text to `parseToolCalls` with the `gemma4` encoding.
   *
   * Fires once at the end rather than per delta, because half a tool call cannot be
   * parsed and a partial parse is worse than waiting.
   *
   * Unnecessary for Qwen3 and Qwen3.5, whose `<tool_call>` tags are `special: false`
   * and therefore survive in the ordinary stream.
   */
  onRawText?: (text: string) => void;
}

/**
 * Default token cap, applied to both runtimes so they truncate consistently.
 *
 * Derived from decode speed, not from the model: at the ~10 tok/s measured for
 * gemma-4 on Chrome, 1024 tokens is roughly 100 seconds of waiting, and Firefox
 * without `subgroups` is several times slower again. That wait is the binding
 * constraint in a browser, so the cap answers "how long will a user sit here"
 * rather than "what does the model allow".
 *
 * **Deriving it from `max_position_embeddings` was considered and rejected.**
 * The models used here report 262144 (Qwen3.5-0.8B and Qwen3-VL-2B alike) and
 * 131072 (gemma-4) — 128x to 256x this value. Any honest formula would be
 * `min(config, a practical ceiling)`, and the ceiling wins every time, so the
 * derivation would add machinery without changing the answer. Two further
 * obstacles: gemma models set `sliding_window: 512`, so KV cache growth is not
 * linear in context length, and the runtimes disagree about what is being
 * counted (see GenerateOptions.maxTokens), so one config number cannot map onto
 * both without separate conversions.
 *
 * Callers who want longer answers pass `maxTokens` explicitly.
 */
export const DEFAULT_MAX_TOKENS = 1024;

export interface LoadOptions {
  onProgress?: (progress: Progress) => void;
  /**
   * Called once with whether the weights came from the cache instead of the
   * network.
   *
   * **Only the MediaPipe path calls this.** transformers.js caches too, but it
   * manages its own storage rather than going through fetchWithProgress, so there
   * is nothing to observe here — a fast load with no progress events is the only
   * hint available on that path.
   */
  onCacheStatus?: (fromCache: boolean) => void;
  /**
   * A probe() result. Without it dtype defaults to q4f16 unconditionally, which
   * fails on browsers carrying a known f16 shader bug.
   */
  capabilities?: Capabilities;
  /**
   * Default token cap for this engine. Required for MediaPipe, which fixes the
   * value at load time and cannot change it afterwards. On transformers.js it sets
   * the default that `GenerateOptions.maxTokens` overrides per call.
   */
  maxTokens?: number;
  /**
   * The inspectModel() result. Without it the adapter assumes a text-only model,
   * since loading a processor for a model that has none fails.
   */
  spec?: ModelSpec;
  /**
   * **MediaPipe only.** Maximum images one query may contain, enabling the vision
   * modality. Load-time like maxTokens, and for the same reason: MediaPipe deletes
   * the model asset after consuming it, so setOptions() afterwards throws.
   *
   * Opt-in rather than inferred, because a `.task` bundle exposes no config to read
   * the modality from, and because the option switches MediaPipe to streaming
   * loading — a cost not worth paying for callers sending only text.
   *
   * Exceeding it throws at generate time, from MediaPipe rather than this library.
   */
  maxImages?: number;
  /**
   * **MediaPipe only.** Enables the audio modality. Load-time, as maxImages is.
   */
  supportAudio?: boolean;
}

export interface Engine {
  /**
   * Yields DELTAS, never cumulative text. Both runtimes emit deltas natively;
   * the contract is stated because assuming cumulative output corrupts display.
   */
  generate(messages: Message[], opts?: GenerateOptions): AsyncIterable<string>;
  close(): void;
}
