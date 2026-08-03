import type { TurnFormat } from './catalog.js';
import type { ToolCall } from './tools.js';

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; src: string | Blob }
  // Two forms because decoding needs AudioContext, which exists only on the main
  // thread: a caller running inference in a worker decodes outside and passes
  // `data`, everyone else passes `src` and lets the library handle it.
  | { type: 'audio'; src: Blob }
  | { type: 'audio'; data: Float32Array };

/** collectAudio's return type: either audio variant, before decoding. */
export type AudioPart = Extract<ContentPart, { type: 'audio' }>;

export interface Message {
  /**
   * 'tool' carries a tool's result back to the model. The chat template wraps it
   * in its own markup — Qwen3 renders it as <tool_response> inside a user turn.
   */
  role: 'user' | 'assistant' | 'tool';
  /**
   * A plain string is shorthand for a single text part, so existing callers keep
   * working. Use parts to attach an image or audio.
   */
  content: string | ContentPart[];
  /**
   * Assistant turns only: the calls the model asked for, replayed so it sees its
   * own request in the next turn. Flat rather than OpenAI's
   * `{function: {...}}` — Qwen3's template unwraps either, and this shape is what
   * parseToolCalls returns.
   */
  tool_calls?: ToolCall[];
}

export function toParts(content: string | ContentPart[]): ContentPart[] {
  return typeof content === 'string' ? [{ type: 'text', text: content }] : content;
}

/** OpenAI's nested form, which is what chat templates read. Never a library input. */
interface NestedToolCall {
  type: 'function';
  function: { name: string; arguments: Record<string, unknown> };
}

/** A message with tool calls rewritten for the template. Not part of the public API. */
export type TemplateMessage = Omit<Message, 'tool_calls'> & {
  tool_calls?: NestedToolCall[];
};

/**
 * Rewrites `tool_calls` into OpenAI's nested shape for the chat template.
 *
 * The library's own shape is flat — `{name, arguments}`, what `parseToolCalls`
 * returns — because that is what a caller wants to read. Templates disagree about
 * which they accept, and gemma-4 is the strict one: the flat form throws
 * `Cannot perform operation + on undefined values` from inside Jinja, naming nothing.
 * Qwen3.5 and Qwen3-VL accept **both**, verified by rendering each, so converting
 * unconditionally needs no per-family branch.
 */
export function toTemplateToolCalls(messages: Message[]): TemplateMessage[] {
  return messages.map(({ tool_calls, ...rest }) => {
    // Dropped entirely when empty rather than passed through as []: a template that
    // tests `if message.tool_calls` would take the call branch on an empty array and
    // render an empty call block.
    if (!tool_calls?.length) return rest;
    return {
      ...rest,
      tool_calls: tool_calls.map((call) => ({
        type: 'function' as const,
        function: { name: call.name, arguments: call.arguments },
      })),
    };
  });
}

// The reverse of toParts, for templates that require a string. Llama-derived ones
// apply `content | trim`, which throws "Unknown ArrayValue filter: trim" on an
// array — the exact opposite of what a vision template needs. Non-text parts are
// dropped because the processor receives them separately as tensors.
export function toText(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content;
  return content
    .flatMap((p) => (p.type === 'text' ? [p.text] : []))
    .join(' ');
}

export function collectImages(messages: Message[]): Array<string | Blob> {
  return messages.flatMap((m) =>
    toParts(m.content).flatMap((p) => (p.type === 'image' ? [p.src] : [])),
  );
}

export function collectAudio(messages: Message[]): AudioPart[] {
  return messages.flatMap((m) =>
    toParts(m.content).flatMap((p) => (p.type === 'audio' ? [p] : [])),
  );
}

export function hasNonTextParts(messages: Message[]): boolean {
  return messages.some((m) => toParts(m.content).some((p) => p.type !== 'text'));
}

/**
 * Renders the turn structure a runtime without chat-template support needs.
 *
 * The result must end by opening the model's turn. Omitting that makes the model
 * generate its answer repeatedly — observed with Gemma 4 (108 chunks, then 48).
 *
 * Non-text parts survive as themselves, because MediaPipe accepts an array of parts
 * and a string has nowhere to put pixels. Adjacent text is merged so the result is
 * the fewest parts that preserve order — the runtime concatenates string chunks
 * anyway, and fewer parts makes the array readable when logged. A text-only
 * conversation therefore collapses to a single string, which is what formatPrompt
 * returns.
 *
 * Deliberately not typed to MediaPipe's `Prompt`: keeping this runtime-agnostic
 * means prompt.ts stays testable without importing a runtime.
 */
export function formatPromptParts(
  messages: Message[],
  format: TurnFormat,
): Array<string | Exclude<ContentPart, { type: 'text' }>> {
  const out: Array<string | Exclude<ContentPart, { type: 'text' }>> = [];
  const pushText = (text: string) => {
    const last = out[out.length - 1];
    if (typeof last === 'string') out[out.length - 1] = last + text;
    else out.push(text);
  };

  for (const message of messages) {
    pushText(message.role === 'user' ? format.user : format.model);
    for (const part of toParts(message.content)) {
      if (part.type === 'text') pushText(part.text);
      else out.push(part);
    }
    pushText(`${format.end}\n`);
  }
  pushText(format.model);
  return out;
}

// The string form, for a runtime that cannot take parts. Built on formatPromptParts
// so the two cannot disagree about turn structure — that agreement used to be a test
// rather than a property of the code. Throws on a non-text part instead of dropping
// it: a silently ignored image yields an answer that does not reflect it, with no way
// for the caller to tell why.
export function formatPrompt(messages: Message[], format: TurnFormat): string {
  return formatPromptParts(messages, format)
    .map((part) => {
      if (typeof part === 'string') return part;
      throw new Error(
        `formatPrompt handles text only, received a "${part.type}" part. ` +
          'This runtime cannot accept that input.',
      );
    })
    .join('');
}

export function truncateAtStop(text: string, stops: string[]): string {
  let cut = -1;
  for (const stop of stops) {
    const at = text.indexOf(stop);
    if (at !== -1 && (cut === -1 || at < cut)) cut = at;
  }
  return cut === -1 ? text : text.slice(0, cut);
}

// Accumulates deltas and truncates at a stop sequence, which may straddle two
// deltas. Needed because MediaPipe's stop tokens are hardcoded to <eos> and
// <|endoftext|> — it never halts on a model-specific terminator like <turn|>.
export function createStopTruncator(stops: string[]): (delta: string) => string {
  let raw = '';
  let done = false;
  let visible = '';

  return (delta: string): string => {
    if (done) return visible;
    raw += delta;
    const truncated = truncateAtStop(raw, stops);
    if (truncated.length !== raw.length) {
      done = true;
    }
    visible = truncated;
    return visible;
  };
}
