import type { ModelRef } from './types.js';
import type { ToolCallFormat } from './tools.js';

export interface TurnFormat {
  /** Prefix for a user turn, e.g. "<|turn>user\n". */
  user: string;
  /** Turn terminator, e.g. "<turn|>". Also used as a stop sequence. */
  end: string;
  /** Prefix that starts the model's turn, e.g. "<|turn>model\n". */
  model: string;
}

export interface CatalogEntry {
  /**
   * MediaPipe only: that runtime cannot apply a chat template, so its turn
   * delimiters must be known in advance. Absent for transformers-only entries.
   */
  turnFormat?: TurnFormat;
  stopSequences?: string[];
  /**
   * How this model frames a tool call in its output. Absent means tool calling is
   * unsupported for it.
   *
   * An entry whose encoding is `gemma4` needs raw text: its delimiters are special
   * tokens, so a caller must pass `GenerateOptions.onRawText` to `generate()` and hand
   * that string to `parseToolCalls`. The streamed deltas will not contain a call.
   */
  toolCallFormat?: ToolCallFormat;
}

// Not a model list — a correction table for runtime gaps. Holds only what
// inspection cannot reveal. Extracted with scripts/inspect-chat-template.mjs.
//
// An entry is matched by substring, so ONE entry can serve the same model on both
// runtimes — `gemma-4` matches the ONNX repo id and the LiteRT URL alike. Its fields
// are not both usable there: `turnFormat` is for MediaPipe, `toolCallFormat` needs
// transformers.js. A caller reading a format off an entry must check `ref.runtime`
// before offering the feature.
export const CATALOG: Record<string, CatalogEntry> = {
  // Gemma 4 uses <|turn> pairs, NOT Gemma 2/3's <start_of_turn>. Passing a raw
  // string makes the model answer repeatedly.
  'gemma-4': {
    turnFormat: {
      user: '<|turn>user\n',
      end: '<turn|>',
      model: '<|turn>model\n',
    },
    stopSequences: ['<turn|>'],
    // Its own dialect, not the <tool_call> shape the Qwen entries share. Both
    // delimiters are special tokens (ids 48 and 49), so this only parses text decoded
    // with skip_special_tokens: false — see GenerateOptions.onRawText. The delimiters
    // and the call syntax come from the export's own tokenizer_config.json
    // `response_schema`, not from reverse-engineering the output.
    //
    // MediaPipe cannot use this even though it shares the turnFormat above: that path
    // has no raw-token hook, only truncated strings.
    toolCallFormat: {
      delimiters: ['<|tool_call>', '<tool_call|>'],
      encoding: 'gemma4',
    },
  },
  // transformers.js only — no turnFormat, because MediaPipe cannot do tools.
  //
  // Qwen3.5 kept ChatML's <tool_call> delimiters but changed the body to XML, so it
  // needs its own entry and must be listed BEFORE 'qwen3': lookupCatalog matches by
  // substring, and "qwen3" is a prefix of "qwen3.5".
  'qwen3.5': {
    toolCallFormat: {
      delimiters: ['<tool_call>', '</tool_call>'],
      encoding: 'xml',
    },
  },
  // Qwen2.5 emits the identical JSON form, but matching is by substring and
  // "qwen2.5" is not one, so adding that model means adding a key.
  qwen3: {
    toolCallFormat: {
      delimiters: ['<tool_call>', '</tool_call>'],
      encoding: 'json',
    },
  },
};

function refKey(ref: ModelRef): string {
  const raw = ref.runtime === 'mediapipe' ? ref.url : 'repo' in ref ? ref.repo : ref.baseUrl;
  return raw.toLowerCase();
}

export function lookupCatalog(ref: ModelRef): CatalogEntry | undefined {
  const key = refKey(ref);
  const match = Object.keys(CATALOG).find((name) => key.includes(name));
  return match ? CATALOG[match] : undefined;
}
