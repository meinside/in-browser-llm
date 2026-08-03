import { describe, expect, it } from 'vitest';
import {
  collectAudio,
  collectImages,
  createStopTruncator,
  formatPrompt,
  formatPromptParts,
  hasNonTextParts,
  toParts,
  toTemplateToolCalls,
  toText,
  truncateAtStop,
} from '../src/prompt.js';
import { CATALOG, lookupCatalog } from '../src/catalog.js';

// turnFormat is optional — transformers-only entries carry none — but gemma-4
// runs on MediaPipe and must have one.
const GEMMA4 = CATALOG['gemma-4']!.turnFormat!;

describe('formatPrompt', () => {
  it('wraps a single user turn and opens the model turn', () => {
    expect(formatPrompt([{ role: 'user', content: 'hi' }], GEMMA4)).toBe(
      '<|turn>user\nhi<turn|>\n<|turn>model\n',
    );
  });

  it('includes prior assistant turns', () => {
    const prompt = formatPrompt(
      [
        { role: 'user', content: 'a' },
        { role: 'assistant', content: 'b' },
        { role: 'user', content: 'c' },
      ],
      GEMMA4,
    );
    expect(prompt).toBe(
      '<|turn>user\na<turn|>\n<|turn>model\nb<turn|>\n<|turn>user\nc<turn|>\n<|turn>model\n',
    );
  });

  it('always ends by opening the model turn', () => {
    // Without this the model has no turn to complete and repeats itself.
    expect(formatPrompt([{ role: 'user', content: 'x' }], GEMMA4).endsWith(GEMMA4.model)).toBe(
      true,
    );
  });
});

// MediaPipe takes an array of parts, so pixels have somewhere to go that a string
// cannot provide. formatPrompt is built on this, so the two cannot disagree about turn
// structure — no test is needed to assert that they match.
describe('formatPromptParts', () => {
  it('collapses a text-only conversation to a single string', () => {
    // Which is what makes formatPrompt a join over this.
    expect(formatPromptParts([{ role: 'user', content: 'hi' }], GEMMA4)).toEqual([
      '<|turn>user\nhi<turn|>\n<|turn>model\n',
    ]);
  });

  it('keeps an image part between the surrounding text', () => {
    const image = { type: 'image' as const, src: 'a.png' };
    expect(
      formatPromptParts(
        [{ role: 'user', content: [image, { type: 'text', text: 'what is this' }] }],
        GEMMA4,
      ),
    ).toEqual([
      '<|turn>user\n',
      image,
      `what is this<turn|>\n<|turn>model\n`,
    ]);
  });

  it('merges adjacent text rather than emitting a part per fragment', () => {
    // The runtime concatenates string chunks anyway; fewer parts keeps the array
    // readable when logged, and makes the assertion above stable.
    const parts = formatPromptParts(
      [{ role: 'user', content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] }],
      GEMMA4,
    );
    expect(parts).toEqual(['<|turn>user\nab<turn|>\n<|turn>model\n']);
  });

  it('preserves order across turns and modalities', () => {
    const image = { type: 'image' as const, src: 'a.png' };
    const audio = { type: 'audio' as const, data: new Float32Array(1) };
    const parts = formatPromptParts(
      [
        { role: 'user', content: [image, { type: 'text', text: 'x' }] },
        { role: 'assistant', content: 'ok' },
        { role: 'user', content: [audio] },
      ],
      GEMMA4,
    );
    expect(parts.filter((p) => typeof p !== 'string')).toEqual([image, audio]);
  });

  it('ends by opening the model turn, as formatPrompt does', () => {
    // Without it the model answers repeatedly — the Gemma 4 symptom.
    const parts = formatPromptParts([{ role: 'user', content: 'x' }], GEMMA4);
    const last = parts[parts.length - 1];
    expect(typeof last === 'string' && last.endsWith(GEMMA4.model)).toBe(true);
  });
});

describe('truncateAtStop', () => {
  it('cuts at the stop sequence', () => {
    expect(truncateAtStop('answer<turn|>extra', ['<turn|>'])).toBe('answer');
  });

  it('returns the text unchanged when no stop appears', () => {
    expect(truncateAtStop('answer', ['<turn|>'])).toBe('answer');
  });

  it('cuts at the earliest of several stops', () => {
    expect(truncateAtStop('a<eos>b<turn|>c', ['<turn|>', '<eos>'])).toBe('a');
  });
});

describe('createStopTruncator', () => {
  it('accumulates deltas and stops emitting after the stop sequence', () => {
    // MediaPipe emits deltas, not cumulative text.
    const truncate = createStopTruncator(['<turn|>']);
    expect(truncate('Hel')).toBe('Hel');
    expect(truncate('lo')).toBe('Hello');
    expect(truncate('<turn|>')).toBe('Hello');
    expect(truncate('junk')).toBe('Hello');
  });

  it('handles a stop sequence split across deltas', () => {
    const truncate = createStopTruncator(['<turn|>']);
    truncate('done');
    truncate('<turn');
    expect(truncate('|>')).toBe('done');
  });
});

describe('toParts', () => {
  it('wraps a plain string as a single text part', () => {
    expect(toParts('hi')).toEqual([{ type: 'text', text: 'hi' }]);
  });

  it('passes an array through unchanged', () => {
    const parts = [{ type: 'image' as const, src: 'a.png' }];
    expect(toParts(parts)).toBe(parts);
  });
});

// gemma-4's template throws "Cannot perform operation + on undefined values" on the
// flat {name, arguments} shape parseToolCalls returns, and needs OpenAI's nested one.
// Qwen3.5 and Qwen3-VL accept both, verified by rendering each — so the conversion is
// unconditional and needs no per-family branch.
describe('toTemplateToolCalls', () => {
  const flat = [{ name: 'get_weather', arguments: { city: 'Seoul' } }];

  it('nests a flat call under type/function', () => {
    expect(toTemplateToolCalls([{ role: 'assistant', content: '', tool_calls: flat }])).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { type: 'function', function: { name: 'get_weather', arguments: { city: 'Seoul' } } },
        ],
      },
    ]);
  });

  it('leaves a message without tool calls untouched', () => {
    const messages = [{ role: 'user' as const, content: 'hi' }];
    expect(toTemplateToolCalls(messages)).toEqual(messages);
  });

  it('drops an empty tool_calls array rather than passing it through', () => {
    // A template testing `if message.tool_calls` takes the call branch on [] and
    // renders an empty call block.
    expect(toTemplateToolCalls([{ role: 'assistant', content: 'x', tool_calls: [] }])).toEqual([
      { role: 'assistant', content: 'x' },
    ]);
  });

  it('does not mutate the caller"s messages', () => {
    // The flat shape is the library's public one; a caller may reuse the array.
    const messages = [{ role: 'assistant' as const, content: '', tool_calls: flat }];
    toTemplateToolCalls(messages);
    expect(messages[0]?.tool_calls).toEqual([
      { name: 'get_weather', arguments: { city: 'Seoul' } },
    ]);
  });

  it('converts every call in a multi-call turn', () => {
    const two = [
      { name: 'a', arguments: {} },
      { name: 'b', arguments: { x: 1 } },
    ];
    const out = toTemplateToolCalls([{ role: 'assistant', content: '', tool_calls: two }]);
    expect(out[0]?.tool_calls?.map((c) => c.function.name)).toEqual(['a', 'b']);
  });
});

describe('catalog tool formats', () => {
  it('gives qwen3 the ChatML tool delimiters', () => {
    // Verified against the model's tokenizer_config.json: it emits
    // <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>.
    const entry = CATALOG['qwen3'];
    expect(entry?.toolCallFormat).toEqual({
      delimiters: ['<tool_call>', '</tool_call>'],
      encoding: 'json',
    });
  });

  it('carries no turn format, being a transformers-only entry', () => {
    // turnFormat exists for MediaPipe, which cannot do tools at all. Inventing
    // one here would make the gate think this model is MediaPipe-ready.
    expect(CATALOG['qwen3']?.turnFormat).toBeUndefined();
  });

  it('gives qwen3.5 the xml encoding, not qwen3\'s json', () => {
    // Qwen3.5 kept the <tool_call> delimiters but switched the body to
    // <function=…><parameter=…>. Verified by rendering the real chat template.
    expect(CATALOG['qwen3.5']?.toolCallFormat?.encoding).toBe('xml');
  });

  it('gives gemma-4 its own dialect, not the shared <tool_call> shape', () => {
    // Both delimiters are special tokens (48, 49), which is why this encoding needs
    // raw text. Taken from the export's tokenizer_config.json `response_schema`.
    expect(CATALOG['gemma-4']?.toolCallFormat).toEqual({
      delimiters: ['<|tool_call>', '<tool_call|>'],
      encoding: 'gemma4',
    });
  });

  it('keeps gemma-4"s turn format alongside its tool format', () => {
    // The MediaPipe path needs the turn format and cannot use the tool format; adding
    // the second must not disturb the first.
    expect(CATALOG['gemma-4']?.turnFormat?.end).toBe('<turn|>');
  });

  it('returns the same gemma-4 entry for the ONNX repo and the LiteRT url', () => {
    // Substring matching means one entry serves both runtimes, so finding a
    // toolCallFormat does NOT mean the runtime can use it — MediaPipe has no raw-token
    // hook. A caller must check ref.runtime too.
    const onnx = lookupCatalog({
      runtime: 'transformers',
      repo: 'onnx-community/gemma-4-E2B-it-ONNX',
    });
    const litert = lookupCatalog({
      runtime: 'mediapipe',
      url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task',
    });
    expect(onnx).toBe(litert);
    expect(litert?.toolCallFormat?.encoding).toBe('gemma4');
  });

  it('matches qwen3.5 before qwen3, since one key is a prefix of the other', () => {
    // lookupCatalog takes the FIRST substring match and Object.keys preserves
    // insertion order, so "qwen3" listed first would silently claim every Qwen3.5
    // model and hand it a JSON parser that rejects every call.
    const ref = { runtime: 'transformers' as const, repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX' };
    expect(lookupCatalog(ref)?.toolCallFormat?.encoding).toBe('xml');
  });
});

// Llama-derived templates apply `content | trim`, which fails on an array with
// "Unknown ArrayValue filter: trim". Vision templates demand the opposite — they
// index content[0]['type']. So the shape is per-family, and this is the reverse of
// toParts for the families that want a string.
describe('toText', () => {
  it('passes a plain string through', () => {
    expect(toText('hi')).toBe('hi');
  });

  it('concatenates the text parts and drops the rest', () => {
    expect(
      toText([
        { type: 'audio', data: new Float32Array(1) },
        { type: 'text', text: 'what is this' },
      ]),
    ).toBe('what is this');
  });

  it('joins multiple text parts with a space', () => {
    expect(
      toText([
        { type: 'text', text: 'first' },
        { type: 'text', text: 'second' },
      ]),
    ).toBe('first second');
  });

  it('returns an empty string when no text part is present', () => {
    // The audio itself carries the turn; an empty prompt is valid.
    expect(toText([{ type: 'audio', data: new Float32Array(1) }])).toBe('');
  });
});

describe('collectImages', () => {
  it('gathers image sources across messages in order', () => {
    const images = collectImages([
      { role: 'user', content: [{ type: 'image', src: 'a.png' }, { type: 'text', text: 'x' }] },
      { role: 'assistant', content: 'ok' },
      { role: 'user', content: [{ type: 'image', src: 'b.png' }] },
    ]);
    expect(images).toEqual(['a.png', 'b.png']);
  });

  it('returns an empty array when there are no images', () => {
    expect(collectImages([{ role: 'user', content: 'hi' }])).toEqual([]);
  });
});

describe('collectAudio', () => {
  it('gathers audio parts, not raw samples', () => {
    // The adapter decides whether to decode, so it needs the part.
    const data = new Float32Array([0.5]);
    expect(collectAudio([{ role: 'user', content: [{ type: 'audio', data }] }])).toEqual([
      { type: 'audio', data },
    ]);
  });

  it('gathers Blob and pre-decoded parts together in order', () => {
    const blob = new Blob([new Uint8Array([1])]);
    const data = new Float32Array([0.5]);
    const parts = collectAudio([
      { role: 'user', content: [{ type: 'audio', src: blob }] },
      { role: 'assistant', content: 'ok' },
      { role: 'user', content: [{ type: 'audio', data }] },
    ]);
    expect(parts).toEqual([
      { type: 'audio', src: blob },
      { type: 'audio', data },
    ]);
  });

  it('returns an empty array when there is no audio', () => {
    expect(collectAudio([{ role: 'user', content: 'hi' }])).toEqual([]);
  });
});

describe('hasNonTextParts', () => {
  it('is false for plain string content', () => {
    expect(hasNonTextParts([{ role: 'user', content: 'hi' }])).toBe(false);
  });

  it('is false for text-only parts', () => {
    expect(hasNonTextParts([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }])).toBe(false);
  });

  it('is true when an image is present', () => {
    expect(hasNonTextParts([{ role: 'user', content: [{ type: 'image', src: 'a.png' }] }])).toBe(
      true,
    );
  });
});

describe('formatPrompt with content parts', () => {
  it('concatenates text parts', () => {
    const prompt = formatPrompt(
      [{ role: 'user', content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] }],
      GEMMA4,
    );
    expect(prompt).toBe('<|turn>user\nab<turn|>\n<|turn>model\n');
  });

  it('throws on an image part rather than dropping it', () => {
    // Silently ignoring the image yields an answer that does not reflect it, with
    // no way for the caller to tell why.
    expect(() =>
      formatPrompt([{ role: 'user', content: [{ type: 'image', src: 'a.png' }] }], GEMMA4),
    ).toThrow(/text only/i);
  });
});
