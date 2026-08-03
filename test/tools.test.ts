import { describe, expect, it, vi } from 'vitest';
import { parseToolCalls, type ToolCallFormat } from '../src/tools.js';

// Qwen3's format. Its tags are special:false in added_tokens_decoder, so they
// survive skip_special_tokens: true and reach the parser.
const QWEN: ToolCallFormat = {
  delimiters: ['<tool_call>', '</tool_call>'],
  encoding: 'json',
};

const call = (body: string) => `<tool_call>\n${body}\n</tool_call>`;

describe('parseToolCalls', () => {
  it('reads a single call', () => {
    expect(
      parseToolCalls(call('{"name": "get_weather", "arguments": {"city": "Seoul"}}'), QWEN),
    ).toEqual([{ name: 'get_weather', arguments: { city: 'Seoul' } }]);
  });

  it('reads several calls in order', () => {
    const text = `${call('{"name": "a", "arguments": {}}')}\n${call('{"name": "b", "arguments": {"x": 1}}')}`;
    expect(parseToolCalls(text, QWEN).map((c) => c.name)).toEqual(['a', 'b']);
  });

  it('ignores prose around a call', () => {
    // Models narrate. "I'll look that up." before the call is normal.
    const text = `Sure, let me check.\n${call('{"name": "a", "arguments": {}}')}\nDone.`;
    expect(parseToolCalls(text, QWEN)).toHaveLength(1);
  });

  it('returns nothing for an ordinary answer', () => {
    // Not an error: a plain reply is indistinguishable from "no tool wanted",
    // and the caller treats both the same way.
    expect(parseToolCalls('The weather is fine.', QWEN)).toEqual([]);
  });

  it('drops a malformed body but keeps the valid ones', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const text = `${call('{"name": "a", "arg')}\n${call('{"name": "b", "arguments": {}}')}`;
    expect(parseToolCalls(text, QWEN).map((c) => c.name)).toEqual(['b']);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toMatch(/not valid JSON/);
    warn.mockRestore();
  });

  it('drops a call whose fields have the wrong types', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const text = `${call('{"name": 42, "arguments": {}}')}\n${call('{"name": "b", "arguments": "nope"}')}`;
    expect(parseToolCalls(text, QWEN)).toEqual([]);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });

  it('treats missing arguments as an empty object', () => {
    // Templates render `"arguments": {}` for a no-argument call, but a model
    // asked for a no-argument tool often omits the key entirely.
    expect(parseToolCalls(call('{"name": "now"}'), QWEN)).toEqual([
      { name: 'now', arguments: {} },
    ]);
  });

  it('drops a name that is not in the allowed list', () => {
    // The likely failure at these sizes: a plausible near-miss of the real name.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const text = call('{"name": "get_wether", "arguments": {}}');
    expect(parseToolCalls(text, QWEN, ['get_weather'])).toEqual([]);
    expect(warn.mock.calls[0]?.[0]).toMatch(/get_wether/);
    warn.mockRestore();
  });

  it('accepts any name when no allowed list is given', () => {
    expect(parseToolCalls(call('{"name": "anything", "arguments": {}}'), QWEN)).toHaveLength(1);
  });

  it('ignores an unterminated call', () => {
    // Hitting the token cap mid-call leaves no closing tag.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls('<tool_call>\n{"name": "a"', QWEN)).toEqual([]);
    warn.mockRestore();
  });
});

// Qwen3.5 shares the <tool_call> delimiters but fills them with XML rather than
// JSON. Its tags are still special:false, so unlike gemma-4 nothing is erased by
// skip_special_tokens and the body survives intact — which is why this encoding is
// worth supporting where gemma-4's is not.
const QWEN35: ToolCallFormat = {
  delimiters: ['<tool_call>', '</tool_call>'],
  encoding: 'xml',
};

// Exactly what apply_chat_template renders for Qwen3.5-0.8B-Text.
const xmlCall = (body: string) => `<tool_call>\n${body}\n</tool_call>`;

describe('parseToolCalls with xml encoding', () => {
  it('reads a function name and one parameter', () => {
    const text = xmlCall(
      '<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>',
    );
    expect(parseToolCalls(text, QWEN35)).toEqual([
      { name: 'get_weather', arguments: { city: 'Seoul' } },
    ]);
  });

  it('reads several parameters', () => {
    const text = xmlCall(
      '<function=book>\n<parameter=city>\nSeoul\n</parameter>\n' +
        '<parameter=nights>\n3\n</parameter>\n</function>',
    );
    // Values stay strings: the XML carries no types, so which "3" is a number cannot be
    // known here. gemma-4's dialect answers the same question by marking strings, which
    // is why only that encoding recovers types.
    expect(parseToolCalls(text, QWEN35)).toEqual([
      { name: 'book', arguments: { city: 'Seoul', nights: '3' } },
    ]);
  });

  it('leaves a Python-spelled boolean exactly as written', () => {
    // Measured in a browser: asked for a `boolean`, Qwen3.5 emitted `True`. Pinned because
    // the tempting fix is to coerce it here, and `value === 'true'` on this string is
    // FALSE — a caller that coerces without folding case inverts the argument silently.
    // The parser has no schema and must not guess which spellings mean true.
    const text = xmlCall(
      '<function=get_weather>\n<parameter=celsius>\nTrue\n</parameter>\n</function>',
    );
    expect(parseToolCalls(text, QWEN35)).toEqual([
      { name: 'get_weather', arguments: { celsius: 'True' } },
    ]);
  });

  it('reads a call with no parameters', () => {
    expect(parseToolCalls(xmlCall('<function=now>\n</function>'), QWEN35)).toEqual([
      { name: 'now', arguments: {} },
    ]);
  });

  it('preserves a multi-line parameter value', () => {
    const text = xmlCall(
      '<function=write>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>',
    );
    expect(parseToolCalls(text, QWEN35)[0]?.arguments['body']).toBe('line one\nline two');
  });

  it('drops a body with no function tag', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(xmlCall('just prose'), QWEN35)).toEqual([]);
    expect(warn.mock.calls[0]?.[0]).toMatch(/no <function=/);
    warn.mockRestore();
  });

  it('still honours the allowed list', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const text = xmlCall('<function=get_wether>\n</function>');
    expect(parseToolCalls(text, QWEN35, ['get_weather'])).toEqual([]);
    expect(warn.mock.calls[0]?.[0]).toMatch(/get_wether/);
    warn.mockRestore();
  });

  it('does not try JSON on an xml body', () => {
    // Guards against the encodings being wired to the same branch.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    parseToolCalls(xmlCall('<function=a>\n</function>'), QWEN35);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});

// gemma-4's dialect. Unlike the other two, its delimiters AND its string marker are
// special tokens, so this only works on text decoded with skip_special_tokens: false
// — see trap 12. Every body below is what apply_chat_template actually rendered for
// the corresponding arguments, not a guess at the syntax.
const GEMMA4: ToolCallFormat = {
  delimiters: ['<|tool_call>', '<tool_call|>'],
  encoding: 'gemma4',
};

const Q = '<|"|>';
const g4 = (body: string) => `<|tool_call>${body}<tool_call|>`;

describe('parseToolCalls with the gemma4 encoding', () => {
  it('reads a name and a string argument', () => {
    expect(parseToolCalls(g4(`call:get_weather{city:${Q}Seoul${Q}}`), GEMMA4)).toEqual([
      { name: 'get_weather', arguments: { city: 'Seoul' } },
    ]);
  });

  // The distinction trap 12 said was unrecoverable. It is — from decoded text; the
  // marker survives here, so these two must NOT produce the same value.
  it('keeps a number a number and a quoted number a string', () => {
    expect(parseToolCalls(g4('call:book{nights:3}'), GEMMA4)[0]?.arguments['nights']).toBe(3);
    expect(parseToolCalls(g4(`call:book{nights:${Q}3${Q}}`), GEMMA4)[0]?.arguments['nights']).toBe(
      '3',
    );
  });

  it('reads booleans as booleans', () => {
    expect(parseToolCalls(g4('call:f{ok:true,no:false}'), GEMMA4)[0]?.arguments).toEqual({
      ok: true,
      no: false,
    });
  });

  it('reads a mixed argument list, preserving each type', () => {
    expect(
      parseToolCalls(g4(`call:book{city:${Q}Seoul${Q},nights:3,ok:true}`), GEMMA4),
    ).toEqual([{ name: 'book', arguments: { city: 'Seoul', nights: 3, ok: true } }]);
  });

  it('reads an array with mixed element types', () => {
    expect(parseToolCalls(g4(`call:f{a:[1,${Q}two${Q}]}`), GEMMA4)[0]?.arguments['a']).toEqual([
      1,
      'two',
    ]);
  });

  it('reads a nested object', () => {
    expect(parseToolCalls(g4(`call:f{a:{b:${Q}c${Q}}}`), GEMMA4)[0]?.arguments['a']).toEqual({
      b: 'c',
    });
  });

  it('reads an empty argument list', () => {
    expect(parseToolCalls(g4('call:now{}'), GEMMA4)).toEqual([{ name: 'now', arguments: {} }]);
  });

  it('reads an empty value as null, which is how the template renders it', () => {
    // Measured: {a: null} renders as `call:f{a:}` — an empty value, not a missing key.
    expect(parseToolCalls(g4('call:f{a:}'), GEMMA4)[0]?.arguments).toEqual({ a: null });
  });

  it('keeps punctuation inside a quoted string out of the structure', () => {
    // The reason this is a scanner and not a regex: a string may contain , { } or ].
    const value = '{"city":"Seoul","celsius":21}';
    expect(parseToolCalls(g4(`call:f{raw:${Q}${value}${Q}}`), GEMMA4)[0]?.arguments['raw']).toBe(
      value,
    );
  });

  it('reads several calls in order', () => {
    const text = `${g4('call:a{}')}${g4('call:b{x:1}')}`;
    expect(parseToolCalls(text, GEMMA4).map((c) => c.name)).toEqual(['a', 'b']);
  });

  it('ignores prose before a call', () => {
    expect(parseToolCalls(`Let me check.${g4('call:a{x:1}')}`, GEMMA4)).toHaveLength(1);
  });

  it('drops a body that is not a call:name{…}', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(g4('just prose'), GEMMA4)).toEqual([]);
    expect(warn.mock.calls[0]?.[0]).toMatch(/gemma-4 call:name/);
    warn.mockRestore();
  });

  it('drops a call whose string is never closed', () => {
    // Generation stopped mid-string at the token cap. Half a value is not a value.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(g4(`call:f{a:${Q}unterminated}`), GEMMA4)).toEqual([]);
    warn.mockRestore();
  });

  it('drops a call whose arguments are truncated', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(g4('call:f{a:1,b'), GEMMA4)).toEqual([]);
    warn.mockRestore();
  });

  it('drops an unquoted value that is not a number or a literal', () => {
    // The dialect has no bare-string form, so accepting one would invent a type the
    // model did not ask for — the exact failure this encoding exists to avoid.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(g4('call:f{a:Seoul}'), GEMMA4)).toEqual([]);
    warn.mockRestore();
  });

  it('honours the allowed list', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseToolCalls(g4('call:get_wether{}'), GEMMA4, ['get_weather'])).toEqual([]);
    expect(warn.mock.calls[0]?.[0]).toMatch(/get_wether/);
    warn.mockRestore();
  });

  it('finds nothing once the markers have been stripped', () => {
    // What skip_special_tokens: true leaves behind. Returning [] rather than a
    // wrongly-typed call is the point: this is why the raw path exists.
    expect(parseToolCalls('call:get_weather{city: Seoul}', GEMMA4)).toEqual([]);
  });
});
