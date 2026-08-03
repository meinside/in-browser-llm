# Tool calling — Design

Date: 2026-07-31

> **Outcome.** Shipped, with three corrections to this design.
>
> 1. **One delimiter pair is not one format.** Qwen3.5 keeps the `<tool_call>` tags
>    but writes the body as `<function=…><parameter=…>` XML, so `ToolCallFormat`
>    gained an `encoding` field. A third value, `gemma4`, was added later for a
>    dialect built entirely from special tokens — that one cannot be parsed from the
>    decoded stream and needs `GenerateOptions.onRawText`.
> 2. **The vision path was reported here as hanging. It was not a hang, and it is
>    fixed.** A sign error in transformers.js's mrope delta sent every decode step
>    outside the trained position range, so generation ran silently to the token cap.
>    The adapter corrects it at load time and the round trip now completes on vision
>    models on both browsers. See CLAUDE.md, "The vision-path 'hang' was a sign error
>    upstream".
> 3. The 0.6B/1.7B models this design planned around were dropped for
>    `Qwen3.5-0.8B`.

## Goal

Let a caller give the model tool definitions, get back what it wants called, and
feed the result into the next turn. Scoped to the transformers.js path and the
ChatML tool format, which covers Qwen3 and Qwen2.5.

## What already works

`GenerateOptions.templateArgs` is forwarded verbatim into `apply_chat_template`, and
transformers.js 4.2.0 accepts a `tools` argument there
(`tokenization_utils.js:747` destructures it and passes it to the Jinja render). So
**passing tool definitions needs no library change** — `templateArgs: { tools }`
reaches the template today.

The gap is the other direction: a tool call arrives as text in the middle of a
stream, and nothing parses it.

## Division of labour

`Engine.generate` yields string deltas, and every caller depends on that. Changing
it to typed events to carry tool calls would break them all, so the library instead
exports a pure function the caller runs over the accumulated text.

**The library provides:** a format-driven parser, and a `Message` shape that can
carry a tool call and a tool result back into the template.

**The library does not provide:** tool execution, conversation history, or retries.
The caller accumulates deltas, parses, executes, and appends messages. This library
exists to gate models and absorb runtime traps; a tool-dispatch loop is the
application's job.

Consequences worth stating: parsing happens after generation completes, not
mid-stream, and the streaming contract is untouched.

## Why format-driven

Making the format a parameter costs nothing over hardcoding one — the same code,
with the delimiters read from the catalog instead of inlined — and it keeps a second
format from requiring a rewrite.

It also already pays off: Qwen2.5-0.5B/1.5B and Qwen3-1.7B use the identical ChatML
format, so one parser covers them. If the 0.6B model proves too weak, moving up is a
dropdown entry, not a code change.

gemma-4 has a tool format too, but a different one
(`<|tool_call>call:get_weather{city:<|"|>Seoul<|"|>}<tool_call|>`), and it is
**deliberately out of scope**: `skip_special_tokens: true` in
`src/engine/transformers.ts` erases its framing tokens *and* its string delimiters,
so `{a: 3}` and `{a: "3"}` decode identically. Supporting it would mean disabling
that flag and hand-stripping special tokens for every model — regressing the
verified plain-text path for one 3.2GB Chrome-only model. A format field leaves the
door open without paying that now.

MediaPipe cannot support tools at all: it has no chat template, so there is nowhere
to put the definitions.

### When to admit another format

A new format costs one of two very different amounts, and the difference is worth
deciding in advance rather than under pressure.

**Cheap — add it.** If the delimiters survive decoding, the work is one `encoding`
variant and one catalog entry. `ToolCallFormat.encoding` is a union for exactly this
reason. Do it without ceremony.

**Not possible from here — decline.** If the format depends on tokens marked
`special: true`, as gemma-4's string delimiters are, no parser downstream of
detokenization can recover the data. `skip_special_tokens: true` has already erased
it, and the loss is of type information, not just framing.

This was researched rather than assumed, and the first version of this rule had the
reasoning backwards. Corrected:

- **The upstream tokenizer marks them special, not the ONNX export.** Verified by
  fetching `google/gemma-4-E2B-it/tokenizer.json`: ids 48 `<|tool_call>`, 49
  `<tool_call|>`, and 52 `<|"|>` are all `special: true`. `tokenizer_config.json`
  names 52 `escape_token`.
- **The reference parsers read the raw string, before detokenization.** HuggingFace
  ships `_gemma4_json_to_json()` in `transformers/utils/chat_parsing_utils.py`, and
  vLLM hand-writes a scanner keyed on `STRING_DELIM = '<|"|>'`. Both run
  server-side on undetokenized text, and both recover `{a: 3}` vs `{a: "3"}` *only*
  because the delimiter is still present. We sit downstream of the point where that
  information exists.
- **The format is not converging on OpenAI JSON — the opposite.** Google added a
  `response_template` to the upstream tokenizer config on 2026-07-20, turning the
  custom syntax into a declarative dialect (`string_delims`, `unquoted_keys`), and
  HuggingFace refactored to support it generically. No public roadmap statement
  mentions adopting OpenAI-compatible function JSON.
- **Even upstream, the format is fragile.** Open issues on ollama (#15315) and
  llama.cpp (#25986) show parse failures when arguments contain quotes or newlines —
  and those implementations *do* see the raw text.

So this is a limit of where this library sits in the pipeline, not a cost trade to
revisit when a second case appears. Supporting it would require decoding with
`skip_special_tokens: false` and stripping tokens per model, which puts the working
plain-text path of every model at risk — and would still be reconstructing type
information from a syntax its own ecosystem parses unreliably.

Qwen's side of the bet holds: `Qwen3-8B`, `Qwen3-4B-Instruct-2507`, and
`Qwen3-Coder-30B` all mark their tool tags `special: false`, with no format-change
issues found.

Unknown: whether a future Gemma marks `escape_token` as `special: false`. No public
signal either way. That, not a second custom format, is what would change this.

## Components

### `src/tools.ts` (new)

A new file rather than an addition to `prompt.ts`, which already owns templates,
truncation, and content parts.

```ts
export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallFormat {
  /** Wraps each call, e.g. ['<tool_call>', '</tool_call>']. */
  delimiters: [string, string];
  /** Body encoding. Only 'json' exists today; gemma-4 would need another. */
  encoding: 'json';
}

export function parseToolCalls(
  text: string,
  format: ToolCallFormat,
  allowed?: string[],
): ToolCall[];
```

**It never throws.** A 0.6B model is expected to emit malformed JSON and invent
function names, so every failure mode drops the offending call and continues:

| Input | Result |
|---|---|
| no delimiters | `[]` — indistinguishable from a plain answer, which is correct |
| body is not valid JSON | that call dropped, others kept |
| `name` not a string, or `arguments` not an object | dropped |
| `name` absent from `allowed` (when supplied) | dropped |
| prose around the call | ignored; only delimited spans are read |

**Every drop logs a `console.warn`**, prefixed `[in-browser-llm]` like the library's
other warnings, with three distinguishable messages: invalid JSON, wrong field
types, and name-not-allowed. Silent dropping would make "the model called no tool"
and "the model called a tool and we discarded it" look identical, and the second is
the likely case at this model size. Offending text is truncated in the message, since
a model can emit a lot of garbage.

These warnings also make the acceptance threshold measurable: they separate a parse
failure from a hallucinated name, which is what decides whether moving to a larger
model would help.

### `src/prompt.ts`

Widen `Message`:

```ts
export interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string | ContentPart[];
  /** Assistant turns only: replayed to the template so the model sees its own call. */
  tool_calls?: ToolCall[];
}
```

`tool_calls` is a flat `ToolCall[]`, matching the parser's output. Verified against
the Qwen3 template: it does `{%- if tool_call.function %}{%- set tool_call =
tool_call.function %}{%- endif %}`, so it accepts both the flat shape and OpenAI's
nested `{function: {...}}`. The flat one needs no repacking.

A tool result is `{role: 'tool', content: string}`; the template wraps it in
`<tool_response>` itself.

### `src/catalog.ts`

Add `toolCallFormat?: ToolCallFormat`, and make `turnFormat` and `stopSequences`
optional — a transformers-only entry has no turn delimiters, and requiring them
would mean inventing unused values.

```ts
qwen3: {
  toolCallFormat: {
    delimiters: ['<tool_call>', '</tool_call>'],
    encoding: 'json',
  },
},
```

`src/engine/mediapipe.ts` already requires `turnFormat` separately from the entry —
its comment anticipates exactly this case — so that guard is unaffected. But
`src/gate.ts` currently blocks a MediaPipe model on `hasCatalogEntry`, which would
now pass for an entry carrying only a tool format. The gate must check for a turn
format specifically. `ModelSpec` gains `hasTurnFormat: boolean`, set by
`inspectModel()`, replacing `hasCatalogEntry` in that check.

## Testing

The parser is pure, and its failure paths are the substance of it, so
`test/tools.test.ts` covers them properly: single call, multiple calls, no arguments,
no delimiters, malformed JSON among valid calls, wrong field types, disallowed name,
and prose surrounding a call. `console.warn` is asserted with `vi.spyOn` — that a
call was dropped is part of the contract, not a side effect.

Gate tests gain a case: a catalog entry with a tool format but no turn format must
still block on the MediaPipe path.

### The example

A dedicated section, separate from the single-turn chat box, driven by one button
that runs the three-turn cycle:

1. Send "what's the weather in Seoul?" with a `get_weather` definition.
2. Parse the reply, display the tool call, and return a **hardcoded** result. No real
   API: a network dependency would make the demo flaky, and the tool path is what is
   being demonstrated.
3. Append the `tool` message and send again for the final answer.

Each step shows its raw text, because when this fails what the model actually emitted
is the whole diagnosis.

### Browser verification

Scenario 16: the same prompt five times, passing if **at least three** produce a
correct function name and parsable JSON. Below that, retry with `Qwen3-1.7B` — same
ChatML format, so only a dropdown entry.

### What tests cannot show

Unit tests prove the parser handles what it is given. They say nothing about whether
a 0.6B model emits usable calls: no benchmark exists at this size, and Qwen's own
model card recommends an external parser, implying raw output needs cleanup. Expect
hallucinated names. That is the risk this feature carries, and only the browser run
settles it.
