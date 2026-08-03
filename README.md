# in-browser-llm

Decide before downloading whether a small LLM will run in the current browser,
then run it on WebGPU.

Inference is done by transformers.js or MediaPipe. This library contributes the
capability gate and absorbs per-runtime traps.

- [Quick start](#quick-start)
- [Verified models](#verified-models)
- [Why the gate matters](#why-the-gate-matters)
- [API](#api)
- Guides — [token cap](#token-cap) · [cache status](#knowing-whether-the-weights-came-from-cache) ·
  [tool calling](#tool-calling) · [sampling](#sampling) ·
  [reasoning models](#reasoning-models) · [images](#images) · [audio](#audio) ·
  [MediaPipe modalities](#mediapipe-modalities) ·
  [picking a model](#picking-a-model)
- [Browser support](#browser-support-measured-macos-apple-silicon)
- [Development](#development) · [Releasing](#releasing)

Manual verification results live in [`docs/verification-log.md`](docs/verification-log.md).

## Quick start

A complete page. Three steps: gate, load, generate.

```html
<!doctype html>
<html lang="en">
  <body>
    <button id="go">Run</button>
    <pre id="out"></pre>

    <!-- Required. The adapters import their runtime by bare specifier, which a
         browser cannot resolve on its own. Include only the runtimes you use. -->
    <script type="importmap">
      {
        "imports": {
          "@huggingface/transformers": "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm",
          "@mediapipe/tasks-genai": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.29/+esm"
        }
      }
    </script>

    <script type="module">
      import { probe, inspectModel, gate, load } from
        'https://cdn.jsdelivr.net/gh/meinside/in-browser-llm@v0.0.1/dist/index.js';

      const out = document.querySelector('#out');
      // Example model, ~450MB at q4f16. Any ONNX repo that transformers.js
      // supports works here.
      const ref = { runtime: 'transformers', repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX' };

      document.querySelector('#go').addEventListener('click', async () => {
        // 1. Refuse before downloading if this browser cannot run the model.
        const caps = await probe();
        const verdict = gate(caps, await inspectModel(ref));
        if (!verdict.ok) {
          out.textContent = verdict.blockers.join('\n');
          return;
        }

        // 2. Load. Multi-GB, so always show progress.
        const engine = await load(ref, {
          capabilities: caps,
          onProgress: ({ ratio }) => {
            out.textContent = `Loading ${(ratio * 100).toFixed(0)}%`;
          },
        });

        // 3. Generate. Values are deltas, so append rather than replace.
        out.textContent = '';
        for await (const delta of engine.generate([{ role: 'user', content: 'Hello!' }])) {
          out.textContent += delta;
        }
      });
    </script>
  </body>
</html>
```

For MediaPipe instead, swap the ref — nothing else changes:

```js
const ref = {
  runtime: 'mediapipe',
  url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task',
};
```

Three things are easy to get wrong:

- **Import from jsDelivr, not `raw.githubusercontent.com`** — the latter sends
  `Content-Type: text/plain`, which browsers refuse to execute as a module. Serving
  the page over `file://` fails too, since ES modules need an HTTP origin.
- **Append the deltas.** `generate()` yields deltas, not cumulative text;
  assigning with `=` leaves only the final fragment.
- **Pass `capabilities`.** Without it, dtype defaults to `q4f16` unconditionally,
  which fails on browsers with a known f16 shader bug.

Skipping the gate and calling `load()` alone does work — and gives up the reason
to use this library. That path is exactly how a 3GB download ends in an
unexplained error.

Neither runtime is bundled, so supply the one you use — via the import map above
for a plain page, or as a dependency when building with a bundler:

    npm install @huggingface/transformers@4.2.0   # or
    npm install @mediapipe/tasks-genai@0.10.29

### Serving `dist/` yourself

jsDelivr is only static hosting, so serving the build from your own origin works
identically — the import map is still needed either way, since it resolves the peer
runtime rather than this library.

**Copy the whole `dist/` directory, not just `index.js`.** The build is code-split:
`index.js` imports shared chunks statically and each engine adapter with a dynamic
`import()`, all by relative path with hashed filenames. Uploading `index.js` alone
gives a page that loads, probes, and gates successfully, then 404s inside `load()` —
the failure lands after the point this library exists to move it before.

## Verified models

Every model below was run in a browser, not merely gated. Nothing here is a
restriction of the library — any ONNX repo transformers.js supports will load, and
these are the ones whose behaviour is known.

| ref | size | text | image | audio | tools | Chrome | Firefox |
|---|---|---|---|---|---|---|---|
| `onnx-community/Qwen3.5-0.8B-Text-ONNX` | ~450MB | ✅ | — | — | `xml` | ✅ | ✅ |
| `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | ~1.3GB | ✅ | ✅ | — | `json` | ✅ | ✅ |
| `onnx-community/gemma-4-E2B-it-ONNX` | ~3.2GB | ✅ | ✅ | ✅ | `gemma4` | ✅ | blocked |
| `…/gemma-4-E2B-it-web.task` (MediaPipe) | ~2.0GB | ✅ | — | — | n/a¹ | ✅ | ✅ |

```js
const ref = { runtime: 'transformers', repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX' };
const ref = { runtime: 'mediapipe', url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task' };
```

Reading the table:

- **Start with `Qwen3.5-0.8B-Text`** for text and tools. It is the default in
  `examples/`, downloads fastest, and is the only one small enough that a first
  try costs seconds rather than minutes.
- **`Qwen3-VL-2B` is the vision model to reach for**, because it is the only one
  that does images on *both* browsers.
- **gemma-4 is blocked on Firefox** — one of its tensors is 1120MiB against a
  1024MiB `maxBufferSize`, so the gate refuses it before any download. That is a
  property of this model, not of Firefox or of image input.
- **The MediaPipe entry is text-only in practice.** The library's image and audio
  options work, but no published web container carries the weights —
  see [that section](#mediapipe-modalities).
- **`tools` names the parser**, which comes from the catalog by substring match on
  the ref. The `gemma4` encoding needs `onRawText` rather than the streamed deltas.

¹ `lookupCatalog` *does* return the `gemma4` format for that URL, since matching is
by substring and the entry is shared with the ONNX build — but MediaPipe has no chat
template to render a tool declaration into and no raw-token hook, so the format is
unusable there. Check `ref.runtime === 'transformers'` before offering the feature.

Before relying on a model absent from this table, run `node
scripts/check-model.mjs <hf-repo-id>` — it fetches only headers and configs. What
that cannot tell you is in [Picking a model](#picking-a-model), which is a
different question from whether it loads.

## Why the gate matters

A model's *total* size is not what fails first — the **largest single tensor** is,
because it must fit in one allocation, capped by the adapter's `maxBufferSize`.
That limit is reported in tiers — 1GiB on Firefox/macOS, and the spec permits as
little as 256MiB, which mobile devices are likely to report — while
large-vocabulary embedding tensors can exceed 1GiB on their own.

A second limit, `maxStorageBufferBindingSize`, caps how much of a buffer one shader
binding may address. ONNX Runtime splits an oversized tensor across several
bindings, so exceeding that limit alone is survivable and this library only warns.
`maxBufferSize` is the one nothing works around.

There is a second, unrelated ceiling. A model whose weights are **embedded in the
`.onnx`** rather than sidecar `.onnx_data` files must be allocated contiguously in
ONNX Runtime's 4GiB wasm heap and duplicated while its protobuf is parsed. A
1360MiB file failed there with `std::bad_alloc` on Firefox, while 949MiB of sidecar
data loaded without trouble — so the discriminator is layout, not total size. The
gate refuses at the documented 2GB protobuf limit and warns above 1GiB, which is a
measurement rather than a documented figure.

When it happens, the runtime error names no cause. One observed example, with
`gemma-4-E2B-it` on Firefox:

    Failed to load external data file "embed_tokens_q4f16.onnx_data",
    error: Unknown error occurred in memory copy.

That was a 1120MiB tensor against a 1024MiB limit, discovered only after
downloading multiple gigabytes. This library reads the ONNX headers with Range
requests — a few MB at most — and refuses before the weight download starts.

## API

Four functions carry the main path; the rest of the exports are for building your
own variant of it.

| | |
|---|---|
| `probe()` | adapter limits and features. Async, no arguments |
| `inspectModel(ref)` | `ModelSpec` from ONNX headers via Range requests |
| `gate(caps, spec, opts?)` | **pure** — `Verdict` with no I/O |
| `load(ref, opts?)` | an `Engine`; imports only the runtime the ref names |

### Reading a verdict

`gate()` returns three fields, and **`warnings` is not optional to read.** It
carries every problem that is real but not certain — the wasm-heap size, missing
`subgroups`, a modality the ref cannot confirm — none of which appears in
`blockers`:

```js
const verdict = gate(caps, spec, { wantsImage: true });
verdict.warnings.forEach((w) => console.warn(w));
if (!verdict.ok) {
  show(verdict.blockers.join('\n'));
  return;
}
```

`ok` is false only for blockers, and the gap between the two lists is wider than it
looks: the wasm-heap check emits **"Loading this model is expected to FAIL"** as a
*warning*, because the real ceiling moves with the browser and with memory free at
the time. A verdict can be `ok: true` and still say that. Dropping warnings on the
floor throws away the more useful half of the answer.

`wantsImage` and `wantsAudio` move a modality check before the download. They also
land in different lists depending on the runtime — a blocker for ONNX, where
`config.json` settles it, but a warning for MediaPipe, whose `.task` bundle carries
no config to read.

### Errors after the gate

The gate answers "will this download be wasted", not "will everything work". Load
and generation throw, typed by cause:

| type | thrown when | carries |
|---|---|---|
| `CapabilityError` | a feature is genuinely absent — no catalog turn format, a MediaPipe modality not enabled at load | — |
| `DownloadError` | non-OK HTTP from the weight fetch | `status`, `resumable` |
| `RuntimeError` | everything else, including the peer runtime being missing | `cause` |

```js
import { DownloadError } from '…/dist/index.js';

try {
  engine = await load(ref, { capabilities: caps, spec });
} catch (e) {
  if (e instanceof DownloadError && e.resumable) retry();
  else show(e.message);   // already written for a human
}
```

The buffer-limit case is deliberately **not** an error type: `gate()` decides it
before downloading, so a caller who gates never reaches a throw for it.

### The rest

`formatPrompt` / `formatPromptParts` / `truncateAtStop` / `createStopTruncator`
build a prompt and cut it at a stop sequence — what the MediaPipe adapter uses
internally, exported for writing another adapter. `parseOnnxHeader` is the
protobuf parser under `inspectModel`, `CATALOG` is the raw correction table behind
`lookupCatalog`, and `createCacheStore` / `requestPersistence` / `fetchWithProgress`
/ `createMetrics` are the loader's parts. `resolveBaseUrl` turns a `ModelRef` into
a URL. None is needed for the path in Quick start.

## Guides

### Token cap

`generate()` accepts `maxTokens` and falls back to `DEFAULT_MAX_TOKENS` (1024).

That default is **a decode-speed budget, not a model capability.** At the ~10 tok/s
measured for gemma-4 on Chrome, 1024 tokens is about 100 seconds of waiting, and
Firefox without `subgroups` is several times slower again. The models here allow far
more — 32K to 128K context — but the wait, not the model, is what binds in a
browser. Pass a larger value when you need longer answers:

```js
engine.generate(messages, { maxTokens: 4096 });
```

Output that stops mid-sentence usually means the cap was reached, not a defect.

The runtimes count differently, and the option is normalized over that rather
than hiding it: transformers.js caps **output** tokens, while MediaPipe caps
**input + output**, so a long prompt reaches the cap sooner on MediaPipe. Note
MediaPipe fixes the cap at load time and cannot change it afterwards, so pass it
to `load()` for that runtime:

```js
await load(ref, { maxTokens: 4096 });
```

### Knowing whether the weights came from cache

```js
await load(ref, { onCacheStatus: (fromCache) => console.log({ fromCache }) });
```

**MediaPipe only.** That path downloads through this library's loader, so the
answer is known. transformers.js caches too but manages its own storage, so the
callback never fires there — a fast load that emits no progress events is the only
available hint.

### Tool calling

Pass tool definitions through `templateArgs`, then parse the reply. **transformers.js
only** — MediaPipe has no chat template to render the tool block into.

```js
// Same jsDelivr URL as Quick start — this package is not on npm.
import { parseToolCalls, lookupCatalog } from '…/dist/index.js';

const tools = [{ type: 'function', function: { name: 'get_weather', /* … */ } }];
let text = '';
for await (const delta of engine.generate(messages, { templateArgs: { tools } })) {
  text += delta;
}

const format = lookupCatalog(ref)?.toolCallFormat;
const calls = format ? parseToolCalls(text, format, ['get_weather']) : [];
```

`parseToolCalls` never throws. A malformed body or a name outside the third argument
is dropped with a `console.warn` naming the reason, so an empty result means either
"no tool wanted" or "everything was rejected" — treat both as an ordinary answer and
read the console to tell them apart. Passing the allowed names matters: inventing a
near-miss of the real name is a common failure at these sizes, and a small model may
answer from memory instead of calling the tool at all.

Feed a result back with the `tool` role, replaying the model's own request:

```js
await engine.generate([
  ...messages,
  { role: 'assistant', content: '', tool_calls: [calls[0]] },
  { role: 'tool', content: JSON.stringify({ celsius: 21 }) },
], { templateArgs: { tools } });
```

The library does not execute tools, keep conversation history, or retry.

Formats live in the catalog per model, because the same reply shape does not follow
from the same delimiters:

| encoding | shape | catalog keys | types |
|---|---|---|---|
| `json` | `{"name": …, "arguments": {…}}` | `qwen3` | preserved |
| `xml` | `<function=name><parameter=key>value</parameter></function>` | `qwen3.5` | all strings |
| `gemma4` | `call:name{city:<\|"\|>Seoul<\|"\|>,n:3}` | `gemma-4` | preserved |

Qwen2.5 and Granite 4.0 emit the same JSON shape but have no entry, because matching
is by substring against the repo id and neither name contains `qwen3`. Adding one is
a catalog key, not a parser. Note `qwen3.5` must stay listed **before** `qwen3`:
`lookupCatalog` takes the first match, so the reverse order hands every Qwen3.5 model
a JSON parser that discards its every call.

XML values arrive as strings, because the markup carries no types. The `gemma4`
dialect does carry them — it wraps only strings in `<|"|>` — so `{n: 3}` and
`{n: "3"}` stay distinct.

**When coercing xml values, fold case.** Asked for a `boolean`, Qwen3.5 returned `True`
— Python's spelling, measured — so `value === 'true'` is false for a value the model
meant as true. `parseXmlBody` deliberately does not coerce: it has no schema, and
guessing which spellings mean true is the same class of invention it refuses elsewhere.

#### gemma-4 needs the raw text, not the stream

Its delimiters *and* its `<|"|>` string marker are **special tokens**, so
`skip_special_tokens: true` erases them from the streamed deltas. Parsing those finds
nothing at all, and would have found the wrong argument types if it found anything.
Pass `onRawText` and parse what it gives you:

```js
let raw = '';
for await (const delta of engine.generate(messages, {
  templateArgs: { tools },
  onRawText: (text) => { raw = text; },   // decoded keeping special tokens
})) {
  process.stdout.write(delta);            // display still uses the clean deltas
}
const calls = parseToolCalls(raw, format, ['get_weather']);
```

It fires once when generation ends, since half a tool call cannot be parsed. Harmless
to pass for the other two encodings, whose `<tool_call>` tags are `special: false` and
therefore survive in the ordinary stream.

Two further differences worth knowing, both measured:

- **The catalog entry is shared with the MediaPipe build.** `lookupCatalog` matches
  `gemma-4` against the LiteRT URL too, so a format comes back for a runtime that
  cannot use it — there is no chat template to render the declaration into and no
  raw-token hook. Check `ref.runtime === 'transformers'` before offering the feature.
- **gemma-4 puts the tool result inside the model's turn and leaves that turn open**,
  where Qwen closes the turn and opens a new one. So `add_generation_prompt` is a no-op
  for that third turn — it declines to reopen a turn after a tool response. The library
  handles this by rendering the model's own template; it matters only if you build
  prompts by hand.

  This is also why an assistant turn's `content` must stay `''` rather than becoming an
  empty text part: the template tests that value for *truthiness*, and a one-element
  array is truthy. Wrapping it closed the turn, and the third turn then returned zero
  chunks with no error, because a complete conversation is answered with EOS.

`tool_calls` is accepted in the flat `{name, arguments}` form that `parseToolCalls`
returns, and converted to OpenAI's nested form for the template. gemma-4 requires the
nested one and fails on the flat one with a Jinja error naming nothing useful; Qwen
accepts either.

### Sampling

By default nothing is overridden, so each model's own `generation_config.json`
applies — usually what you want, since the model authors tuned it. To change it,
pass `overrides`, which is merged verbatim into the generation call:

```js
engine.generate(messages, { overrides: { do_sample: false } });   // deterministic
engine.generate(messages, { overrides: { temperature: 0.3 } });
```

**Check that your model specifies sampling at all.** transformers.js defaults
`do_sample: false`, so a `generation_config.json` carrying only `eos_token_id` decodes
greedily — and a small model decoding greedily tends to repeat one phrase until the
token cap. That is not the model being weak; it is a missing default. `load()` warns
when it sees this, and the fix is to pass the values the source model's card
recommends:

```js
engine.generate(messages, {
  overrides: { do_sample: true, temperature: 1.0, top_p: 1.0, top_k: 20 },
});
```

`Qwen3.5-0.8B-Text-ONNX` is one such export, and `examples/main.js` shows the values
used for it. Note transformers.js has no `presence_penalty`; use
`repetition_penalty` or `no_repeat_ngram_size` if you need that kind of control.

Only transformers.js honours `overrides`; MediaPipe fixes its sampling parameters
at load time.

### Reasoning models

Thinking mode is disabled by default (`enable_thinking: false` is passed to the
chat template), because Qwen3 and similar models otherwise spend most of the token
budget reasoning. Re-enable it, or pass any other template argument, with
`templateArgs`:

```js
engine.generate(messages, { templateArgs: { enable_thinking: true } });
```

**The reasoning is not wrapped in a `<think>` block — it arrives first, bare, followed
by `</think>` and then the answer.** The template writes the *opening* tag into the
prompt rather than leaving the model to emit it (Qwen3.5's does `{{- '<think>\n' }}`
right after `add_generation_prompt`), and the streamed deltas skip the prompt. So
generation begins already inside the block and only the closing tag crosses the stream.
Turning the flag off does not remove the tag either: that branch pre-fills an empty
`<think>\n\n</think>` pair in the prompt, which is *how* it suppresses reasoning — so
nothing is streamed and every delta is answer.

`</think>` itself is **not** a special token (id 248069, `special: false` in
`tokenizer.json`), so it survives `skip_special_tokens: true` and is yours to split on:

```js
let reasoning = '', answer = '', done = false;
for await (const delta of engine.generate(messages, {
  templateArgs: { enable_thinking: true },
})) {
  if (done) { answer += delta; continue; }
  // Accumulate: the streamer flushes on whitespace, so '</' and 'think>' can arrive
  // in separate deltas. See examples/main.js for the buffering version.
  reasoning += delta;
  const at = reasoning.indexOf('</think>');
  if (at !== -1) {
    answer = reasoning.slice(at + 8);
    reasoning = reasoning.slice(0, at);
    done = true;
  }
}
```

Measured on `Qwen3.5-0.8B-Text-ONNX`. Split only when you asked for thinking: with the
flag off no closing tag ever arrives, so a splitter waiting for one would file the whole
answer as reasoning. And **a reasoning block with no answer after it means the token cap
landed inside the block** — raise `maxTokens` rather than reading it as an empty reply.

#### gemma-4 needs the raw text for this too, and the split cannot be live

Its dialect is different and so is the answer. gemma-4 wraps reasoning in
`<|channel>thought … <channel|>` and puts `<|think|>` in a synthesized *system* turn
rather than at the end of the prompt:

```
enable_thinking: true  → <bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n
enable_thinking: false → <bos><|turn>user\nhi<turn|>\n<|turn>model\n
```

But `<|channel>` and `<channel|>` are **`special: true`** (ids 100 and 101), so
`skip_special_tokens: true` erases both and the reasoning arrives silently merged into
the answer with no boundary left:

```
skip_special_tokens: true  → "thought\nlet me think\nthe answer"
skip_special_tokens: false → "<|channel>thought\nlet me think\n<channel|>the answer"
```

This is the same shape as the tool-call problem — a format built from special tokens
cannot be read from the decoded stream — so the same escape applies: `onRawText` gives you
the second form. **But it fires once at completion, so this split cannot be live**; the
merged text streams as it arrives and is redistributed at the end. That is a real UX
difference from Qwen3.5, not a shortcut.

The redistribution needs no regex, because **only the delimiters are special**: the
reasoning body appears in the clean stream too, so deleting the delimiters from raw
reproduces clean exactly, making the summed body length the offset where the answer starts.
The answer is `clean.slice(consumed)` and nothing has to strip EOS or turn framing out of
the raw form. `examples/main.js` implements this in `splitRawChannels`; a block left
unterminated by the token cap yields all reasoning and no answer, which the page reports
rather than showing as empty.

**The two families are opposites, so do not carry a `</think>` assumption across them.**
The example keeps a per-model table naming which *mechanism* applies rather than a bare
delimiter, and a model absent from it gets no split. Note gemma-4's template ships a
`strip_thinking` macro for replaying history — upstream expects callers to handle the split
themselves.

### Images

Two things differ from the text path, and both are easy to miss: `gate()` needs to
know a modality is coming, and `load()` needs the spec. The whole path:

```js
const ref = { runtime: 'transformers', repo: 'onnx-community/gemma-4-E2B-it-ONNX' };

const caps = await probe();
const spec = await inspectModel(ref);

// wantsImage refuses a text-only model here rather than at generate time.
const verdict = gate(caps, spec, { wantsImage: true });
verdict.warnings.forEach((w) => console.warn(w));
if (!verdict.ok) return show(verdict.blockers.join('\n'));

// spec picks the model class — a vision model needs a processor, which a
// tokenizer cannot substitute for. Omit it and the adapter assumes text-only.
const engine = await load(ref, { capabilities: caps, spec });

for await (const delta of engine.generate([
  {
    role: 'user',
    content: [
      { type: 'image', src: fileFromAnInput },
      { type: 'text', text: 'What is in this picture?' },
    ],
  },
])) {
  out.textContent += delta;
}
```

`src` takes a `Blob` — a `File` straight from an `<input type="file">` — or a URL
string, so an object URL or an `https:` URL works equally. A `Blob` avoids having to
revoke anything afterwards.

Images are gathered in the order they appear and handed to the processor separately
from the rendered text, so the model's chat template decides where the image token
lands — not the position of the part in the array.

Only models whose `config.json` declares `vision_config` accept images. Two are
verified: `Qwen3-VL-2B` (~1.3GB), which runs on **both** browsers, and
`gemma-4-E2B` (~3.2GB), which is **Chrome-only** — Firefox refuses it on the
tensor limit, a property of that model rather than of image input. Sending an image
to a model without a vision encoder fails with a named reason rather than an answer
that quietly ignores it.

A smaller vision model would be preferable, but the one tried was rejected on three
defects that metadata cannot reveal: a quantized vision encoder can be a malformed
graph that loads anyway and emits meaningless embeddings, a processor may throw when
called without an image, and a vision chat template may reject plain-string content.
Verify any replacement in a browser rather than trusting its size.

Video is not possible at all: transformers.js has no processor that accepts it.

### Audio

Same shape as [Images](#images), including `gate()` needing to be told and `load()`
needing the spec — pass `{ wantsAudio: true }` in place of `wantsImage`. Attach it
either as a `Blob` for the library to decode or as pre-decoded 16kHz mono samples:

```js
await engine.generate([
  {
    role: 'user',
    content: [
      { type: 'audio', src: fileFromAnInput },       // library decodes
      { type: 'text', text: 'What is said here?' },
    ],
  },
]);

// Or, if you decoded it yourself — necessary inside a worker, where AudioContext
// does not exist:
// { type: 'audio', data: float32MonoAt16kHz }
```

Only models whose `config.json` declares `audio_config` accept audio — currently
`gemma-4-E2B`, which is Chrome-only.

**An image and audio in the same turn works**, and needs both flags at the gate:

```js
const verdict = gate(caps, spec, { wantsImage: true, wantsAudio: true });

await engine.generate([{ role: 'user', content: [
  { type: 'image', src: photo },
  { type: 'audio', src: clip },
  { type: 'text', text: 'Does the recording describe this picture?' },
]}]);
```

Verified on gemma-4. Each modality is merged under its own token id, and a count
mismatch throws rather than passing silently — so a turn that reads as though one
modality was ignored is more likely sampling variance than a dropped input. Re-run
with `overrides: { do_sample: false }` before suspecting otherwise.

### MediaPipe modalities

**Supported by the library, blocked by the weights.**

The options exist and are load-time only, since MediaPipe deletes the model asset
once consumed:

```js
const engine = await load(ref, { maxImages: 1, supportAudio: true });
```

Content parts then work as they do on transformers.js, and sending an image without
`maxImages` fails with a named reason rather than a count mismatch against zero.

**But no published gemma-4 web container has vision or audio weights**, so this
currently cannot work in a browser. Both `gemma-4-E2B-it-web.litertlm` and `-web.task`
declare `model_type: tf_lite_artisan_text_decoder`; the builds that do carry
`tf_lite_vision_encoder` declare `backend_constraint: "cpu"` and are not browser
targets.

MediaPipe accepts the options, loads, and then fails at generate:
`Llm{Vision,Audio}InferenceCalculator` reports "Image models could not be created" or
"Audio models could not be created". Neither message mentions the weights, so the
library rewrites both into one error that does. A load-time variant ("Audio options
should not be null when loading audio runner") was seen once and is matched too, though
it has not been reproducible — see [the verification log](docs/verification-log.md).

**Use transformers.js for image or audio input.** Note also that `gate()` cannot
pre-empt this: a LiteRT container exposes no config, so `wantsImage` yields a warning
rather than a verdict.

Two limits are worth knowing, both of which the library warns about rather than
working around:

- The model **ignores everything past 30 seconds** (`audio_seq_length` 750 ×
  `audio_ms_per_token` 40ms). It does not error, so the answer would silently omit
  the tail.
- Supplying **more than one audio part** feature-extracts only the first; the rest
  affect token counts alone.

### Picking a model

Small does not mean suitable, and packaging fitness is a different question from
usefulness. A survey that only checks whether a model loads will recommend the wrong
one. `Qwen3.5-0.8B-Text` (~450MB) is the example here; four checks decided it, and
they are the ones worth repeating for any replacement:

- **Tokenizer efficiency in the language you use.** One non-Latin-script sentence
  costs 16 tokens here against 35 on `granite-4.0-h-350m`, 21 of those being broken
  UTF-8 fragments — it splits a single three-character word into a character plus two
  partial bytes plus another character. Fragments waste context and decode time in
  proportion. Measure it in your own language rather than assuming: tokenize a real
  sentence and count the ids whose `decode` yields U+FFFD.
- **Benchmark numbers, not the card's language list.** IBM lists broad multilingual
  support for Granite 4.0 while its own multilingual benchmarks in the same card sit
  at chance level for the 350M — MMMLU 27.95 on four-way multiple choice.
- **Weights in sidecar `.onnx_data`**, which sidesteps the wasm-heap ceiling that
  stops larger single-file exports regardless of total size.
- **It actually calls tools.** A published score is not proof: a model tried here
  answered from memory while ignoring a tool block confirmed present in the rendered
  prompt. Verify in a browser.
- **Its `generation_config.json` specifies sampling.** An export that omits it decodes
  greedily under transformers.js and repeats itself, which is easy to misread as the
  model being too small. `check-model.mjs` reports this.

A model built as a fine-tuning base rather than for chat will ramble on open-ended
prompts no matter how cleanly it loads — that is the model, not the runtime.

## Browser support (measured, macOS Apple Silicon)

These are **example figures** from one session with `gemma-4-E2B-it`, kept to convey
rough scale. They are not tracked and will drift as models and browsers change —
run `probe()` and the example page for current numbers rather than trusting these.

| | MediaPipe | transformers.js |
|---|---|---|
| Chrome 150 | ~14 tok/s | ~10 tok/s |
| Firefox 153 | ~3 tok/s | works, model permitting |

The durable part is the *reason*, not the numbers: Firefox lacks the `subgroups`
feature, so decode is several times slower. It is not a defect.

Whether transformers.js runs on Firefox depends on the model, not the runtime.
`Qwen3.5-0.8B` and `Qwen3-VL-2B` work there. `gemma-4-E2B` does not: one of its tensors is larger
than Firefox's buffer limit and no fallback exists, so the gate refuses it up
front.

Separately, on browsers with a known f16 shader bug — currently Firefox ≤153 —
every model is loaded as `q4` rather than `q4f16`, because some exports contain
f16 guard nodes that fail to compile there. This is decided per browser, not per
model, since no capability query predicts it.

## Development

    npm install
    npm test          # pure functions only: gate, onnx-parse, prompt, loader, inspect, transformers, mediapipe, tools, audio
    npm run build     # produces dist/

`check-model.mjs`, referenced under [Verified models](#verified-models), reports
what would otherwise fail silently — an unrecognized session layout that disables
the tensor gate, a missing quantized variant that means an fp32 download, f16
shader guards that break Firefox, and sampling defaults worth leaving alone. A
second script prints a model's turn delimiters, which the MediaPipe path needs in
the catalog:

    node scripts/inspect-chat-template.mjs <hf-repo-id>

Real inference cannot be tested automatically — headless browsers expose no
WebGPU adapter and CI cannot fetch multi-GB weights. Verify by hand:

    npm run build
    npm run example      # or any static server over examples/

## Manual verification log

Pass/fail per scenario, with what each one settled, in
[`docs/verification-log.md`](docs/verification-log.md). Read it before changing
`transformers.ts`, the catalog, or anything touching modalities — several entries
record a conclusion that was wrong twice before it was right, and the wrong
readings are kept deliberately.

## Releasing

`dist/` is gitignored, so `master` never carries build output. The Action builds it
and attaches the release tag.

Bump the version in **two** places — `package.json` and the import URL in
[Quick start](#quick-start) — then tag:

    npm version 0.0.2 --no-git-tag-version   # updates package-lock.json too
    # edit the jsDelivr URL in this README's Quick start example to match
    git commit -am 'release: v0.0.2'
    git tag src-v0.0.2
    git push origin master src-v0.0.2

`--no-git-tag-version` is a safety flag, not a convenience: without it `npm version`
tags the source commit `v0.0.2`, and a later `git push --tags` would publish that name
with no `dist/` in it. Never create a `vX.Y.Z` tag by hand — it is the Action's output,
and the two names must differ because they point at different commits: `src-vX.Y.Z` at
the source on `master`, `vX.Y.Z` at the build commit the Action creates from it. Ordinary
pushes carry no tag at all, so nothing runs.

The workflow then runs the tests, typechecks, builds, verifies no peer dependency was
inlined, commits `dist/` with `git add -f`, and pushes tag `vX.Y.Z`. jsDelivr serves
that tag:

    https://cdn.jsdelivr.net/gh/meinside/in-browser-llm@vX.Y.Z/dist/index.js

**Those two places are the only pinned versions in the repo**, and the workflow checks
both against the tag before building — a mismatch fails the release rather than
publishing a README whose example points at a version that does not exist. Everywhere
else writes `vX.Y.Z` deliberately: a stale URL in prose looks usable, which is worse
than an obvious placeholder.

Note jsDelivr does **not** serve GitHub Release assets, which is why the workflow
commits `dist/` to the tag instead of attaching it. It also caches a tag permanently,
so a bad release cannot be fixed in place — only superseded by the next version.

That permanence is what the two-tag scheme buys: `vX.Y.Z` comes into existence only
after the tests, typecheck, build, and peer check have all passed, so a failed release
leaves the name jsDelivr reads untouched. Pushing `vX.Y.Z` directly would publish it
before `dist/` exists, and one request during that window caches an empty tree forever.
