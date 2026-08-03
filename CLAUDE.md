# in-browser-llm

A TypeScript library for running small LLMs/SLMs in the browser on WebGPU.
Personal-use; distributed by importing directly from GitHub via jsDelivr.

Design: `docs/superpowers/specs/` — one spec per feature, each recording why it was
built that way rather than what the code now does: image input, audio input, tool
calling, audio-only models, and two gate corrections (buffer-limit, wasm-heap).

**A new spec means updating that sentence**, which is an index over a directory and
so is never wrong until the directory grows — exactly when nothing reminds you. Read
the code for current behaviour; a spec that contradicts it has been superseded, and
that is worth a note at the top of the spec rather than silent divergence.

## Why this library exists

transformers.js and MediaPipe do the inference. What this library sells is the
ability to **decide, before downloading, whether a model will run**, plus
absorbing per-runtime traps.

In an earlier prototype a 3.4GB download completed and then failed to load with
`Failed to load external data file … Unknown error occurred in memory copy`. The
cause was a single 1120MiB tensor against a 1GiB buffer limit — not the total size
— and the message named none of that. That experience is the basis for this whole
design; the traps below are what came out of it.

## Never forget these (traps confirmed by measurement)

All of these were actually hit, and most of them fail **silently**.

1. **The largest single tensor blows up before total size does.**
   In `gemma-4`, `model_embed_tokens_per_layer_weight_quant` is 1120MiB, which
   exceeds Firefox's `maxBufferSize` of 1GiB. Chrome passes at 4GiB−4.
   **Two limits matter and they are not interchangeable:** `maxBufferSize` caps one
   allocation and is fatal, because ORT calls `CreateBuffer` once per tensor;
   `maxStorageBufferBindingSize` caps one binding, and ORT splits an oversized
   tensor across several bindings over that same buffer (PR #25962), so exceeding
   it alone is survivable. Firefox/macOS reports 1GiB for both, which is why the
   distinction stayed hidden — the WebGPU defaults are 256MiB against 128MiB, a 2×
   gap that mobile adapters are expected to land in.
   For *that tensor* q4 and q4f16 are the same size (only the scale dtype
   differs), so quantization does not solve it — but this is not a general rule;
   `scripts/check-model.mjs` prints per-variant totals.

2. **A missing dtype session key silently downloads fp32.**
   That is why dtype is passed as a single string, which applies to every session
   whatever the export's layout is. Never hardcode session names as dtype keys.

3. **Gemma 4's turn delimiters are `<|turn>role ... <turn|>`.**
   They are **not** Gemma 2/3's `<start_of_turn>`. Passing a raw string makes the
   model generate its answer repeatedly. MediaPipe's stop tokens are hardcoded to
   `<eos>` and `<|endoftext|>`, so it never halts on `<turn|>`, and there is no
   `stopTokens` option in its API.

4. **MediaPipe's streaming `partial` is a delta**, not cumulative. Completion
   arrives as `("", true)`.

5. **`AudioContext` does not exist in Workers.** Audio must be decoded on the
   main thread and passed across as a `Float32Array`.

6. **The WebGPU wasm builds ship no CPU kernels.**
   `GatherBlockQuantized` symbol counts are 0 in the `asyncify`/`jspi` builds and
   24 in `jsep`/default. So promising "falls back to CPU if GPU fails" would be a lie.

7. **Do not override a model's sampling defaults — but check that it has any.**
   Each model ships a `generation_config.json`; `gemma-3-270m` asks for
   `do_sample: true` with `top_k: 64`, `top_p: 0.95`. Hardcoding `do_sample: false`
   made it greedy, and a 270M model decoding greedily rambles and repeats. Callers
   who want determinism use `GenerateOptions.overrides`. Note `eos_token_id` can be
   an **array** (`[1, 106]` here) — transformers.js handles that, so do not
   reimplement it.
   **The other half of this, found later and more insidious: a config may specify
   nothing, and the runtime's fallback is greedy.**
   `Qwen3.5-0.8B-Text-ONNX`'s `generation_config.json` holds only `eos_token_id`, and
   transformers.js defaults `do_sample = false` (`configuration_utils.js:68`). So the
   default example model was decoding greedily and answered a conversational prompt
   with one phrase repeated to the token cap. **Do not attribute repetition to model
   size before reading the config** — this looked exactly like "0.8B is just weak",
   and it was not. The failure mode is identical to overriding badly; only the cause
   differs, so "we don't override" is not sufficient protection.
   Qwen's card recommends `temperature: 1.0, top_p: 1.0, top_k: 20`,
   `presence_penalty: 2.0` for non-thinking text (transformers.js has no
   `presence_penalty`; `no_repeat_ngram_size` and `repetition_penalty` are the
   available levers). The example passes those; the library only *warns*
   (`warnIfGreedyByDefault`) rather than injecting them, because picking values for a
   model whose author picked none is the very thing the first half of this trap
   forbids. Compare before assuming: `Qwen3-VL-2B` and `gemma-4` both ship
   `do_sample: true`, which is why only this model showed it.
   **Distinguish the two symptoms, because sampling fixes one and not the other.**
   Verified in a browser: with the card's values the verbatim repetition is gone,
   while *some* prompts still get verbose or meandering answers. The second is a 0.8B
   model at `temperature: 1.0` and is not a defect to chase. The signature of the real
   bug is **one phrase repeated identically to the token cap**; if that returns, check
   whether `overrides` is reaching `generate()` before suspecting the model.

8. **MediaPipe deletes `baseOptions.modelAssetBuffer` after consuming the stream.**
   Calling `setOptions()` afterwards re-validates against an empty asset and throws
   "No model asset provided." That is why `maxTokens` is load-time only for
   MediaPipe.

9. **A stream from the Cache API may belong to another realm.** MediaPipe checks
   `reader instanceof ReadableStreamDefaultReader` and rejects a foreign-realm
   reader with the same "No model asset provided." error. `loader.ts` pipes cached
   bodies through a local `TransformStream` to fix the identity.

10. **`content-length` is invisible in a browser and wrong in Node — read
    `x-linked-size`.** HuggingFace's `access-control-expose-headers` does not list
    Content-Length, so a browser sees `null` and any size read collapses to 0. That
    silently disabled every size check on the gate while all Node tests passed —
    the worst shape of bug this project can have. And Node is not a reference
    either: LFS-backed files report the *pointer* size there (974 bytes for a 1.3GB
    model) unless redirects are followed. `x-linked-size` carries the real byte
    count and **is** on the exposed list. Verified against live response headers.
    `test/inspect.test.ts` now stubs a response with no `content-length` at all, so
    the browser condition is covered.

11. **Embedded weights hit a wasm-heap limit that no tensor check sees.**
    `Qwen3-1.7B` (q4f16, one 1360MiB `.onnx`) threw
    `Can't create a session. ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc` on
    Firefox **after** the gate cleared it and the download finished. Cause:
    `wasm-core-impl.ts` does `_malloc(model.byteLength)` then `HEAPU8.set(...)`, so
    an embedded file needs one contiguous block, and ORT allocates the initializers
    again while parsing the protobuf — inside a wasm32 heap of `maximum:65536`
    pages (4GiB, growth refused above `4294901760`). Sidecar `.onnx_data` never
    takes that path: bytes stay JS-side and wasm copies per-tensor slices.
    **The discriminator is storage layout, not total size.** Measured: 543MiB
    embedded works, 949MiB of `.onnx_data` works, 1360MiB embedded fails. Note
    1360MiB is under *every* documented ceiling (protobuf 2GB, Chrome ArrayBuffer
    2046MiB, wasm32 4GiB), so the 1GiB warning threshold in `gate.ts` is arbitrary
    and says so; only the 2GiB blocker is documented.

12. **A tool-call format built from special tokens cannot be parsed from decoded
    text — but the decoded text is not the only thing available.**
    Qwen3's `<tool_call>` tags are `special: false`, so they survive
    `skip_special_tokens: true` and reach `parseToolCalls`. gemma-4's framing tokens
    *and* its string delimiter `<|"|>` (id 52, named `escape_token`) are
    `special: true` **in Google's own upstream tokenizer**, not just the ONNX export.
    Confirmed by reading `tokenizer.json`: ids 46–52 and 100–106 all carry
    `"special": true`. And the ambiguity is real — measured, not inferred:

    ```
    encode('{"a": <|"|>3<|"|>}') → decode skip_special_tokens:true → '{"a": 3}'
    encode('{"a": 3}')           → decode skip_special_tokens:true → '{"a": 3}'
    ```

    Identical. So no parser downstream of that decode can recover the types, and the
    reference implementations (HuggingFace's `_gemma4_json_to_json`, vLLM's scanner)
    work only because they read the raw string first.

    **The earlier conclusion — "this library sits downstream of that, do not plan to
    add it later" — was wrong, and the error was assuming the position rather than
    checking it.** `TextStreamer` accepts a `token_callback_function`, invoked at
    `streamers.js:94` with the raw ids **before** the special-token branch that drops
    them (`:96`). `modeling_utils.js:1014` supplies them per step. Decoding those ids
    with `skip_special_tokens: false` returns the delimiters intact, and an
    incrementally accumulated decode equals a whole-string decode — both verified.
    So the same position the reference parsers use is reachable from here.

    **This is a class, not one bug — reasoning blocks are the second instance, and
    `onRawText` is the escape there too.** Verified against both tokenizers, and the two
    families are opposites: `Qwen3.5`'s `</think>` is `special: false` (id 248069) and
    survives the stream, so reasoning splits out of the deltas; `gemma-4` wraps reasoning
    in `<|channel>thought … <channel|>` with **both** `special: true` (ids 100, 101), so
    `skip_special_tokens: true` erases the boundary and the reasoning merges silently into
    the answer. Measured:

    ```
    encode('<|channel>thought\nlet me think\n<channel|>the answer')
      → skip:true  → 'thought\nlet me think\nthe answer'
      → skip:false → '<|channel>thought\nlet me think\n<channel|>the answer'
    ```

    gemma-4 also puts `<|think|>` in a synthesized *system* turn rather than at the
    prompt's end, so "where does the model start" differs too. **Before splitting any
    structured region out of generated text, check `special` on its delimiters** — the
    answer is per family and the failure is silent either way.

    **I concluded "not separable" first, which is the third time in this trap's history
    that *not in the decoded stream* got read as *not available*.** What actually differs
    is only that `onRawText` fires once at completion, so a channel split cannot be live
    the way Qwen's is. And the redistribution needs no regex, because only the delimiters
    are special: the reasoning body is ordinary text and appears in the clean stream too,
    so the summed body lengths give the offset where the answer starts and
    `clean.slice(consumed)` is the answer — no stripping of EOS or turn framing required.

    Types survive because the dialect marks only strings:

    ```
    {a:'x'}       → call:f{a:<|"|>x<|"|>}
    {a:3}         → call:f{a:3}
    {a:true}      → call:f{a:true}
    {a:[1,'two']} → call:f{a:[1,<|"|>two<|"|>]}
    ```

    Do not expect the format to change: Google made the custom dialect first-class
    upstream, and the ONNX export's `tokenizer_config.json` now ships a
    `response_schema` giving the parse rules outright — `x-regex-iterator`
    `<\|tool_call>(.*?)<tool_call\|>`, then `call\:(?P<name>\w+)(?P<arguments>\{.*\})`.
    **Read that field rather than reverse-engineering the dialect.** Even ollama and
    llama.cpp, which do see raw text, have open parse-failure issues on quoted
    arguments, so the parser needs its own tests rather than trust.

13. **MediaPipe multimodal input fails on the published gemma-4 web builds, and the
    reason is the weights, not the API or this library.** Settled by measurement —
    do not re-litigate the API surface.
    The API is real: `genai.d.ts` declares `maxNumImages`, `supportAudio`, and
    `Prompt = PromptPart | PromptPart[]` where `PromptPart = string | Image | Audio`.
    Both options are accepted; the failure comes from the weights.
    **One cause, more than one message** — the reason `explainModalityFailure` matches
    several patterns rather than one:
    - *Predict time, both modalities.* The graph builds, then
      `Llm{Vision,Audio}InferenceCalculator` dies with **"… models could not be
      created"** (`llm_gpu_runner_manager.cc:509`). Reproducible on demand with
      `maxImages` or `supportAudio`, alone or together. Both calculator symbols are
      present in `genai_wasm_internal.wasm`.
    - *Load time*, inside `createFromOptions` (`:572`), before a prompt exists:
      **"Audio options should not be null when loading audio runner"**. Names no
      calculator and shares no wording with the predict-time form.

    **The load-time message was seen once and has not been reproduced since.** It was
    first recorded here as what `supportAudio` always does; when audio alone turned out to
    reach the calculator, the claim was revised to "both options together" — which did not
    reproduce either. **Both were one observation restated as a rule**, the very thing this
    file's "tag facts separately from research" rule exists to prevent. The pattern stays
    in `explainModalityFailure` because it costs one regex and nothing else would catch it,
    but its trigger is **unknown**: do not attribute it to an option again without
    reproducing it, and do not add an example entry claiming to exercise it — one was added
    and removed for exactly that reason.

    The wasm states the real cause outright — `strings` on
    `genai_wasm_internal.wasm` yields **"Audio is enabled but
    audio_encoder_parameters are not defined in the model nor well known models"**
    alongside `model_params.has_audio_encoder_parameters()`. When MediaPipe's message
    is unhelpful, grep the wasm: these diagnostics are more direct than anything the
    JS layer surfaces.
    **Cause, from the container's own header:** both
    `gemma-4-E2B-it-web.litertlm` and `-web.task` declare
    `model_type: tf_lite_artisan_text_decoder` and contain **no** vision or audio
    encoder section. The non-web builds (`gemma-4-E2B-it.litertlm`, the intel and
    qualcomm variants) *do* — `tf_lite_vision_encoder`, `tf_lite_vision_adapter`,
    `tf_lite_audio_encoder_hw` — but they declare `backend_constraint: "cpu"` and
    are 2468–2877MiB, i.e. not browser targets. So there is currently **no web-ready
    LiteRT container with vision weights**, and no option can conjure one.
    Two things this corrects, both of which cost time:
    - The `.task` is **not a ZIP**. It begins with `TFL3`, same as a bare tflite
      file; the bundle's ZIP branch is for some other format. Do not go looking for
      a central directory.
    - The docs' claim that these options are "not compatible with converted models"
      is **not what the code enforces**. `generateResponse` checks only the image
      count against `maxNumImages` and the `supportAudio` flag; the converted-model
      refusals fire for multi-response and LoRA only. The container format was never
      the obstacle.
    `mediapipe.ts` keeps the options and rewrites that failure via
    `explainModalityFailure`, because MediaPipe's own message names every layer
    except the missing weights. The example page keeps a deliberately-failing entry
    so the next attempt does not mistake this for a regression here.

## Picking a model: what actually decided the current lineup

Mechanical fitness and usefulness are different questions, and a survey that only
checks packaging will recommend the wrong model. What settled each choice:

The lineup is `Qwen3.5-0.8B-Text` (text + tools), `Qwen3-VL-2B` (text + image, both
browsers), `gemma-4-E2B` (text + image + audio, Chrome-only), and gemma-4 over
MediaPipe. `Qwen3-0.6B` and `Qwen3-1.7B` were dropped; the reasons below are why.

- **Multilingual tokenization, measured.** One non-Latin-script sentence, same text
  through three tokenizers: `Qwen3.5-0.8B` 16 tokens with 0 byte-fragments,
  `Qwen3-0.6B` 21 tokens / 2 fragments, `granite-4.0-h-350m` 35 tokens / **21
  fragments** — it splits a single three-character word into one character plus two
  partial-UTF-8 bytes plus another character. Fragments cost context and decode time
  in proportion. Reproduce with `AutoTokenizer` and count the ids whose `decode`
  contains U+FFFD. Do this in whatever language you actually plan to use: a
  tokenizer efficient in one non-Latin script is not automatically efficient in
  another.
- **A card's language list is not evidence.** IBM lists broad multilingual support
  for Granite 4.0, and IBM's own benchmarks in the same card contradict it: MMMLU
  27.95 and INCLUDE 27.09 for the 350M, i.e. chance level on four-way multiple
  choice. Read the numbers, not the list.
- **Storage layout over total size.** `Qwen3.5-0.8B` is 448MiB with sidecar
  `.onnx_data`; the 543MiB `Qwen3-0.6B` it replaced embeds its weights. Sidecar
  avoids trap 11 outright.
- **A published tool-calling score does not mean a model calls tools.**
  `Qwen3-0.6B` has no BFCL number and answered from memory instead of calling the
  supplied tool — verified in a browser, with the rendered prompt confirmed to
  contain the tool block.
- **One capable model beats two weak ones as an example.** `Qwen3-0.6B` and
  `Qwen3-1.7B` were both in the lineup and neither demonstrated anything the 0.8B
  does not do better; the 1.7B additionally trips trap 11. Two entries testing the
  same path is not coverage.

## Tool call formats are per family, and one needs a different input

Three shapes seen so far. The first two share the `<tool_call>` delimiters:

- **JSON** — `{"name": …, "arguments": {…}}`. Qwen3, Qwen2.5, Granite 4.0.
- **XML** — `<function=name><parameter=key>value</parameter></function>`. Qwen3.5.
  Values come back as strings; the markup carries no types. **And not necessarily in a
  spelling JSON accepts:** asked for a `boolean` it returned `True` — measured in a
  browser — so `value === 'true'` is *false* for a value meant as true, inverting the
  argument with nothing to see. Coercion stays with the caller who knows the declared
  type; folding case in `parseXmlBody` would be guessing with a longer list of guesses.
  The same call on gemma-4 returned a real `true`, so this belongs to the encoding rather
  than to the model.
- **gemma-4's DSL** — `<|tool_call>call:name{key:<|"|>str<|"|>,n:3}<tool_call|>`.
  Types are carried: only strings get `<|"|>`. But every delimiter is a *special*
  token, so this one cannot be parsed from the decoded stream the other two use — it
  needs the raw token ids (trap 12).

`ToolCallFormat.encoding` selects the parser, which is what made adding Qwen3.5 cost
one branch and one catalog entry. gemma-4 cost more, because the *input* differs
rather than only the syntax. **Render the real template and parse its output back
before trusting a format** — Qwen3.5 was first reported as JSON because its prompt
*describes* JSON while its assistant turn emits XML.

**The `tool_calls` shape a template accepts is also per family, and gemma-4 is the
strict one.** It requires OpenAI's nested `{type: 'function', function: {name,
arguments}}` and throws `Cannot perform operation + on undefined values` on the flat
`{name, arguments}` that `parseToolCalls` returns. Qwen3.5 and Qwen3-VL accept
**both**, verified — so normalizing to the nested shape needs no per-family branch,
and the flat shape stays the library's public one.

`lookupCatalog` matches by substring against the first hit, so `'qwen3.5'` must be
listed before `'qwen3'`; otherwise every Qwen3.5 model gets the JSON parser and every
call is discarded. A test pins that order.

## The vision-path "hang" was a sign error upstream, and it was never a hang

Root cause found by reading `node_modules`, and the earlier guess (rope-index or
KV-cache handling) was right about the area and useless about the mechanism. Recorded
in full because the shape of this bug recurs.

`Qwen2VLForConditionalGeneration.get_rope_index` returns
`[position_ids, mrope_position_deltas]` from two branches that **disagree about what
the delta means**:

| branch | formula | file |
|---|---|---|
| multimodal | `max(positions) + 1 - seq_len` | `modeling_qwen2_vl.js:257` |
| text-only | `max(positions) + 1 `**`+`**` seq_len` | `modeling_qwen2_vl.js:37` |

One is an offset; the other is a `+` where the first has a `-`. For a full attention
mask `max(positions)` is `seq_len - 1`, so the correct delta is **0** and the shipped
one is **`2 * seq_len`**. `prepare_inputs_for_generation` then places every decode
step at `past_length + delta`, so the first generated token gets position
`3 * seq_len`. Positions land far outside the trained range, logits degenerate, EOS
never arrives, and generation runs to `max_new_tokens` emitting nothing usable.

**So it was not a deadlock — it was a long silent loop.** That is exactly why no
error appeared, and why "hangs" was the wrong word: with a low `maxTokens` it would
have returned, empty. Verify the same shape by timing rather than by waiting.

**Only the delta is wrong. The `position_ids` from that branch were always correct**
(`cumsum_masked_fill` yields `0..L-1`). A first attempt at this fix assumed both were
broken and supplied `position_ids` to `generate()`; that produced `TheTheThe…` instead
of nothing — better, and still wrong. The reason is worth remembering:
`prepare_inputs_for_generation` does return early when `position_ids` are set, but
`_update_model_kwargs_for_generation` assigns `position_ids = null` after **every**
step (`modeling_utils.js:687`) to force a recompute. So a per-call override fixes the
prefill and nothing after it. **A value recomputed each step must be fixed at its
source, not passed in once.**

Three details explain why it looked model-class-specific:

- A **text-only turn on a vision model** takes the buggy branch, because the
  processor emits `image_grid_thw` only when images are present
  (`processing_qwen2_vl.js`). Turns *with* an image were always fine.
- The error **scales with prompt length**, so a short first turn survives while a
  third turn carrying a tool call and its result does not. Nothing about tools is
  special; they just make the prompt long.
- The affected export **does take `position_ids`** (confirmed present in
  `decoder_model_merged_q4f16.onnx`'s header), so the bad values really reach the
  model rather than being dropped.

The fix is `patchMropeDeltas` in `transformers.ts`, applied once at load: it wraps the
loaded model's `_get_text_only_rope_index` and recomputes the delta with the
multimodal branch's sign, leaving `position_ids` untouched. It works because that
method is a prototype method dispatched through `this`, so an instance-level
assignment shadows it — verified, not assumed. Scoped by `config.model_type` to
`qwen2_vl` / `qwen2_5_vl` / `qwen3_vl`; turns carrying an image never reach the
wrapped method, so vision keeps its real 3D positions with no explicit condition.
The delta is derived from the returned positions rather than recomputed from the mask,
so it survives upstream changing how positions are built. **Remove the whole function
when upstream fixes the sign.**

**Verified in a browser:** the three-turn tool cycle completes on `Qwen3-VL-2B` on
both Firefox and Chrome. So tool calling is no longer restricted to text-only models,
and the example page no longer hides the demo for vision.

Also worth keeping: `gemma-4` is a vision model that never showed this, because it
does not use mrope. "Vision model" was not the right category; "uses mrope" was.

Ruled out and not worth re-checking: the prompt, `enable_thinking` (Qwen3-VL's
template ignores it), the processor call, and the model class.

## Check these three things before trusting a new vision model

All three were found on SmolVLM-256M while trying to add it as a smaller image
example, after gemma-4 already worked on the same code path. SmolVLM was dropped,
so 1 and 2 are no longer live in this codebase — they are here because the next
vision model can hit them, and nothing in the model's metadata warns you:

1. **A quantized variant can be a broken graph.** SmolVLM's `vision_encoder_q4f16`
   (and `fp16`) fails on the CPU EP with a `SimplifiedLayerNormFusion` cast error,
   and on WebGPU loads anyway while emitting meaningless embeddings — real subwords
   in nonsense order plus runs of newlines. Size tells you nothing about this.
2. **A processor may not tolerate a missing image.** SmolVLM's reads
   `image_inputs.rows` unconditionally, so calling it text-only throws
   `Cannot read properties of undefined (reading 'rows')`. gemma-4 guards the same
   access, and its processor is safe to call either way.
3. **Vision chat templates index content as an array.** They read
   `message['content'][0]['type']` and loop over it; a plain string makes the loop
   iterate characters and fail with `Expected iterable or object type in for loop:
   got StringValue`. Content is normalized to parts before rendering — this one is
   real and still in the code.

SmolVLM was **dropped**: working around (1) and (2) required per-model special
cases that would grow with every new model, and the payoff was only a smaller
example.

`Qwen3-VL-2B` is verified on **both** browsers and is the one to reach for on
Firefox — largest tensor 148MiB, zero Clip nodes, and its processor takes
`(text, images, ...args)`, matching the adapter. gemma-4 still works too but stays
Chrome-only because of its 1120MiB tensor, not because Firefox cannot do vision.

None of these is detectable without running the model, and none produces an error
naming its own cause. Test a new vision model in a browser before trusting it.

## Audio

**Processor arity AND argument order are fixed per model family.** gemma-4's
processor is always `(text, images, audio, options)`, so a text-only turn still
passes `null` into the audio slot. Omitting the argument puts the options object
there instead, and its feature extractor rejects it with "expects a Float32Array …
but got Object". But order differs too: qwen2/3-vl is `(text, images, ...args)`
while `UltravoxProcessor` is `(text, audio, kwargs)`, with audio where gemma-4 puts
images. So the call shape comes from `spec.vision` and `spec.audio` together, which
is why `inspectModel()`'s result must reach `load()`. **Read the family's
`processing_*.js` in `node_modules` before adding a model** — these signatures are
documented nowhere else. `processorCallArgs` builds the argument list from those two
flags alone and is unit-tested per combination, so a new family's shape goes there
rather than into `generate()`.

**The chat template's `content` shape is per-family too, and the requirements are
opposite.** Vision templates index `message['content'][0]['type']` and loop over it,
so a plain string fails with "Expected iterable or object type in for loop: got
StringValue". Llama-derived templates apply `content | trim`, which fails on an
array with "Unknown ArrayValue filter: trim". Hence `toParts` for vision and
`toText` otherwise.

**But do NOT wrap empty content, and the reason is truthiness rather than shape.**
`toParts('')` gives `[{type: 'text', text: ''}]`, and a one-element array is *truthy*
where `''` is falsy. Templates branch on that. gemma-4's does:

```jinja
{%- elif not (ns_tr_out.flag and not message.get('content')) -%}
    {{- '<turn|>\n' -}}
```

An assistant turn carrying only `tool_calls` has empty content by construction, so the
wrap closed the model's turn after the tool response — and `add_generation_prompt`
declines to reopen it when the previous block was a `tool_response`. The third turn
therefore handed the model a *finished* conversation, which it correctly answered with
EOS: **zero chunks, no error, nothing naming the cause.** Found by rendering the text
path and the vision path side by side and diffing the last 60 characters; the text path
was always correct, so it looked like a vision bug and was not one.

Two lessons worth more than the fix. **A template reads content for truthiness as well
as for shape, so a normalization that preserves meaning can still change behaviour.**
And **empty output with zero chunks means the prompt was already complete** — check
whether the model's turn is open before suspecting the model. Count `<|turn>` against
`<turn|>`: a prompt ready for generation has one more opener than closer.

Image and audio in one turn **is** supported: `modeling_gemma3n.js` merges each
modality in its own `if` block rather than `else if`, the token ids differ
(258880 / 258881), and the masked scatter throws on a count mismatch — so a
dropped modality would error rather than pass silently. A turn that looks like it
ignored the audio is more likely sampling variance or an unclear prompt. **This was
chased once for a long session and there was no bug** — the same prompt worked on
re-run. Re-test with `overrides: { do_sample: false }` before suspecting the merge
logic.

### Audio without vision does not work yet, and the reason is not ours

`gemma-4` is still the only model that accepts audio here. The adapter *has* an
audio-only branch — `AutoModelForAudioTextToText` plus a `(text, audio, kwargs)`
call — because transformers.js registers all four audio-text models
(`granite_speech`, `ultravox`, `voxtral`, `voxtral_realtime`) under that class
alone, so any other class throws "Unsupported model type". **That branch is
UNVERIFIED: no model has reached inference through it.**

Ultravox-1B was the attempt and failed in three layers, each hidden behind the
last. The first two were ours and are fixed; the third is not:

1. Wrong model class → `Unsupported model type: ultravox`.
2. `content` normalized to parts → `Unknown ArrayValue filter: trim`, because its
   Llama-derived template does the opposite of what vision templates need.
3. **The export's `preprocessor_config.json` declares
   `"processor_class": "WhisperProcessor"`.** `AutoProcessor` trusts that field
   unconditionally (`models/auto/processing_auto.js:55`), so it never constructs
   `UltravoxProcessor` — and `WhisperProcessor._call(audio)` takes audio *first*,
   which is why the prompt string landed in the audio slot:
   `WhisperFeatureExtractor expects ... but got String instead`.

Only 3 is fatal, and only hardcoding `UltravoxProcessor` past `AutoProcessor` would
work around it — a per-model special case, which is what got SmolVLM dropped.
**Check `preprocessor_config.json`'s `processor_class` before trying another audio
model**; if it names a bare feature-extractor processor rather than the model's own,
that export cannot work through `AutoProcessor`.

## Adding or changing a supported model

Run this first. It downloads only headers and JSON configs, never weights:

```bash
node scripts/check-model.mjs onnx-community/gemma-3-270m-it-ONNX
```

Every check in it exists because that thing failed silently once. Adding
gemma-3-270m produced four separate bugs, in this order:

1. **Session names** — the export ships a single `model` session, but `inspect.ts`
   only probed `embed_tokens`/`decoder_model_merged`. Every size read as 0, which
   the gate interpreted as "fits", disabling the tensor-limit check entirely.
2. **dtype keys** — a per-session dtype object with gemma-4's key names resolved to
   null for this layout and silently downloaded **fp32**: 260MB became 1087MB.
3. **f16 Clip guards** — the q4f16 variant carries fp16 overflow guards that
   Firefox cannot compile. q4 has none, so dtype downgrade rescues it — but note
   gemma-4's *encoders* keep their Clip nodes even at q4, where downgrading does
   not help. The script distinguishes these two cases.
4. **Sampling** — forcing `do_sample: false` on a 270M model made it ramble. The
   model's own `generation_config.json` asks for sampling.

The script reports MUST FIX for anything that would break the gate, and CHECK for
things needing a decision. Neither replaces opening a browser: it cannot prove
inference works, so the browser steps it prints at the end still apply.

**Adding or replacing a model updates TWO documents, and they have different
readers.** README's `Verified models` table is what a caller consults to pick one, so
it carries the ref, the size, the modality columns, and the tool encoding — and a row
belongs there only after the model ran in a browser, not after it gated. Every
*reason* goes in `docs/verification-log.md` instead. Dropping a model means removing
its row from the table while **keeping** the account of why it was dropped, which is
the whole point of the split: the table would otherwise grow a graveyard, and the
reasoning would otherwise be lost with the row. `Qwen3-0.6B`, `Qwen3-1.7B` and
SmolVLM are the existing examples — named nowhere in README, since a caller does not
need them, but kept in this file under "Picking a model" and "Check these three
things", where the next survey will look.

## Finding a new model's turn delimiters

A new or changed model needs a catalog entry. Do not guess — run the script.

```bash
node scripts/inspect-chat-template.mjs onnx-community/gemma-4-E2B-it-ONNX
```

It fetches only `tokenizer_config.json` (a few KB) and prints:

- `eos_token` / `bos_token` / `pad_token`
- every special token appearing in the template
- literals from the `add_generation_prompt` block → **the model's turn-start string**
- closing tokens of the form `<...|>` → **stop-sequence candidates**

Verified output for Gemma 4:

```
generation prompt literal : "<|turn>model\n"
turn-end candidates       : "<turn|>", "<channel|>", "<tool|>", …
```

### Do not trust the output blindly

**It is not a generator.** Templates are Jinja with conditional branches, and a
human must decide which branch is the default path. In the output above,
`<tool|>` and `<channel|>` are for tool calls and thinking mode, not the
end of an ordinary conversational turn. Only `<turn|>` was actually needed.

### How to confirm

**Generate something and watch for repetition.** If the answer appears several
times, the turn-end string is wrong — the model ended its turn, the runtime
failed to stop, so the model opened a new turn and answered again. This exact
symptom is how the Gemma 4 problem surfaced (108 chunks, then 48 after the fix).

### Same family does not mean same format

Gemma 2/3 use `<start_of_turn>`/`<end_of_turn>`; Gemma 4 uses `<|turn>`/`<turn|>`.
**Never assume by family — check every time.** The script's regex matches both shapes.

### Mostly unnecessary on the transformers.js path

`processor.apply_chat_template()` interprets the Jinja template there. This
procedure is mainly for the **MediaPipe path**, which has no template support and
hardcoded stop tokens.

## Working principles

### Do not assert without verifying

Most facts in this project were confirmed by **opening package sources or
binaries directly**. Documentation repeatedly disagreed with reality:

- MediaPipe had a reputation for being Chrome-only, but a Firefox Asyncify build
  had already been merged
- caniuse lists Firefox WebGPU as "disabled by default", yet it was enabled on
  Apple Silicon + Tahoe
- research claimed the CPU EP supports f16 scales, but the shipped wasm build had
  no such kernel

When handling a new runtime or model, grep `node_modules` directly, and inspect
wasm binary symbols when necessary.

### Tag facts separately from research

This is why the design notes distinguish measured findings from things only read
in documentation. Writing unverified claims as facts leads to designing on top of them.

### Do not declare completion unilaterally

Real inference cannot be verified automatically — headless browsers expose no
WebGPU adapter and CI cannot fetch multi-GB weights. Therefore:

- pure functions (`gate`, `prompt`, `onnx-parse`) **are tested with Vitest**
- real inference is checked by the user opening `examples/` in a browser
- "it works" is a judgement only the user's run can make

**That judgement is recorded in `docs/verification-log.md`, and recording it is part
of the work rather than bookkeeping after it.** A browser run that is not written
down is indistinguishable later from one that never happened, and this file is the
only place the distinction survives — nothing in CI can rebuild it.

Three rules, each of which the file already demonstrates:

- **Write what the run settled, not just pass or fail.** A ✅ next to a negative
  scenario means "failed for the documented reason", which is unreadable without the
  sentence explaining it. Scenarios 21 and 22 are that shape.
- **When a previous conclusion turns out wrong, keep the wrong reading.** Do not
  quietly correct it. Row 22 was wrong twice and scenarios 27/28 three times, always
  in the same direction — *absent from the decoded stream* read as *absent
  altogether* — and it is precisely that repetition, visible only because the earlier
  readings were left in, that turned trap 12 from one bug into a class. An overwritten
  history costs the next person the same three attempts.
- **State what a ✅ does not cover.** `processorCallArgs` is unit-tested for shape and
  cannot show that a real processor accepts the array, so the log names the two models
  that exercise both arities. A cell that overstates its reach is worse than a blank
  one, because a blank invites re-testing and an overstatement forbids it.

Since the log moved out of README, nothing puts it in front of you — so treat a
browser-verified change as incomplete until it is updated.

An engine adapter is not exempt just because it needs a browser. The decisions inside
it that fail *silently* — processor argument order, model-class selection, content
shape — are pure functions and are extracted as such (`processorCallArgs`,
`selectModelClass`, `normalizeContent` in `transformers.ts`). **Prefer that over
asserting against the adapter's own source text**, which an earlier round of tests did:
a regex over `transformers.ts` breaks when a variable is renamed and passes when the
argument order is wrong, so it tests spelling rather than behaviour. Two such
assertions remain deliberately, both about *wiring* rather than logic — that
`patchMropeDeltas` is installed at all, and that the failed `position_ids` approach has
not crept back — because neither is reachable without an adapter and real weights.

### git

**The user manages commits.** Do not run `git add` or `git commit` unless asked.

## Distribution (jsDelivr constraints)

The `/gh/` path serves **only files committed to git**. GitHub Release assets are
not served. `raw.githubusercontent.com` responds with `Content-Type: text/plain`,
which browsers refuse to execute as an ES module.

Workflow:

1. `master` holds sources only (`dist/` is not committed)
2. push a `src-vX.Y.Z` tag
3. an Action builds, creates a commit including `dist/`, and attaches tag `vX.Y.Z`
4. jsDelivr serves `dist/` from `vX.Y.Z`

```
https://cdn.jsdelivr.net/gh/meinside/in-browser-llm@vX.Y.Z/dist/index.js
```

**Only one place carries a real version: the Quick start example in README.** Everywhere else
writes `vX.Y.Z`, so a release touches `package.json` and that one snippet. The release
workflow verifies the two agree with the tag and fails if they do not — see
`.github/workflows/release.yml`. Do not reintroduce a pinned version in prose; a stale
URL in documentation is worse than a placeholder because it looks usable.

transformers.js and MediaPipe are **not bundled** (treated as peers).

## Environment notes

- Requires Node.js 20.19+, 22.12+, or 24+ (odd majors are excluded by vitest).
  CI runs 24, the current LTS.
- **TypeScript stays on 5.x.** TypeScript 7 breaks `tsup`'s declaration build with
  `Cannot read properties of undefined (reading 'useCaseSensitiveFileNames')`.
  `tsc --noEmit` passes either way, so the failure only appears in `npm run build`
  and only for `dist/index.d.ts` — which a library cannot ship without. Verify
  `npm run build` emits the DTS line before upgrading.
- Root-owned files in `~/.cache/npm` can make `npm install` fail with EPERM.
  Fix: `sudo chown -R $(id -u):$(id -g) ~/.cache/npm`
  Workaround: `npm install --cache "$TMPDIR/npm-cache"`
- Dev servers and `huggingface.co` access may be blocked by the sandbox

## Measured baselines (macOS Apple Silicon M1 Pro)

Example figures from one session with gemma-4-E2B, kept for rough scale. Not
maintained — they will drift as models and browsers change.

| Runtime | Browser | TTFT | decode |
|---|---|---|---|
| MediaPipe (LiteRT 2.0GB) | Chrome 150 | 432ms | 13.7 tok/s |
| MediaPipe (LiteRT 2.0GB) | Firefox 153 | 718ms | 3.3 tok/s |
| transformers.js (ONNX q4f16 3.4GB) | Chrome 150 | 568ms | 10.4 tok/s |

Firefox decoding at a quarter of Chrome's rate is due to missing `subgroups` —
not a defect.

| Feature | Firefox 153 | Chrome 150 |
|---|---|---|
| `shader-f16` | yes | yes |
| `subgroups` | no | yes |
| `maxBufferSize` | 1GiB | 4GiB−4 |
| `maxStorageBufferBindingSize` | 1GiB | 4GiB−4 |

Both browsers report "Intel Mac OS X" in the UA — **never trust the UA.**

## Settled — do not re-investigate

Both of these look like obvious improvements and are not. The evidence is recorded
so the investigation does not get repeated.

- **ORT splitting is already in use.** The large-tensor
  splitting of microsoft/onnxruntime#25962 (merged 2025-10, `535f1f37c4`) is an
  ancestor of the `1.26.0-dev` commit transformers.js pins, so no upgrade is
  needed or possible (the pin is exact, and 4.2.0 is the latest release). Verified
  three ways: merge-commit ancestry via `gh api compare/535f1f37...`,
  `CHUNK_SIZE_IN_ELEMENTS` present in `ort-wasm-simd-threaded.asyncify.wasm` (the
  build transformers.js loads) and absent from `jsep`, and `program_manager.cc`
  reading the device limit with no session option required. What remains open is
  whether a tensor between the two limits really loads: it needs an adapter where
  `maxBufferSize > maxStorageBufferBindingSize`, which Firefox/macOS is not.
- **`DEFAULT_MAX_TOKENS` is not worth deriving.** Deriving 1024 from
  `max_position_embeddings` sounds better than it is. The current models report
  262144 / 262144 / 131072, so the formula collapses to `min(config, practical
  ceiling)` with the ceiling always winning — machinery, no behaviour change. Two
  further obstacles: gemma sets `sliding_window: 512`, so KV cache growth is not
  linear in context length, and the runtimes count different things
  (transformers.js caps output, MediaPipe caps input + output), so one config
  number needs two conversions. 1024 is now documented as a decode-speed budget
  (~100s at 10 tok/s) rather than an arbitrary constant.

## Worth revisiting

- Safari untested
- Mobile untested — `maxBufferSize` is likely as low as 256MiB there, so trap 1
  should occur far more often. Such adapters are also the ones where the two limits
  diverge, making the gate's segmentation warning reachable for the first time
- **MediaPipe multimodal input** is implemented and **blocked on an asset**, not on
  code (trap 13). The plumbing is done: `maxImages`/`supportAudio` in `LoadOptions`,
  `formatPromptParts` building a `Prompt` array. What is missing is a web-ready
  LiteRT container carrying vision weights. Re-check `litert-community` for one
  rather than re-reading the API; if a `*-web*.litertlm` appears whose header names
  `tf_lite_vision_encoder`, this should work as-is.
- **Electron untested, and the version floor is the first thing to check.** WebGPU
  shipped in Chromium 113 (Electron 25+), so anything older cannot run this at all —
  the app that prompted the question was Electron 21.4.3 / Chromium 106, whose
  framework binary carries `enable-unsafe-webgpu` but **zero** occurrences of
  `shader-f16`, which `gate.ts:178` treats as a hard MediaPipe requirement. Beyond the
  floor, `subgroups` decides speed rather than support, and `maxBufferSize` is set by
  the GPU, so **trap 1 survives any Electron upgrade**. Three things are ours rather
  than Chromium's and would need work:
  - `file://` is not a secure context, so `typeof caches === 'undefined'` and
    `storage.ts:13,18` **silently no-op** — every launch re-downloads gigabytes with no
    error. Register a custom scheme with `{ secure: true, supportFetchAPI: true,
    stream: true }` or serve over local HTTP.
  - `mediapipe.ts:18` hardcodes the jsDelivr `WASM_BASE`, so that path cannot run
    offline as written; making it an option is the fix.
  - transformers.js manages its own storage, so a packaged app has **two** cache
    locations and `onCacheStatus` observes only one.
  Probe rather than infer: print `process.versions.chrome`, `isSecureContext`,
  `typeof caches`, and the adapter's features and `maxBufferSize` in the target build.
- **Nothing is published yet.** The repo has never been pushed, so README's jsDelivr URL
  is dead until `master` and a matching `src-v*` tag are pushed. Versioning starts in the
  0.0.x range because jsDelivr caches a tag permanently: a bad release cannot be fixed in
  place, only superseded, so the early numbers are cheap to burn.
