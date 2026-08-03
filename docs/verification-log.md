# Manual verification log

Real inference cannot be tested automatically — headless browsers expose no WebGPU
adapter and CI cannot fetch multi-GB weights. So the scenarios below are checked by
hand in `examples/`, and this file records what each one settled.

It is kept out of the README because most entries answer "why is the code shaped
this way", not "how do I use this". Several were wrong before they were right, and
the wrong readings are left in on purpose: they are the part that stops the same
mistake being made twice.

Pass/fail only — sizes and speeds belong to whichever model was used at the time,
so they are not worth maintaining here. Blank means untested, `—` means not
applicable on that browser.

| # | Scenario | Chrome 150 | Firefox 153 |
|---|---|---|---|
| 0 | transformers.js loads and generates | ✅ | ✅ |
| 1 | Gate blocks a model whose largest tensor exceeds `maxBufferSize` | — | ✅ |
| 2 | Gate passes MediaPipe, warning about `subgroups` where absent | ✅ | ✅ |
| 3 | First load shows monotonic progress | ✅ | ✅ |
| 4 | Revisit reports a cache hit (MediaPipe only) | ✅ | ✅ |
| 5 | Generation streams without repetition | ✅ | ✅ |
| 6 | Size reported for a model with weights embedded in the `.onnx` | ✅ | ✅ |
| 7 | Reasoning model emits no `<think>` block by default | ✅ | ✅ |
| 8 | Vision model answers about an image | ✅ | — |
| 9 | Image picker hidden for a text-only model | ✅ | ✅ |
| 10 | Vision model answers a text-only turn | ✅ | — |
| 11 | Answer reflects supplied audio | ✅ | — |
| 12 | Image and audio in one turn are both reflected | ✅ | — |
| 13 | Audio picker hidden for a model without an audio encoder | ✅ | ✅ |
| 14 | Clip longer than 30s logs a console warning | ✅ | — |
| 15 | Vision model answers about an image on Firefox | ✅ | ✅ |
| 16 | Tool call parsed from the reply | ✅ | ✅ |
| 17 | Tool result fed back and reflected in the answer | ✅ | ✅ |
| 18 | XML tool-call encoding parsed end to end | ✅ | ✅ |
| 19 | Non-English prompt answered in the same language | ✅ | ✅ |
| 19b | Conversational prompt does not degenerate into repetition | ✅ | ✅ |
| 20 | Tool round-trip completes on a vision model (mrope fix) | ✅ | ✅ |
| 21 | MediaPipe image input refused at generate, naming the missing weights | ✅ | ✅ |
| 22 | MediaPipe audio input refused at generate, naming the same | ✅ | ✅ |
| 22c | Text-only generation still works on both those engines | ✅ | ✅ |
| 23 | Tool round-trip completes on gemma-4 via the raw-text path | ✅ | — |
| 24 | Tool demo stays hidden for the MediaPipe entries | ✅ | ✅ |
| 25 | A boolean tool argument stays a boolean under `gemma4` | ✅ | — |
| 26 | The same argument arrives as a string under `xml` | ✅ | ✅ |
| 27 | Thinking mode on request separates reasoning from the answer (Qwen3.5) | ✅ | ✅ |
| 28 | gemma-4 reasoning separated from the answer via the raw-text channel split | ✅ | — |

Tested on macOS Apple Silicon. The table was last swept end to end on both browsers
against the change set that added reasoning separation, the typed tool argument, and the
per-modality MediaPipe entries — no scenario regressed, and every applicable cell is now
filled. `—` marks a scenario that cannot apply on that browser, which for gemma-4 is
always the same `maxBufferSize` limit.

Scenarios 23 and 24 cover gemma-4 tool calling. 23 is Chrome-only for the usual
reason — gemma-4's 1120MiB tensor exceeds Firefox's `maxBufferSize`, so the gate
refuses before any download. 24 is a regression guard rather than a feature: adding the
`gemma-4` catalog entry makes `lookupCatalog` return a tool format for the LiteRT URL
as well, and the demo must stay hidden there.

Everything except the model's own behaviour was verified against the real tokenizer:
the declaration renders, all eight argument shapes round-trip through the parser, and
the same text with special tokens stripped yields nothing. What a browser added was the
one thing that pre-check could not reach — the *third* turn came back empty, because an
empty assistant `content` was being wrapped into a text part and gemma-4's template
reads that value for truthiness. Turn 1 had worked all along, which is why only a full
round-trip in a browser found it.

Scenario 20 was re-run after that fix, since `Qwen3-VL-2B` goes through the same
normalization: no regression on either browser.

Scenarios 25 and 26 read the same tool through two encodings. They exist because the demo
tool used to take one string argument, under which all three parsers look identical and a
regression that flattened every value would pass. It now also takes a `celsius` boolean:
`Number('3')` coerces away harmlessly, but the string `'false'` is **truthy**, so the demo
applies the naive `Boolean(args.celsius)` a caller would write and renders the answer in
whichever unit results — a wrong type becomes a wrong temperature rather than a log note.

Both need the model to actually emit the call: the demo substitutes a correctly-typed
stand-in when parsing yields nothing, so a run reporting "No usable tool call" has not
tested the parser.

Both passed, and each transcript said more than the scenario asked.

**26 returned `"True"`** — Python's capitalization for a declared `boolean`. So xml is
worse than "types become strings": it is not a spelling `JSON.parse` accepts, and
`value === 'true'` is **false** for a value meant as true. Fold case when coercing. The
parser does not, because it has no schema and guessing spellings is the invention it
refuses elsewhere.

**25's gemma-4 transcript settled three things at once:**

```
> call:get_weather{celsius:true,city:Seoul}
--- parsed: get_weather — celsius: true (boolean), city: "Seoul" (string) ---
```

A boolean and a string from the same call, so the dialect's "only strings wrapped" rule
does real work. The model **reordered** the arguments against the declaration, which the
scanner absorbs by reading keys as it finds them. And `city:Seoul` displays unwrapped,
which by the parser's own rules is not a string at all — an unquoted non-literal returns
`undefined` and discards the call. It parsed, so `<|"|>` was in the raw text and stripped
only for display: the `onRawText` path confirmed by a value's type rather than by
inspecting the stream.

Scenarios 27 and 28 cover `templateArgs` in the other direction — the library defaults
`enable_thinking` to false, so passing true had never been exercised — and they took
**three readings, all wrong the same way**: no visible `<think>` block → "unobservable" →
"not separable" → separable, just not live. Each step read *absent from the decoded
stream* as *absent altogether*, which is trap 12 for the third time. Qwen3.5's `</think>`
is `special: false` and had always been in the stream; gemma-4's `<|channel>` pair is
special, and `onRawText` — already there for tool calls — already handled that. **The
tokenizer's `special` field settles this in one lookup; the visible output does not.**

Both pass now: Qwen3.5 splits live, gemma-4 redistributes once `onRawText` arrives. So
`REASONING` in `examples/main.js` names a mechanism per model rather than a delimiter, and
a model absent from it gets no split. MediaPipe is excluded explicitly — it matches
gemma-4's catalog substring but has neither the template nor the raw-token hook, the same
shared-entry trap as the tool format.

Writing the splitter surfaced one bug worth keeping: holding back seven characters as a
possible partial `</think>` loses them when the tag never comes, which is the ordinary
outcome when the cap lands inside the block. Hence the explicit flush.

A later refactor moved the processor call's argument list into `processorCallArgs`,
which is the one change in that pass that unit tests cannot fully close: they pin the
array's shape, not whether a real processor accepts it. Re-checked on `Qwen3-VL-2B`
(three arguments, vision only) and `gemma-4-E2B` (four, vision plus audio), covering
both branches. **A future change to that function needs the same two models**, since
no other pair exercises both arities.

Scenario 1 is not applicable to Chrome: its adapter limit is high enough that no
tensor in the models tried triggers the block. On Firefox the gate refused before
any weight download, naming the tensor and the `maxBufferSize` it exceeded. The
`subgroups` warning in scenario 2 appears on Firefox only, since Chrome has the
feature.

Scenario 4 covers the MediaPipe path only, and the qualifier is not a shortcut.
`fetchWithProgress` reports `fromCache`, which the MediaPipe adapter forwards to
`LoadOptions.onCacheStatus`. transformers.js caches as well, but it manages its own
storage instead of going through the loader, so this library has nothing to observe
there and says so rather than guessing.

Scenarios 0, 6, and 7 used an earlier text model whose weights are embedded in the
`.onnx`: the size came from the file length rather than an external-data index,
leaving the largest tensor unknowable, and answers were short and coherent with no
`<think>` block, confirming `enable_thinking: false` reaches the chat template. The
current example has sidecar weights, so scenario 6 needs an embedded-weight model to
re-run — `Qwen3-1.7B` is one, though it is now blocked by the wasm-heap warning.

Scenarios 8 and 10 used `gemma-4-E2B` on Chrome, which is the only verified vision
model and is unavailable on Firefox. A vision model routes even a text-only turn
through its processor, since `apply_chat_template` alone leaves the pixel tensors
null.

Scenarios 18 and 19 used `Qwen3.5-0.8B-Text`, now the default example: its XML
tool-call encoding parsed and the three-turn cycle completed, and a non-English
prompt was answered in its own language rather than drifting to English. Both were
reasons it replaced the earlier text model.

Scenario 19b exists because 19 passing did not mean the output was good. A short,
factual prompt answered fine while a conversational one collapsed into the same phrase
repeated to the token cap — this export ships no sampling parameters, so it was
decoding greedily. The example now passes the values from Qwen's card, and the
collapse is gone.

**Residual, and expected:** some prompts still draw verbose or meandering answers,
just far less severely. That is a 0.8B model at `temperature: 1.0`, not the same
defect — the greedy failure was a single phrase repeated verbatim to the cap, which no
longer occurs. If it reappears in that exact form, suspect the `overrides` are not
reaching `generate()` rather than the model. Callers wanting tighter output can lower
`temperature` or add `repetition_penalty`; the example keeps the card's values instead
of tuning for the demo, so the defaults stay honest.

Test this with a casual, open-ended prompt rather than a task-shaped one. The failure
only appeared when the model had latitude in how to answer, which is why scenario 19
missed it.

Scenarios 16, 17 and 20 cover tool calling, and 20 exists because it used to fail.
`Qwen3-VL-2B` emitted a well-formed `<tool_call>` and `parseToolCalls` read it, but
feeding the result back produced nothing on the third turn, while the same cycle
completed on text-only models.

**That is fixed, and it was never a hang.** transformers.js 4.2.0 computes mrope
position deltas with a `+` where its own multimodal branch uses a `-`
(`modeling_qwen2_vl.js:37` against `:257`), which for a text-only turn yields
`2 × seq_len` instead of `0`. Every decode step then lands at roughly three times the
prompt length, far outside the trained position range, so EOS never arrives and
generation runs to the token cap. It looked like a hang because the cap is 1024
tokens; a smaller cap would have returned an empty answer instead.

The adapter corrects that delta at load time for `qwen2_vl` / `qwen2_5_vl` /
`qwen3_vl`, and the three-turn cycle now completes on both browsers. Turns carrying an
image are untouched, since those take the correct branch already. `gemma-4` never
showed this at all — it is a vision model that does not use mrope, so the relevant
category was never "vision".

Note the delta is recomputed for **every** generated token, so correcting it once per
call is not enough: an earlier attempt that passed `position_ids` into `generate()`
fixed the first token only, and the symptom changed from empty output to one word
repeated to the cap. The fix has to sit where the value is produced.

Scenarios 21 and 22 are **negative** cases: ✅ means each failed for the documented
reason, not that the modality worked. The container headers explain both — the web
builds hold a text decoder only. Each is loaded with one option (`maxImages` or
`supportAudio`), and both fail the same way: `load()` succeeds, then
`Llm{Vision,Audio}InferenceCalculator` reports "… models could not be created" at
generate. They are separate rows because the two calculators are separate code paths, and
both reproduce identically on both browsers — so the timing is a property of the container
rather than of one engine. The example page keeps these deliberately-failing entries so a
future attempt does not read the failure as a regression here.

**Row 22c is what makes the other two mean anything.** A text-only prompt on those same
engines still generates normally, so the option is not breaking the session and the failure
is scoped to the turn actually carrying an image or audio. Without that control, "MediaPipe
threw" would be equally consistent with the option corrupting the load, and the rewritten
error message would be naming the wrong cause.

Row 22 previously read "refused at load time" and was **wrong twice**: the load-time
message was seen once, written down as what `supportAudio` always does, then revised to
"both options together" when audio alone proved to fail at generate. That did not reproduce
either. **Both versions were one observation restated as a rule.** The regex in
`explainModalityFailure` stays — it was really seen and costs nothing — but its trigger is
unknown, and there is deliberately no example entry claiming to test it.

Scenario 15 used `Qwen3-VL-2B` and supersedes the earlier claim that image input is
Chrome-only. That was a property of `gemma-4`'s 1120MiB tensor, not of vision
models: this one's largest is 148MiB, well under Firefox's 1024MiB limit.

Scenarios 11, 12, and 14 used the same model, the only one here with an audio
encoder. In 12 the answer addressed both the picture and the speech, and compared
them as the prompt asked. Scenario 14 confirms only that the warning fires — that
the tail is genuinely ignored is read from the model's config, not measured.

