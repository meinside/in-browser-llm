# Image input — design

Date: 2026-07-30

> **Superseded on audio.** This spec reserved the audio part type and specified
> that it throws. `2026-07-30-audio-input-design.md` implemented it; audio input
> works. Everything else here still holds.

Extends `in-browser-llm` so a model can be asked about an image, not only text.

## Goal

Accept image input on the transformers.js path, and refuse it clearly everywhere
else. The example aimed for a model small enough for both Chrome and Firefox; the
only vision model that actually works is Chrome-only (see the correction below).

## Why this is not just an example change

`Message.content` is `string`, so there is no way to attach an image today. The
transformers adapter also loads `AutoTokenizer` + `AutoModelForCausalLM`, which
cannot process pixels. Both are public-facing, so this is a library change with an
example attached rather than the reverse.

## Verified constraints

Checked directly against `@huggingface/transformers` 4.2.0 and the HF API.

| Fact | Consequence |
|---|---|
| No processor accepts video — `grep video src/models/*/processing_*.js` is empty across all 35 processors, and `pixel_values_videos` is commented out in Qwen2-VL | Video is out of scope, not merely unimplemented |
| No `stablediffusion`/`unet`/`latentdiffusion` in the model registry | Image *generation* is impossible here |
| gemma-4 processor is `_call(text, images, audio, options)` | 4 positional arguments |
| SmolVLM (idefics3) processor is `_call(text, images, options)` | 3 positional arguments — passing 4 puts audio where options belong |
| `SmolVLMForConditionalGeneration` is in the registry | The model class exists |
| `HuggingFaceTB/SmolVLM-256M-Instruct` q4f16 totals **180MB** (73 + 54 + 52), Apache-2.0, ungated | Looked like a fit — but see the correction below |

**SmolVLM was dropped during implementation.** Three defects surfaced, none
visible from its size or config:

1. `vision_encoder_q4f16` (and `fp16`) is a malformed graph — it fails on the CPU EP
   with a `SimplifiedLayerNormFusion` cast error and, on WebGPU, loads while
   emitting meaningless embeddings.
2. Its processor reads `image_inputs.rows` unconditionally, so it cannot be called
   for a text-only turn.
3. Its chat template indexes `message['content'][0]['type']`, so string content
   breaks the template loop.

Only (3) generalizes, and the adapter handles it. Working around (1) and (2)
required per-model special cases for a payoff of a smaller example, so gemma-4 —
already verified for image input, Chrome-only — remains the only vision model.

An earlier theory blamed `transformers.js_config.kv_cache_dtype`. That was wrong:
Qwen3-0.6B and gemma-4 declare the same field and work fine at q4f16.

## Public API

`content` becomes a union. Existing string usage keeps working unchanged.

```ts
export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; src: string | Blob }
  | { type: 'audio'; data: Float32Array };

export interface Message {
  role: 'user' | 'assistant';
  content: string | ContentPart[];
}
```

`audio` is declared now because gemma-4 supports it and reserving the slot avoids
a breaking change later. **It is not implemented in v1**: an audio part throws
rather than being silently dropped. A type that exists without behaviour behind it
is worse than no type at all.

## Detecting what a model accepts

`inspectModel` fetches `config.json` (a few KB) and reads `architectures`:

```ts
interface ModelSpec {
  // …existing fields
  vision: boolean;   // needs AutoProcessor + AutoModelForImageTextToText
  audio: boolean;    // accepts audio input
}
```

This reverses an earlier decision. The original spec said `config.json` would be
read to pick dtype session keys, and that proved unnecessary — a single dtype
string works regardless of layout. **Model class is different: it cannot be
guessed.** `AutoTokenizer` and `AutoProcessor` are distinct objects, and choosing
wrong fails at load.

The reversal pays for itself elsewhere: sending an image to a text-only model is
caught by `gate()` before the download, not at `generate()` afterwards.

When `config.json` is unavailable, both flags are `false` — the conservative
choice, since it declines image input rather than attempting an unsupported path.

## Adapter structure

The transformers adapter branches once, at load:

```
spec.vision || spec.audio → AutoProcessor  + AutoModelForImageTextToText
otherwise                 → AutoTokenizer  + AutoModelForCausalLM
```

Both expose `apply_chat_template`, so prompt construction stays shared.

Processor arity is resolved by whether audio is present:

```ts
const inputs = audio
  ? await processor(prompt, images, audio, { add_special_tokens: false })
  : await processor(prompt, images, { add_special_tokens: false });
```

One branch covers both families. A model without audio support never reaches the
4-argument form, because `spec.audio === false` rejects the input earlier.

Images load through `load_image`. A `Blob` is converted with
`URL.createObjectURL` and revoked after use. Unlike audio, images need no
main-thread-only API, so this would work in a worker — but no worker is introduced
here; that is separate work.

## Refusing unsupported input

Three layers, each naming its own reason:

1. **`gate()`** — image requested against a model whose `spec.vision` is false.
   Caught before downloading. Note `gate()` receives the spec, not the messages,
   so this requires the caller to state intent; the example passes it, and callers
   who do not simply fall through to layer 2 or 3 at generate time.
2. **MediaPipe adapter** — any non-text part throws `CapabilityError`: the
   runtime is text-only and cannot see images. MediaPipe specs always report
   `vision: false`, since `.task` files expose no `config.json`.
3. **`formatPrompt`** — accepts array content but concatenates text parts only,
   and throws on anything else. This is the backstop for callers who skip the gate.

Layer 3 matters because silently dropping an image produces an answer that
ignores it, and the user cannot tell why. That failure mode is exactly what this
library exists to prevent.

## Testing

Testable as pure functions, which is where this change's risk concentrates:

| Target | Assertions |
|---|---|
| `formatPrompt` | string content unchanged (regression); text parts concatenated; throws on image part |
| part normalization | string content normalized to parts; text/image/audio separated |
| `inspectModel` | `vision`/`audio` derived from `architectures`; both false when `config.json` is missing |
| `gate` | blocks an image request against a text-only model |

Not testable here: the actual processor call and inference. Whether the arity
branch is correct shows up only in a browser. To compensate,
`scripts/check-model.mjs` gains `architectures` plus the detected vision and audio
flags, so a new model's shape is visible before any browser run.

## Example

The example shows a file picker only when the selected model reports `vision`.
Letting the UI offer a combination the library rejects would be a worse
demonstration than hiding it. In practice that means only the gemma-4 ONNX entry
exposes it.

The gemma-4 ONNX entry carries this. It is Chrome-only, but it is the only
verified image path.

## Out of scope for v1

Audio input (type slot reserved, throws if used), video (no processor path
exists), image generation (no diffusion models in the registry), audio generation
(needs a separate non-streaming API — decide later), workers.

## Verification

Browser-only, as before:

1. gemma-4 + image on Chrome — answer reflects the image
2. gemma-4, text only, on Chrome — the vision path handles a turn with no image
3. Image sent to a text-only model — error names the reason, before download
4. Image sent to a MediaPipe model — `CapabilityError`, not a silent text answer
