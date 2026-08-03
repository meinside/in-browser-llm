# Audio-only models, and a Firefox-capable vision model — Design

Date: 2026-07-31

> **Outcome.** The vision half shipped: `Qwen3-VL-2B` is verified on Chrome and
> Firefox, making image input work on Firefox for the first time. The audio half
> did **not**: Ultravox was dropped after failing in a third layer this design did
> not anticipate — its export declares `"processor_class": "WhisperProcessor"`, and
> `AutoProcessor` trusts that field, so `UltravoxProcessor` is never constructed.
> The library changes remain, unverified, and `scripts/check-model.mjs` now detects
> this class of broken export up front. See CLAUDE.md, "Audio without vision does
> not work yet".

## Goal

Two changes that share a verification pass:

1. Add `Qwen3-VL-2B-Instruct` to the examples. **No library change** — it is the
   first vision model whose tensors fit Firefox's buffer limit.
2. Support **audio-only** models (`vision: false, audio: true`) so `Ultravox-1B`
   loads. This needs a library change: one model class and one processor call
   shape.

## Why an audio-only model fails today

`src/engine/transformers.ts` lets one flag decide two things. `isVision` selects
both the processor-vs-tokenizer and the model class, so a model with audio but no
vision gets a tokenizer and `AutoModelForCausalLM`. `ultravox` is absent from that
registry mapping, so `from_pretrained` throws `Unsupported model type: ultravox`.

Verified: `node_modules/@huggingface/transformers/src/models/registry.js:408-413`
registers exactly four `MODEL_FOR_AUDIO_TEXT_TO_TEXT` entries — `granite_speech`,
`ultravox`, `voxtral`, `voxtral_realtime` — and none of them appears in the
CausalLM or ImageTextToText mappings. `AutoModelForAudioTextToText` exists
(`models/auto/modeling_auto.js:382`) and is in the public type surface via
`types/transformers.d.ts:3`.

The two decisions must separate:

| Needed | Condition |
|---|---|
| processor rather than tokenizer | `vision \|\| audio` — audio needs feature extraction too |
| model class | `vision` → ImageTextToText; `audio && !vision` → AudioTextToText; else CausalLM |

`vision && audio` (gemma-4) stays on ImageTextToText: it is registered there and
that path is already verified. This change must not alter it.

## Processor argument order differs three ways

This is the trap that matters, and it extends what CLAUDE.md already records.
Processor arity is fixed per model family — but so is argument *order*, and the
current code keys the choice on `spec.audio` alone, which cannot express the third
case.

| spec | signature | evidence |
|---|---|---|
| vision + audio (gemma-4) | `(text, images, audio, options)` | current code, verified in browser |
| vision only (Qwen3-VL) | `(text, images, ...args)` | `models/qwen2_vl/processing_qwen2_vl.js:18`, inherited by qwen3_vl |
| **audio only (Ultravox)** | **`(text, audio, kwargs)`** | `models/ultravox/processing_ultravox.js:17` |

So for Ultravox the audio goes in the **second** position, where gemma-4 puts
images. Keying on `spec.audio` would call
`(prompt, null, audioArg, options)` — audio in the wrong slot and options past the
end of the signature. The selection becomes:

```ts
inputs = isVision
  ? processorTakesAudioArg
    ? await call(prompt, imageArg, audioArg, processorOptions)  // gemma-4
    : await call(prompt, imageArg, processorOptions)            // Qwen3-VL
  : await call(prompt, audioArg, processorOptions);             // Ultravox
```

The `else` branch is reached only when `needsProcessor && !isVision`, which is
audio-only by construction.

## Why Qwen3-VL-2B needs no code change

- `config.json` declares `vision_config` and no `audio_config`, so
  `inspectModel()` reports `vision: true, audio: false` — the existing
  three-argument path.
- Largest tensor **148MiB** against Firefox's 1024MiB `maxBufferSize`. gemma-4's is
  1120MiB, which is why it is refused there.
- **Zero `Clip` op nodes** in all six ONNX files, so the Firefox f16 shader bug
  does not apply. Counted by matching the protobuf pattern for
  `NodeProto.op_type`, not by substring.
- `q4f16` and `q4` present for every session, so the single-string dtype resolves.
  A missing variant is what disqualified gemma-3n.
- Text-only turns are guarded inside the processor, so it does not throw the way
  SmolVLM's did.

## Changes

### `src/engine/transformers.ts`

Split the flags: add `hasAudio` and `needsProcessor = isVision || hasAudio`, and
use `needsProcessor` to pick processor vs tokenizer. Select the model class into a
variable rather than duplicating the option object across three branches, since
`device`, `dtype`, and `progress_callback` are identical:

```ts
const ModelClass = isVision
  ? AutoModelForImageTextToText
  : hasAudio
    ? AutoModelForAudioTextToText
    : AutoModelForCausalLM;
```

Then apply the three-way processor call above. Import
`AutoModelForAudioTextToText` alongside the existing classes.

### `examples/main.js`

Two `REFS` entries and two `<option>`s in `examples/index.html`:
`onnx-community/Qwen3-VL-2B-Instruct-ONNX` and
`onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX`. The image and audio pickers are
already driven independently by `spec.vision` and `spec.audio`, so Ultravox shows
only the audio picker with no further wiring.

### Unchanged

`src/gate.ts` — `wantsImage` and `wantsAudio` are already checked independently, so
an audio-only model correctly refuses an image. `src/catalog.ts` — it supplies turn
delimiters for the MediaPipe path; the transformers path uses the Jinja template.

## Testing

Engines are not unit-tested: they need a browser with a WebGPU adapter, which CI
cannot provide. `test/dtype.test.ts` already asserts against the adapter's source
text for decisions that would otherwise fail silently, and the two below qualify —
swapping the model classes, or handing an audio-only model a tokenizer, produces a
runtime error far from its cause.

1. The model-class selection contains all three branches, with `AudioTextToText`
   guarded by audio and `ImageTextToText` by vision.
2. The processor is chosen on `vision || audio`, not on vision alone.

> **Superseded on 2026-08-03.** Both were implemented as described, then the tests
> were rewritten: the source-text assertions became behaviour tests over extracted
> pure functions (`selectModelClass`, `processorCallArgs`, `normalizeContent`), and
> `test/dtype.test.ts` is now `test/transformers.test.ts`. The reasoning above about
> *what* needs covering still holds; the mechanism does not. See CLAUDE.md,
> "Do not declare completion unilaterally".

Gate behaviour for an audio-only spec needs no new test: the existing
`wantsImage`/`wantsAudio` cases cover it.

### Browser verification

| # | Scenario | Browsers |
|---|---|---|
| 15 | Qwen3-VL answers about an image | Chrome and **Firefox** |
| 16 | Gate passes Qwen3-VL on Firefox | Firefox |
| 17 | Ultravox answers about supplied audio | Chrome, Firefox |
| 18 | Image picker hidden for an audio-only model | both |

Scenario 15 is the point of the vision half: whether image input works on Firefox
at all.

### Two risks that header inspection cannot settle

- **A quantized vision encoder can load and emit nonsense.** SmolVLM did exactly
  that — real subwords in nonsense order. Only asking about an image reveals it.
  Related and unresolved: `vision_config` carries eight
  `deepstack_visual_indexes`, and grepping transformers.js for `deepstack` finds
  nothing, while the exported encoder emits a single `image_features` output. The
  export appears to have folded it in, but that is inference, not verification.
- **Ultravox's chat template may not tolerate the parts normalization** applied
  before rendering. Vision templates require it; an audio template may take a
  plain string.

If either model fails, drop that model. The library change stands on its own — it
is what makes any audio-only model loadable.
