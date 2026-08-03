# Audio input — design

Date: 2026-07-30

Extends the image-input work (`2026-07-30-image-input-design.md`) so a model can
also be asked about a sound.

## Goal

Accept audio input on the transformers.js path for models that declare an audio
encoder, and refuse it clearly everywhere else. Chrome-only in practice, since
gemma-4-E2B is the only such model and Firefox rejects it on the tensor limit.

## Why this is mostly plumbing already

The image work laid most of the groundwork:

| Already present | Consequence |
|---|---|
| `ContentPart` has an `audio` variant | The public type does not change shape |
| `collectAudio()` with tests | Gathering works |
| `spec.audio`, read from `config.json`'s `audio_config` | Capability detection works |
| `processorTakesAudioArg`, the 4-argument processor branch | gemma-4's `_call(text, images, audio, options)` is already handled — it currently passes `null` |
| `hasNonTextParts` in the MediaPipe adapter | Audio is already refused there, with a correct message |

What remains is decoding, the gate option, and replacing a `throw` with real work.

## Public API

The `audio` part accepts either form, mirroring how `image` accepts a URL or a Blob:

```ts
export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; src: string | Blob }
  | { type: 'audio'; src: Blob }              // library decodes
  | { type: 'audio'; data: Float32Array };    // already 16kHz mono
```

Accepting both matters because decoding needs `AudioContext`, which exists only on
the main thread. A caller who later moves inference into a worker can decode
outside and pass `data`; everyone else passes `src` and ignores the problem.

`collectAudio()` changes from returning `Float32Array[]` to returning the audio
**parts**, because deciding whether to decode belongs to the adapter.

## Decoding

A new `src/audio.ts` holds everything that touches `AudioContext`:

```ts
export const AUDIO_SAMPLE_RATE = 16000;
export const AUDIO_MAX_SECONDS = 30;

export function downmixToMono(channels: Float32Array[]): Float32Array;
export async function decodeAudio(blob: Blob): Promise<Float32Array>;
export function warnIfTooLong(samples: Float32Array): void;
```

`decodeAudio` resamples to 16kHz via `new AudioContext({ sampleRate })` and
downmixes stereo with a `1/sqrt(2)` factor — the same factor transformers.js uses
in `load_audio`, so both paths produce identical samples. This code was verified in
an earlier prototype.

Keeping it in its own file draws the main-thread boundary explicitly, and lets
`downmixToMono` be unit-tested without a browser.

### The 30-second limit

gemma-4 clamps audio at `audio_seq_length: 750` × `audio_ms_per_token: 40` = 30
seconds. It does not error; the tail is simply ignored, so a user gets an answer
that silently omits part of the input.

`warnIfTooLong` logs a warning and passes the samples through unchanged. It does
not truncate, because truncating would make the library responsible for choosing
which 30 seconds matter.

`AUDIO_MAX_SECONDS` is a constant, **not read from `config.json`**. Other models may
differ, but gemma-4 is currently the only audio model, and warning with a wrong
number would be worse than a comment stating where 30 came from.

## Adapter

Replaces the current `throw` in `src/engine/transformers.ts`:

```ts
const audioParts = collectAudio(messages);

if (audioParts.length > 0 && !processorTakesAudioArg) {
  throw new RuntimeError(
    'This model has no audio encoder, so it cannot accept audio input.',
  );
}

const audio = await Promise.all(
  audioParts.map((p) => ('data' in p ? p.data : decodeAudio(p.src))),
);
audio.forEach(warnIfTooLong);

const audioArg = audio.length > 0 ? audio : null;
inputs = processorTakesAudioArg
  ? await call(prompt, imageArg, audioArg, processorOptions)
  : await call(prompt, imageArg, processorOptions);
```

The 4-argument branch already exists; only the `null` becomes real data.

One condition changes. The processor path is currently entered on `isVision`
alone, which would miss an audio-only model:

```ts
if (isVision || processorTakesAudioArg) {
```

gemma-4 reports both, so this is not currently observable — it is corrected
because the existing condition is wrong, not because a model needs it today.

## Gate

`wantsAudio` joins `wantsImage`, same shape:

```ts
if (opts.wantsAudio && !spec.audio) {
  blockers.push(
    'This model has no audio encoder, so it cannot accept audio input. ' +
      'Pick a model whose config.json declares audio_config.',
  );
}
```

Two separate flags rather than one, because a model can accept images and refuse
audio — SmolVLM was exactly that, and the distinction should stay expressible.

MediaPipe needs no change: `hasNonTextParts` already refuses any non-text part and
names the runtime as the reason.

## Testing

| Target | Assertions |
|---|---|
| `collectAudio` | returns parts (shape change); handles mixed Blob and Float32Array; empty when absent |
| `downmixToMono` | mono passthrough; stereo scaled by `1/sqrt(2)`; length preserved |
| `warnIfTooLong` | warns above 30s, silent at or below |
| `gate` | `wantsAudio` blocks when `spec.audio` is false, passes when true |
| adapter rules | source check that `audioArg` reaches the 4-argument call |

**`decodeAudio` is not unit-tested.** `AudioContext` does not exist in Node, and
mocking it would verify nothing about resampling or downmixing while creating the
appearance of coverage. `downmixToMono` is split out precisely so the part that
*can* be tested is.

## Example

An audio picker sits beside the image picker, shown only when `spec.audio` is true
— in practice only the gemma-4 ONNX entry.

The example passes a **Blob**, exercising the library's decoding path rather than
the pre-decoded one. Between two paths, the example should cover the less obvious.

## Out of scope

Microphone capture (file selection only), workers, video, audio generation.

## Verification

Chrome only, since gemma-4 is the only audio model:

1. Audio + text — the answer reflects the speech
2. Image + audio + text — both are reflected
3. Audio longer than 30s — a console warning appears
4. A model without `audio_config` selected — the audio picker stays hidden
5. Text only — no regression
