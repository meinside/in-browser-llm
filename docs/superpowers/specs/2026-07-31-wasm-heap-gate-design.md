# Gating the wasm-heap allocation limit — Design

Date: 2026-07-31

## Goal

The gate checks the largest single *tensor* against WebGPU's `maxBufferSize`. That
misses a second failure mode with a different cause: a model whose weights are
embedded in the `.onnx` must be buffered whole into the wasm heap, and a large
enough file fails there with `std::bad_alloc`. Add a check for it.

## The failure

`onnx-community/Qwen3-1.7B-ONNX` (q4f16) on Firefox 153 / macOS:

```
Error: Can't create a session. ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc
```

The gate cleared it. The download completed. Then it failed — the exact shape of
failure this library exists to prevent.

### Why the existing check does not fire

These models embed their weights in the `.onnx` rather than a sidecar
`.onnx_data`, so there is no `external_data` index and `largestTensorBytes` reads
0. `gate()` warns that the tensor check was skipped and passes.

### What actually happens

Verified in the shipped onnxruntime-web:

- `lib/wasm/wasm-core-impl.ts` does `const modelDataOffset = wasm._malloc(model.byteLength)`
  followed by `wasm.HEAPU8.set(model, modelDataOffset)`. An embedded file needs
  **one contiguous allocation of the whole file**, and ORT then allocates the
  initializers again while parsing the protobuf — a peak of roughly twice the file.
- External data avoids that path entirely: bytes stay in a JS-side map and wasm
  copies per-tensor slices.
- The heap this happens inside is wasm32: `new WebAssembly.Memory({initial:256,
  maximum:65536, shared:true})` — 4GiB — with growth refused above `4294901760`
  (4GiB−64KiB). Both values confirmed in `dist/ort-wasm-simd-threaded.jsep.mjs`.

`std::bad_alloc` rather than ORT's own "failed to allocate a buffer of size N"
means the `_malloc` succeeded and the parse-time allocation is what failed.

### The measurements that identify the discriminator

All q4f16, same browser and machine:

| model | largest single file | weights | result |
|---|---|---|---|
| Qwen3-0.6B | 543MiB | embedded | works |
| Qwen3-VL-2B | 949MiB | external, 3 sessions | works |
| Qwen3-1.7B | 1360MiB | embedded | `std::bad_alloc` |

Total download does not separate these — VL-2B totals ~1308MiB and works. **How the
weights are stored does.**

## What to measure

**The largest single file, not the total.** `ModelSpec.totalBytes` sums every
session, but `_malloc` happens per session, so a summed value overstates the
constraint for a multi-session model. Qwen3-1.7B has one session, which is why the
two numbers coincided there.

`inspectOnnx` already distinguishes the two cases — the branch taken when
`header.tensors.length === 0` is the embedded one, and it already fetches
`contentLength` there. Today that number is added to the total and otherwise
discarded. Return its maximum as well.

`ModelSpec` gains:

```ts
/**
 * Largest single .onnx that must be buffered whole, or 0 when weights live in
 * sidecar .onnx_data files. ORT mallocs an embedded file contiguously, then
 * duplicates it while parsing the protobuf; external data is copied per tensor.
 */
largestEmbeddedFileBytes: number;
```

## Decision

| Condition | Verdict | Basis |
|---|---|---|
| `largestEmbeddedFileBytes >= 2GiB` | block | ONNX protobuf caps a message at 2GB; Chrome caps `ArrayBuffer` at 2,145,386,496 bytes. Both documented. |
| `> 1GiB` | pass, with warning | Measured: 949MiB loaded, 1360MiB did not. |
| `=== 0` | no check | External data never takes the contiguous path. |

**The 1GiB threshold is arbitrary and the code says so.** It is a round number
between the only two observations, not a documented limit — no ONNX Runtime or
transformers.js source names a per-file size above which embedded weights fail, and
1360MiB is below *every* documented ceiling (protobuf 2GB, Chrome ArrayBuffer
2046MiB, wasm32 4GiB). The real cause is fragmentation plus parse-time duplication,
which no source quantifies. That uncertainty is why this tier warns rather than
blocks: a hard refusal at an invented number would produce false negatives, which is
the failure the library just spent a session removing from the tensor check.

The warning names the mechanism — a contiguous allocation plus a parse-time copy
inside a 4GiB wasm32 heap — and points at the documented remedy: an export whose
weights sit in `.onnx_data`.

## Testing

`gate()` is pure, so these are Vitest cases:

1. An embedded file at or above 2GiB blocks, and the message names the protobuf
   limit.
2. **1360MiB passes with a warning.** The real failure sits here; the case fixes
   that warn-not-block is deliberate.
3. **949MiB with `largestEmbeddedFileBytes: 0` produces no warning.** The control
   that passed. This is the case that pins "storage layout, not total size" — a
   threshold applied to `totalBytes` would flag it wrongly.
4. 949MiB embedded produces no warning either, being under 1GiB.

`inspect.test.ts` gains one case: a session with no external tensors records its
file length in `largestEmbeddedFileBytes`, and a session with external tensors
leaves it 0.

### What tests cannot show

They prove the decision logic. Whether 1GiB is the *right* threshold would need
models at several sizes downloaded and tried, which was not done. The browser check
is narrower: that Qwen3-1.7B now warns **before** the download rather than failing
after it.

## Documentation

- **CLAUDE.md** — a new trap entry. This one fails silently and names nothing about
  its own cause, so record the citable evidence (`_malloc(model.byteLength)` in
  `wasm-core-impl.ts`, `maximum:65536` in the glue, the protobuf 2GB limit) together
  with the 949-passes/1360-fails control data. Without the evidence the next person
  re-derives it.
- **README, "Why the gate matters"** — it currently explains only the tensor limit.
  Add the second mode and keep the two distinct: one is a GPU buffer binding, the
  other a wasm heap allocation, and a model can hit either alone.
- **`scripts/check-model.mjs`** — print whether each session embeds its weights and
  how large the file is, and emit CHECK above 1GiB. Catching this from headers is
  the whole point of that script.
