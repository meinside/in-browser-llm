# Correcting the tensor-size gate — Design

Date: 2026-07-31

## Goal

The gate blocks a model when its largest tensor exceeds
`maxStorageBufferBindingSize`. That check is wrong in two ways: it names a limit
that ONNX Runtime now works around, and it misses the limit that actually fails.
Replace it with a check against `maxBufferSize`, and warn — rather than block —
in the window between the two.

## What changed underneath

ONNX Runtime PR #25962 ("[webgpu] Split large inputs into smaller buffers to
bypass maxStorageBufferBindingSize limit", merged 2025-10-09, `535f1f37c4`) binds
an oversized tensor as several bind-group entries over one buffer, and rewrites
the generated WGSL to index through a `get_<name>_by_offset` accessor.

Three facts about it were verified before designing this, because the project's
notes had assumed the opposite:

- **It is already in the version we use.** `535f1f37c4` is an ancestor of
  `v1.26.0` and of the exact dev commit transformers.js 4.2.0 pins
  (`b7804b056c`, `behind_by=0`). The previous note claimed the pin was what held
  the feature back; the feature predates the pin by about six months. No upgrade
  is available anyway — the pin is exact, not a caret, and 4.2.0 is the latest
  release.
- **It ships in the binary that is loaded.** `CHUNK_SIZE_IN_ELEMENTS` appears in
  `ort-wasm-simd-threaded.asyncify.wasm` and the `jspi` build, and not in `jsep`
  or the default build. transformers.js loads the asyncify build.
- **It is on by default.** `program_manager.cc` reads the device's real limit;
  the `maxStorageBufferBindingSize` session option only forces a *smaller* value
  for testing. No caller action is needed.

`GatherBlockQuantized` — the op holding gemma-4's oversized tensor — indexes
through the rerouted accessors, so it is on the covered path.

**Segmentation splits bindings, not allocation.** `buffer_manager.cc` still makes
one `CreateBuffer` call for the whole tensor. So `maxBufferSize` is the limit that
genuinely cannot be worked around.

## Why the verdict for gemma-4 on Firefox does not change

Measured on Firefox 153 / macOS Apple Silicon: `maxBufferSize` and
`maxStorageBufferBindingSize` are **both 1024MiB**. gemma-4's 1120MiB tensor
exceeds both, so it is still blocked — correct verdict, wrong stated reason.

The window between the two limits is empty on that adapter. It is not empty in
general: the WebGPU defaults are 256MiB for `maxBufferSize` against 128MiB for
`maxStorageBufferBindingSize`, a 2× gap, and mobile adapters are expected to
report values in that range. On any such adapter the current gate refuses models
that would load.

A related worry was ruled out: WebGPU grants a device only default limits unless
`requestDevice` is given `requiredLimits`, which would have made the adapter's
1024MiB irrelevant. ORT's `webgpu_context.cc` does pass `requiredLimits`, asking
for the adapter's maximum for both limits, and the shipped
`ort.webgpu.bundle.min.mjs` contains `requiredLimits` and all four limit names.
Comparing against adapter-reported values is therefore correct.

## Decision model

Let `N` be `spec.largestTensorBytes`.

| Condition | Verdict | Reason |
|---|---|---|
| `N > maxBufferSize` | block | one allocation must hold the tensor; nothing splits it |
| `maxStorageBufferBindingSize < N ≤ maxBufferSize` | pass, with warning | ORT segments the bindings |
| `N ≤ both` | pass | — |

The middle case warns rather than passing silently because two conditions remain
unverifiable from here:

- The segment count must not exceed `maxStorageBuffersPerShaderStage`, and total
  bindings must stay under `maxBindingsPerBindGroup`. Both are enforced across
  *all* variables in a program, so a per-tensor check cannot predict them.
- `maxBufferSize` is compared against a normalized allocation size, which bucket
  rounding can push above `N`. A tensor just under the limit may still fail.

Neither is worth modelling: doing so needs numbers the ONNX headers do not carry.
The warning states that the tensor exceeds the binding limit, that ORT is
expected to split it, and that this combination has not been verified in a
browser.

MediaPipe is unaffected. Its `.task` files expose no tensor sizes, so
`largestTensorBytes` is 0 and neither branch fires.

## Changes

### `src/gate.ts`

Split the single check into block and warn branches as tabled above. The blocker
message names `maxBufferSize`, explains that segmentation covers bindings but not
allocation, and keeps the existing note that falling back to wasm does not help
(the WebGPU wasm build has no CPU kernel for quantized gather).

Extend the adapter-absent guard to cover `maxBufferSize === 0`. It currently
tests only `maxStorageBufferBindingSize`; without this, a zero limit would make
the new check block every model and report it as a size problem.

### `src/errors.ts`

Delete `TensorLimitError` and its export from `src/index.ts`. It carries the same
incorrect explanation, and nothing throws it — `gate()` returns structured
`blockers` and the loader does not use the class. Correcting a dead type would
mean maintaining a second copy of this reasoning. Removing it is free now and
would be a breaking change after the first tag.

### `src/capability.ts`

No behaviour change; both limits are already probed. Add a comment recording that
the two are consumed differently, since the field names alone do not convey that
one is a hard ceiling and the other is a segmentation threshold.

### `examples/main.js`

Show `maxBufferSize` alongside the binding limit in the gate output. The value
that decides a block is currently absent from the page, which makes a refusal
impossible to interpret.

## Testing

`gate()` is pure and holds the riskiest decision in the library, so these are
Vitest cases:

1. `N > maxBufferSize` blocks, and the message names `maxBufferSize`.
2. **A tensor in the window passes with a warning.** Needs a new fixture whose
   two limits differ — 256MiB `maxBufferSize` against 128MiB
   `maxStorageBufferBindingSize`, the WebGPU defaults. The existing fixtures set
   both limits equal, so they cannot express this case. This is the regression
   guard against reintroducing the false negative.
3. `FIREFOX_153` with gemma-4 still blocks, confirming the verdict is unchanged
   where both limits are 1024MiB.
4. `maxBufferSize === 0` reports an absent adapter, independent of tensor size.

### What tests cannot show

Unit tests prove the logic, not the runtime. Confirming that a tensor in the
window actually loads requires an adapter whose limits differ, and Firefox/macOS
is not one. Until such a device is tested, the warning says the load is expected
to work, not that it does.

Firefox needs one browser re-run to confirm gemma-4 is still refused before any
weight download, now naming `maxBufferSize`.

## Documentation

- **CLAUDE.md** — remove the "Worth revisiting" item about the ORT version; its
  premise is refuted. Fold the correction into trap 1 under "Never forget these":
  the hard limit is `maxBufferSize`, the binding limit is segmented around, and
  the checks that establish this are the merge-commit ancestry and the wasm
  symbol counts. Recording the method matters as much as the conclusion, so the
  next person does not repeat the investigation.
- **README, "Why the gate matters"** — the observed gemma-4 failure on Firefox
  stands, but attribute it to the allocation limit. Keep the point that total
  size is not what fails first.
- **README verification log** — scenario 1's note reasons from the binding limit;
  correct it, and mark the scenario for re-verification under the new message.

