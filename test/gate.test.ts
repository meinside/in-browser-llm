import { describe, expect, it } from 'vitest';
import { gate } from '../src/gate.js';
import type { Capabilities } from '../src/capability.js';
import type { ModelSpec } from '../src/gate.js';

// Measured on macOS Apple Silicon M1 Pro.
const CHROME_150: Capabilities = {
  hasWebGPU: true,
  hasF16: true,
  knownF16ShaderIssues: [],
  hasSubgroups: true,
  maxStorageBufferBindingSize: 4294967292,
  maxBufferSize: 4294967292,
  adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
};

const FIREFOX_153: Capabilities = {
  hasWebGPU: true,
  hasF16: true,
  knownF16ShaderIssues: ["Firefox 153 fails ONNX Runtime's f16 Clip kernel"],
  hasSubgroups: false,
  maxStorageBufferBindingSize: 1073741824,
  maxBufferSize: 1073741824,
  adapterInfo: { vendor: '', architecture: '' },
};

const NO_WEBGPU: Capabilities = {
  hasWebGPU: false,
  hasF16: false,
  knownF16ShaderIssues: [],
  hasSubgroups: false,
  maxStorageBufferBindingSize: 0,
  maxBufferSize: 0,
  adapterInfo: { vendor: '', architecture: '' },
};

// WebGPU spec defaults: maxBufferSize is 2x the binding limit. Mobile adapters are
// expected to report values in this range. No fixture had differing limits before,
// so the segmentation window was untestable.
const DEFAULT_LIMITS: Capabilities = {
  hasWebGPU: true,
  hasF16: true,
  knownF16ShaderIssues: [],
  hasSubgroups: true,
  maxStorageBufferBindingSize: 134217728,
  maxBufferSize: 268435456,
  adapterInfo: { vendor: '', architecture: '' },
};

// gemma-4 ONNX: the 1120MiB tensor that broke Firefox.
const GEMMA4_ONNX: ModelSpec = {
  runtime: 'transformers',
  largestTensorBytes: 1174405120,
  totalBytes: 3565158400,
  largestEmbeddedFileBytes: 0,
  hasTurnFormat: true,
  vision: true,
  audio: true,
};

const GEMMA4_TASK: ModelSpec = {
  runtime: 'mediapipe',
  largestTensorBytes: 0,
  totalBytes: 2003697664,
  largestEmbeddedFileBytes: 0,
  hasTurnFormat: true,
  // The LiteRT web build is text-only even though the model supports more.
  vision: false,
  audio: false,
};

describe('gate', () => {
  it('blocks when WebGPU is absent', () => {
    const verdict = gate(NO_WEBGPU, GEMMA4_ONNX);
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/WebGPU/);
  });

  it('passes gemma-4 ONNX on Chrome', () => {
    expect(gate(CHROME_150, GEMMA4_ONNX).ok).toBe(true);
  });

  it('blocks gemma-4 ONNX on Firefox, naming the allocation limit', () => {
    const verdict = gate(FIREFOX_153, GEMMA4_ONNX);
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/1120MiB/);
    expect(verdict.blockers.join(' ')).toMatch(/maxBufferSize/);
  });

  it('explains that segmentation cannot rescue an oversized allocation', () => {
    // ORT splits bindings but still allocates one buffer, and moving the session
    // to wasm fails too: the WebGPU wasm build has no GatherBlockQuantized kernel.
    const blockers = gate(FIREFOX_153, GEMMA4_ONNX).blockers.join(' ');
    expect(blockers).toMatch(/single (buffer|allocation)/i);
    expect(blockers).toMatch(/wasm/);
  });

  it('passes a tensor between the two limits, with a warning', () => {
    // 200MiB: over the 128MiB binding limit, under the 256MiB allocation limit.
    // ORT segments the bindings, so blocking here would be a false negative.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestTensorBytes: 209715200 };
    const verdict = gate(DEFAULT_LIMITS, spec);
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/maxStorageBufferBindingSize/);
  });

  it('blocks a tensor above the allocation limit even on a segmenting adapter', () => {
    // 300MiB: over both limits.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestTensorBytes: 314572800 };
    expect(gate(DEFAULT_LIMITS, spec).ok).toBe(false);
  });

  it('reports an absent adapter rather than a size problem when limits are zero', () => {
    // maxBufferSize 0 would otherwise make every tensor look oversized.
    const noAdapter: Capabilities = { ...NO_WEBGPU, hasWebGPU: true };
    const verdict = gate(noAdapter, GEMMA4_ONNX);
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/adapter/i);
    expect(verdict.blockers.join(' ')).not.toMatch(/MiB/);
  });

  it('warns about missing subgroups without blocking', () => {
    const verdict = gate(FIREFOX_153, GEMMA4_TASK);
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/subgroups/);
  });

  it('blocks MediaPipe when shader-f16 is missing', () => {
    // MediaPipe passes requiredFeatures: ["shader-f16"], so createDevice rejects.
    const noF16 = { ...FIREFOX_153, hasF16: false };
    const verdict = gate(noF16, GEMMA4_TASK);
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/shader-f16/);
  });

  it('allows transformers.js without shader-f16 but warns about the dtype downgrade', () => {
    const noF16 = { ...CHROME_150, hasF16: false };
    const verdict = gate(noF16, GEMMA4_ONNX);
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/q4/);
  });

  it('blocks MediaPipe when the catalog entry has no turn format', () => {
    // An entry may exist purely to carry a tool-call format. MediaPipe needs turn
    // delimiters specifically, so entry-exists is not the right question.
    const verdict = gate(CHROME_150, { ...GEMMA4_TASK, hasTurnFormat: false });
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/turn/i);
  });

  it('skips the tensor check when the size is unknown', () => {
    expect(gate(FIREFOX_153, GEMMA4_TASK).ok).toBe(true);
  });

  it('surfaces known f16 shader bugs as warnings for transformers.js', () => {
    // Firefox 153 compiles every f16 canary and still fails ORT's Clip kernel at
    // inference time, so the condition is tracked as a known issue rather than
    // pretended to be probe-detectable.
    expect(gate(FIREFOX_153, GEMMA4_ONNX).warnings.join(' ')).toMatch(/Clip/);
  });

  it('does not surface f16 shader bugs for MediaPipe, which does not use ORT', () => {
    expect(gate(FIREFOX_153, GEMMA4_TASK).warnings.join(' ')).not.toMatch(/Clip/);
  });

  it('warns when ONNX tensor sizes could not be read', () => {
    // A zero here means inspection failed, not that the model fits. Passing
    // silently would imply the model was cleared when it was never checked.
    const verdict = gate(FIREFOX_153, { ...GEMMA4_ONNX, largestTensorBytes: 0 });
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/could not read/i);
  });

  it('does not warn about unreadable sizes for MediaPipe', () => {
    // A .task file genuinely cannot expose tensor sizes, so 0 is expected there.
    expect(gate(CHROME_150, GEMMA4_TASK).warnings.join(' ')).not.toMatch(/could not read/i);
  });

  it('blocks an embedded .onnx at the protobuf limit', () => {
    // 2GiB: ONNX caps a protobuf message at 2GB and Chrome caps ArrayBuffer at
    // 2046MiB, so this one is documented rather than guessed.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestEmbeddedFileBytes: 2147483648 };
    const verdict = gate(CHROME_150, spec);
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/protobuf|2GB/i);
  });

  it('warns but does not block at the size that actually failed', () => {
    // 1360MiB is Qwen3-1.7B, which threw std::bad_alloc on Firefox. It is below
    // every documented ceiling, so the threshold is a measurement and blocking on
    // it would risk false negatives.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestEmbeddedFileBytes: 1426063360 };
    const verdict = gate(CHROME_150, spec);
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/wasm heap|contiguous/i);
  });

  it('says a failure is expected at a size seen to fail, not merely possible', () => {
    // The warning has to be readable as a reason not to download. A caller who is
    // told "a file this size threw std::bad_alloc" and then shown a clean verdict
    // will spend the bandwidth anyway.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestEmbeddedFileBytes: 1426063360 };
    expect(gate(CHROME_150, spec).warnings.join(' ')).toMatch(/expected to FAIL/);
  });

  it('hedges for an embedded file above the threshold but below the failure', () => {
    // 1100MiB: over the arbitrary 1GiB threshold, under the 1360MiB that failed.
    // Nothing was measured here, so the wording must not claim it will fail.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestEmbeddedFileBytes: 1153433600 };
    const warnings = gate(CHROME_150, spec).warnings.join(' ');
    expect(warnings).toMatch(/may fail/);
    expect(warnings).not.toMatch(/expected to FAIL/);
  });

  it('says nothing about a model whose weights are external', () => {
    // The control: Qwen3-VL-2B's largest file is 949MiB but it is .onnx_data, so
    // nothing is buffered whole and it loaded fine. A threshold applied to
    // totalBytes would have flagged it wrongly.
    const spec: ModelSpec = {
      ...GEMMA4_ONNX,
      largestEmbeddedFileBytes: 0,
      totalBytes: 1371537408,
    };
    const verdict = gate(CHROME_150, spec);
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).not.toMatch(/wasm heap|contiguous/i);
  });

  it('says nothing about an embedded file under the threshold', () => {
    // 543MiB is Qwen3-0.6B, which works.
    const spec: ModelSpec = { ...GEMMA4_ONNX, largestEmbeddedFileBytes: 569425920 };
    expect(gate(CHROME_150, spec).warnings.join(' ')).not.toMatch(/wasm heap|contiguous/i);
  });
});

// Small enough that every size check passes, so only the modality flags decide.
const TEXT_ONLY: ModelSpec = {
  runtime: 'transformers',
  largestTensorBytes: 1000,
  totalBytes: 500_000_000,
  largestEmbeddedFileBytes: 0,
  hasTurnFormat: false,
  vision: false,
  audio: false,
};

describe('gate with image intent', () => {
  const VISION: ModelSpec = { ...TEXT_ONLY, vision: true };

  it('blocks an image request against a text-only model', () => {
    const verdict = gate(CHROME_150, TEXT_ONLY, { wantsImage: true });
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/image/i);
  });

  it('allows an image request against a vision model', () => {
    expect(gate(CHROME_150, VISION, { wantsImage: true }).ok).toBe(true);
  });

  it('allows a text-only model when no image is requested', () => {
    expect(gate(CHROME_150, TEXT_ONLY).ok).toBe(true);
  });
});

describe('gate with audio intent', () => {
  // A model can accept images and refuse audio, so the flags stay separate.
  const IMAGE_ONLY: ModelSpec = { ...TEXT_ONLY, vision: true };
  const BOTH: ModelSpec = { ...TEXT_ONLY, vision: true, audio: true };

  it('blocks an audio request against a model with no audio encoder', () => {
    const verdict = gate(CHROME_150, TEXT_ONLY, { wantsAudio: true });
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/audio/i);
  });

  it('blocks audio on an image-only model', () => {
    const verdict = gate(CHROME_150, IMAGE_ONLY, { wantsAudio: true });
    expect(verdict.ok).toBe(false);
    expect(verdict.blockers.join(' ')).toMatch(/audio/i);
  });

  it('allows audio on a model that declares an audio encoder', () => {
    expect(gate(CHROME_150, BOTH, { wantsAudio: true }).ok).toBe(true);
  });

  it('allows both modalities together', () => {
    expect(gate(CHROME_150, BOTH, { wantsImage: true, wantsAudio: true }).ok).toBe(true);
  });

  it('ignores audio capability when no audio is requested', () => {
    expect(gate(CHROME_150, IMAGE_ONLY, { wantsImage: true }).ok).toBe(true);
  });
});

// vision/audio are false on a MediaPipe spec because a .task bundle has no
// config.json to read them from — "unknown", not "absent". Blocking on that would
// refuse a model that works, since the runtime does support both modalities.
describe('gate modality intent on MediaPipe, where the spec cannot know', () => {
  const TASK: ModelSpec = {
    runtime: 'mediapipe',
    largestTensorBytes: 0,
    totalBytes: 2_000_000_000,
    largestEmbeddedFileBytes: 0,
    hasTurnFormat: true,
    vision: false,
    audio: false,
  };

  it('warns rather than blocking an image request', () => {
    const verdict = gate(CHROME_150, TASK, { wantsImage: true });
    expect(verdict.ok).toBe(true);
    expect(verdict.blockers).toEqual([]);
    // The warning has to name the escape hatch, or it is just discouragement.
    expect(verdict.warnings.join(' ')).toMatch(/maxImages/);
  });

  it('warns rather than blocking an audio request', () => {
    const verdict = gate(CHROME_150, TASK, { wantsAudio: true });
    expect(verdict.ok).toBe(true);
    expect(verdict.warnings.join(' ')).toMatch(/supportAudio/);
  });

  it('still blocks the same request on transformers, where config.json is definite', () => {
    // The contrast is the point: same intent, same false flags, opposite verdict,
    // because only one of the two runtimes was actually asked.
    const onnx: ModelSpec = { ...TASK, runtime: 'transformers', hasTurnFormat: false };
    expect(gate(CHROME_150, onnx, { wantsImage: true }).ok).toBe(false);
  });
});
