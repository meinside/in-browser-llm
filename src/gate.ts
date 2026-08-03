import type { Capabilities } from './capability.js';

export interface ModelSpec {
  runtime: 'transformers' | 'mediapipe';
  /** 0 when unknown, as with MediaPipe .task files. */
  largestTensorBytes: number;
  totalBytes: number;
  /**
   * Largest single .onnx that must be buffered whole, or 0 when weights live in
   * sidecar .onnx_data files. ORT mallocs an embedded file contiguously, then
   * duplicates it while parsing the protobuf; external data is copied per tensor.
   */
  largestEmbeddedFileBytes: number;
  /**
   * Whether the catalog supplies turn delimiters. Not merely whether an entry
   * exists: an entry may carry only a tool-call format, which MediaPipe cannot
   * use.
   */
  hasTurnFormat: boolean;
  /** Needs AutoProcessor + AutoModelForImageTextToText rather than a tokenizer. */
  vision: boolean;
  /** Accepts audio input, so the processor takes a fourth argument. */
  audio: boolean;
}

export interface GateOptions {
  /** Set when the caller intends to send an image, so the check runs early. */
  wantsImage?: boolean;
  /**
   * Set when the caller intends to send audio. Separate from wantsImage because a
   * model can accept one and refuse the other.
   */
  wantsAudio?: boolean;
}

export interface Verdict {
  ok: boolean;
  blockers: string[];
  warnings: string[];
}

const mib = (n: number) => Math.round(n / 1024 / 1024);

// ONNX caps a protobuf message at 2GB, and Chrome caps a single ArrayBuffer at
// 2,145,386,496 bytes. Both are documented, so this tier blocks.
const PROTOBUF_LIMIT_BYTES = 2 * 1024 * 1024 * 1024;

// ARBITRARY, and deliberately so: a round number between the only two
// observations — 949MiB of sidecar data loaded, a 1360MiB embedded file did not.
// No ONNX Runtime or transformers.js source names a per-file limit below 2GB, and
// 1360MiB is under every documented ceiling; the real cause is heap fragmentation
// plus parse-time duplication, which nothing quantifies. Hence a warning.
const EMBEDDED_WARN_BYTES = 1024 * 1024 * 1024;

// Where a load was actually seen to fail: Qwen3-1.7B q4f16, one 1360MiB .onnx,
// "std::bad_alloc" on Firefox 153/macOS, reproduced. Above this the warning says a
// failure is expected rather than possible. Still not a blocker — the real ceiling
// moves with the browser and with memory free at the time — but a caller reading
// "this size failed" should not also read a verdict that looks clean.
const OBSERVED_FAILURE_BYTES = 1360 * 1024 * 1024;

// Pure: no network, no globals. This is what makes the riskiest decision testable.
export function gate(caps: Capabilities, spec: ModelSpec, opts: GateOptions = {}): Verdict {
  const blockers: string[] = [];
  const warnings: string[] = [];

  if (!caps.hasWebGPU) {
    return {
      ok: false,
      blockers: ['This browser does not support WebGPU (navigator.gpu is absent).'],
      warnings,
    };
  }

  // Both limits, because the checks below compare against both. A zero limit would
  // otherwise make every tensor look oversized and report a size problem.
  if (caps.maxStorageBufferBindingSize === 0 || caps.maxBufferSize === 0) {
    return {
      ok: false,
      blockers: ['Could not obtain a GPU adapter (requestAdapter returned null).'],
      warnings,
    };
  }

  // For ONNX the size is readable, so 0 means inspection failed rather than
  // "fits". Say so instead of implying the model was cleared.
  if (spec.runtime === 'transformers' && spec.largestTensorBytes === 0) {
    warnings.push(
      'Could not read tensor sizes from the ONNX headers, so the largest-tensor ' +
        'check was skipped. Loading may still fail on the buffer limit.',
    );
  }

  // The non-obvious failure: total size fits but one tensor does not. Two limits
  // matter and they are not interchangeable. maxBufferSize caps one allocation and
  // is fatal; maxStorageBufferBindingSize caps one binding, and ONNX Runtime splits
  // an oversized tensor across several bindings over the same buffer.
  if (spec.largestTensorBytes > caps.maxBufferSize) {
    blockers.push(
      `Largest tensor is ${mib(spec.largestTensorBytes)}MiB but this adapter's ` +
        `maxBufferSize is ${mib(caps.maxBufferSize)}MiB. A tensor must fit in a ` +
        'single allocation: ONNX Runtime splits oversized tensors across bindings, ' +
        'but still allocates one buffer for the whole tensor. Moving the session to ' +
        'wasm does not help either, because the WebGPU wasm build ships no CPU ' +
        'kernel for quantized gather.',
    );
  } else if (spec.largestTensorBytes > caps.maxStorageBufferBindingSize) {
    warnings.push(
      `Largest tensor is ${mib(spec.largestTensorBytes)}MiB, over this adapter's ` +
        `maxStorageBufferBindingSize of ${mib(caps.maxStorageBufferBindingSize)}MiB ` +
        'but within maxBufferSize. ONNX Runtime is expected to split it across ' +
        'bindings, so the load should succeed — but this combination has not been ' +
        "verified in a browser, and a segment count exceeding the adapter's " +
        'per-stage storage buffer limit would still fail.',
    );
  }

  // A second, independent failure mode with a different cause. The tensor checks
  // above concern GPU buffer bindings; this one is the wasm heap, and a model can
  // hit either alone. Embedded weights force ORT to malloc the whole .onnx
  // contiguously and then duplicate it while parsing the protobuf, which surfaces
  // as "std::bad_alloc" and names nothing about its cause.
  if (spec.largestEmbeddedFileBytes >= PROTOBUF_LIMIT_BYTES) {
    blockers.push(
      `The model is one ${mib(spec.largestEmbeddedFileBytes)}MiB .onnx with its ` +
        'weights embedded, at or over the 2GB ONNX protobuf limit — and Chrome caps ' +
        'a single ArrayBuffer at 2046MiB besides. Use an export whose weights sit ' +
        'in sidecar .onnx_data files.',
    );
  } else if (spec.largestEmbeddedFileBytes > EMBEDDED_WARN_BYTES) {
    const observed = spec.largestEmbeddedFileBytes >= OBSERVED_FAILURE_BYTES;
    warnings.push(
      (observed
        ? `Loading this model is expected to FAIL. It is one ` +
          `${mib(spec.largestEmbeddedFileBytes)}MiB .onnx with embedded weights, and a ` +
          'file this size threw "std::bad_alloc" on Firefox. Downloading it is not ' +
          'recommended. '
        : `This model is one ${mib(spec.largestEmbeddedFileBytes)}MiB .onnx with ` +
          'embedded weights, which may fail to load. ') +
        'ONNX Runtime must allocate such a file contiguously in a 4GiB wasm heap and ' +
        'then duplicate it while parsing, whereas sidecar .onnx_data is copied per ' +
        'tensor — 949MiB of it loaded fine. Prefer an export using .onnx_data. This ' +
        'is not a hard refusal because the exact ceiling depends on the browser and ' +
        'on memory available at the time.',
    );
  }

  // A .task bundle carries no config.json, so vision/audio are false there for
  // "unknown" rather than for "absent" — blocking on that would refuse a model
  // that works. The runtime does support both modalities, so this warns instead
  // and the load-time option decides. On transformers.js the spec is read from
  // config.json and the answer is definite, so it blocks.
  const modalityUnknown = spec.runtime === 'mediapipe';

  if (opts.wantsImage && !spec.vision) {
    (modalityUnknown ? warnings : blockers).push(
      modalityUnknown
        ? 'Whether this MediaPipe model accepts images cannot be read from a .task ' +
          'bundle, which carries no config. Pass { maxImages: n } to load() to try ' +
          'it; MediaPipe throws at generate time if the model has no vision encoder.'
        : 'This model has no vision encoder, so it cannot accept image input. ' +
          'Pick a model whose config.json declares vision_config.',
    );
  }

  if (opts.wantsAudio && !spec.audio) {
    (modalityUnknown ? warnings : blockers).push(
      modalityUnknown
        ? 'Whether this MediaPipe model accepts audio cannot be read from a .task ' +
          'bundle, which carries no config. Pass { supportAudio: true } to load() ' +
          'to try it.'
        : 'This model has no audio encoder, so it cannot accept audio input. ' +
          'Pick a model whose config.json declares audio_config.',
    );
  }

  if (spec.runtime === 'mediapipe') {
    // MediaPipe passes requiredFeatures: ["shader-f16"] — a hard requirement.
    if (!caps.hasF16) {
      blockers.push(
        'MediaPipe requires the shader-f16 WebGPU feature, which this adapter lacks.',
      );
    }
    // No chat template support, so the turn format must come from the catalog.
    if (!spec.hasTurnFormat) {
      blockers.push(
        'No catalog entry for this model. The MediaPipe path cannot apply a chat ' +
          'template, so turn delimiters must be known in advance.',
      );
    }
  } else if (!caps.hasF16) {
    warnings.push(
      'shader-f16 is unavailable, so dtype falls back to q4. Size and quality differ.',
    );
  }

  if (spec.runtime === 'transformers' && caps.knownF16ShaderIssues.length > 0) {
    warnings.push(...caps.knownF16ShaderIssues);
  }

  if (!caps.hasSubgroups) {
    warnings.push(
      'The subgroups feature is unavailable. Generation will work but decode is ' +
        'substantially slower (measured at roughly a quarter of Chrome). This is not a defect.',
    );
  }

  return { ok: blockers.length === 0, blockers, warnings };
}
