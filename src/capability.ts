export interface Capabilities {
  hasWebGPU: boolean;
  hasF16: boolean;
  hasSubgroups: boolean;
  /**
   * Non-empty when this browser is known to miscompile f16 shaders that some
   * ONNX exports rely on, regardless of what feature probing reports.
   *
   * Discovered empirically: Firefox 153 rejects ORT's Clip kernel with
   * "ShaderModule with 'Clip' label is invalid" during inference, while every
   * WGSL canary we could construct compiles fine. Rather than pretend a
   * capability probe can predict it, the condition is recorded as a known issue
   * and f16 dtypes are avoided when it applies.
   */
  knownF16ShaderIssues: string[];
  /**
   * Caps a single shader binding, not a tensor. ONNX Runtime splits a tensor that
   * exceeds this across several bindings over one buffer, so on its own this is a
   * warning rather than a blocker. See gate().
   */
  maxStorageBufferBindingSize: number;
  /**
   * Caps a single allocation, which ONNX Runtime always makes whole per tensor.
   * This is the hard ceiling nothing works around.
   */
  maxBufferSize: number;
  adapterInfo: { vendor: string; architecture: string };
}

// A factory, not a shared constant: spreading a constant would leave every
// caller sharing one knownF16ShaderIssues array and one adapterInfo object.
function empty(): Capabilities {
  return {
    hasWebGPU: false,
    hasF16: false,
    hasSubgroups: false,
    knownF16ShaderIssues: [],
    maxStorageBufferBindingSize: 0,
    maxBufferSize: 0,
    adapterInfo: { vendor: '', architecture: '' },
  };
}

// Engine bugs that no capability query exposes. Detecting this one needs the
// actual failing workload, which costs a multi-GB download first.
//
// This is deliberately a UA check, which is otherwise avoided in this file, and
// it is justified narrowly: the bug lives in the browser's shader compiler rather
// than in anything `adapter.features` describes, and the alternative is crashing
// after a long download. Note the canary cannot tell you when to remove this —
// every WGSL shape we could construct already compiles on Firefox. Retest the
// real model instead.
function detectKnownF16ShaderIssues(): string[] {
  if (typeof navigator === 'undefined') return [];
  const gecko = /rv:(\d+)\.\d+\) Gecko/.exec(navigator.userAgent);
  if (!gecko) return [];
  const version = Number(gecko[1]);
  if (version <= 153) {
    return [
      `Firefox ${version} fails ONNX Runtime's f16 Clip kernel during inference ` +
        `("ShaderModule with 'Clip' label is invalid"), so f16 model variants are ` +
        'avoided on this browser.',
    ];
  }
  return [];
}

// Probe runtime values only. Both Firefox and Chrome report "Intel Mac OS X" in
// the UA on Apple Silicon, so user-agent sniffing is actively misleading.
export async function probe(): Promise<Capabilities> {
  if (!navigator.gpu) return empty();

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...empty(), hasWebGPU: true };

  // Firefox blanks adapter info as a fingerprinting defence; Chrome fills it in.
  const info = adapter.info ?? ({} as GPUAdapterInfo);

  return {
    hasWebGPU: true,
    hasF16: adapter.features.has('shader-f16'),
    knownF16ShaderIssues: detectKnownF16ShaderIssues(),
    hasSubgroups: adapter.features.has('subgroups'),
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize ?? 0,
    maxBufferSize: adapter.limits.maxBufferSize ?? 0,
    adapterInfo: {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
    },
  };
}
