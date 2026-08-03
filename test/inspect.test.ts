import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { inspectModel } from '../src/inspect.js';
import { resolveBaseUrl } from '../src/types.js';

// Wrapped in Uint8Array so `.slice()` yields an exact-length view; a raw Node
// Buffer's `.buffer` spans the shared 64KB pool.
const header = new Uint8Array(
  readFileSync(new URL('./fixtures/gemma4-embed_tokens_q4f16.onnx', import.meta.url)),
);

function stubFetch(bodies: Record<string, Uint8Array | object>): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const url = String(input);
    const key = Object.keys(bodies).find((k) => url.includes(k));
    if (!key) return new Response(null, { status: 404 });
    const body = bodies[key]!;
    if (body instanceof Uint8Array) {
      // slice() copies into a tight view. Passing `.buffer` instead would hand
      // over Node's whole 64KB Buffer pool, which silently pads the response.
      return new Response(body.slice(), {
        status: 206,
        headers: { 'content-length': String(body.length) },
      });
    }
    return new Response(JSON.stringify(body), { status: 200 });
  }) as typeof fetch;
}

describe('resolveBaseUrl', () => {
  it('builds a HuggingFace resolve URL from a repo id', () => {
    expect(resolveBaseUrl({ runtime: 'transformers', repo: 'org/model' })).toBe(
      'https://huggingface.co/org/model/resolve/main',
    );
  });

  it('strips a trailing slash from a base URL', () => {
    expect(resolveBaseUrl({ runtime: 'transformers', baseUrl: '/models/x/' })).toBe('/models/x');
  });
});

describe('inspectModel', () => {
  it('reports the largest tensor for a transformers model', async () => {
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/gemma-4-E2B-it-ONNX' },
      { fetch: stubFetch({ embed_tokens: header, 'config.json': { model_type: 'gemma4' } }) },
    );
    expect(spec.largestTensorBytes).toBe(1174405120);
    expect(spec.runtime).toBe('transformers');
  });

  it('reports an unknown tensor size for a .task bundle', async () => {
    // A .task exposes no tensor index, so the gate's largest-tensor check cannot
    // run on this runtime and must see 0 rather than a guess.
    const task = new Uint8Array(8);
    const spec = await inspectModel(
      { runtime: 'mediapipe', url: 'https://example.test/model.task' },
      { fetch: stubFetch({ 'model.task': task }) },
    );
    expect(spec.runtime).toBe('mediapipe');
    expect(spec.largestTensorBytes).toBe(0);
  });

  it('leaves the tensor size unknown when the header cannot be read', async () => {
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'org/missing' },
      { fetch: stubFetch({}) },
    );
    expect(spec.largestTensorBytes).toBe(0);
  });

  it('sums every session rather than stopping at the first', async () => {
    // Regression: the inspection window was once smaller than the decoder header
    // (5.6KB for embed_tokens vs 673KB for decoder_model_merged), so the decoder
    // parsed as zero tensors and the reported total covered embed_tokens alone.
    const requested: string[] = [];
    const doFetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      requested.push(url);
      if (url.includes('embed_tokens') || url.includes('decoder_model_merged')) {
        return new Response(header.slice(), { status: 206 });
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch;

    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'org/model' },
      { fetch: doFetch },
    );

    // Both sessions contributed, so the total is double one header's worth.
    expect(spec.totalBytes).toBe(1590689792 * 2);
    expect(requested.some((u) => u.includes('decoder_model_merged'))).toBe(true);
  });

  it('inspects a single-session export named "model"', async () => {
    // Regression: only embed_tokens/decoder_model_merged were probed, so smaller
    // text models (gemma-3-270m ships onnx/model_q4f16.onnx) reported every size
    // as 0 — which the gate then read as "fits", disabling the check entirely.
    const doFetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/onnx/model')) {
        return new Response(header.slice(), { status: 206 });
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch;

    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/gemma-3-270m-it-ONNX' },
      { fetch: doFetch },
    );
    expect(spec.largestTensorBytes).toBeGreaterThan(0);
  });

  it('falls back to Content-Length when weights are embedded in the .onnx', async () => {
    // Regression: Qwen3-0.6B ships a 569MB .onnx with a 15-byte .onnx_data stub,
    // so there are no external_data entries to parse. The old code skipped such
    // sessions entirely and reported a total of 0MiB.
    const noExternalData = new Uint8Array([0x08, 0x0a]); // parses, yields no tensors
    const doFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (!url.includes('/onnx/model')) return new Response(null, { status: 404 });
      if (init?.method === 'HEAD') {
        return new Response(null, { status: 200, headers: { 'content-length': '569789750' } });
      }
      return new Response(noExternalData.slice(), { status: 206 });
    }) as typeof fetch;

    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/Qwen3-0.6B-ONNX' },
      { fetch: doFetch },
    );
    expect(spec.totalBytes).toBe(569789750);
    // Unknowable without the sidecar index — reported as unknown, not as "fits".
    expect(spec.largestTensorBytes).toBe(0);
    // Embedded weights: ORT must malloc this whole file contiguously, so its size
    // is the constraint the wasm heap sees.
    expect(spec.largestEmbeddedFileBytes).toBe(569789750);
  });

  it('reads x-linked-size when content-length is hidden, as in a browser', async () => {
    // The bug this pins: HuggingFace omits Content-Length from
    // access-control-expose-headers, so a browser sees null and every size read 0 —
    // silently disabling the size checks while Node tests passed. For LFS-backed
    // files content-length is also just the pointer's size, so x-linked-size is the
    // right source in both environments.
    const noExternalData = new Uint8Array([0x08, 0x0a]); // parses, yields no tensors
    const doFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (!url.includes('/onnx/model')) return new Response(null, { status: 404 });
      if (init?.method === 'HEAD') {
        // No content-length at all, exactly what CORS leaves visible.
        return new Response(null, { status: 200, headers: { 'x-linked-size': '1426069098' } });
      }
      return new Response(noExternalData.slice(), { status: 206 });
    }) as typeof fetch;

    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/Qwen3-1.7B-ONNX' },
      { fetch: doFetch },
    );
    expect(spec.totalBytes).toBe(1426069098);
    expect(spec.largestEmbeddedFileBytes).toBe(1426069098);
  });

  it('prefers x-linked-size over a content-length describing the LFS pointer', async () => {
    const noExternalData = new Uint8Array([0x08, 0x0a]);
    const doFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (!url.includes('/onnx/model')) return new Response(null, { status: 404 });
      if (init?.method === 'HEAD') {
        // 974 bytes is a real LFS pointer size; the weights are 1.3GB.
        return new Response(null, {
          status: 200,
          headers: { 'content-length': '974', 'x-linked-size': '1426069098' },
        });
      }
      return new Response(noExternalData.slice(), { status: 206 });
    }) as typeof fetch;

    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/Qwen3-1.7B-ONNX' },
      { fetch: doFetch },
    );
    expect(spec.largestEmbeddedFileBytes).toBe(1426069098);
  });

  it('reports no embedded file when weights live in a sidecar', async () => {
    // gemma-4 splits weights into .onnx_data, so nothing is buffered whole and the
    // wasm-heap limit does not apply — 0 means "not applicable", not "unknown".
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/gemma-4-E2B-it-ONNX' },
      { fetch: stubFetch({ embed_tokens: header, 'config.json': { model_type: 'gemma4' } }) },
    );
    expect(spec.largestTensorBytes).toBeGreaterThan(0);
    expect(spec.largestEmbeddedFileBytes).toBe(0);
  });

  it('requests a window large enough for a big decoder header', async () => {
    let requestedRange = '';
    const doFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (!requestedRange) {
        requestedRange = (init?.headers as Record<string, string>)?.Range ?? '';
      }
      return new Response(header.slice(), { status: 206 });
    }) as typeof fetch;

    await inspectModel({ runtime: 'transformers', repo: 'org/model' }, { fetch: doFetch });

    // Real decoder headers reach ~673KB; a 64KB window silently truncates them.
    const end = Number(requestedRange.replace('bytes=0-', ''));
    expect(end).toBeGreaterThan(700_000);
  });
});

describe('modality detection', () => {
  // Verified against real configs: vision_config is present in gemma-4-E2B-it-ONNX
  // and in SmolVLM-256M-Instruct, absent in Qwen3-0.6B-ONNX and gemma-3-270m-it-ONNX.
  // audio_config appears only in gemma-4. SmolVLM is not a supported model — it was
  // tried and rejected — but its Idefics3 config is kept here because detection must
  // key off the sub-config rather than the architecture name, and gemma-4 alone
  // cannot show that.
  function configFetch(config: object): typeof fetch {
    return (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('config.json')) {
        return new Response(JSON.stringify(config), { status: 200 });
      }
      if (init?.method === 'HEAD') {
        return new Response(null, { status: 200, headers: { 'content-length': '1000' } });
      }
      if (url.includes('/onnx/model')) {
        return new Response(header.slice(), { status: 206 });
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch;
  }

  it('reports vision for a model with vision_config', async () => {
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'HuggingFaceTB/SmolVLM-256M-Instruct' },
      {
        fetch: configFetch({
          architectures: ['Idefics3ForConditionalGeneration'],
          vision_config: {},
        }),
      },
    );
    expect(spec.vision).toBe(true);
    expect(spec.audio).toBe(false);
  });

  it('reports both for a model with vision_config and audio_config', async () => {
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/gemma-4-E2B-it-ONNX' },
      {
        fetch: configFetch({
          architectures: ['Gemma4ForConditionalGeneration'],
          vision_config: {},
          audio_config: {},
        }),
      },
    );
    expect(spec.vision).toBe(true);
    expect(spec.audio).toBe(true);
  });

  it('reports neither for a text-only model', async () => {
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'onnx-community/Qwen3-0.6B-ONNX' },
      { fetch: configFetch({ architectures: ['Qwen3ForCausalLM'] }) },
    );
    expect(spec.vision).toBe(false);
    expect(spec.audio).toBe(false);
  });

  it('defaults to neither when config.json is unavailable', async () => {
    // Conservative: declining image input beats attempting an unsupported path.
    const spec = await inspectModel(
      { runtime: 'transformers', repo: 'org/model' },
      { fetch: stubFetch({ embed_tokens: header }) },
    );
    expect(spec.vision).toBe(false);
  });

  it('reports neither for MediaPipe, which exposes no config.json', async () => {
    const task = new Uint8Array(8);
    task.set([0x54, 0x46, 0x4c, 0x33], 4);
    const spec = await inspectModel(
      { runtime: 'mediapipe', url: 'https://example.test/model.task' },
      { fetch: stubFetch({ 'model.task': task }) },
    );
    expect(spec.vision).toBe(false);
  });
});
