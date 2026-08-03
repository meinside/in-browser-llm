import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseOnnxHeader } from '../src/onnx-parse.js';

const fixture = new Uint8Array(
  readFileSync(new URL('./fixtures/gemma4-embed_tokens_q4f16.onnx', import.meta.url)),
);

describe('parseOnnxHeader', () => {
  it('finds every external tensor', () => {
    expect(parseOnnxHeader(fixture).tensors).toHaveLength(6);
  });

  it('reports the largest tensor exactly', () => {
    // Per-Layer Embeddings: 1120MiB. Exceeds Firefox's 1GiB buffer limit.
    expect(parseOnnxHeader(fixture).largestTensorBytes).toBe(1174405120);
  });

  it('names the largest tensor', () => {
    const { tensors } = parseOnnxHeader(fixture);
    const largest = tensors.reduce((a, t) => (t.bytes > a.bytes ? t : a));
    expect(largest.name).toBe('model_embed_tokens_per_layer_weight_quant');
  });

  it('sums total external bytes', () => {
    // Matches content-length of embed_tokens_q4f16.onnx_data exactly (verified
    // over HTTP), which confirms every external tensor was found.
    expect(parseOnnxHeader(fixture).totalExternalBytes).toBe(1590689792);
  });

  it('records which sidecar file each tensor lives in', () => {
    // The presence of a location is what distinguishes a sidecar export from an
    // embedded one, which is the discriminator for the wasm-heap limit.
    const { tensors } = parseOnnxHeader(fixture);
    expect(new Set(tensors.map((t) => t.location))).toEqual(
      new Set(['embed_tokens_q4f16.onnx_data']),
    );
  });

  it('returns empty results for a non-ONNX buffer instead of throwing', () => {
    const junk = new Uint8Array([0xff, 0xff, 0xff, 0xff]);
    expect(parseOnnxHeader(junk).tensors).toEqual([]);
  });

  it('does not throw on a truncated header', () => {
    expect(() => parseOnnxHeader(fixture.subarray(0, 100))).not.toThrow();
  });
});
