import { parseOnnxHeader } from './onnx-parse.js';
import { resolveBaseUrl, type ModelRef } from './types.js';
import type { ModelSpec } from './gate.js';
import { lookupCatalog } from './catalog.js';

// Header sizes vary by orders of magnitude: gemma-4's embed_tokens .onnx is
// 5.6KB while its decoder_model_merged is 673KB, because the decoder graph holds
// thousands of nodes. 4MB covers that with room to spare and still costs far less
// than the multi-GB weights. Setting this too low is silent: the parser tolerates
// truncated input by returning fewer tensors, so an undersized limit understates
// both the total and the largest tensor.
const HEADER_BYTES = 4 * 1024 * 1024;

// Session file names to look for. Exports vary in layout: multimodal models split
// into embed_tokens / decoder_model_merged / encoders, while smaller text models
// ship a single `model`. Missing the actual name is silent — every size reads as
// 0, which disables the gate — so cover both layouts.
const CANDIDATE_SESSIONS = [
  'model',
  'embed_tokens',
  'decoder_model_merged',
  'vision_encoder',
  'audio_encoder',
];
const DTYPE_SUFFIXES = ['_q4f16', '_q4', ''];

interface InspectOptions {
  fetch?: typeof fetch;
}

async function fetchRange(
  doFetch: typeof fetch,
  url: string,
  bytes: number,
): Promise<Uint8Array | null> {
  try {
    const res = await doFetch(url, { headers: { Range: `bytes=0-${bytes - 1}` } });
    if (!res.ok) return null;
    return new Uint8Array(await res.arrayBuffer());
  } catch {
    return null;
  }
}

// Reads x-linked-size before content-length, because on HuggingFace the latter is
// wrong in Node and invisible in a browser.
//
// Large files are LFS-backed: content-length describes the pointer, not the weights.
// Worse, `access-control-expose-headers` omits Content-Length entirely, so a browser
// sees null and this returned 0 — silently disabling every size check while Node
// tests passed. x-linked-size carries the real byte count and IS on the exposed
// list. Verified against the live headers, not documentation.
async function contentLength(doFetch: typeof fetch, url: string): Promise<number> {
  try {
    const res = await doFetch(url, { method: 'HEAD' });
    if (!res.ok) return 0;
    const linked = Number(res.headers.get('x-linked-size'));
    if (linked > 0) return linked;
    return Number(res.headers.get('content-length')) || 0;
  } catch {
    return 0;
  }
}

interface Modalities {
  vision: boolean;
  audio: boolean;
}

// A model's class cannot be guessed: AutoTokenizer and AutoProcessor are distinct
// objects and picking wrong fails at load. config.json is a few KB, so read it.
//
// Detection keys off the encoder sub-configs rather than the architecture name,
// which varies per family (Idefics3ForConditionalGeneration vs
// Gemma4ForConditionalGeneration) while the sub-config keys do not.
async function inspectModalities(doFetch: typeof fetch, base: string): Promise<Modalities> {
  const unknown: Modalities = { vision: false, audio: false };
  try {
    const res = await doFetch(`${base}/config.json`);
    if (!res.ok) return unknown;
    const config = (await res.json()) as Record<string, unknown>;

    return {
      vision: typeof config['vision_config'] === 'object' && config['vision_config'] !== null,
      audio: typeof config['audio_config'] === 'object' && config['audio_config'] !== null,
    };
  } catch {
    // Conservative default: refuse image input rather than attempt a path the
    // model may not support.
    return unknown;
  }
}

async function inspectOnnx(
  doFetch: typeof fetch,
  base: string,
): Promise<{ largestTensorBytes: number; totalBytes: number; largestEmbeddedFileBytes: number }> {
  let largest = 0;
  let total = 0;
  // Per session, not summed: ORT mallocs each session's file separately.
  let largestEmbedded = 0;

  for (const session of CANDIDATE_SESSIONS) {
    for (const suffix of DTYPE_SUFFIXES) {
      const url = `${base}/onnx/${session}${suffix}.onnx`;
      const bytes = await fetchRange(doFetch, url, HEADER_BYTES);
      if (!bytes) continue;

      const header = parseOnnxHeader(bytes);
      if (header.tensors.length > 0) {
        // Weights live in a sidecar .onnx_data, so the header describes them.
        // A header longer than the window parses partially and understates sizes,
        // which would weaken the gate — worth saying out loud.
        if (bytes.length >= HEADER_BYTES) {
          console.warn(
            `[in-browser-llm] ${url} exceeds the ${HEADER_BYTES}-byte inspection ` +
              'window; tensor sizes may be understated.',
          );
        }
        largest = Math.max(largest, header.largestTensorBytes);
        total += header.totalExternalBytes;
      } else {
        // Weights are embedded in the .onnx itself (Qwen3-1.7B ships a 1360MiB
        // .onnx with a 15-byte .onnx_data LFS stub). No external_data entries
        // exist, so the total comes from the file length and the largest tensor
        // stays unknown — gate() reports that rather than implying it fits.
        const fileBytes = await contentLength(doFetch, url);
        total += fileBytes;
        largestEmbedded = Math.max(largestEmbedded, fileBytes);
      }
      break; // First available dtype for this session is enough.
    }
  }

  return {
    largestTensorBytes: largest,
    totalBytes: total,
    largestEmbeddedFileBytes: largestEmbedded,
  };
}

// Only the size: a .task bundle exposes no tensor index, so Content-Length is all
// there is to learn without downloading it.
async function inspectTask(doFetch: typeof fetch, url: string): Promise<{ totalBytes: number }> {
  try {
    const res = await doFetch(url, { method: 'HEAD' });
    return { totalBytes: Number(res.headers.get('content-length')) || 0 };
  } catch {
    return { totalBytes: 0 };
  }
}

// Learns a model's constraints from a few KB, so a multi-GB download can be
// refused up front.
export async function inspectModel(ref: ModelRef, opts: InspectOptions = {}): Promise<ModelSpec> {
  const doFetch = opts.fetch ?? globalThis.fetch;
  const base = resolveBaseUrl(ref);

  if (ref.runtime === 'mediapipe') {
    const { totalBytes } = await inspectTask(doFetch, base);
    return {
      runtime: 'mediapipe',
      largestTensorBytes: 0, // Not visible inside a .task file.
      totalBytes,
      largestEmbeddedFileBytes: 0, // A .task is not an ONNX protobuf.
      hasTurnFormat: lookupCatalog(ref)?.turnFormat !== undefined,
      // A .task bundle exposes no config.json to read these from, and this
      // library's MediaPipe path sends text only regardless (see CLAUDE.md
      // trap 13 — the runtime itself does support both).
      vision: false,
      audio: false,
    };
  }

  const [{ largestTensorBytes, totalBytes, largestEmbeddedFileBytes }, modalities] = await Promise.all([
    inspectOnnx(doFetch, base),
    inspectModalities(doFetch, base),
  ]);

  return {
    runtime: 'transformers',
    largestTensorBytes,
    totalBytes,
    largestEmbeddedFileBytes,
    hasTurnFormat: lookupCatalog(ref)?.turnFormat !== undefined,
    vision: modalities.vision,
    audio: modalities.audio,
  };
}
