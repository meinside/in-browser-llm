export { probe, type Capabilities } from './capability.js';
export { gate, type ModelSpec, type Verdict } from './gate.js';
export { inspectModel } from './inspect.js';
export { parseOnnxHeader, type ExternalTensor, type OnnxHeader } from './onnx-parse.js';
export {
  formatPrompt,
  formatPromptParts,
  truncateAtStop,
  createStopTruncator,
  type ContentPart,
  type Message,
} from './prompt.js';
export {
  CATALOG,
  lookupCatalog,
  type CatalogEntry,
  type TurnFormat,
} from './catalog.js';
export { parseToolCalls, type ToolCall, type ToolCallFormat } from './tools.js';
export { fetchWithProgress, type Progress } from './loader.js';
export { createCacheStore, requestPersistence, type CacheStore } from './storage.js';
export { createMetrics, type Metrics } from './metrics.js';
export {
  CapabilityError,
  DownloadError,
  RuntimeError,
} from './errors.js';
export { DEFAULT_MAX_TOKENS } from './engine/types.js';
export type { Engine, GenerateOptions, LoadOptions } from './engine/types.js';
export { resolveBaseUrl, type ModelRef } from './types.js';

import type { Engine, LoadOptions } from './engine/types.js';
import type { ModelRef } from './types.js';

// Adapters are imported dynamically so an unused runtime never loads.
export async function load(ref: ModelRef, opts: LoadOptions = {}): Promise<Engine> {
  if (ref.runtime === 'mediapipe') {
    const { createMediaPipeEngine } = await import('./engine/mediapipe.js');
    return createMediaPipeEngine(ref, opts);
  }
  const { createTransformersEngine } = await import('./engine/transformers.js');
  return createTransformersEngine(ref, opts);
}
