export interface Metrics {
  ttftMs: number | null;
  tokensPerSec: number;
  chunkCount: number;
}

// Chrome and Firefox differed by 4x on identical code, so measurement is a
// feature rather than decoration.
export function createMetrics(): { onChunk(): void; finish(): Metrics } {
  const started = performance.now();
  let ttftMs: number | null = null;
  let chunkCount = 0;

  return {
    onChunk() {
      if (ttftMs === null) ttftMs = performance.now() - started;
      chunkCount += 1;
    },
    finish(): Metrics {
      const elapsed = performance.now() - started;
      const decodeMs = ttftMs === null ? elapsed : elapsed - ttftMs;
      return {
        ttftMs,
        tokensPerSec: decodeMs > 0 ? (chunkCount / decodeMs) * 1000 : 0,
        chunkCount,
      };
    },
  };
}
