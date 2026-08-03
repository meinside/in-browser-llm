// Bridges a callback-based streaming API to an async iterator.
//
// Both runtimes hand tokens to a callback while generation runs as a promise, so
// both need the same buffering. Extracted rather than duplicated per adapter.
export function createDeltaStream(): {
  push(delta: string): void;
  finish(): void;
  fail(error: unknown): void;
  iterate(): AsyncGenerator<string, void, undefined>;
} {
  const queue: string[] = [];
  let notify: (() => void) | null = null;
  let finished = false;
  let failure: unknown = null;

  const wake = () => {
    notify?.();
    notify = null;
  };

  return {
    push(delta) {
      if (finished) return;
      queue.push(delta);
      wake();
    },
    finish() {
      finished = true;
      wake();
    },
    fail(error) {
      failure = error;
      finished = true;
      wake();
    },
    async *iterate() {
      while (true) {
        if (queue.length > 0) {
          yield queue.shift()!;
          continue;
        }
        if (failure) throw failure;
        if (finished) return;
        await new Promise<void>((resolve) => {
          notify = resolve;
        });
      }
    },
  };
}
