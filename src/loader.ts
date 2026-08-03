import { DownloadError } from './errors.js';
import { createCacheStore, type CacheStore } from './storage.js';

export interface Progress {
  loaded: number;
  total: number;
  ratio: number;
}

export interface FetchResult {
  stream: ReadableStream<Uint8Array>;
  fromCache: boolean;
  totalBytes: number;
}

interface FetchOptions {
  store?: CacheStore;
  onProgress?: (progress: Progress) => void;
  fetch?: typeof fetch;
  /** Range-resume attempts after an interrupted transfer. Default 3. */
  retries?: number;
}

// Re-creates a stream in this realm.
//
// Consumers may test `reader instanceof ReadableStreamDefaultReader` — MediaPipe
// does, and throws "No model asset provided." when it fails. A body handed back by
// the Cache API can belong to a different realm, so its reader fails that check
// even though it is a perfectly good stream. Piping through a locally constructed
// stream makes the identity check pass.
function reWrap(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>());
}

// MediaPipe offers no download progress hook, so progress is derived here from a
// byte count over the response stream.
//
// Multi-GB transfers get interrupted in practice, so a dropped connection is
// resumed with a Range request rather than restarting from zero. HuggingFace
// serves accept-ranges: bytes.
export async function fetchWithProgress(
  url: string,
  opts: FetchOptions = {},
): Promise<FetchResult> {
  const doFetch = opts.fetch ?? globalThis.fetch;
  const store = opts.store ?? createCacheStore();
  const maxRetries = opts.retries ?? 3;
  const onProgress = opts.onProgress;

  const cached = await store.match(url);
  if (cached?.body) {
    const totalBytes = Number(cached.headers.get('content-length')) || 0;
    // Report completion even though nothing streamed, so a progress bar does not
    // sit frozen at zero on a cache hit.
    onProgress?.({ loaded: totalBytes, total: totalBytes, ratio: 1 });
    return { stream: reWrap(cached.body), fromCache: true, totalBytes };
  }

  const first = await doFetch(url);
  if (!first.ok || !first.body) {
    throw new DownloadError({
      status: first.status,
      resumable: first.headers.get('accept-ranges') === 'bytes',
      url,
    });
  }

  const totalBytes = Number(first.headers.get('content-length')) || 0;
  const resumable = first.headers.get('accept-ranges') === 'bytes' && totalBytes > 0;

  let loaded = 0;

  // Reads one response body into the output stream. Returns normally on
  // completion; throws if the connection drops mid-body.
  async function pump(
    body: ReadableStream<Uint8Array>,
    controller: ReadableStreamDefaultController<Uint8Array>,
  ): Promise<void> {
    const reader = body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      loaded += value.byteLength;
      controller.enqueue(value);
      onProgress?.({
        loaded,
        total: totalBytes,
        ratio: totalBytes ? loaded / totalBytes : 0,
      });
    }
  }

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let body: ReadableStream<Uint8Array> | null = first.body;
      let attempt = 0;

      while (true) {
        try {
          if (body) await pump(body, controller);

          // A partial transfer mistaken for success produces an inscrutable
          // error at load time, so verify the byte count.
          if (totalBytes && loaded < totalBytes) throw new Error('truncated transfer');

          controller.close();
          return;
        } catch (error) {
          if (!resumable || attempt >= maxRetries) {
            controller.error(new DownloadError({ status: 0, resumable, url }));
            return;
          }
          attempt += 1;
          console.warn(
            `[in-browser-llm] transfer interrupted at ${loaded}/${totalBytes}, ` +
              `resuming (attempt ${attempt}/${maxRetries})`,
            error,
          );
          const resumed = await doFetch(url, { headers: { Range: `bytes=${loaded}-` } });
          // 206 Partial Content is required; 200 would restart and duplicate bytes.
          if (resumed.status !== 206 || !resumed.body) {
            controller.error(
              new DownloadError({ status: resumed.status, resumable: false, url }),
            );
            return;
          }
          body = resumed.body;
        }
      }
    },
  });

  // Tee so the caller consumes one branch while the cache stores the other.
  // Caching first.clone() instead would be wrong twice over: an unconsumed clone
  // branch stalls and reorders the error relative to buffered bytes, and an
  // interrupted transfer would be cached truncated. Teeing the resumed output
  // means only complete, resume-repaired bytes reach the cache.
  const [forCaller, forCache] = stream.tee();

  void store
    .put(
      url,
      new Response(forCache, {
        headers: totalBytes ? { 'content-length': String(totalBytes) } : {},
      }),
    )
    .catch((error) => {
      console.warn('[in-browser-llm] caching failed, continuing without cache:', error);
    });

  return { stream: reWrap(forCaller), fromCache: false, totalBytes };
}
