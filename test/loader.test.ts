import { describe, expect, it, vi } from 'vitest';
import { fetchWithProgress } from '../src/loader.js';
import { DownloadError } from '../src/errors.js';

const EMPTY_STORE = { match: async () => undefined, put: async () => {} };

function bodyOf(bytes: number[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(bytes));
      controller.close();
    },
  });
}

// Models a real interrupted transfer: bytes are delivered to the consumer, and
// only then does the connection drop. Calling controller.error() in the same tick
// would discard the queued chunk, which is not how a dropped socket behaves.
function failingBody(bytes: number[]): ReadableStream<Uint8Array> {
  let sent = false;
  return new ReadableStream({
    pull(controller) {
      if (!sent) {
        sent = true;
        controller.enqueue(new Uint8Array(bytes));
        return;
      }
      controller.error(new Error('connection dropped'));
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<number[]> {
  const out: number[] = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(...value);
  }
  return out;
}

describe('fetchWithProgress', () => {
  it('reports progress and passes bytes through', async () => {
    const doFetch = vi.fn(
      async () => new Response(bodyOf([1, 2, 3, 4]), { headers: { 'content-length': '4' } }),
    ) as unknown as typeof fetch;

    const seen: number[] = [];
    const result = await fetchWithProgress('https://example.test/m', {
      store: EMPTY_STORE,
      fetch: doFetch,
      onProgress: (p) => seen.push(p.loaded),
    });

    expect(await collect(result.stream)).toEqual([1, 2, 3, 4]);
    expect(result.totalBytes).toBe(4);
    expect(seen).toEqual([4]);
  });

  it('returns the cached body on a hit', async () => {
    const store = {
      match: async () => new Response(bodyOf([9]), { headers: { 'content-length': '1' } }),
      put: async () => {},
    };
    const result = await fetchWithProgress('https://example.test/m', {
      store,
      fetch: (() => {
        throw new Error('should not fetch on a cache hit');
      }) as unknown as typeof fetch,
    });
    expect(result.fromCache).toBe(true);
  });

  it('reports completion on a cache hit so the progress bar does not freeze', async () => {
    const store = {
      match: async () => new Response(bodyOf([1, 2]), { headers: { 'content-length': '2' } }),
      put: async () => {},
    };
    const seen: number[] = [];
    await fetchWithProgress('https://example.test/m', {
      store,
      fetch: (() => {
        throw new Error('should not fetch');
      }) as unknown as typeof fetch,
      onProgress: (p) => seen.push(p.ratio),
    });
    expect(seen).toEqual([1]);
  });

  it('resumes with a Range request after an interrupted transfer', async () => {
    let call = 0;
    const doFetch = vi.fn(async (_url: string, init?: RequestInit) => {
      call += 1;
      if (call === 1) {
        return new Response(failingBody([1, 2]), {
          headers: { 'content-length': '4', 'accept-ranges': 'bytes' },
        });
      }
      expect((init?.headers as Record<string, string>).Range).toBe('bytes=2-');
      return new Response(bodyOf([3, 4]), { status: 206 });
    }) as unknown as typeof fetch;

    const result = await fetchWithProgress('https://example.test/m', {
      store: EMPTY_STORE,
      fetch: doFetch,
    });

    // Resumed bytes must append, never duplicate.
    expect(await collect(result.stream)).toEqual([1, 2, 3, 4]);
  });

  it('fails when the server answers 200 instead of 206 on resume', async () => {
    let call = 0;
    const doFetch = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        return new Response(failingBody([1]), {
          headers: { 'content-length': '4', 'accept-ranges': 'bytes' },
        });
      }
      return new Response(bodyOf([1, 2, 3, 4]), { status: 200 });
    }) as unknown as typeof fetch;

    const result = await fetchWithProgress('https://example.test/m', {
      store: EMPTY_STORE,
      fetch: doFetch,
    });
    await expect(collect(result.stream)).rejects.toThrow(DownloadError);
  });

  it('throws DownloadError with the status on a failed request', async () => {
    const doFetch = vi.fn(
      async () => new Response(null, { status: 404 }),
    ) as unknown as typeof fetch;
    await expect(
      fetchWithProgress('https://example.test/m', { store: EMPTY_STORE, fetch: doFetch }),
    ).rejects.toThrow(DownloadError);
  });

  it('does not cache a truncated body when the transfer cannot be resumed', async () => {
    // This is what tee() exists for: caching first.clone() would store whatever
    // arrived before the connection dropped, and a later run would load it as if
    // complete.
    const stored: number[] = [];
    const store = {
      match: async () => undefined,
      put: async (_url: string, response: Response) => {
        const reader = response.body!.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          stored.push(...value);
        }
      },
    };
    const doFetch = (async () =>
      new Response(failingBody([1, 2]), {
        headers: { 'content-length': '4' }, // no accept-ranges, so no resume
      })) as unknown as typeof fetch;

    const result = await fetchWithProgress('https://example.test/m', {
      store,
      fetch: doFetch,
    });
    await expect(collect(result.stream)).rejects.toThrow(DownloadError);
    // The cache branch errored too, so no complete body was ever stored.
    expect(stored.length).toBeLessThan(4);
  });

  it('does not retry when the server does not accept ranges', async () => {
    const doFetch = vi.fn(
      async () => new Response(failingBody([1]), { headers: { 'content-length': '4' } }),
    ) as unknown as typeof fetch;

    const result = await fetchWithProgress('https://example.test/m', {
      store: EMPTY_STORE,
      fetch: doFetch,
    });
    await expect(collect(result.stream)).rejects.toThrow(DownloadError);
    expect(doFetch).toHaveBeenCalledTimes(1);
  });
});
