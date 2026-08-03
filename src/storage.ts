const DEFAULT_CACHE = 'in-browser-llm';

export interface CacheStore {
  match(url: string): Promise<Response | undefined>;
  put(url: string, response: Response): Promise<void>;
}

// Wrapped so OPFS can replace the Cache API later. OPFS is the only option with a
// streaming write path, which matters for multi-GB weights, but it is out of v0 scope.
export function createCacheStore(name: string = DEFAULT_CACHE): CacheStore {
  return {
    async match(url) {
      if (typeof caches === 'undefined') return undefined;
      const cache = await caches.open(name);
      return cache.match(url);
    },
    async put(url, response) {
      if (typeof caches === 'undefined') return;
      const cache = await caches.open(name);
      await cache.put(url, response);
    },
  };
}

// Best-effort storage is evicted under quota pressure — whole origin at a time.
// Firefox shows a permission prompt here; Chrome and Safari decide heuristically.
// Expose the result so callers can explain repeated downloads to the user.
export async function requestPersistence(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.persist) return false;
  try {
    if (await navigator.storage.persisted()) return true;
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}
