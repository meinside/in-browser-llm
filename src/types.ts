export type ModelRef =
  | { runtime: 'transformers'; repo: string }
  | { runtime: 'transformers'; baseUrl: string }
  | { runtime: 'mediapipe'; url: string };

export const HF_BASE = 'https://huggingface.co';

// Both repo ids and explicit base URLs are accepted so self-hosted copies work.
export function resolveBaseUrl(ref: ModelRef): string {
  if (ref.runtime === 'mediapipe') return ref.url;
  if ('repo' in ref) return `${HF_BASE}/${ref.repo}/resolve/main`;
  return ref.baseUrl.replace(/\/$/, '');
}
