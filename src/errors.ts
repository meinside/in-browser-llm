// Named types so callers can branch on the cause instead of matching message text.
// The buffer-limit case is deliberately absent: it is decided before downloading by
// gate(), which reports it as a blocker string rather than an exception.

export class CapabilityError extends Error {
  // `cause` is optional because most capability refusals originate here rather than
  // wrapping anything. It matters when a runtime's own error is replaced by a clearer
  // one: the original stays reachable instead of being discarded.
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'CapabilityError';
  }
}

export class DownloadError extends Error {
  readonly status: number;
  readonly resumable: boolean;

  constructor(opts: { status: number; resumable: boolean; url: string }) {
    super(`Download failed with HTTP ${opts.status}: ${opts.url}`);
    this.name = 'DownloadError';
    this.status = opts.status;
    this.resumable = opts.resumable;
  }
}

export class RuntimeError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'RuntimeError';
  }
}
