// Frame images: fetched as JPEG from the server, decoded off-thread to
// ImageBitmaps, kept in a small LRU and prefetched around the current frame.

export class FrameLoader {
  constructor(baseUrl, meta, maxCached = 160) {
    this.base = baseUrl;
    this.meta = meta;
    this.cache = new Map(); // key -> ImageBitmap (Map keeps insertion order = LRU)
    this.inflight = new Map();
    this.max = maxCached;
    this.previewW = Math.min(meta.width, 1920);
    this.queue = [];
    this.active = 0;
  }

  key(f, w) { return `${f}@${w}`; }

  cached(f, w = this.previewW) {
    const b = this.cache.get(this.key(f, w));
    if (b) { this.cache.delete(this.key(f, w)); this.cache.set(this.key(f, w), b); }
    return b || null;
  }

  // best cached image for frame f (full resolution preferred)
  best(f) { return this.cached(f, 0) || this.cached(f, this.previewW); }

  load(f, w = this.previewW, hint = []) {
    const k = this.key(f, w);
    if (this.cache.has(k)) return Promise.resolve(this.cached(f, w));
    if (this.inflight.has(k)) return this.inflight.get(k);
    const qs = `w=${w === this.meta.width ? 0 : w}${hint.length ? `&next=${hint.join(',')}` : ''}`;
    const p = fetch(`${this.base}/frame/${f}?${qs}`)
      .then((r) => { if (!r.ok) throw new Error(`frame ${f}: ${r.status}`); return r.blob(); })
      .then((b) => createImageBitmap(b))
      .then((bmp) => {
        this.cache.set(k, bmp);
        while (this.cache.size > this.max) {
          const [old, ob] = this.cache.entries().next().value;
          this.cache.delete(old);
          if (ob.close) ob.close();
        }
        return bmp;
      })
      .finally(() => this.inflight.delete(k));
    this.inflight.set(k, p);
    return p;
  }

  // Background loads, most important first; replaces the previous wish list.
  prefetch(frames, w = this.previewW) {
    const n = this.meta.n_frames;
    this.queue = frames.filter((f) => f >= 0 && f < n && !this.cache.has(this.key(f, w))).map((f) => [f, w]);
    this.pump();
  }

  pump() {
    while (this.active < 3 && this.queue.length) {
      const [f, w] = this.queue.shift();
      this.active++;
      this.load(f, w).catch(() => {}).finally(() => { this.active--; this.pump(); });
    }
  }
}
