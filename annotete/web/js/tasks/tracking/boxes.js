// Per-frame boxes from the server, fetched in chunks and cached until an
// edit invalidates the frames it touched.
import { get } from '../../api.js';

const CHUNK = 50;

export class BoxCache {
  constructor(base) {
    this.base = base;
    this.chunks = new Map(); // chunk index -> {frames} | Promise
    this.gen = 0;            // bumps on invalidation, so late answers are dropped
  }

  peek(f) {
    const c = this.chunks.get(Math.floor(f / CHUNK));
    return c && !(c instanceof Promise) ? c[f] || { d: [], t: [] } : null;
  }

  async get(f) {
    const k = Math.floor(f / CHUNK);
    let c = this.chunks.get(k);
    if (!c) c = this._fetch(k);
    if (c instanceof Promise) c = await c;
    return (c && c[f]) || { d: [], t: [] };
  }

  prefetch(f) { const k = Math.floor(f / CHUNK); if (!this.chunks.has(k)) this._fetch(k); }

  _fetch(k) {
    const gen = this.gen;
    const p = get(`${this.base}/boxes?a=${k * CHUNK}&b=${k * CHUNK + CHUNK - 1}`).then((r) => {
      if (gen !== this.gen) return r.frames; // stale, do not cache
      this.chunks.set(k, r.frames);
      return r.frames;
    }).catch((e) => { this.chunks.delete(k); throw e; });
    this.chunks.set(k, p);
    return p;
  }

  invalidate(range) {
    this.gen++;
    if (!range) { this.chunks.clear(); return; }
    const [a, b] = range;
    for (const k of [...this.chunks.keys()]) {
      if (k * CHUNK <= b && k * CHUNK + CHUNK - 1 >= a) this.chunks.delete(k);
    }
  }

  // boxes of one track over [a, b] from whatever is cached (for trails)
  trail(tid, a, b) {
    const pts = [];
    for (let f = a; f <= b; f++) {
      const fr = this.peek(f);
      if (!fr) continue;
      const t = fr.t.find((x) => x[0] === tid);
      if (t) pts.push([f, t[1] + t[3] / 2, t[2] + t[4] / 2]);
    }
    return pts;
  }
}
