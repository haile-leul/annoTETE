// Timeline: frame ruler (with the work range, keyframe step ticks and a strip
// showing where unassigned detections remain) and a track "gantt" chart.
// Wheel zooms the time axis (Ctrl/Alt+wheel over the track rows), drag on the
// ruler scrubs, click on a track row selects that track at that frame.
import { idColor, clamp } from './util.js';

export class Timeline {
  constructor(ruler, gantt, wrap, handler) {
    this.r = ruler; this.g = gantt; this.wrap = wrap; this.h = handler;
    this.d = { n: 1, range: [0, 0], frame: 0, step: 10, tracks: [], coverage: null, active: null, keys: [], fps: 25 };
    this.t0 = 0; this.t1 = 1;
    this._raf = null;
    this._bind();
    new ResizeObserver(() => this.render()).observe(wrap.parentElement);
  }

  set(patch, refit = false) {
    Object.assign(this.d, patch);
    if (refit) { this.t0 = this.d.range[0]; this.t1 = this.d.range[1] + 1; }
    this.render();
  }

  // keep the current frame visible when zoomed in
  reveal(f) {
    const span = this.t1 - this.t0;
    if (f < this.t0 || f >= this.t1) {
      this.t0 = clamp(f - span * 0.25, 0, Math.max(0, this.d.n - span));
      this.t1 = this.t0 + span;
    }
  }

  x(f, w) { return ((f - this.t0) / (this.t1 - this.t0)) * w; }
  f(x, w) { return Math.floor(this.t0 + (x / w) * (this.t1 - this.t0)); }

  rows() { return this.d.tracks.filter((t) => t.visible); }

  rowH() {
    const n = this.rows().length;
    return n <= 8 ? 14 : n <= 20 ? 10 : n <= 60 ? 7 : 5;
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._drawRuler(); this._drawGantt(); });
  }

  _canvas(c, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.parentElement.clientWidth || c.clientWidth;
    if (cssH != null) c.style.height = `${cssH}px`;
    const hh = cssH != null ? cssH : c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hh * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr);
    }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [ctx, w, hh];
  }

  _drawRuler() {
    const [ctx, W, H] = this._canvas(this.r);
    const d = this.d;
    ctx.fillStyle = '#12161b'; ctx.fillRect(0, 0, W, H);
    // outside the work range
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    const ra = this.x(d.range[0], W), rb = this.x(d.range[1] + 1, W);
    if (ra > 0) ctx.fillRect(0, 0, ra, H);
    if (rb < W) ctx.fillRect(rb, 0, W - rb, H);
    // coverage strip: red where free detections remain
    const cov = d.coverage;
    if (cov && cov.bins.length) {
      const n = cov.end - cov.start + 1;
      const max = Math.max(1, ...cov.bins);
      cov.bins.forEach((c, i) => {
        if (!c) return;
        const fa = cov.start + (i * n) / cov.bins.length, fb = cov.start + ((i + 1) * n) / cov.bins.length;
        const xa = this.x(fa, W), xb = Math.max(xa + 1, this.x(fb, W));
        if (xb < 0 || xa > W) return;
        ctx.fillStyle = `rgba(239, 91, 91, ${0.25 + 0.75 * (c / max)})`;
        ctx.fillRect(xa, H - 6, xb - xa, 6);
      });
    }
    // ticks
    const span = this.t1 - this.t0;
    const px = W / span;
    const steps = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000];
    const major = steps.find((s) => s * px >= 80) || 100000;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    if (d.step * px >= 6) { // keyframe step grid
      ctx.fillStyle = 'rgba(55,195,166,0.35)';
      for (let f = Math.ceil(this.t0 / d.step) * d.step; f < this.t1; f += d.step) ctx.fillRect(this.x(f, W), H - 12, 1, 5);
    }
    for (let f = Math.ceil(this.t0 / major) * major; f < this.t1; f += major) {
      const xx = this.x(f, W);
      ctx.fillStyle = '#3a4350'; ctx.fillRect(xx, 0, 1, 10);
      ctx.fillStyle = '#8a96a6'; ctx.fillText(String(f), xx + 3, 2);
    }
    const minor = major / 5;
    if (minor * px >= 8) {
      ctx.fillStyle = '#2a313b';
      for (let f = Math.ceil(this.t0 / minor) * minor; f < this.t1; f += minor) ctx.fillRect(this.x(f, W), 0, 1, 5);
    }
    // active track keys
    if (d.keys && d.keys.length) {
      ctx.fillStyle = d.active != null ? idColor(d.active) : '#fff';
      for (const [f, flags] of d.keys) {
        const xx = this.x(f + 0.5, W);
        if (xx < -4 || xx > W + 4) continue;
        if (flags & 1) { ctx.fillStyle = '#8a96a6'; ctx.fillRect(xx - 3, 17, 6, 2); ctx.fillStyle = idColor(d.active); continue; }
        if (px < 9) { ctx.fillRect(xx - 0.5, 14, 1, 8); continue; } // dense keys: ticks
        ctx.beginPath(); ctx.moveTo(xx, 14); ctx.lineTo(xx + 4, 18); ctx.lineTo(xx, 22); ctx.lineTo(xx - 4, 18); ctx.fill();
      }
    }
    this._cursor(ctx, W, H);
  }

  _cursor(ctx, W, H) {
    const d = this.d;
    const xa = this.x(d.frame, W), xb = this.x(d.frame + 1, W);
    ctx.fillStyle = 'rgba(245,165,36,0.25)';
    if (xb - xa > 2) ctx.fillRect(xa, 0, xb - xa, H);
    ctx.fillStyle = '#f5a524';
    ctx.fillRect(Math.round((xa + xb) / 2) - 1, 0, 2, H);
  }

  _drawGantt() {
    const rows = this.rows();
    const rh = this.rowH();
    const H = Math.max(this.wrap.clientHeight - 2, rows.length * rh + 4);
    const [ctx, W] = this._canvas(this.g, H);
    const d = this.d;
    ctx.fillStyle = '#0f1216'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    const ra = this.x(d.range[0], W), rb = this.x(d.range[1] + 1, W);
    if (ra > 0) ctx.fillRect(0, 0, ra, H);
    if (rb < W) ctx.fillRect(rb, 0, W - rb, H);
    rows.forEach((t, i) => {
      const y = 2 + i * rh;
      if (t.id === d.active) { ctx.fillStyle = 'rgba(255,255,255,0.07)'; ctx.fillRect(0, y - 1, W, rh); }
      ctx.fillStyle = idColor(t.id, t.id === d.active ? 1 : 0.8);
      for (const [a, b] of t.runs) {
        const xa = this.x(a, W), xb = this.x(b + 1, W);
        if (xb < 0 || xa > W) continue;
        ctx.fillRect(xa, y, Math.max(1, xb - xa), rh - 2);
      }
    });
    if (d.active != null && d.keys && W / (this.t1 - this.t0) >= 4) { // key ticks on the active row (if not too dense)
      const i = rows.findIndex((t) => t.id === d.active);
      if (i >= 0) {
        ctx.fillStyle = '#fff';
        for (const [f, flags] of d.keys) {
          if (flags & 1) continue;
          const xx = this.x(f + 0.5, W);
          if (xx >= 0 && xx <= W) ctx.fillRect(xx - 0.5, 2 + i * rh, 1.5, rh - 2);
        }
      }
    }
    this._cursor(ctx, W, H);
  }

  rowAt(y) { const i = Math.floor((y - 2) / this.rowH()); return this.rows()[i] || null; }

  _bind() {
    const seekAt = (ev, c) => {
      const r = c.getBoundingClientRect();
      return clamp(this.f(ev.clientX - r.left, r.width), 0, this.d.n - 1);
    };
    this.r.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      this.h.seek(seekAt(ev, this.r));
      const mv = (e) => this.h.seek(seekAt(e, this.r));
      const up = () => { window.removeEventListener('mousemove', mv); window.removeEventListener('mouseup', up); };
      window.addEventListener('mousemove', mv);
      window.addEventListener('mouseup', up);
    });
    const zoom = (ev, c) => {
      ev.preventDefault();
      const r = c.getBoundingClientRect();
      const W = r.width;
      if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) {
        const dx = (ev.deltaX || ev.deltaY) / W * (this.t1 - this.t0);
        const span = this.t1 - this.t0;
        this.t0 = clamp(this.t0 + dx, 0, Math.max(0, this.d.n - span));
        this.t1 = this.t0 + span;
      } else {
        const fx = this.t0 + ((ev.clientX - r.left) / W) * (this.t1 - this.t0);
        const k = Math.exp(ev.deltaY * 0.002);
        const span = clamp((this.t1 - this.t0) * k, 20, this.d.n);
        this.t0 = clamp(fx - ((fx - this.t0) / (this.t1 - this.t0)) * span, 0, Math.max(0, this.d.n - span));
        this.t1 = this.t0 + span;
      }
      this.render();
    };
    this.r.addEventListener('wheel', (ev) => zoom(ev, this.r), { passive: false });
    this.g.addEventListener('wheel', (ev) => { if (ev.ctrlKey || ev.altKey) zoom(ev, this.g); }, { passive: false });
    this.g.addEventListener('mousedown', (ev) => {
      const r = this.g.getBoundingClientRect();
      const t = this.rowAt(ev.clientY - r.top);
      const f = seekAt(ev, this.g);
      if (t) this.h.selectTrack(t.id, f); else this.h.seek(f);
    });
    this.g.addEventListener('mousemove', (ev) => {
      const r = this.g.getBoundingClientRect();
      const t = this.rowAt(ev.clientY - r.top);
      const f = seekAt(ev, this.g);
      this.g.title = t ? `#${t.id}  frames ${t.runs[0][0]}-${t.runs[t.runs.length - 1][1]}  (${t.n} boxes)  · frame ${f}` : `frame ${f}`;
    });
  }
}
