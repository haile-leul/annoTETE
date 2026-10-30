// Canvas image viewer: zoom (wheel, at the cursor), pan (drag on empty space
// or middle mouse), fit (double click / Ctrl+0). Coordinates handed to the
// task are image (video pixel) coordinates.
//
// The task supplies a handler:
//   draw(ctx, view)            overlay, in screen pixels (use view.rect(box))
//   down(p, ev) -> 'drag'|'pan'  pointer pressed; 'drag' = the task handles move/up
//   drag(p, ev), up(p, ev, moved)
//   click(p, ev)               click (no drag) that the task did not consume on down
//   hover(p, ev), context(p, ev)

export class Viewer {
  constructor(canvas, handler) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.h = handler;
    this.iw = 1; this.ih = 1;
    this.scale = 1; this.ox = 0; this.oy = 0;
    this.img = null;
    this.fitted = true;
    this._raf = null;
    this.mouse = null;
    this.dead = false;
    this._ac = new AbortController();
    this._bind();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(canvas.parentElement);
  }

  // Stop drawing and listening: the canvas is shared with the next video's viewer.
  destroy() {
    this.dead = true;
    this._ac.abort();
    this._ro.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.img = null;
  }

  setImageSize(w, h) { this.iw = w; this.ih = h; this.fit(); }

  setImage(bmp) { this.img = bmp; this.render(); }

  _resize() {
    if (this.dead) return;
    const r = this.c.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.cw = r.width; this.ch = r.height;
    this.c.width = Math.round(r.width * dpr);
    this.c.height = Math.round(r.height * dpr);
    if (this.fitted) this.fit(); else this.render();
  }

  fit() {
    if (!this.cw) return;
    const s = Math.min(this.cw / this.iw, this.ch / this.ih) * 0.98;
    this.scale = s;
    this.ox = (this.cw - this.iw * s) / 2;
    this.oy = (this.ch - this.ih * s) / 2;
    this.fitted = true;
    this.render();
  }

  // screen (CSS px relative to canvas) <-> image coordinates
  toImage(sx, sy) { return { x: (sx - this.ox) / this.scale, y: (sy - this.oy) / this.scale }; }
  toScreen(x, y) { return { x: x * this.scale + this.ox, y: y * this.scale + this.oy }; }
  rect(b) { return [b[0] * this.scale + this.ox, b[1] * this.scale + this.oy, b[2] * this.scale, b[3] * this.scale]; }

  zoomAt(sx, sy, factor) {
    const p = this.toImage(sx, sy);
    const minS = Math.min(this.cw / this.iw, this.ch / this.ih) * 0.2;
    this.scale = Math.max(minS, Math.min(40, this.scale * factor));
    this.ox = sx - p.x * this.scale;
    this.oy = sy - p.y * this.scale;
    this.fitted = false;
    this.render();
  }

  // Keep box b visible: center on it, optionally zooming so it fills ~`fill` of the view.
  centerOn(b, fill = 0) {
    if (fill) {
      const s = Math.min(this.cw * fill / Math.max(b[2], 8), this.ch * fill / Math.max(b[3], 8));
      this.scale = Math.max(Math.min(this.cw / this.iw, this.ch / this.ih), Math.min(12, s));
    }
    this.ox = this.cw / 2 - (b[0] + b[2] / 2) * this.scale;
    this.oy = this.ch / 2 - (b[1] + b[3] / 2) * this.scale;
    this.fitted = false;
    this.render();
  }

  // only move if the box is (partly) outside the central area
  follow(b) {
    const [x, y, w, hh] = this.rect(b);
    const m = 0.18;
    if (x < this.cw * m || y < this.ch * m || x + w > this.cw * (1 - m) || y + hh > this.ch * (1 - m)) this.centerOn(b);
  }

  // image px per screen px (> 1 means the frame is shown smaller than native)
  get density() { return 1 / this.scale; }

  render() {
    if (this._raf || this.dead) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._draw(); });
  }

  _draw() {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, this.c.width, this.c.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.img) {
      ctx.imageSmoothingEnabled = this.scale < 2;
      ctx.drawImage(this.img, this.ox, this.oy, this.iw * this.scale, this.ih * this.scale);
    }
    this.h.draw(ctx, this);
  }

  _pt(ev) {
    const r = this.c.getBoundingClientRect();
    const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    return { ...this.toImage(sx, sy), sx, sy };
  }

  _bind() {
    const c = this.c;
    const sig = { signal: this._ac.signal };
    let mode = null, start = null, moved = false, last = null;
    c.addEventListener('mousedown', (ev) => {
      if (ev.button === 2) return;
      ev.preventDefault();
      const p = this._pt(ev);
      start = p; last = p; moved = false;
      mode = ev.button === 1 ? 'pan' : (this.h.down(p, ev) || 'pan');
      const move = (e) => {
        const q = this._pt(e);
        if (Math.abs(q.sx - start.sx) + Math.abs(q.sy - start.sy) > 3) moved = true;
        if (mode === 'pan') {
          if (moved) {
            this.ox += q.sx - last.sx; this.oy += q.sy - last.sy;
            this.fitted = false;
            this.render();
            c.style.cursor = 'grabbing';
          }
        } else this.h.drag(q, e);
        last = q;
      };
      const up = (e) => {
        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
        c.style.cursor = '';
        const q = this._pt(e);
        if (mode === 'pan') { if (!moved) this.h.click(q, e); } else this.h.up(q, e, moved);
        mode = null;
      };
      window.addEventListener('mousemove', move);
      window.addEventListener('mouseup', up);
    }, sig);
    c.addEventListener('mousemove', (ev) => { if (!mode) { this.mouse = this._pt(ev); this.h.hover(this.mouse, ev); } }, sig);
    c.addEventListener('mouseleave', () => { this.mouse = null; this.h.hover(null); }, sig);
    c.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const p = this._pt(ev);
      const k = ev.deltaMode === 1 ? 0.05 : 0.0015;
      this.zoomAt(p.sx, p.sy, Math.exp(-ev.deltaY * k));
    }, { passive: false, signal: this._ac.signal });
    c.addEventListener('contextmenu', (ev) => { ev.preventDefault(); this.h.context(this._pt(ev), ev); }, sig);
    c.addEventListener('dblclick', (ev) => { if (!this.h.dblclick || !this.h.dblclick(this._pt(ev), ev)) this.fit(); }, sig);
  }
}
