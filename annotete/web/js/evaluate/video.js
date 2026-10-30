// evaluATE video view: the frames with ground truth + tracker boxes and their
// match status, a per object timeline (which tracker id covered it when), and
// the list of events (swaps, id jumps, lost tracks, ghost tracks).
import { get } from '../api.js';
import { $, h, idColor, toast, clamp, fmtTime } from '../util.js';
import { Viewer } from '../viewer.js';
import { FrameLoader } from '../frames.js';
import { M, EVENTS, fmt, delta } from './app.js';

const MISS = '#ef5b5b';      // missed although the detector had a box
const NODET = '#8a6fd8';     // missed, no detection there
const GHOST = '#ef5b5b';
const EVWIN = 12;            // frames around an event where it is highlighted

export class VideoView {
  constructor(app, seq) {
    this.app = app;
    this.seq = seq;
    this.rid = app.summary.id;
    this.ti = 0;
    this.frame = 0;
    this.playing = false;
    this.sel = null;          // selected event
    this.selGt = null;        // highlighted ground truth object
    this.typeFilter = new Set(Object.keys(EVENTS));
    this.details = {};
    this.dead = false;
    this.ac = new AbortController();
    this.viewer = new Viewer($('#canvas'), this);
    this.tl = { t0: 0, t1: 1 };
    for (const id of ['optGt', 'optTrk', 'optTrails', 'optLabels']) $(`#${id}`).onchange = () => this.viewer.render();
    for (const b of document.querySelectorAll('#video .transport [data-act]')) {
      b.onclick = () => { this.act(b.dataset.act); b.blur(); };
    }
    $('#fitBtn').onclick = () => this.viewer.fit();
    this.bindTimeline();
    this.load(0);
  }

  destroy() {
    this.dead = true;
    this.playing = false;
    this.viewer.destroy();
    this.ac.abort();
  }

  pause() { this.playing = false; this.updatePlay(); }

  get D() { return this.details[this.ti]; }

  async load(ti) {
    this.ti = ti;
    if (!this.details[ti]) {
      $('#loading').classList.remove('hidden');
      try {
        this.details[ti] = await get(`/api/eval/${this.rid}/seq/${encodeURIComponent(this.seq)}/${ti}`);
      } catch (e) { toast(e.message, 'err', 6000); return; } finally { $('#loading').classList.add('hidden'); }
      if (this.dead) return;
    }
    const D = this.D;
    if (!this.frames && D.has_video) {
      try {
        const meta = await get(`/api/eval/${this.rid}/video/${encodeURIComponent(this.seq)}/meta`);
        this.frames = new FrameLoader(`/api/eval/${this.rid}/video/${encodeURIComponent(this.seq)}`, meta);
        this.size = [meta.width, meta.height];
      } catch (e) { toast(e.message, 'err', 6000); }
    }
    if (!this.size) this.size = this.extent();
    this.viewer.setImageSize(this.size[0], this.size[1]);
    this.tl = { t0: 0, t1: D.n_frames };
    this.renderSide();
    this.go(this.frame, true);
  }

  extent() { // no video: size from the boxes
    let w = this.D.width || 0, hh = this.D.height || 0;
    if (!w || !hh) {
      for (const fr of this.D.frames) for (const b of fr.g.concat(fr.t)) { w = Math.max(w, b[1] + b[3]); hh = Math.max(hh, b[2] + b[4]); }
    }
    return [Math.max(64, Math.ceil(w)), Math.max(64, Math.ceil(hh))];
  }

  // ------------------------------------------------------------- navigation
  go(f, force) {
    if (this.dead || !this.D) return;
    f = clamp(Math.round(f), 0, this.D.n_frames - 1);
    if (f === this.frame && !force && this.shown === f) return;
    this.frame = f;
    $('#frameLabel').textContent = `${f} / ${this.D.n_frames - 1} · ${fmtTime(f, this.D.fps)}`;
    if (this.frames) {
      const img = this.frames.best(f);
      if (img) this.show(f, img);
      else this.frames.load(f, this.frames.previewW, [f + 1, f + 2]).then((im) => { if (this.frame === f) this.show(f, im); }).catch(() => this.show(f, null));
      this.frames.prefetch(this.playing ? [1, 2, 3, 4, 5, 6].map((k) => f + k) : [f + 1, f - 1, f + 5, f - 5]);
    } else this.show(f, null);
    this.drawTimeline();
    this.updateBanner();
  }

  show(f, img) {
    if (this.dead) return;
    this.shown = f;
    this.viewer.setImage(img);
  }

  act(a) {
    if (a === 'back') { this.app.show('overview'); return; }
    if (a === 'prev') this.go(this.frame - 1);
    if (a === 'next') this.go(this.frame + 1);
    if (a === 'play') this.togglePlay();
    if (a === 'prevEvent') this.stepEvent(-1);
    if (a === 'nextEvent') this.stepEvent(1);
  }

  onKey(e) {
    const k = e.key;
    if (k === 'ArrowRight') { this.go(this.frame + (e.shiftKey ? 10 : 1)); return true; }
    if (k === 'ArrowLeft') { this.go(this.frame - (e.shiftKey ? 10 : 1)); return true; }
    if (k === ' ') { this.togglePlay(); return true; }
    if (k === ']') { this.stepEvent(1); return true; }
    if (k === '[') { this.stepEvent(-1); return true; }
    if (k === 'Escape') { if (this.sel) { this.select(null); } else this.app.show('overview'); return true; }
    if (k === 'Home') { this.go(0); return true; }
    if (k === 'End') { this.go(this.D.n_frames - 1); return true; }
    return false;
  }

  togglePlay() {
    this.playing = !this.playing;
    this.updatePlay();
    if (!this.playing) return;
    let last = performance.now(), acc = 0;
    const tick = (now) => {
      if (!this.playing || this.dead) return;
      acc += (now - last) / 1000 * (this.D.fps || 25);
      last = now;
      if (acc >= 1 && this.shown === this.frame) {
        const k = Math.min(Math.floor(acc), 3);
        acc -= k;
        if (this.frame + k >= this.D.n_frames) { this.pause(); return; }
        this.go(this.frame + k);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  updatePlay() { const b = document.querySelector('#video .transport .play'); if (b) b.innerHTML = this.playing ? '&#10074;&#10074;' : '&#9654;'; }

  events() { return this.D.events.filter((e) => this.typeFilter.has(e.type)); }

  stepEvent(dir) {
    const ev = this.events();
    if (!ev.length) return;
    const i = this.sel ? ev.indexOf(this.sel) : -1;
    let j;
    if (i >= 0) j = clamp(i + dir, 0, ev.length - 1);
    else j = dir > 0 ? ev.findIndex((e) => e.frame > this.frame) : ev.map((e) => e.frame < this.frame).lastIndexOf(true);
    if (j < 0) j = dir > 0 ? ev.length - 1 : 0;
    this.select(ev[j]);
  }

  select(e) {
    this.sel = e;
    this.selGt = e ? (e.tracks[0] ?? null) : null;
    for (const el of document.querySelectorAll('#vside .ev')) el.classList.toggle('on', el._ev === e);
    const on = document.querySelector('#vside .ev.on');
    if (on) { const list = on.parentElement; const top = on.offsetTop - list.offsetTop; if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - 40) list.scrollTop = top - 60; }
    if (e) {
      this.pause();
      // show the moment just before: playing on reveals what happens
      this.go(Math.max(0, e.frame - (e.type === 'lost' || e.type === 'false' ? 2 : 3)), true);
      this.focusOn(e);
    }
    this.viewer.render();
    this.drawTimeline();
  }

  focusOn(e) { // zoom so the objects of the event are well visible
    const fr = this.D.frames[clamp(e.frame, 0, this.D.n_frames - 1)];
    const boxes = fr.g.filter((b) => e.tracks.includes(b[0])).concat(fr.t.filter((b) => (e.trk || []).includes(b[0])));
    if (!boxes.length) return;
    const x0 = Math.min(...boxes.map((b) => b[1])), y0 = Math.min(...boxes.map((b) => b[2]));
    const x1 = Math.max(...boxes.map((b) => b[1] + b[3])), y1 = Math.max(...boxes.map((b) => b[2] + b[4]));
    const [W, H] = this.size;
    if ((x1 - x0) * (y1 - y0) > 0.15 * W * H) { this.viewer.fit(); return; }
    this.viewer.centerOn([x0, y0, x1 - x0, y1 - y0], 0.45);
  }

  updateBanner() {
    const f = this.frame;
    const near = this.D.events.filter((e) => this.typeFilter.has(e.type) && f >= e.frame - EVWIN && f <= (e.end ?? e.frame) + EVWIN);
    const e = this.sel && near.includes(this.sel) ? this.sel : near[0];
    $('#hudEvent').innerHTML = '';
    if (e) {
      $('#hudEvent').append(h('span', { class: `evchip ${e.type}` }, EVENTS[e.type][0]), ' ',
        f < e.frame ? `in ${e.frame - f} fr: ` : '', e.msg);
    }
    $('#hudFrame').textContent = `${f}`;
  }

  // ---------------------------------------------------------------- drawing
  // viewer handler
  down() { return 'pan'; }
  drag() {}
  up() {}
  click(p) {
    const hit = this.hit(p);
    this.selGt = hit ? hit : null;
    this.viewer.render();
    this.drawTimeline();
  }
  dblclick() { return false; }
  context() {}
  hover(p) {
    if (!p) { $('#hudHover').textContent = ''; return; }
    const fr = this.D && this.D.frames[this.frame];
    if (!fr) return;
    let best = null;
    for (const b of fr.t) if (inside(p, b)) best = !best || b[3] * b[4] < best[3] * best[4] ? ['t', b] : best;
    for (const b of fr.g) if (inside(p, b) && (!best || b[3] * b[4] < best[1][3] * best[1][4])) best = ['g', b];
    if (!best) { $('#hudHover').textContent = ''; return; }
    const [kind, b] = best;
    $('#hudHover').textContent = kind === 't'
      ? (b[5] != null ? `tracker id ${b[0]} → object ${b[5]}` : `tracker id ${b[0]}: false box (no object here)`)
      : (b[6] != null ? `object ${b[0]} ← tracker id ${b[6]}${b[5] === 3 ? ' (interpolated in the annotation)' : ''}`
        : `object ${b[0]} missed${b[5] === 1 || b[5] === 3 ? ': the detector had no box' : ': the detector had a box'}`);
  }

  hit(p) {
    const fr = this.D.frames[this.frame];
    let best = null;
    for (const b of fr.g) if (inside(p, b) && (!best || b[3] * b[4] < best[3] * best[4])) best = b;
    return best ? best[0] : null;
  }

  draw(ctx, view) {
    const D = this.D;
    if (!D) return;
    $('#fitBtn').classList.toggle('hidden', view.whole);
    const fr = D.frames[this.frame];
    if (!fr) return;
    const showGt = $('#optGt').checked, showTrk = $('#optTrk').checked, labels = $('#optLabels').checked;
    const evGt = new Set(this.sel ? this.sel.tracks : []);
    const evTrk = new Set(this.sel ? this.sel.trk || [] : []);
    if (this.selGt != null) evGt.add(this.selGt);
    const focus = evGt.size > 0;
    // trails: path of the highlighted objects, colored by the tracker id that had them
    if ($('#optTrails').checked) {
      const ids = focus ? [...evGt] : fr.g.map((b) => b[0]);
      for (const gid of ids) this.trail(ctx, view, gid, focus ? 45 : 12, focus ? 3 : 1.5);
    }
    if (showGt) {
      for (const b of fr.g) {
        const [x, y, w, hh] = view.rect(b.slice(1, 5));
        const missed = b[6] == null, nodet = b[5] === 1 || b[5] === 3;
        const hl = evGt.has(b[0]);
        ctx.save();
        if (missed) {
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = nodet ? NODET : MISS;
          ctx.lineWidth = hl ? 3 : 2;
          ctx.fillStyle = nodet ? 'rgba(138,111,216,0.12)' : 'rgba(239,91,91,0.12)';
          ctx.fillRect(x, y, w, hh);
        } else {
          ctx.strokeStyle = 'rgba(255,255,255,0.75)';
          ctx.lineWidth = hl ? 2 : 1;
        }
        if (hl) { ctx.shadowColor = '#f5a524'; ctx.shadowBlur = 14; }
        ctx.strokeRect(x, y, w, hh);
        ctx.restore();
        if (labels && (missed || hl)) tag(ctx, x, y + hh + 15, missed ? `obj ${b[0]} missed${nodet ? ' · no det' : ''}` : `obj ${b[0]}`, missed ? (nodet ? NODET : MISS) : '#ddd', true);
      }
    }
    if (showTrk) {
      for (const b of fr.t) {
        const [x, y, w, hh] = view.rect(b.slice(1, 5));
        const fp = b[5] == null;
        const hl = evTrk.has(b[0]) || (b[5] != null && evGt.has(b[5]));
        const col = fp ? GHOST : idColor(b[0]);
        ctx.save();
        ctx.strokeStyle = col;
        ctx.lineWidth = hl ? 3.5 : 2;
        if (fp) ctx.setLineDash([3, 3]);
        if (hl) { ctx.shadowColor = col; ctx.shadowBlur = 12; }
        if (focus && !hl) ctx.globalAlpha = 0.45;
        ctx.strokeRect(x, y, w, hh);
        ctx.restore();
        if (labels || hl) tag(ctx, x, y, fp ? `T${b[0]} · false` : `T${b[0]} → obj ${b[5]}`, col, fp);
      }
    }
  }

  trail(ctx, view, gid, span, width) {
    const D = this.D;
    const a = Math.max(0, this.frame - span), z = Math.min(D.n_frames - 1, this.frame + span);
    let prev = null;
    for (let f = a; f <= z; f++) {
      const b = D.frames[f].g.find((x) => x[0] === gid);
      if (!b) { prev = null; continue; }
      const s = view.toScreen(b[1] + b[3] / 2, b[2] + b[4] / 2);
      if (prev) {
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(s.x, s.y);
        ctx.setLineDash(b[6] == null ? [3, 4] : []);
        ctx.strokeStyle = b[6] == null ? (b[5] === 1 || b[5] === 3 ? NODET : MISS) : idColor(b[6]);
        ctx.globalAlpha = f > this.frame ? 0.45 : 0.95;
        ctx.lineWidth = width;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.setLineDash([]);
      }
      prev = s;
    }
  }

  // ----------------------------------------------------------------- sidebar
  renderSide() {
    const D = this.D;
    const S = this.app.summary;
    const side = $('#vside');
    side.innerHTML = '';
    const v = this.app.variant;
    const mine = S.results[this.ti].seqs[this.seq], other = S.results.length === 2 ? S.results[1 - this.ti].seqs[this.seq] : null;
    const r = mine[v] || mine.all;
    const tile = (k, main) => h('div', { class: `tile${main ? ' hi' : ''}`, title: M[k][1] },
      h('div', { class: 'k' }, h('span', {}, M[k][0]), other && other[v] && this.ti === 0 ? delta(k, r[k], other[v][k], true) : ''),
      h('div', { class: 'v' }, fmt(k, r[k])));
    const card = h('div', { class: 'card' },
      h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('h4', { style: { margin: 0 } }, this.seq),
        S.results.length === 2 ? h('div', { class: 'seg' }, S.results.map((res, i) => h('button', { class: i === this.ti ? 'on' : '', onclick: () => this.load(i) }, res.name))) : ''),
      h('div', { class: 'mtiles', style: { marginTop: '8px' } }, ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN'].map((k, i) => tile(k, i === 0))),
      h('div', { class: 'muted small', style: { marginTop: '6px' } }, v === 'det' ? 'detector-visible boxes only' : 'all annotated boxes', ' · ', D.align),
    );
    const legend = h('div', { class: 'card' }, h('h4', {}, 'On the video'),
      h('div', { class: 'legend' },
        h('span', { style: { '--c': '#37c3a6' } }, 'tracker box (color = tracker id) → object'),
        h('span', { style: { '--c': '#ddd' } }, 'annotated object, tracked'),
        h('span', { style: { '--c': MISS } }, 'missed (detector had a box)'),
        h('span', { style: { '--c': NODET } }, 'missed (no detection)'),
        h('span', { style: { '--c': GHOST } }, 'dashed red: false box')),
      h('div', { class: 'muted small', style: { marginTop: '6px' } }, 'Trails and timeline rows are colored by the tracker id that had the object: a color change is an id switch.'));
    const chips = h('div', { class: 'chiprow' }, Object.entries(EVENTS).map(([k, [label]]) => {
      const n = D.events.filter((e) => e.type === k).length;
      return h('label', {}, h('input', { type: 'checkbox', checked: this.typeFilter.has(k), onchange: (e) => {
        e.target.checked ? this.typeFilter.add(k) : this.typeFilter.delete(k); this.renderEvents(); this.drawTimeline(); this.updateBanner();
      } }), h('span', { class: `evchip ${k}` }, `${label} ${n}`));
    }));
    this.evList = h('div', { class: 'list evlist' });
    side.append(card, legend, h('div', { class: 'card' }, h('h4', {}, 'Events'), chips,
      h('div', { class: 'muted small', style: { marginTop: '6px' } }, 'Click one to jump there. [ ] walk through them.',
        mine.detector.flicker ? ` Not listed: ${mine.detector.flicker} short detector gaps the tracker bridged with the same id.` : '')), this.evList);
    side.style.display = 'flex';
    side.style.flexDirection = 'column';
    side.style.gap = '10px';
    side.style.padding = '10px';
    side.style.overflow = 'auto';
    this.renderEvents();
  }

  renderEvents() {
    const list = this.evList;
    list.innerHTML = '';
    const ev = this.events();
    if (!ev.length) { list.append(h('div', { class: 'empty' }, this.D.events.length ? 'none of the selected kinds' : 'No problems found in this video 🎉')); return; }
    const compare = this.app.summary.results.length === 2;
    for (const e of ev) {
      const badge = compare && e.diff ? h('span', { class: `evchip ${e.diff === 'new' ? 'newdiff' : 'fixed'}`, title: e.diff === 'new'
        ? `${this.app.summary.results[0].name} has this, ${this.app.summary.results[1].name} did not` : `gone in ${this.app.summary.results[0].name}` },
      e.diff === 'new' ? 'new' : 'fixed') : '';
      const el = h('div', { class: `ev${e === this.sel ? ' on' : ''}`, onclick: () => this.select(e) },
        h('div', { class: 'f' }, `f ${e.frame}`, e.end != null && e.end !== e.frame ? h('div', {}, `→ ${e.end}`) : ''),
        h('div', { class: 'm' }, h('div', { class: 't' }, h('span', { class: `evchip ${e.type}` }, EVENTS[e.type][0]), ' ', badge), e.msg));
      el._ev = e;
      list.append(el);
    }
  }

  // ---------------------------------------------------------------- timeline
  bindTimeline() {
    const r = $('#tlRuler'), rows = $('#tlRows');
    const sig = { signal: this.ac.signal };
    const fAt = (ev, c) => { const b = c.getBoundingClientRect(); return clamp(Math.floor(this.tl.t0 + (ev.clientX - b.left) / b.width * (this.tl.t1 - this.tl.t0)), 0, this.D.n_frames - 1); };
    r.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      this.go(fAt(ev, r));
      const mv = (e) => this.go(fAt(e, r));
      const up = () => { window.removeEventListener('mousemove', mv); window.removeEventListener('mouseup', up); };
      window.addEventListener('mousemove', mv);
      window.addEventListener('mouseup', up);
    }, sig);
    rows.addEventListener('mousedown', (ev) => {
      const b = rows.getBoundingClientRect();
      const i = Math.floor((ev.clientY - b.top - 14) / this.rowH());
      const row = this.D.gt_rows[i];
      // an event marker under the mouse?
      const f = fAt(ev, rows);
      const mark = this._marks && this._marks.find((m) => Math.abs(m.x - (ev.clientX - b.left)) <= 5 && Math.abs(m.y - (ev.clientY - b.top)) <= 7);
      if (mark) { this.select(mark.e); return; }
      if (row) this.selGt = row.id;
      this.go(f);
      this.viewer.render();
    }, sig);
    const zoom = (ev, c) => {
      ev.preventDefault();
      const b = c.getBoundingClientRect();
      const n = this.D.n_frames;
      const fx = this.tl.t0 + (ev.clientX - b.left) / b.width * (this.tl.t1 - this.tl.t0);
      const span = clamp((this.tl.t1 - this.tl.t0) * Math.exp(ev.deltaY * 0.002), 20, n);
      this.tl.t0 = clamp(fx - (fx - this.tl.t0) / (this.tl.t1 - this.tl.t0) * span, 0, Math.max(0, n - span));
      this.tl.t1 = this.tl.t0 + span;
      this.drawTimeline();
    };
    r.addEventListener('wheel', (ev) => zoom(ev, r), { passive: false, signal: this.ac.signal });
    rows.addEventListener('wheel', (ev) => { if (ev.ctrlKey || ev.altKey) zoom(ev, rows); }, { passive: false, signal: this.ac.signal });
    rows.addEventListener('mousemove', (ev) => {
      const b = rows.getBoundingClientRect();
      const i = Math.floor((ev.clientY - b.top - 14) / this.rowH());
      const row = this.D && this.D.gt_rows[i];
      rows.title = row ? `object ${row.id}` : '';
    }, sig);
    new ResizeObserver(() => this.drawTimeline()).observe($('#tlWrap'));
  }

  rowH() { const n = this.D ? this.D.gt_rows.length : 1; return n <= 10 ? 14 : n <= 30 ? 9 : 6; }

  drawTimeline() {
    if (this.dead || !this.D) return;
    const D = this.D;
    const dpr = window.devicePixelRatio || 1;
    const x = (f, W) => (f - this.tl.t0) / (this.tl.t1 - this.tl.t0) * W;
    // ruler + false box lane
    const r = $('#tlRuler');
    const W = r.parentElement.clientWidth;
    r.width = W * dpr; r.height = 30 * dpr;
    let c = r.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#12161b'; c.fillRect(0, 0, W, 30);
    const span = this.tl.t1 - this.tl.t0;
    const steps = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];
    const major = steps.find((s) => s * W / span >= 80) || 10000;
    c.font = '10px ui-monospace, monospace'; c.textBaseline = 'top';
    for (let f = Math.ceil(this.tl.t0 / major) * major; f < this.tl.t1; f += major) {
      c.fillStyle = '#3a4350'; c.fillRect(x(f, W), 0, 1, 8);
      c.fillStyle = '#8a96a6'; c.fillText(String(f), x(f, W) + 3, 1);
    }
    for (let f = Math.floor(this.tl.t0); f < this.tl.t1; f++) { // false boxes per frame
      const n = D.frames[f] ? D.frames[f].t.filter((b) => b[5] == null).length : 0;
      if (n) { c.fillStyle = `rgba(239,91,91,${Math.min(1, 0.35 + 0.2 * n)})`; c.fillRect(x(f, W), 22, Math.max(1, x(f + 1, W) - x(f, W)), 8); }
    }
    cursor(c, x(this.frame, W), x(this.frame + 1, W), 30);
    // rows: one per object, colored by the tracker id that had it
    const rows = $('#tlRows');
    const rh = this.rowH();
    const H = Math.max($('#tlWrap').clientHeight - 2, D.gt_rows.length * rh + 18);
    rows.style.height = `${H}px`;
    rows.width = W * dpr; rows.height = H * dpr;
    c = rows.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#0f1216'; c.fillRect(0, 0, W, H);
    const sel = this.sel ? new Set(this.sel.tracks) : new Set();
    D.gt_rows.forEach((row, i) => {
      const y = 14 + i * rh;
      if (sel.has(row.id) || row.id === this.selGt) { c.fillStyle = 'rgba(245,165,36,0.12)'; c.fillRect(0, y - 1, W, rh); }
      for (const [a, b, v] of row.segs) {
        const xa = x(a, W), xb = x(b + 1, W);
        if (xb < 0 || xa > W) continue;
        c.fillStyle = v === -1 ? MISS : v === -2 ? NODET : idColor(v);
        c.globalAlpha = v < 0 ? 0.75 : 0.95;
        c.fillRect(xa, y, Math.max(1, xb - xa), rh - 2);
        c.globalAlpha = 1;
      }
    });
    // event markers
    this._marks = [];
    const rowOf = new Map(D.gt_rows.map((r2, i) => [r2.id, i]));
    for (const e of this.events()) {
      const xx = x(e.frame + 0.5, W);
      if (xx < -5 || xx > W + 5) continue;
      const i = e.tracks.length ? rowOf.get(e.tracks[0]) : null;
      const y = i != null ? 14 + i * rh + rh / 2 : 6;
      c.fillStyle = { swap: '#ff6b6b', transfer: '#ff6b6b', new_id: '#ffc14d', switch: '#ffc14d', lost: '#8fbaff', false: '#c3b3ff' }[e.type];
      c.beginPath(); c.moveTo(xx, y - 6); c.lineTo(xx + 5, y); c.lineTo(xx, y + 6); c.lineTo(xx - 5, y); c.fill();
      if (e === this.sel) { c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.stroke(); }
      this._marks.push({ x: xx, y, e });
    }
    cursor(c, x(this.frame, W), x(this.frame + 1, W), H);
  }
}

function inside(p, b) { return p.x >= b[1] && p.x <= b[1] + b[3] && p.y >= b[2] && p.y <= b[2] + b[4]; }

function cursor(c, xa, xb, H) {
  c.fillStyle = 'rgba(245,165,36,0.25)';
  if (xb - xa > 2) c.fillRect(xa, 0, xb - xa, H);
  c.fillStyle = '#f5a524';
  c.fillRect(Math.round((xa + xb) / 2) - 1, 0, 2, H);
}

function tag(ctx, x, y, text, col, ghost) {
  ctx.font = '11px ui-monospace, monospace';
  const w = ctx.measureText(text).width + 8;
  const yy = y - 15 < 0 ? y + 1 : y - 15;
  ctx.fillStyle = ghost ? 'rgba(10,12,15,0.85)' : col;
  ctx.fillRect(x, yy, w, 14);
  ctx.fillStyle = ghost ? col : '#0a0c0f';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + 4, yy + 7.5);
}
