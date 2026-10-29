// Tracking task UI: canvas overlay + interactions, keyboard, proposals,
// timeline sync, sidebar panels (panels.js).
//
// Interaction model ("id brush"):
//   - the active track is the brush; clicking a free (white, dashed) detection
//     gives it the brush id. With no brush, a click starts a new track
//     (Shift+click always starts a new track).
//   - clicking a colored track box makes it the brush.
//   - Space accepts the dashed proposals and jumps one step ahead, so a clean
//     stretch costs one key press per step; Shift+Space keeps going on its own
//     until something needs you.
import { get, post, item } from '../../api.js';
import { FrameLoader } from '../../frames.js';
import { Viewer } from '../../viewer.js';
import { Timeline } from '../../timeline.js';
import { $, h, idColor, toast, debounce, fmtTime, menu, prompt, confirm, clamp, store } from '../../util.js';
import { BoxCache } from './boxes.js';
import * as panels from './panels.js';

const KIND = { KEY: 0, MANUAL: 1, AUTO: 2, INTERP: 3 };
const KIND_NAME = ['key', 'key (drawn)', 'auto (detection)', 'interpolated'];
const HANDLE = 6;

export class TrackingTask {
  constructor(app, meta) {
    this.app = app;
    this.meta = meta;
    this.name = meta.name;
    this.base = item(meta.name);
    this.frames = new FrameLoader(this.base, meta);
    this.boxes = new BoxCache(this.base);
    this.step = app.settings.step || 10;
    this.frame = clamp(store.get(`frame:${this.name}`, meta.range[0]), 0, meta.n_frames - 1);
    this.cur = { d: [], t: [] };
    this.shownFrame = -1;
    this.props = { proposals: [], lost: [] };
    this.tracks = [];
    this.trackById = new Map();
    this.active = null;      // brush id (may be an id that does not exist yet)
    this.activeDetail = null;
    this.hovered = null;
    this.pending = null;     // {mode: 'swap'|'merge'|'draw'}
    this.idbuf = '';
    this.dragging = null;
    this.playing = false;
    this.issues = [];
    this.issueIdx = -1;
    this.opts = Object.assign({ labels: true, free: true, trails: true, follow: false, props: true, onlyHere: false },
      store.get('opts', {}));
    this.token = 0;
    this.busy = 0;

    this.viewer = new Viewer($('#canvas'), this);
    this.viewer.setImageSize(meta.width, meta.height);
    this.timeline = new Timeline($('#ruler'), $('#gantt'), $('#ganttWrap'), {
      seek: (f) => this.go(f),
      selectTrack: (tid, f) => { this.setActive(tid); this.go(f); },
    });
    this.timeline.set({ n: meta.n_frames, range: meta.range, frame: this.frame, step: this.step, fps: meta.fps }, true);

    this.refreshTracks = debounce(() => this._refreshTracks(), 120);
    this.refreshCoverage = debounce(() => this._refreshCoverage(), 700);
    this.refreshProps = debounce(() => this._refreshProps(), 60);

    panels.build(this);
    this._bindTransport();
    this._refreshTracks();
    this._refreshCoverage();
    this.go(this.frame, true);
    this.updateChips();
  }

  destroy() {
    this.playing = false;
    store.set(`frame:${this.name}`, this.frame);
  }

  // ------------------------------------------------------------ navigation
  get range() { return this.meta.range; }

  go(f, force = false) {
    f = clamp(Math.round(f), 0, this.meta.n_frames - 1);
    if (f === this.frame && !force && this.shownFrame === f) return;
    const dir = f >= this.frame ? 1 : -1;
    this.frame = f;
    $('#frameInput').value = f;
    $('#timeLabel').textContent = fmtTime(f, this.meta.fps);
    this.timeline.reveal(f);
    this.timeline.set({ frame: f });
    const tok = ++this.token;
    // fast path: everything cached
    const img = this.frames.best(f);
    const bx = this.boxes.peek(f);
    if (img && bx) this._show(f, img, bx);
    else {
      $('#loading').classList.remove('hidden');
      Promise.all([img || this.frames.load(f, this.frames.previewW, this._hint(f, dir)), bx || this.boxes.get(f)])
        .then(([im, b]) => { if (tok === this.token) this._show(f, im, b); })
        .catch((e) => { if (tok === this.token) { $('#loading').classList.add('hidden'); toast(e.message, 'err'); } });
    }
    this._prefetch(f, dir);
    if (!this.playing) this.refreshProps();
    else this.props = { proposals: [], lost: [] };
  }

  _hint(f, dir) {
    const s = this.step;
    return dir > 0 ? [f + 1, f + s, f + 2 * s] : [f - 1, f - s];
  }

  _prefetch(f, dir) {
    const s = this.step;
    const list = this.playing ? [1, 2, 3, 4, 5, 6, 8, 10].map((k) => f + k)
      : dir > 0 ? [f + s, f + 1, f + 2 * s, f - s, f + 2, f - 1] : [f - s, f - 1, f - 2 * s, f + s, f + 1];
    this.frames.prefetch(list);
    for (const g of [f, f + s, f + 2 * s, f - s]) if (g >= 0 && g < this.meta.n_frames) this.boxes.prefetch(g);
    clearTimeout(this._hiresT);
    if (this.viewer.density < 0.9 * this.meta.width / this.frames.previewW && this.frames.previewW < this.meta.width) {
      // zoomed in beyond the preview resolution: fetch the native frame when idle
      this._hiresT = setTimeout(() => this.frames.load(f, this.meta.width).then((im) => {
        if (this.frame === f) { this.viewer.setImage(im); }
      }).catch(() => {}), 180);
    }
  }

  _show(f, img, boxes) {
    $('#loading').classList.add('hidden');
    this.shownFrame = f;
    this.cur = boxes;
    this.viewer.setImage(img);
    if (this.opts.follow && this.active != null) {
      const b = this.trackBox(this.active);
      if (b) this.viewer.follow(b.slice(1, 5));
    }
    this.updateHud();
    panels.markHere(this);
  }

  async reloadFrame() {
    const f = this.frame;
    const b = await this.boxes.get(f);
    if (f === this.frame) { this.cur = b; this.viewer.render(); this.updateHud(); panels.markHere(this); }
  }

  async _refreshProps() {
    if (!this.opts.props) { this.props = { proposals: [], lost: [] }; return; }
    const f = this.frame;
    try {
      const r = await get(`${this.base}/proposals?f=${f}`);
      if (f !== this.frame) return;
      this.props = r;
      this.viewer.render();
      this.updateHud();
    } catch (e) { /* not critical */ }
  }

  stepBy(k) { this.go(this.frame + k); }

  // -------------------------------------------------------------- commands
  async cmd(op, args = {}, { quiet = false } = {}) {
    this.busy++;
    this.app.setSave('busy');
    try {
      const r = await post(`${this.base}/cmd`, { op, ...args });
      if (r.changed) {
        this.boxes.invalidate(r.dirty ? [r.dirty[0] - 1, r.dirty[1] + 1] : null);
        await this.reloadFrame();
        this.refreshProps();
        this.refreshTracks();
        this.refreshCoverage();
        if (this.active != null) this.loadActiveDetail();
      }
      this.meta.next_id = r.next_id;
      this.updateChips();
      $('#undoBtn').disabled = !r.undo;
      $('#redoBtn').disabled = !r.redo;
      this.app.setSave('ok');
      return r.result;
    } catch (e) {
      this.app.setSave('err');
      if (!quiet) toast(e.message, 'err', 4000);
      throw e;
    } finally {
      this.busy--;
    }
  }

  async undo() { const l = await this.cmd('undo'); toast(l ? `Undid ${l}` : 'Nothing to undo'); }
  async redo() { const l = await this.cmd('redo'); toast(l ? `Redid ${l}` : 'Nothing to redo'); }

  // brush assignment of a free detection
  async paint(det, forceNew = false) {
    const prop = this.props.proposals.find((p) => p.det === det[0] && p.frame === this.frame);
    let tid = forceNew ? null : this.active;
    if (tid == null && !forceNew && prop && !this.brushNew) tid = prop.track;
    const id = await this.cmd('assign', { frame: this.frame, det: det[0], track: tid });
    this.setActive(id);
    this.brushNew = false;
    this.updateChips();
  }

  async acceptAndStep() {
    if (this.props.proposals.length) await this.cmd('accept', { frame: this.frame });
    this.go(Math.min(this.frame + this.step, this.meta.n_frames - 1));
  }

  async accept() {
    const n = await this.cmd('accept', { frame: this.frame });
    toast(n ? `Accepted ${n} proposal${n > 1 ? 's' : ''}` : 'No proposals here');
  }

  async advance() {
    toast('Auto-advancing ...', '', 1200);
    const r = await this.cmd('advance', { frame: this.frame, step: this.step });
    this.go(r.frame);
    toast(`${r.steps} step${r.steps === 1 ? '' : 's'}, ${r.accepted} box${r.accepted === 1 ? '' : 'es'} accepted. Stopped: ${r.reason}`,
      r.reason === 'end' ? 'ok' : '', 4500);
  }

  async newTracksHere() {
    const ids = await this.cmd('new_tracks', { frame: this.frame });
    toast(ids.length ? `${ids.length} new track${ids.length > 1 ? 's' : ''}: #${ids.join(', #')}` : 'No free detections here');
  }

  // -------------------------------------------------------------- tracks
  setActive(tid) {
    this.active = tid == null ? null : Number(tid);
    this.brushNew = false;
    this.activeDetail = null;
    if (this.active != null && this.trackById.has(this.active)) this.loadActiveDetail();
    else this.timeline.set({ active: this.active, keys: [] });
    this.updateChips();
    panels.renderActive(this);
    panels.highlightList(this);
    this.viewer.render();
  }

  async loadActiveDetail() {
    const tid = this.active;
    if (tid == null || !this.trackById.has(tid)) { this.activeDetail = null; this.timeline.set({ active: tid, keys: [] }); return; }
    try {
      const d = await get(`${this.base}/track/${tid}`);
      if (tid !== this.active) return;
      this.activeDetail = d;
      this.timeline.set({ active: tid, keys: d.keys });
      panels.renderActive(this);
    } catch (e) { /* deleted meanwhile */ }
  }

  async _refreshTracks() {
    const r = await get(`${this.base}/tracks`);
    this.tracks = r.tracks;
    this.trackById = new Map(r.tracks.map((t) => [t.id, t]));
    this.timeline.set({ tracks: r.tracks });
    panels.renderList(this);
    panels.renderActive(this);
    this.app.refreshStats();
  }

  async _refreshCoverage() {
    try {
      const c = await get(`${this.base}/coverage?bins=1500`);
      this.timeline.set({ coverage: c });
    } catch (e) { /* ignore */ }
  }

  trackBox(tid) { return this.cur.t.find((t) => t[0] === tid) || null; }

  nearestFrame(tid) {
    const t = this.trackById.get(tid);
    if (!t) return null;
    let best = null;
    for (const [a, b] of t.runs) {
      const g = clamp(this.frame, a, b);
      if (best == null || Math.abs(g - this.frame) < Math.abs(best - this.frame)) best = g;
    }
    return best;
  }

  selectAndShow(tid) {
    this.setActive(tid);
    if (!this.trackBox(tid)) { const f = this.nearestFrame(tid); if (f != null) this.go(f); }
  }

  jumpKey(dir) {
    const d = this.activeDetail;
    if (!d) return;
    const ks = d.keys.map((k) => k[0]);
    const f = dir > 0 ? ks.find((k) => k > this.frame) : [...ks].reverse().find((k) => k < this.frame);
    if (f != null) this.go(f);
  }

  cycleActive(dir) {
    const here = [...this.cur.t].sort((a, b) => a[1] - b[1]);
    if (!here.length) return;
    let i = here.findIndex((t) => t[0] === this.active);
    i = (i + dir + here.length) % here.length;
    this.setActive(here[i][0]);
  }

  needActive() {
    if (this.active == null || !this.trackById.has(this.active)) { toast('Select a track first (click its box)', 'err'); return false; }
    return true;
  }

  async startPending(mode) {
    if (!this.needActive()) return;
    this.pending = { mode };
    this.updateChips();
  }

  async swapWith(other) {
    const a = this.active;
    await this.cmd('swap', { frame: this.frame, a, b: other });
    toast(`Swapped #${a} and #${other} from frame ${this.frame} on`, 'ok');
  }

  async mergeIn(other) {
    const into = this.active;
    await this.cmd('merge', { into, other });
    toast(`Merged #${other} into #${into}`, 'ok');
  }

  async split() {
    if (!this.needActive()) return;
    const n = await this.cmd('split', { frame: this.frame, track: this.active });
    toast(`Frames from ${this.frame} on are now #${n}`, 'ok');
    this.setActive(n);
  }

  async deleteBox() {
    if (!this.needActive()) return;
    if (!this.trackBox(this.active)) { toast(`#${this.active} has no box on this frame`); return; }
    await this.cmd('delete_box', { frame: this.frame, track: this.active });
  }

  async endHere() {
    if (!this.needActive()) return;
    await this.cmd('delete_track', { track: this.active, frame: this.frame, from_frame: true });
    toast(`#${this.active} now ends before frame ${this.frame}`);
  }

  async deleteTrack() {
    if (!this.needActive()) return;
    const tid = this.active;
    const t = this.trackById.get(tid);
    if (t && t.n > 50 && !(await confirm('Delete track', `Delete track #${tid} (${t.n} boxes)?`, 'Delete', true))) return;
    await this.cmd('delete_track', { track: tid });
    this.setActive(null);
  }

  async outside() {
    if (!this.needActive()) return;
    await this.cmd('outside', { frame: this.frame, track: this.active });
    toast(`#${this.active} absent from here until its next keyframe`);
  }

  async editId() {
    if (!this.needActive()) return;
    const v = await prompt('Change track id', `New id for #${this.active} (an existing id merges the two tracks)`, String(this.active));
    if (v == null || !/^\d+$/.test(v.trim())) return;
    const n = await this.cmd('relabel', { track: this.active, new_id: Number(v) });
    this.setActive(n);
  }

  async fillGaps() {
    if (!this.needActive()) return;
    const n = await this.cmd('fill_gaps', { track: this.active });
    toast(n ? `Filled ${n} gap${n > 1 ? 's' : ''} of #${this.active}` : 'No gaps to fill');
  }

  async setClass(tid, cls) { await this.cmd('set_class', { track: tid, cls }); }

  // ------------------------------------------------------- canvas handler
  hitTest(p) {
    // proposal / lost-track tags first: clicking one makes that track the brush
    const sp = this.viewer.toScreen(p.x, p.y);
    for (const g of this._tags || []) {
      const [x, y, w, hh] = g.rect;
      if (sp.x >= x && sp.x <= x + w && sp.y >= y && sp.y <= y + hh) return { kind: 'tag', item: g.tid, area: 0 };
    }
    const tol = 3 / this.viewer.scale;
    let best = null;
    const consider = (kind, arr, box) => {
      const [x, y, w, hh] = box;
      if (p.x >= x - tol && p.x <= x + w + tol && p.y >= y - tol && p.y <= y + hh + tol) {
        const area = w * hh;
        if (!best || area < best.area) best = { kind, item: arr, area };
      }
    };
    for (const t of this.cur.t) consider('track', t, t.slice(1, 5));
    if (this.opts.free) for (const d of this.cur.d) consider('det', d, d.slice(1, 5));
    if (!best && this.opts.props) {
      for (const l of this.props.lost) if (!l.exited) consider('lost', l, l.box);
      if (best) best = { kind: 'tag', item: best.item.track, area: 0 };
    }
    return best;
  }

  handleAt(p) {
    if (this.active == null) return null;
    const t = this.trackBox(this.active);
    if (!t) return null;
    const [x, y, w, hh] = this.viewer.rect(t.slice(1, 5));
    const s = this.viewer.toScreen(p.x, p.y);
    const pts = { nw: [x, y], ne: [x + w, y], sw: [x, y + hh], se: [x + w, y + hh], n: [x + w / 2, y], s: [x + w / 2, y + hh], w: [x, y + hh / 2], e: [x + w, y + hh / 2] };
    for (const [k, [hx, hy]] of Object.entries(pts)) if (Math.abs(s.x - hx) <= HANDLE && Math.abs(s.y - hy) <= HANDLE) return k;
    return null;
  }

  down(p, ev) {
    if (ev.button !== 0) return 'pan';
    if (this.pending && this.pending.mode === 'draw' || (ev.ctrlKey || ev.metaKey) && !this.pending) {
      this.dragging = { type: 'draw', a: p, b: p };
      return 'drag';
    }
    if (this.pending) return 'pan'; // resolved on click
    const hd = this.handleAt(p);
    const t = this.active != null ? this.trackBox(this.active) : null;
    if (hd && t) { this.dragging = { type: 'resize', handle: hd, box: t.slice(1, 5), orig: t.slice(1, 5), start: p }; return 'drag'; }
    const hit = this.hitTest(p);
    if (hit && hit.kind === 'track' && hit.item[0] === this.active) {
      this.dragging = { type: 'move', box: hit.item.slice(1, 5), orig: hit.item.slice(1, 5), start: p };
      return 'drag';
    }
    return 'pan';
  }

  drag(p) {
    const d = this.dragging;
    if (!d) return;
    if (d.type === 'draw') d.b = p;
    else if (d.type === 'move') {
      d.box = [d.orig[0] + p.x - d.start.x, d.orig[1] + p.y - d.start.y, d.orig[2], d.orig[3]];
    } else if (d.type === 'resize') {
      let [x, y, w, hh] = d.orig;
      let x2 = x + w, y2 = y + hh;
      if (d.handle.includes('w')) x = p.x;
      if (d.handle.includes('e')) x2 = p.x;
      if (d.handle.includes('n')) y = p.y;
      if (d.handle.includes('s')) y2 = p.y;
      d.box = [Math.min(x, x2), Math.min(y, y2), Math.abs(x2 - x), Math.abs(y2 - y)];
    }
    this.viewer.render();
  }

  async up(p, ev, moved) {
    const d = this.dragging;
    this.dragging = null;
    if (!d) return;
    if (d.type === 'draw') {
      const box = [Math.min(d.a.x, d.b.x), Math.min(d.a.y, d.b.y), Math.abs(d.b.x - d.a.x), Math.abs(d.b.y - d.a.y)];
      if (this.pending && this.pending.mode === 'draw') { this.pending = null; this.updateChips(); }
      if (box[2] * this.viewer.scale < 4 || box[3] * this.viewer.scale < 4) { this.viewer.render(); return; }
      const only = (this.app.settings.classes || []).length === 1 ? this.app.settings.classes[0] : null;
      const id = await this.cmd('set_box', { frame: this.frame, box: this.clampBox(box), track: this.brushNew ? null : this.active, cls: only });
      this.setActive(id);
      return;
    }
    if (!moved) { this.viewer.render(); this.click(p, ev); return; }
    await this.cmd('set_box', { frame: this.frame, box: this.clampBox(d.box), track: this.active });
  }

  clampBox(b) {
    const W = this.meta.width, H = this.meta.height;
    const x = clamp(b[0], 0, W - 1), y = clamp(b[1], 0, H - 1);
    return [x, y, Math.max(1, Math.min(b[0] + b[2], W) - x), Math.max(1, Math.min(b[1] + b[3], H) - y)];
  }

  async click(p, ev) {
    const hit = this.hitTest(p);
    if (this.pending) {
      const mode = this.pending.mode;
      const other = hit && (hit.kind === 'track' ? hit.item[0] : hit.kind === 'tag' ? hit.item : null);
      if (other != null && other !== this.active) {
        this.pending = null; this.updateChips();
        if (mode === 'swap') await this.swapWith(other);
        else if (mode === 'merge') await this.mergeIn(other);
      } else if (!hit) { this.pending = null; this.updateChips(); }
      return;
    }
    if (!hit) {
      if (this.active != null) { this.setActive(null); }
      return;
    }
    if (hit.kind === 'det') await this.paint(hit.item, ev.shiftKey);
    else if (hit.kind === 'tag') this.setActive(hit.item);
    else this.setActive(hit.item[0]);
  }

  dblclick(p) {
    const hit = this.hitTest(p);
    if (hit && hit.kind !== 'tag') { this.viewer.centerOn(hit.item.slice(1, 5), 0.35); return true; }
    return false;
  }

  hover(p) {
    const prev = this.hovered;
    this.hovered = p ? this.hitTest(p) : null;
    const hd = p ? this.handleAt(p) : null;
    const c = $('#canvas');
    c.style.cursor = this.pending ? 'crosshair' : hd ? (['n', 's'].includes(hd) ? 'ns-resize' : ['e', 'w'].includes(hd) ? 'ew-resize' : ['nw', 'se'].includes(hd) ? 'nwse-resize' : 'nesw-resize')
      : this.hovered ? (this.hovered.kind === 'track' && this.hovered.item[0] === this.active ? 'move' : 'pointer') : '';
    if ((prev && prev.item) !== (this.hovered && this.hovered.item)) { this.viewer.render(); this.updateHover(); }
  }

  context(p, ev) {
    const hit = this.hitTest(p);
    const f = this.frame;
    if (hit && hit.kind === 'det') {
      const d = hit.item;
      const items = [{ header: `detection · ${this.className(d[6])} · conf ${d[5]}` }];
      if (this.active != null) items.push({ label: `Give id #${this.active}`, key: 'click', act: () => this.paint(d) });
      items.push({ label: 'New track', key: 'Shift+click', act: () => this.paint(d, true) });
      const prop = this.props.proposals.find((q) => q.det === d[0]);
      if (prop) items.push({ label: `Accept proposal #${prop.track}`, act: () => this.cmd('assign', { frame: f, det: d[0], track: prop.track }) });
      items.push({ label: 'Give id ...', act: async () => {
        const v = await prompt('Assign id', 'Track id (new or existing)', String(this.meta.next_id));
        if (v && /^\d+$/.test(v.trim())) { const id = await this.cmd('assign', { frame: f, det: d[0], track: Number(v) }); this.setActive(id); }
      } });
      items.push('-', { label: 'All free detections here → new tracks', key: 'Shift+N', act: () => this.newTracksHere() });
      menu(ev.clientX, ev.clientY, items);
      return;
    }
    if (hit && hit.kind === 'track') {
      const tid = hit.item[0];
      this.setActive(tid);
      menu(ev.clientX, ev.clientY, [
        { header: `#${tid} · ${KIND_NAME[hit.item[5]]}` },
        { label: 'Swap id with ... (from here on)', key: 'X', act: () => this.startPending('swap') },
        { label: 'Merge another track into this', key: 'M', act: () => this.startPending('merge') },
        { label: 'Split: new id from this frame', key: 'K', act: () => this.split() },
        { label: 'Change id ...', key: 'E', act: () => this.editId() },
        '-',
        { label: 'Delete box on this frame', key: 'Del', act: () => this.deleteBox() },
        { label: 'Absent until next keyframe', key: 'O', act: () => this.outside() },
        { label: 'Track ends here', key: 'Shift+Del', act: () => this.endHere() },
        { label: 'Fill gaps (interpolate)', key: 'G', act: () => this.fillGaps() },
        { label: 'Delete whole track', key: 'Ctrl+Del', danger: true, act: () => this.deleteTrack() },
      ]);
      return;
    }
    menu(ev.clientX, ev.clientY, [
      { label: 'All free detections here → new tracks', key: 'Shift+N', act: () => this.newTracksHere() },
      { label: 'Draw a box', key: 'B', act: () => { this.pending = { mode: 'draw' }; this.updateChips(); } },
      { label: 'Fit view', key: 'dbl-click', act: () => this.viewer.fit() },
    ]);
  }

  className(c) {
    const names = this.app.settings.class_names || {};
    if (c == null || c < 0) return 'no class';
    return names[String(c)] ? `${names[String(c)]} (${c})` : `cls ${c}`;
  }

  // --------------------------------------------------------------- drawing
  draw(ctx, view) {
    this._tags = [];
    if (this.shownFrame !== this.frame && !this.playing) ctx.globalAlpha = 0.85;
    const hov = this.hovered && this.hovered.item;
    const propDets = new Map(this.props.proposals.filter((p) => p.frame === this.frame).map((p) => [p.det, p]));
    // free detections
    if (this.opts.free) {
      for (const d of this.cur.d) {
        if (propDets.has(d[0]) && this.opts.props) continue;
        const [x, y, w, hh] = view.rect(d.slice(1, 5));
        const isH = d === hov;
        ctx.setLineDash(isH ? [] : [5, 3]);
        ctx.lineWidth = isH ? 2 : 1.25;
        ctx.strokeStyle = isH ? '#fff' : 'rgba(255,255,255,0.8)';
        if (isH) { ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(x, y, w, hh); }
        ctx.strokeRect(x, y, w, hh);
      }
      ctx.setLineDash([]);
    }
    // trails of the active track
    if (this.opts.trails && this.active != null) {
      const pts = this.boxes.trail(this.active, this.frame - 60, this.frame + 60);
      if (pts.length > 1) {
        ctx.strokeStyle = idColor(this.active, 0.55);
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach(([f, cx, cy], i) => {
          const s = view.toScreen(cx, cy);
          if (i && f !== pts[i - 1][0] + 1) ctx.moveTo(s.x, s.y); else if (i) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y);
        });
        ctx.stroke();
      }
    }
    // tracks
    const order = [...this.cur.t].sort((a, b) => (a[0] === this.active) - (b[0] === this.active));
    for (const t of order) {
      const isA = t[0] === this.active;
      const box = this.dragging && isA && this.dragging.box ? this.dragging.box : t.slice(1, 5);
      const [x, y, w, hh] = view.rect(box);
      const col = idColor(t[0]);
      ctx.save();
      if (isA) { ctx.shadowColor = col; ctx.shadowBlur = 10; }
      ctx.strokeStyle = col;
      ctx.lineWidth = isA ? 3 : t[5] <= KIND.MANUAL ? 2.25 : 1.75;
      if (t[5] === KIND.INTERP) ctx.setLineDash([7, 4]);
      if (t === hov && !isA) { ctx.fillStyle = idColor(t[0], 0.15); ctx.fillRect(x, y, w, hh); }
      ctx.strokeRect(x, y, w, hh);
      ctx.restore();
      if (t[5] <= KIND.MANUAL) { // key marker
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.moveTo(x + w, y); ctx.lineTo(x + w - 9, y); ctx.lineTo(x + w, y + 9); ctx.closePath();
        if (t[5] === KIND.MANUAL) { ctx.lineWidth = 1.5; ctx.strokeStyle = col; ctx.stroke(); } else ctx.fill();
      }
      if (this.opts.labels || isA || t === hov) this._label(ctx, x, y, `${t[0]}`, col, isA);
      if (isA && !this.dragging) this._handles(ctx, x, y, w, hh, col);
    }
    // proposals and lost tracks
    if (this.opts.props) {
      for (const p of this.props.proposals) {
        const box = p.frame === this.frame ? p.box : p.pred;
        const [x, y, w, hh] = view.rect(box);
        const col = idColor(p.track);
        ctx.setLineDash(p.frame === this.frame ? [4, 3] : [2, 4]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = p.sure ? col : '#f5a524';
        ctx.fillStyle = idColor(p.track, 0.12);
        ctx.fillRect(x, y, w, hh);
        ctx.strokeRect(x, y, w, hh);
        ctx.setLineDash([]);
        this._label(ctx, x, y, `${p.track}${p.sure ? '' : '?'}${p.coast ? ' (missed)' : ''}`, p.sure ? col : '#f5a524', false, true, p.track);
      }
      for (const l of this.props.lost) {
        if (l.exited) continue;
        const [x, y, w, hh] = view.rect(l.box);
        ctx.setLineDash([1, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = idColor(l.track, 0.7);
        ctx.strokeRect(x, y, w, hh); ctx.setLineDash([]);
        this._label(ctx, x, y, `${l.track} lost`, idColor(l.track, 0.7), false, true, l.track);
      }
    }
    // rubber band
    if (this.dragging && this.dragging.type === 'draw') {
      const a = view.toScreen(this.dragging.a.x, this.dragging.a.y), b = view.toScreen(this.dragging.b.x, this.dragging.b.y);
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.strokeStyle = this.active != null && !this.brushNew ? idColor(this.active) : '#fff';
      ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  }

  _label(ctx, x, y, text, col, strong, ghost, tid) {
    ctx.font = `${strong ? 'bold ' : ''}11px ui-monospace, monospace`;
    const w = ctx.measureText(text).width + 8;
    const yy = y - 15 < 0 ? y + 1 : y - 15;
    if (tid != null) this._tags.push({ tid, rect: [x, yy, w, 14] });
    if (tid != null && tid === this.active) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, yy - 1, w + 2, 16); }
    ctx.fillStyle = ghost ? 'rgba(10,12,15,0.8)' : col;
    ctx.fillRect(x, yy, w, 14);
    ctx.fillStyle = ghost ? col : '#0a0c0f';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + 4, yy + 7.5);
  }

  _handles(ctx, x, y, w, hh, col) {
    ctx.fillStyle = '#0a0c0f'; ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    for (const [hx, hy] of [[x, y], [x + w, y], [x, y + hh], [x + w, y + hh], [x + w / 2, y], [x + w / 2, y + hh], [x, y + hh / 2], [x + w, y + hh / 2]]) {
      ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7); ctx.strokeRect(hx - 3.5, hy - 3.5, 7, 7);
    }
  }

  // ------------------------------------------------------------------ HUD
  updateHud() {
    const f = this.frame;
    const [a, b] = this.range;
    const onStep = (f - a) % this.step === 0;
    $('#hudFrame').innerHTML = `${f} <span class="muted">/ ${this.meta.n_frames - 1}</span>${onStep ? ' <span style="color:#37c3a6">◆ step</span>' : ''}${f < a || f > b ? ' <span style="color:#f5a524">outside range</span>' : ''}`;
    const P = this.props.proposals.filter((p) => p.frame === f || p.coast);
    const nSure = P.filter((p) => p.sure).length;
    const lost = this.props.lost.filter((l) => !l.exited).length;
    const free = this.cur.d.length;
    const parts = [];
    if (P.length) parts.push(`<b>${P.length}</b> proposal${P.length > 1 ? 's' : ''} (${nSure} sure) · <kbd>Space</kbd> accept + next`);
    if (lost) parts.push(`<span style="color:#f5a524">${lost} track${lost > 1 ? 's' : ''} lost</span> (click its detection, or ignore if it ended)`);
    if (free - P.length > 0) parts.push(`${free - P.length} free detection${free - P.length > 1 ? 's' : ''} · <kbd>Shift</kbd>+<kbd>N</kbd> all → new`);
    $('#hudProps').innerHTML = parts.join('<br>');
    this.updateHover();
  }

  updateHover() {
    const hv = this.hovered;
    let s = '';
    if (hv && hv.kind === 'det') {
      const d = hv.item;
      s = `free detection · ${this.className(d[6])} · conf ${d[5]} · click: ${this.active != null && !this.brushNew ? `give #${this.active}` : 'new track'}`;
    } else if (hv && hv.kind === 'tag') {
      const pr = this.props.proposals.find((q) => q.track === hv.item);
      s = `#${hv.item} ${pr ? (pr.sure ? 'proposal' : 'unsure proposal') : 'lost'} · click: make #${hv.item} the brush, then click its right box`;
    } else if (hv && hv.kind === 'track') {
      const t = hv.item;
      const tr = this.trackById.get(t[0]);
      s = `#${t[0]}${tr ? ` · ${this.className(tr.cls)}` : ''} · ${KIND_NAME[t[5]]}${tr ? ` · ${tr.runs[0][0]}–${tr.runs[tr.runs.length - 1][1]}` : ''}`;
    }
    $('#hudHover').textContent = s;
  }

  updateChips() {
    const bc = $('#brushChip');
    if (this.brushNew || this.active == null) {
      bc.innerHTML = `brush: <b>new id</b> <span class="muted">(#${this.meta.next_id})</span>`;
    } else {
      const exists = this.trackById.has(this.active);
      bc.innerHTML = `<span class="sw" style="background:${idColor(this.active)}"></span>brush <b>#${this.active}</b>${exists ? '' : ' <span class="muted">(new)</span>'}`;
    }
    const pc = $('#pendingChip');
    const p = this.pending;
    pc.textContent = !p ? '' : p.mode === 'swap' ? `Swap #${this.active} with … click a track (Esc cancels)`
      : p.mode === 'merge' ? `Merge into #${this.active}: click the other track (Esc cancels)`
        : 'Draw a box: drag on the image (Esc cancels)';
    $('#modeChip').textContent = this.meta.mode === 'tracks' ? 'correcting tracks' : 'assigning ids';
  }

  // -------------------------------------------------------------- keyboard
  onKey(e) {
    const k = e.key;
    const ctrl = e.ctrlKey || e.metaKey;
    // id typing: digits then Enter
    if (/^[0-9]$/.test(k) && !ctrl && !e.altKey) {
      this.idbuf = (this.idbuf + k).slice(0, 7);
      this._showIdBuf();
      return true;
    }
    if (this.idbuf) {
      if (k === 'Backspace') { this.idbuf = this.idbuf.slice(0, -1); this._showIdBuf(); return true; }
      if (k === 'Enter') { this._applyIdBuf(); return true; }
      if (k === 'Escape') { this.idbuf = ''; this._showIdBuf(); return true; }
    }
    if (ctrl && (k === 'z' || k === 'Z')) { e.shiftKey ? this.redo() : this.undo(); return true; }
    if (ctrl && (k === 'y' || k === 'Y')) { this.redo(); return true; }
    if (ctrl && k === '0') { this.viewer.fit(); return true; }
    if (ctrl && (k === 'Delete' || k === 'Backspace')) { this.deleteTrack(); return true; }
    if (ctrl) return false;
    const big = e.shiftKey ? 10 : 1;
    switch (k) {
      case 'ArrowRight': if (e.altKey) this.jumpKey(1); else this.stepBy(big); return true;
      case 'ArrowLeft': if (e.altKey) this.jumpKey(-1); else this.stepBy(-big); return true;
      case 'd': case 'D': case 'PageDown': this.stepBy(this.step * (e.shiftKey ? 5 : 1)); return true;
      case 'a': case 'A': case 'PageUp': this.stepBy(-this.step * (e.shiftKey ? 5 : 1)); return true;
      case 'Home': this.go(this.range[0]); return true;
      case 'End': this.go(this.range[1]); return true;
      case ' ': if (e.shiftKey) this.advance(); else this.acceptAndStep(); return true;
      case 'Enter': this.accept(); return true;
      case 'p': case 'P': this.togglePlay(); return true;
      case 'n': this.brushNew = true; this.active = null; this.updateChips(); this.viewer.render(); toast('Next click on a free detection starts a new track'); return true;
      case 'N': this.newTracksHere(); return true;
      case 'Escape':
        if (this.pending) { this.pending = null; this.updateChips(); } else this.setActive(null);
        return true;
      case 'x': case 'X': this.startPending('swap'); return true;
      case 'm': case 'M': this.startPending('merge'); return true;
      case 'k': case 'K': this.split(); return true;
      case 'Delete': case 'Backspace': if (e.shiftKey) this.endHere(); else this.deleteBox(); return true;
      case 'o': this.outside(); return true;
      case 'e': case 'E': this.editId(); return true;
      case 'g': case 'G': this.fillGaps(); return true;
      case 'b': case 'B': this.pending = this.pending && this.pending.mode === 'draw' ? null : { mode: 'draw' }; this.updateChips(); return true;
      case 'f': case 'F': this.opts.follow = !this.opts.follow; this._saveOpts(); toast(`Follow active track: ${this.opts.follow ? 'on' : 'off'}`); return true;
      case 'z': case 'Z': {
        const t = this.active != null && this.trackBox(this.active);
        if (t && !this._zoomed) { this.viewer.centerOn(t.slice(1, 5), 0.3); this._zoomed = true; } else { this.viewer.fit(); this._zoomed = false; }
        return true;
      }
      case 'l': case 'L': this.opts.labels = !this.opts.labels; this._saveOpts(); this.viewer.render(); return true;
      case 'h': case 'H': this.opts.free = !this.opts.free; this._saveOpts(); this.viewer.render(); toast(`Free detections ${this.opts.free ? 'shown' : 'hidden'}`); return true;
      case 't': case 'T': this.opts.trails = !this.opts.trails; this._saveOpts(); this.viewer.render(); return true;
      case 'I': this.setRangeEnd(0); return true;
      case 'O': this.setRangeEnd(1); return true;
      case ']': panels.nextIssue(this, 1); return true;
      case '[': panels.nextIssue(this, -1); return true;
      case 'Tab': this.cycleActive(e.shiftKey ? -1 : 1); return true;
      default: return false;
    }
  }

  _showIdBuf() {
    const el = $('#idBuffer');
    if (!this.idbuf) { el.classList.add('hidden'); return; }
    el.classList.remove('hidden');
    const exists = this.trackById.has(Number(this.idbuf));
    const hv = this.hovered && this.hovered.kind === 'det' ? 'give it to the hovered detection' : 'set as brush';
    el.innerHTML = `#${this.idbuf}<small>${exists ? 'existing track' : 'new id'} · Enter: ${hv}</small>`;
  }

  async _applyIdBuf() {
    const id = Number(this.idbuf);
    this.idbuf = '';
    this._showIdBuf();
    if (!id) return;
    if (this.hovered && this.hovered.kind === 'det') {
      const r = await this.cmd('assign', { frame: this.frame, det: this.hovered.item[0], track: id });
      this.setActive(r);
    } else {
      this.setActive(id);
      if (!this.trackBox(id) && this.trackById.has(id)) toast(`Brush #${id} (not on this frame)`);
    }
  }

  _saveOpts() { store.set('opts', this.opts); panels.renderData(this); }

  // -------------------------------------------------------------- playback
  togglePlay() {
    this.playing = !this.playing;
    $('.transport .play').innerHTML = this.playing ? '&#10074;&#10074;' : '&#9654;';
    if (!this.playing) { this.refreshProps(); return; }
    let last = performance.now();
    let acc = 0;
    const tick = (now) => {
      if (!this.playing) return;
      acc += (now - last) / 1000 * this.meta.fps;
      last = now;
      if (acc >= 1 && this.shownFrame === this.frame) {
        const k = Math.min(Math.floor(acc), 3);
        acc -= k;
        if (this.frame + k > this.range[1]) { this.togglePlay(); return; }
        this.go(this.frame + k);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ range
  async setRangeEnd(which) {
    const r = [...this.range];
    r[which] = this.frame;
    if (r[1] < r[0]) r.reverse();
    await this.app.setRange(r);
  }

  onMeta(meta) {
    this.meta = Object.assign(this.meta, meta);
    this.timeline.set({ range: meta.range });
    $('#rangeInfo').textContent = meta.full_range ? 'whole video' :
      `range ${meta.range[0]}–${meta.range[1]} (${fmtTime(meta.range[1] - meta.range[0] + 1, meta.fps)})`;
    this.updateHud();
  }

  setStep(s) {
    this.step = Math.max(1, s | 0);
    $('#stepInput').value = this.step;
    this.timeline.set({ step: this.step });
    this.updateHud();
  }

  _bindTransport() {
    $('#stepInput').value = this.step;
    $('#frameInput').max = this.meta.n_frames - 1;
    $('#rangeInfo').textContent = this.meta.full_range ? 'whole video' : `range ${this.range[0]}–${this.range[1]}`;
    this.onMeta(this.meta);
  }

  transport(act) {
    switch (act) {
      case 'start': this.go(this.range[0]); break;
      case 'end': this.go(this.range[1]); break;
      case 'prev': this.stepBy(-1); break;
      case 'next': this.stepBy(1); break;
      case 'prevStep': this.stepBy(-this.step); break;
      case 'nextStep': this.stepBy(this.step); break;
      case 'play': this.togglePlay(); break;
      case 'in': this.setRangeEnd(0); break;
      case 'out': this.setRangeEnd(1); break;
      default: break;
    }
  }
}

export const KEYMAP = [
  ['Navigate', [
    ['→ / ←', 'next / previous frame (Shift: ±10)'],
    ['D / A', 'one keyframe step forward / back (Shift: 5 steps)'],
    ['Home / End', 'start / end of the range'],
    ['P', 'play / pause'],
    ['Alt+→ / Alt+←', 'next / previous keyframe of the active track'],
    ['[ / ]', 'previous / next review issue'],
  ]],
  ['Assign ids', [
    ['click free box', 'give it the brush id (no brush: new track)'],
    ['Shift+click', 'always a new track'],
    ['Space', 'accept proposals, jump one step'],
    ['Shift+Space', 'auto-advance until something needs you'],
    ['Enter', 'accept proposals here'],
    ['Shift+N', 'every free detection here → new track'],
    ['N', 'brush = new id'],
    ['12 Enter', 'type an id: brush (or give it to the hovered box)'],
    ['click track box', 'make it the brush (active track)'],
    ['Tab', 'cycle the active track among boxes on this frame'],
    ['Esc', 'cancel / clear the brush'],
  ]],
  ['Fix tracks (active track)', [
    ['X then click', 'swap ids from this frame on (ID switch)'],
    ['M then click', 'merge the clicked track into the active one'],
    ['K', 'split: new id from this frame on'],
    ['E', 'change id (existing id = merge)'],
    ['Del', 'delete box (keyframe: re-interpolate)'],
    ['O', 'absent (occluded) until the next keyframe'],
    ['Shift+Del', 'track ends here'],
    ['Ctrl+Del', 'delete whole track'],
    ['G', 'fill gaps of the track by interpolation'],
    ['drag box / handles', 'move / resize (becomes a keyframe)'],
    ['B or Ctrl+drag', 'draw a box (missed detection)'],
  ]],
  ['View', [
    ['wheel / drag', 'zoom / pan (double click: fit or zoom to box)'],
    ['Z', 'zoom to the active track / fit'],
    ['F', 'follow the active track'],
    ['L / H / T', 'labels / free detections / trails'],
    ['Shift+I / Shift+O', 'range start / end at this frame'],
    ['Ctrl+Z / Ctrl+Shift+Z', 'undo / redo'],
    ['Ctrl+S', 'save now (autosave is always on)'],
  ]],
];

export function legend() {
  return h('div', { class: 'help' },
    h('p', {}, 'White dashed = free detection (no id yet). Colored = track; solid with a corner flag = keyframe, solid = matched detection, dashed = interpolated (no detection). Dashed with fill = proposal (orange "?" = unsure).'));
}
