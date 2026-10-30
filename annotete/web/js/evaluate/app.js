// evaluATE: inputs -> overview (all videos) -> one video (video.js).
import { get, post } from '../api.js';
import { $, h, toast, modal, store, debounce, switchApp } from '../util.js';
import { VideoView } from './video.js';

// metric key -> label, help, percent?, which direction is better
export const M = {
  HOTA: ['HOTA', 'Overall tracking accuracy: √(DetA × AssA), averaged over IoU thresholds 0.05–0.95', true, 1],
  DetA: ['DetA', 'Detection accuracy: are the objects found (TP / (TP + FN + FP))', true, 1],
  AssA: ['AssA', 'Association accuracy: does each object keep one tracker id over time', true, 1],
  IDF1: ['IDF1', 'Identity F1: share of boxes with the right id under the best global id mapping', true, 1],
  MOTA: ['MOTA', 'CLEAR MOTA: 1 − (FN + FP + IDSW) / GT boxes (mostly detection)', true, 1],
  MOTP: ['MOTP', 'Mean IoU of matched boxes', true, 1],
  DetRe: ['DetRe', 'Detection recall', true, 1],
  DetPr: ['DetPr', 'Detection precision', true, 1],
  IDSW: ['IDSW', 'Identity switches: an object continues under a different tracker id', false, -1],
  Frag: ['Frag', 'Fragmentations: an object stops being tracked and is picked up again', false, -1],
  CLR_FP: ['FP', 'False boxes (IoU 0.5)', false, -1],
  CLR_FN: ['FN', 'Missed boxes (IoU 0.5)', false, -1],
  MT: ['MT', 'Mostly tracked objects (> 80% of their frames)', false, 1],
  ML: ['ML', 'Mostly lost objects (< 20% of their frames)', false, -1],
  IDs: ['IDs', 'Tracker ids', false, 0],
  GT_IDs: ['GT ids', 'Annotated objects', false, 0],
};
export const EVENTS = {
  swap: ['swap', 'two objects exchange tracker ids'],
  transfer: ['id jump', 'a tracker id jumps to another object'],
  new_id: ['new id', 'the object restarts with a new tracker id'],
  switch: ['switch', 'the object moves to another existing tracker id'],
  lost: ['lost', 'the object is not tracked for a while'],
  false: ['ghost', 'a tracker id that follows nothing'],
};

export const fmt = (k, v) => (v == null ? '–' : M[k] && M[k][2] ? (v * 100).toFixed(1) : String(v));

export function delta(k, a, b) {
  if (b == null || a == null) return '';
  const d = M[k][2] ? (a - b) * 100 : a - b;
  if (Math.abs(d) < (M[k][2] ? 0.05 : 0.5)) return h('span', { class: 'delta same' }, '±0');
  const good = Math.sign(d) === M[k][3];
  return h('span', { class: `delta ${M[k][3] === 0 ? 'same' : good ? 'up' : 'down'}`, title: 'current − previous' },
    `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(M[k][2] ? 1 : 0)}`);
}

class EvalApp {
  constructor() {
    this.summary = null;
    this.variant = store.get('ev:variant', 'all');
    this.sort = store.get('ev:sort', ['name', 1]);
    this.video = null;
    $('#brand').onclick = (e) => switchApp(e);
    $('#setupBtn').onclick = () => this.show('setup');
    $('#helpBtn').onclick = () => helpDialog();
    window.addEventListener('popstate', () => this.route());
    document.addEventListener('keydown', (e) => {
      if ($('#modalRoot').children.length || ['input', 'select', 'textarea'].includes((e.target.tagName || '').toLowerCase())) return;
      if (this.video && !$('#video').classList.contains('hidden') && this.video.onKey(e)) e.preventDefault();
    });
    this.boot();
  }

  async boot() {
    const q = new URLSearchParams(location.search);
    const last = store.get('ev:inputs', {});
    let gt = q.get('gt') || last.gt || '';
    if (!gt) { try { gt = (await get('/api/eval/defaults')).gt; } catch (e) { /* no project */ } }
    this.inputs = { gt, a: q.get('a') || last.a || '', an: q.get('an') || last.an || '', b: q.get('b') ?? last.b ?? '',
      bn: q.get('bn') || last.bn || '', shift: last.shift || 'auto', classes: last.classes || 'gt' };
    this.renderSetup();
    if (q.get('run') && this.inputs.gt && this.inputs.a) this.run();
    else this.show('setup');
  }

  show(view) {
    for (const id of ['setup', 'overview', 'video']) $(`#${id}`).classList.toggle('hidden', id !== view);
    if (view !== 'video' && this.video) this.video.pause();
    this.crumbs(view);
  }

  crumbs(view) {
    const c = $('#crumbs');
    c.innerHTML = '';
    if (!this.summary) { c.append('tracker evaluation'); return; }
    const t = this.summary.trackers;
    c.append(h('a', { href: '#', onclick: (e) => { e.preventDefault(); this.show('overview'); } },
      t.length === 2 ? `${t[0].name} vs ${t[1].name}` : t[0].name));
    if (view === 'video' && this.video) c.append(' › ', h('span', {}, this.video.seq));
  }

  route() { /* keep it simple: back always returns to the overview */ if (this.summary) this.show('overview'); }

  // ------------------------------------------------------------------ inputs
  renderSetup() {
    const root = $('#setup');
    root.innerHTML = '';
    const I = this.inputs;
    const path = (v, ph) => {
      const dl = h('datalist', { id: `dl${Math.random().toString(36).slice(2)}` });
      const inp = h('input', { type: 'text', value: v, placeholder: ph, list: dl.id, spellcheck: 'false', autocomplete: 'off' });
      const complete = debounce(async () => {
        try {
          const r = await get(`/api/ls?path=${encodeURIComponent(inp.value || '~')}`);
          dl.innerHTML = '';
          for (const e of r.entries.slice(0, 200)) dl.append(h('option', { value: e.path + (e.dir ? '/' : '') }));
        } catch (e) { /* ignore */ }
      }, 200);
      inp.addEventListener('input', complete);
      inp.addEventListener('focus', complete);
      return [inp, dl];
    };
    const [gt, d1] = path(I.gt, '…/my_project/export   (annoTETE export folder)');
    const [a, d2] = path(I.a, 'folder of tracker output files, named like the videos');
    const an = h('input', { type: 'text', value: I.an, placeholder: 'name (e.g. v2)' });
    const [b, d3] = path(I.b, 'optional: a previous tracker output to compare with');
    const bn = h('input', { type: 'text', value: I.bn, placeholder: 'name (e.g. v1)' });
    const shift = h('select', {}, h('option', { value: 'auto' }, 'detect automatically'),
      h('option', { value: '0' }, 'tracker ran on the exported (trimmed) videos'));
    shift.value = I.shift === 'auto' ? 'auto' : '0';
    const classes = h('select', {}, h('option', { value: 'gt' }, 'only the annotated classes'), h('option', { value: 'all' }, 'all tracker boxes'));
    classes.value = I.classes;
    const go = h('button', { class: 'primary', onclick: () => {
      this.inputs = { gt: gt.value.trim(), a: a.value.trim(), an: an.value.trim(), b: b.value.trim(), bn: bn.value.trim(),
        shift: shift.value, classes: classes.value };
      this.run();
    } }, 'Evaluate');
    root.append(h('div', { class: 'wrap' },
      h('h2', {}, 'Evaluate a tracker', h('small', {}, 'scores like TrackEval (HOTA, IDF1, MOTA …) plus where and why it fails')),
      h('div', { class: 'card' }, h('div', { class: 'form' },
        h('label', { class: 'l' }, 'Ground truth'), gt, h('span'),
        h('div', { class: 'note' }, 'An annoTETE export folder. Uses annotations_mot/ (ground truth), annotations_json/ (which boxes had a detection) and videos/ (trimmed clips) when present.'),
        h('label', { class: 'l' }, 'Tracker output'), a, an,
        h('div', { class: 'note' }, 'One file per video, named like the video (cam1.txt, cam1_tracks.txt …), any layout annoTETE reads (space / comma, MOT, 0- or 1-based).'),
        h('label', { class: 'l' }, 'Compare with'), b, bn,
        h('div', { class: 'note' }, 'Optional: the output of your previous tracker version. Scores then show the change, and events are marked fixed / new.'),
        h('label', { class: 'l' }, 'Frame alignment'), shift, h('span'),
        h('label', { class: 'l' }, 'Classes'), classes, h('span'),
        d1, d2, d3)),
      h('div', { class: 'row', style: { justifyContent: 'flex-end' } }, go)));
  }

  async run() {
    const I = this.inputs;
    store.set('ev:inputs', I);
    const trackers = [{ name: I.an, path: I.a }];
    if (I.b) trackers.push({ name: I.bn, path: I.b });
    const root = $('#overview');
    root.innerHTML = '';
    const log = h('pre', { class: 'log' }, 'evaluating …');
    root.append(h('div', { class: 'wrap' }, h('h2', {}, 'Evaluating …'), log));
    this.show('overview');
    try {
      let job = await post('/api/eval/run', { gt: I.gt, trackers, shift: I.shift, classes: I.classes });
      while (!job.done) {
        await new Promise((r) => setTimeout(r, 300));
        job = await get(`/api/jobs/${job.id}`);
        log.textContent = job.log.slice(-12).join('\n');
      }
      if (job.error) throw new Error(job.error);
      this.summary = await get(`/api/eval/${job.result}/summary`);
      this.report = job.report;
      $('#runInfo').textContent = `${this.summary.seqs.length} video${this.summary.seqs.length > 1 ? 's' : ''} · ${this.summary.seconds}s`;
      if (!this.summary.has_kinds && this.variant === 'det') this.variant = 'all';
      this.renderOverview();
      this.show('overview');
    } catch (e) {
      toast(e.message, 'err', 7000);
      this.show('setup');
    }
  }

  // ---------------------------------------------------------------- overview
  renderOverview() {
    const S = this.summary;
    const A = S.results[0], B = S.results[1];
    const v = this.variant;
    const root = $('#overview');
    root.innerHTML = '';
    const ca = A.combined[v], cb = B && B.combined[v];
    const varSeg = h('div', { class: 'seg' }, [['all', 'All annotations'], ['det', 'Detector-visible only']].map(([k, label]) =>
      h('button', { class: k === v ? 'on' : '', disabled: k === 'det' && !S.has_kinds,
        title: k === 'det' ? (S.has_kinds ? 'Annotated boxes without a detection (interpolated / drawn) are ignored: the score the tracker can reach with this detector'
          : 'Needs annotations_json/ in the export (tick JSON when exporting)') : 'Every annotated box counts',
        onclick: () => { this.variant = k; store.set('ev:variant', k); this.renderOverview(); } }, label)));
    const tile = (k, main) => h('div', { class: `tile${main ? ' hi' : ''}`, title: M[k][1] },
      h('div', { class: 'k' }, h('span', {}, M[k][0]), cb ? delta(k, ca[k], cb[k]) : ''),
      h('div', { class: 'v' }, fmt(k, ca[k])),
      cb ? h('div', { class: 's' }, `was ${fmt(k, cb[k])}`) : '');
    const wrap = h('div', { class: 'wrap' });
    wrap.append(
      h('div', { class: 'row', style: { justifyContent: 'space-between' } },
        h('h2', {}, B ? `${A.name} vs ${B.name}` : A.name, h('small', {}, `${S.seqs.length} video${S.seqs.length > 1 ? 's' : ''} · ground truth ${S.gt_dir}`)),
        varSeg),
      h('div', { class: 'tiles' }, ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'MT', 'ML', 'IDs'].map((k, i) => tile(k, i === 0))),
    );
    if (S.warnings.length) wrap.append(h('div', { class: 'card warnlist' }, S.warnings.slice(0, 12).map((w) => h('div', {}, '⚠ ', w))));
    wrap.append(h('div', { class: 'row2' }, this.detectorCard(A, B), this.curveCard(A, B)));
    wrap.append(this.eventsCard(A, B));
    wrap.append(this.videoTable(A, B));
    if (this.report) wrap.append(h('div', { class: 'muted small' }, `Report (JSON + CSV) saved in ${this.report}`));
    root.append(wrap);
  }

  detectorCard(A, B) {
    const d = A.combined.detector;
    const card = h('div', { class: 'card' }, h('h4', {}, 'Detector vs tracker'));
    if (!this.summary.has_kinds || !d.gt) {
      card.append(h('p', { class: 'muted' }, 'Export with "JSON" ticked to see which annotated boxes had a detection, and how much the detector limits the tracker.'));
      return card;
    }
    const pct = (x) => (100 * x / Math.max(1, d.gt)).toFixed(1);
    const bar = (parts) => h('div', { class: 'bar' }, parts.filter(([n]) => n > 0).map(([n, c, t]) =>
      h('i', { style: { width: `${100 * n / Math.max(1, parts.reduce((s, p) => s + p[0], 0))}%`, background: c }, title: t })));
    const all = A.combined.all.HOTA, det = A.combined.det ? A.combined.det.HOTA : null;
    card.append(
      h('div', { class: 'muted small' }, `${d.gt} annotated boxes:`),
      bar([[d.gt_det, 'var(--good)', 'detector had a box'], [d.gt_interp, 'var(--warn)', 'interpolated (no detection)'], [d.gt_manual, 'var(--nodet)', 'drawn by hand (no detection)']]),
      h('div', { class: 'legend' }, h('span', { style: { '--c': 'var(--good)' } }, `detector had a box ${pct(d.gt_det)}%`),
        h('span', { style: { '--c': 'var(--warn)' } }, `interpolated ${pct(d.gt_interp)}%`), h('span', { style: { '--c': 'var(--nodet)' } }, `drawn ${pct(d.gt_manual)}%`)),
      h('div', { class: 'muted small', style: { marginTop: '10px' } }, `${d.fn} boxes missed by the tracker:`),
      bar([[d.fn_nodet, 'var(--nodet)', 'no detection there'], [d.fn_det, 'var(--bad)', 'the detector had a box: the tracker dropped it']]),
      h('div', { class: 'legend' }, h('span', { style: { '--c': 'var(--nodet)' } }, `no detection there ${d.fn_nodet}`),
        h('span', { style: { '--c': 'var(--bad)' } }, `dropped by the tracker ${d.fn_det}`)),
    );
    const lines = [];
    if (det != null) lines.push(h('div', {}, 'Scored only where the detector had a box, HOTA would be ', h('b', {}, (det * 100).toFixed(1)),
      ` (${det >= all ? '+' : ''}${((det - all) * 100).toFixed(1)}): `, det - all > 0.005 ? 'that is what the detector misses cost the tracker.' : 'the detector misses cost little.'));
    if (d.fn) lines.push(h('div', {}, h('b', {}, `${Math.round(100 * d.fn_nodet / d.fn)}%`), ' of the missed boxes had no detection (a detector problem); ',
      h('b', {}, `${Math.round(100 * d.fn_det / d.fn)}%`), ' had one that the tracker did not use (a tracker problem).'));
    if (d.gt - d.gt_det) lines.push(h('div', {}, `The tracker itself filled ${d.tp_nodet} of the ${d.gt - d.gt_det} boxes the detector missed.`));
    if (d.flicker) lines.push(h('div', {}, `The detector blinked ${d.flicker} times (${d.flicker_frames} frames with no box, the tracker kept the id): not listed as events.`));
    if (B && B.combined.det && A.combined.det) {
      lines.push(h('div', {}, 'Change vs previous: all annotations ', delta('HOTA', A.combined.all.HOTA, B.combined.all.HOTA), ' · detector-visible ', delta('HOTA', A.combined.det.HOTA, B.combined.det.HOTA), ' HOTA'));
    }
    card.append(h('div', { class: 'insight' }, lines));
    return card;
  }

  curveCard(A, B) {
    const W = 520, H = 190, P = 28;
    const x = (i) => P + (i / 18) * (W - P - 8), y = (v) => H - 20 - v * (H - 34);
    const svg = (tag, attrs) => { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, val] of Object.entries(attrs)) el.setAttribute(k, val); return el; };
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'curve' });
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      s.append(svg('line', { x1: P, x2: W - 8, y1: y(g), y2: y(g), stroke: '#2a313b' }));
      const t = svg('text', { x: 2, y: y(g) + 3 }); t.textContent = String(g * 100); s.append(t);
    }
    for (const [i, lab] of [[0, '0.05'], [9, '0.5'], [18, '0.95']]) { const t = svg('text', { x: x(i) - 10, y: H - 4 }); t.textContent = `α ${lab}`; s.append(t); }
    const colors = { HOTA: '#f5a524', DetA: '#37c3a6', AssA: '#7fb2ff' };
    const v = this.variant;
    for (const [res, dash] of [[A, ''], [B, '5 4']]) {
      if (!res || !res.combined[v]) continue;
      for (const [k, c] of Object.entries(colors)) {
        const pts = res.combined[v].curves[k].map((val, i) => `${x(i)},${y(val)}`).join(' ');
        s.append(svg('polyline', { points: pts, fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-dasharray': dash }));
      }
    }
    return h('div', { class: 'card' }, h('h4', {}, 'Scores over the IoU threshold'), s,
      h('div', { class: 'legend' }, Object.entries(colors).map(([k, c]) => h('span', { style: { '--c': c } }, k)),
        B ? h('span', { style: { '--c': 'transparent' } }, `solid: ${A.name}, dashed: ${B.name}`) : ''),
      h('div', { class: 'muted small' }, 'A curve that falls early means boxes that are in the right place but loose (localisation); a low AssA line means identity problems.'));
  }

  eventsCard(A, B) {
    const e = A.combined.events;
    const chips = Object.entries(EVENTS).map(([k, [label, desc]]) => h('span', { class: `evchip ${k}`, title: desc },
      `${label} ${e[k]}`, B ? ` (was ${B.combined.events[k]})` : ''));
    const fl = A.combined.detector.flicker;
    if (fl) chips.push(h('span', { class: 'evchip', title: 'short gaps where the detector had no box and the tracker kept the id: not listed' }, `detector flicker ${fl}`));
    const card = h('div', { class: 'card' }, h('h4', {}, 'What goes wrong'), h('div', { class: 'evchips', style: { justifyContent: 'flex-start' } }, chips));
    if (B) {
      const nNew = this.summary.seqs.reduce((s, n) => s + (A.seqs[n].new || 0), 0);
      const nFix = this.summary.seqs.reduce((s, n) => s + (A.seqs[n].fixed || 0), 0);
      card.append(h('div', { class: 'insight' }, h('b', {}, `${nFix} fixed`), ` problems of ${B.name} are gone, `, h('b', {}, `${nNew} new`),
        ` ones appear in ${A.name}. Open a video to see them on the frames.`));
    } else card.append(h('div', { class: 'muted small', style: { marginTop: '6px' } }, 'Open a video to see each of them on the frames.'));
    return card;
  }

  videoTable(A, B) {
    const v = this.variant;
    const cols = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN'];
    const rows = this.summary.seqs.map((n) => ({ n, a: A.seqs[n], b: B && B.seqs[n] }));
    const [sk, sd] = this.sort;
    rows.sort((p, q) => (sk === 'name' ? p.n.localeCompare(q.n) : (p.a[v] ? p.a[v][sk] : 0) - (q.a[v] ? q.a[v][sk] : 0)) * sd);
    const th = (k, label) => h('th', { class: sk === k ? 'on' : '', title: k === 'name' ? '' : M[k] && M[k][1],
      onclick: () => { this.sort = [k, sk === k ? -sd : (k === 'name' ? 1 : -1)]; store.set('ev:sort', this.sort); this.renderOverview(); } },
    label, sk === k ? (sd > 0 ? ' ▲' : ' ▼') : '');
    const table = h('table', { class: 'vt' }, h('tr', {}, th('name', 'video'), cols.map((k) => th(k, M[k][0])), h('th', {}, 'detector'), h('th', {}, 'events')));
    for (const { n, a, b } of rows) {
      const r = a[v];
      const d = a.detector;
      const cov = d.gt ? d.gt_det / d.gt : null;
      const ev = Object.entries(EVENTS).filter(([k]) => a.events[k]).map(([k, [label]]) => h('span', { class: `evchip ${k}` }, `${a.events[k]} ${label}`));
      if (b) {
        if (a.fixed) ev.push(h('span', { class: 'evchip fixed' }, `${a.fixed} fixed`));
        if (a.new) ev.push(h('span', { class: 'evchip newdiff' }, `${a.new} new`));
      }
      table.append(h('tr', { class: 'r', onclick: () => this.openVideo(n), title: a.align },
        h('td', {}, n),
        cols.map((k) => h('td', {}, r ? fmt(k, r[k]) : '–', k === 'HOTA' && r ? h('span', { class: 'heat' }, h('i', { style: { width: `${r.HOTA * 100}%` } })) : '',
          b && r && b[v] ? delta(k, r[k], b[v][k]) : '')),
        h('td', { title: 'share of the annotated boxes that had a detection' }, cov != null && this.summary.has_kinds ? `${(cov * 100).toFixed(0)}%` : '–'),
        h('td', {}, h('span', { class: 'evchips' }, ev))));
    }
    return h('div', { class: 'card' }, h('h4', {}, 'Videos', h('span', { class: 'muted', style: { textTransform: 'none', letterSpacing: 0, fontWeight: 400 } }, ' · click one to see where it goes wrong')), table);
  }

  openVideo(seq) {
    if (this.video) this.video.destroy();
    this.video = new VideoView(this, seq);
    this.show('video');
  }
}

function helpDialog() {
  modal('What the numbers mean', () => h('div', { class: 'help' },
    h('p', {}, 'Scores are computed like TrackEval (same formulas, checked against it): HOTA averaged over IoU thresholds 0.05–0.95; CLEAR (MOTA, IDSW, Frag, MT/ML) and Identity (IDF1) at IoU 0.5.'),
    h('div', { class: 'keys' }, Object.entries(M).map(([, [name, desc]]) => h('div', { class: 'k' }, h('span', {}, desc), h('kbd', {}, name)))),
    h('p', { style: { marginTop: '12px' } }, h('b', {}, 'All annotations'), ' counts every annotated box. ', h('b', {}, 'Detector-visible only'),
      ' ignores the annotated boxes that had no detection (interpolated / drawn): tracker boxes there are neither right nor wrong. The gap between the two is what the detector costs.'),
    h('div', { class: 'keys' }, Object.entries(EVENTS).map(([k, [name, desc]]) => h('div', { class: 'k' }, h('span', {}, desc), h('span', { class: `evchip ${k}` }, name)))),
    h('p', {}, 'Video view keys: ← → frame, Space play, [ ] previous / next event, Esc back to all videos.')), { wide: true });
}

window.evaluate = new EvalApp();
