// App shell: project / item switching, settings, ranges, export, help.
import { get, post, item } from './api.js';
import { $, $$, h, toast, modal, debounce, store, switchApp } from './util.js';
import { renderSetup } from './setup.js';
import { TrackingTask, KEYMAP, legend } from './tasks/tracking/index.js';
import * as panels from './tasks/tracking/panels.js';

const TASKS = { tracking: TrackingTask };

class App {
  constructor() {
    this.state = null;
    this.task = null;
    this.refreshStats = debounce(() => this._refreshStats(), 800);
    this._bindGlobal();
    this.boot();
  }

  get project() { return this.state && this.state.project; }
  get settings() { return (this.project && this.project.settings) || {}; }

  async boot() {
    try {
      this.state = await get('/api/state');
    } catch (e) {
      document.body.innerHTML = `<p style="padding:20px">Cannot reach the annoTETE server: ${e.message}</p>`;
      return;
    }
    if (!this.project) { this.showSetup(); return; }
    this.showWork();
  }

  showSetup() {
    if (this.task) { this.task.destroy(); this.task = null; }
    $('#work').classList.add('hidden');
    $('#setup').classList.remove('hidden');
    renderSetup($('#setup'), this.project, (state) => { if (state) this.state = state; this.showWork(); });
  }

  showWork() {
    $('#setup').classList.add('hidden');
    $('#work').classList.remove('hidden');
    this.renderItems();
    const items = this.project.items;
    const last = store.get(`item:${this.project.root}`);
    const pick = items.find((i) => i.name === last) || items.find((i) => i.status !== 'done') || items[0];
    if (pick) this.openItem(pick.name);
  }

  renderItems() {
    const sel = $('#itemSelect');
    sel.innerHTML = '';
    const icon = { todo: '○', progress: '◐', done: '●' };
    for (const it of this.project.items) {
      const p = it.stats && it.stats.progress != null ? ` · ${Math.round(it.stats.progress * 100)}%` : '';
      sel.append(h('option', { value: it.name }, `${icon[it.status] || '○'} ${it.name}${p}${it.range ? ' ✂' : ''}${it.boxes ? '' : ' (no boxes)'}`));
    }
    if (this.task) sel.value = this.task.name;
  }

  async openItem(name) {
    if (this.task) this.task.destroy();
    this.task = null;
    const tok = (this._openTok = (this._openTok || 0) + 1);
    $('#loading').classList.remove('hidden');
    let meta;
    try {
      meta = await get(`${item(name)}/meta`);
      if (tok !== this._openTok) return; // another video was picked meanwhile
    } catch (e) {
      if (tok !== this._openTok) return;
      $('#loading').classList.add('hidden');
      toast(`Cannot open ${name}: ${e.message}`, 'err', 6000);
      return;
    }
    if (meta.settings) this.project.settings = meta.settings; // opening can widen the class filter (new box file)
    store.set(`item:${this.project.root}`, name);
    $('#itemSelect').value = name;
    $('#itemStatus').value = meta.status || 'todo';
    $('#undoBtn').disabled = !meta.undo;
    $('#redoBtn').disabled = !meta.redo;
    const Task = TASKS[this.settings.task || 'tracking'];
    this.task = new Task(this, meta);
    for (const w of meta.warnings || []) toast(w, 'err', 6000);
    document.title = `${name} · annoTETE`;
  }

  async _refreshStats() {
    if (!this.task) return;
    const T = this.task;
    const meta = await get(`${T.base}/meta`);
    if (T !== this.task) return;
    T.meta.stats = meta.stats;
    const it = this.project.items.find((i) => i.name === meta.name);
    if (it) { it.stats = meta.stats; }
    panels.renderStats(this.task);
    this.renderItems();
  }

  setSave(s) {
    const el = $('#saveState');
    el.className = `savestate ${s === 'busy' ? 'busy' : s === 'err' ? 'err' : ''}`;
    el.textContent = s === 'busy' ? 'saving…' : s === 'err' ? 'error' : 'saved';
  }

  // Settings changes run one after another, so quick successive changes (a
  // slider being dragged) can never land out of order.
  saveSettings(patch) {
    this._settingsQ = (this._settingsQ || Promise.resolve()).then(() => this._saveSettings(patch));
    return this._settingsQ;
  }

  async _saveSettings(patch) {
    try {
      this.state = await post('/api/settings', patch);
    } catch (e) { toast(e.message, 'err'); return; }
    const T = this.task;
    if (!T) return;
    if ('step' in patch) T.setStep(patch.step);
    if ('classes' in patch || 'min_conf' in patch || 'max_gap' in patch) {
      T.boxes.invalidate(null);
      await T.reloadFrame();
      T.refreshTracks();
      T.refreshCoverage();
      T.refreshProps();
      this.refreshStats();
    }
    if ('class_names' in patch) { T.viewer.render(); panels.renderList(T); }
    T.updateChips();
  }

  async setRange(r) {
    if (!this.task) return;
    try {
      const meta = await post(`${this.task.base}/range`, { range: r });
      this.task.onMeta(meta);
      const it = this.project.items.find((i) => i.name === meta.name);
      if (it) it.range = meta.full_range ? null : meta.range;
      this.renderItems();
      toast(r ? `Range ${meta.range[0]}–${meta.range[1]}` : 'Using the whole video');
    } catch (e) { toast(e.message, 'err'); }
  }

  allRanges() {
    modal('Same range for every video', (close) => {
      const start = h('input', { type: 'text', value: '0', placeholder: 'e.g. 0, 1:30, 90s, 2250' });
      const dur = h('input', { type: 'text', value: '60s', placeholder: 'e.g. 60s, 1:00, 1500 (frames)' });
      const go = async (clear) => {
        try {
          this.state = await post('/api/ranges', clear ? { clear: true } : { start: start.value, duration: dur.value });
          this.renderItems();
          if (this.task) this.openItem(this.task.name);
          close(true);
          toast(clear ? 'Ranges cleared' : 'Range set for every video (clipped to each video length)', 'ok');
        } catch (e) { toast(e.message, 'err'); }
      };
      return h('div', {},
        h('p', { class: 'muted' }, 'Times accept frames (2250), seconds (90s) or m:s (1:30). The range limits the work area; Export → "trimmed" writes the cut videos, their annotations and input boxes, renumbered from frame 0.'),
        h('label', { class: 'blk' }, 'Start'), start, h('label', { class: 'blk' }, 'Duration'), dur,
        h('div', { class: 'actions' }, h('button', { onclick: () => go(true) }, 'Clear all ranges'), h('div', { class: 'spacer' }),
          h('button', { onclick: () => close(null) }, 'Cancel'), h('button', { class: 'primary', onclick: () => go(false) }, 'Apply to all')));
    });
  }

  exportDialog() {
    const p = this.project;
    const anyRange = p.items.some((i) => i.range);
    modal('Export', (close) => {
      // default: <project folder>/export; a folder you pick is remembered for this project only
      const defOut = `${p.root}/export`;
      const out = h('input', { type: 'text', value: store.get(`exportDir:${p.root}`, defOut) });
      const resetOut = h('a', { href: '#', onclick: (e) => { e.preventDefault(); out.value = defOut; } }, 'project default');
      const fmt = (k, label, on) => h('label', { class: 'tog' }, h('input', { type: 'checkbox', value: k, checked: on, class: 'fmt' }), label);
      const trimmed = h('input', { type: 'checkbox', checked: anyRange });
      const interp = h('input', { type: 'checkbox', checked: true });
      const video = h('input', { type: 'checkbox', checked: store.get(`exportVideo:${p.root}`, false) });
      const which = h('select', {}, h('option', { value: 'all' }, `all videos (${p.items.length})`),
        this.task ? h('option', { value: 'cur' }, `only ${this.task.name}`) : '');
      const log = h('pre', { class: 'log hidden' });
      const run = h('button', { class: 'primary' }, 'Export');
      run.onclick = async () => {
        store.set(`exportDir:${p.root}`, out.value);
        store.set(`exportVideo:${p.root}`, video.checked);
        const formats = $$('input.fmt', log.parentElement).filter((c) => c.checked).map((c) => c.value);
        run.disabled = true;
        log.classList.remove('hidden');
        log.textContent = 'starting ...';
        try {
          let job = await post('/api/export', { out: out.value, trimmed: trimmed.checked, interp: interp.checked, formats, video: video.checked,
            items: which.value === 'cur' ? [this.task.name] : null });
          while (!job.done) {
            await new Promise((r) => setTimeout(r, 400));
            job = await get(`/api/jobs/${job.id}`);
            log.textContent = job.log.join('\n') + (job.warnings.length ? `\n\nwarnings:\n${job.warnings.join('\n')}` : '');
          }
          if (job.error) { log.textContent += `\n\nERROR: ${job.error}`; toast(job.error, 'err'); } else toast('Export finished', 'ok');
        } catch (e) { log.textContent += `\n${e.message}`; toast(e.message, 'err'); }
        run.disabled = false;
      };
      return h('div', {},
        h('label', { class: 'blk' }, 'Output folder ', h('span', { class: 'muted' }, '(', resetOut, ')')), out,
        h('label', { class: 'blk' }, 'Annotation formats'),
        h('div', { class: 'row' }, fmt('txt', 'same layout as the input (extra columns kept)', true)),
        h('div', { class: 'row' }, fmt('mot', 'MOTChallenge gt (1-based)', true), fmt('json', 'JSON (with box kinds)', true)),
        h('div', { class: 'muted small' }, 'evaluATE uses the MOT + JSON files of an export as ground truth.'),
        h('label', { class: 'blk' }, 'Options'),
        h('div', { class: 'row' }, h('label', { class: 'tog' }, trimmed, 'trimmed: only each video\'s range, renumbered from 0, plus the cut video and its input boxes')),
        h('div', { class: 'row' }, h('label', { class: 'tog' }, interp, 'include interpolated boxes (no detection behind them)')),
        h('div', { class: 'row' }, h('label', { class: 'tog' }, video, 'annotated video: boxes and ids drawn on the frames (dashed = interpolated); follows the trimmed option')),
        h('div', { class: 'row', style: { marginTop: '8px' } }, which),
        h('p', { class: 'muted' }, 'Writes annotations/<video>.txt (and annotations_mot/, annotations_json/); with trimmed also videos/<video>.mp4 and detections/<video>.txt; with annotated video videos_annotated/<video>.mp4.'),
        log,
        h('div', { class: 'actions' }, h('button', { onclick: () => close(null) }, 'Close'), run));
    }, { wide: true });
  }

  projectDialog() {
    const p = this.project;
    modal('Project', (close) => h('div', {},
      h('div', { class: 'stat' }, h('span', {}, 'folder'), h('b', { class: 'mono' }, p.root)),
      h('div', { class: 'stat' }, h('span', {}, 'videos'), h('b', { class: 'mono' }, p.videos || '')),
      h('div', { class: 'stat' }, h('span', {}, 'boxes'), h('b', { class: 'mono' }, p.boxes || 'none')),
      h('div', { class: 'stat' }, h('span', {}, 'mode'), h('b', {}, p.settings.mode)),
      h('div', { class: 'stat' }, h('span', {}, 'videos done'), h('b', {}, `${p.items.filter((i) => i.status === 'done').length} / ${p.items.length}`)),
      h('p', { class: 'muted' }, 'Annotations are autosaved every few seconds to the project folder (annotations/<video>.json, with a .bak copy).'),
      h('div', { class: 'actions' },
        h('button', { onclick: async () => { await post('/api/save'); toast('Saved', 'ok'); } }, 'Save now'),
        h('button', { onclick: () => { close(); this.showSetup(); } }, 'Open / create another project…'),
        h('button', { class: 'primary', onclick: () => close() }, 'Close'))));
  }

  helpDialog() {
    modal('Shortcuts & workflow', () => h('div', {},
      h('div', { class: 'help' },
        h('p', {}, h('b', {}, 'Detections → tracks: '), 'on the first frame press ', h('kbd', {}, 'Shift+N'), ' (every detection becomes a track). Then press ',
          h('kbd', {}, 'Space'), ' to jump one step: dashed boxes are proposals where each track probably is now; fix the wrong ones by clicking (brush) or ',
          h('kbd', {}, 'X'), ' (swap), then ', h('kbd', {}, 'Space'), ' again. Frames in between are filled automatically by snapping to your detections. ',
          h('kbd', {}, 'Shift+Space'), ' advances by itself until something is uncertain.'),
        h('p', {}, h('b', {}, 'Tracker output → clean tracks: '), 'play / scrub, fix ID switches with ', h('kbd', {}, 'X'), ', join fragments with ', h('kbd', {}, 'M'),
          ' (or from the Review tab), cut with ', h('kbd', {}, 'K'), '. The Review tab lists likely mistakes.')),
      legend(),
      h('div', { class: 'keys' }, KEYMAP.map(([title, rows]) => [h('h5', {}, title),
        ...rows.map(([k, d]) => h('div', { class: 'k' }, h('span', {}, d), h('kbd', {}, k)))]))), { wide: true });
  }

  _bindGlobal() {
    $('#itemSelect').addEventListener('change', (e) => this.openItem(e.target.value));
    const move = (d) => {
      const items = this.project.items;
      const i = items.findIndex((x) => x.name === (this.task && this.task.name));
      const n = items[(i + d + items.length) % items.length];
      if (n) this.openItem(n.name);
    };
    $('#prevItem').onclick = () => move(-1);
    $('#nextItem').onclick = () => move(1);
    $('#itemStatus').addEventListener('change', async (e) => {
      if (!this.task) return;
      await post(`${this.task.base}/status`, { status: e.target.value });
      const it = this.project.items.find((i) => i.name === this.task.name);
      if (it) it.status = e.target.value;
      this.renderItems();
    });
    $('#undoBtn').onclick = () => this.task && this.task.undo();
    $('#redoBtn').onclick = () => this.task && this.task.redo();
    $('#exportBtn').onclick = () => this.exportDialog();
    $('#projectBtn').onclick = () => this.projectDialog();
    $('#helpBtn').onclick = () => this.helpDialog();
    $('#brand').onclick = (e) => switchApp(e);
    $('#fitBtn').onclick = (e) => { if (this.task) this.task.viewer.fit(); e.currentTarget.blur(); };
    for (const b of $$('.tabs button')) {
      b.onclick = () => {
        $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
        $$('.tab').forEach((t) => t.classList.toggle('on', t.id === `tab-${b.dataset.tab}`));
        if (b.dataset.tab === 'data' && this.task) panels.renderData(this.task);
        if (b.dataset.tab === 'review' && this.task && !this.task.issues.length) panels.refreshIssues(this.task);
      };
    }
    for (const b of $$('.transport [data-act]')) {
      b.onclick = () => {
        const a = b.dataset.act;
        if (a === 'clearRange') this.setRange(null);
        else if (a === 'allRanges') this.allRanges();
        else if (this.task) this.task.transport(a);
        b.blur();
      };
    }
    $('#frameInput').addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') { this.task && this.task.go(Number(e.target.value)); e.target.blur(); }
      if (e.key === 'Escape') e.target.blur();
    });
    $('#stepInput').addEventListener('keydown', (e) => e.stopPropagation());
    $('#stepInput').addEventListener('change', (e) => this.saveSettings({ step: Math.max(1, Number(e.target.value) | 0) }));
    document.addEventListener('keydown', (e) => {
      if ($('#modalRoot').children.length) return;
      const tag = (e.target.tagName || '').toLowerCase();
      if (['input', 'select', 'textarea'].includes(tag)) return;
      if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
        e.preventDefault();
        post('/api/save').then(() => toast('Saved', 'ok'));
        return;
      }
      if (e.key === '?') { this.helpDialog(); e.preventDefault(); return; }
      if (this.task && this.task.onKey(e)) e.preventDefault();
    });
    window.addEventListener('beforeunload', () => { if (this.task) this.task.destroy(); navigator.sendBeacon && navigator.sendBeacon('/api/save', '{}'); });
  }
}

window.annotete = new App();
