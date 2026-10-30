// Sidebar panels of the tracking task: Tracks, Review, Data.
import { get } from '../../api.js';
import { $, h, idColor, toast, clamp, debounce } from '../../util.js';

export function build(T) {
  const tracks = $('#tab-tracks');
  tracks.innerHTML = '';
  T.ui = {};
  T.ui.active = h('div', { class: 'card' });
  T.ui.search = h('input', { type: 'text', placeholder: 'find id ...', style: { flex: 1, minWidth: 0 } });
  T.ui.onlyHere = h('input', { type: 'checkbox', checked: T.opts.onlyHere });
  T.ui.list = h('div', { class: 'list' });
  T.ui.count = h('span', { class: 'muted' });
  T.ui.search.addEventListener('input', () => renderList(T));
  T.ui.search.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { const id = Number(T.ui.search.value); if (T.trackById.has(id)) T.selectAndShow(id); T.ui.search.blur(); }
    if (e.key === 'Escape') T.ui.search.blur();
  });
  T.ui.onlyHere.addEventListener('change', () => { T.opts.onlyHere = T.ui.onlyHere.checked; renderList(T); });
  tracks.append(
    T.ui.active,
    h('div', { class: 'row' }, T.ui.search, h('label', { class: 'tog', title: 'Only tracks on this frame' }, T.ui.onlyHere, 'here'), T.ui.count),
    T.ui.list,
  );
  buildReview(T);
  renderData(T);
  renderActive(T);
}

// ------------------------------------------------------------------ tracks
export function renderList(T) {
  const q = T.ui.search.value.trim();
  const here = new Set(T.cur.t.map((t) => t[0]));
  let rows = T.tracks.filter((t) => t.visible);
  if (q) rows = rows.filter((t) => String(t.id).startsWith(q));
  if (T.opts.onlyHere) rows = rows.filter((t) => here.has(t.id));
  rows = [...rows].sort((a, b) => a.id - b.id);
  T.ui.count.textContent = `${rows.length}`;
  const list = T.ui.list;
  list.innerHTML = '';
  if (!rows.length) {
    list.append(h('div', { class: 'empty' }, T.tracks.length ? 'no match' : 'No tracks yet. Click a white detection to start one, or Shift+N to make every detection on this frame a track.'));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const t of rows.slice(0, 1500)) {
    const a = t.runs[0][0], b = t.runs[t.runs.length - 1][1];
    frag.append(h('div', {
      class: `li${t.id === T.active ? ' on' : ''}${here.has(t.id) ? ' here' : ''}`, 'data-id': t.id,
      onclick: () => T.selectAndShow(t.id),
      ondblclick: () => { T.setActive(t.id); T.go(a); },
    }, h('span', { class: 'sw', style: { background: idColor(t.id) } }), h('span', { class: 'id' }, `#${t.id}`),
    h('span', { class: 'meta' }, `${T.className(t.cls)} · ${a}–${b}${t.runs.length > 1 ? ` · ${t.runs.length - 1} gap${t.runs.length > 2 ? 's' : ''}` : ''}`),
    h('span', { class: 'meta' }, `${t.n}`)));
  }
  list.append(frag);
}

export function highlightList(T) {
  for (const el of T.ui.list.querySelectorAll('.li')) el.classList.toggle('on', Number(el.dataset.id) === T.active);
  const on = T.ui.list.querySelector('.li.on');
  if (on) on.scrollIntoView({ block: 'nearest' });
}

export function markHere(T) {
  if (T.opts.onlyHere) { renderList(T); return; }
  const here = new Set(T.cur.t.map((t) => t[0]));
  for (const el of T.ui.list.querySelectorAll('.li')) el.classList.toggle('here', here.has(Number(el.dataset.id)));
}

export function renderActive(T) {
  const c = T.ui.active;
  c.innerHTML = '';
  const tid = T.active;
  const t = tid != null ? T.trackById.get(tid) : null;
  if (tid == null || !t) {
    c.append(h('h4', {}, 'Brush'),
      h('div', { class: 'help' },
        h('p', {}, tid == null ? 'No active track: clicking a free detection starts a new track.' : `#${tid} does not exist yet: the next detection you click becomes #${tid}.`),
        h('p', {}, 'Click a colored box to make that track the brush, then click its detections on later frames. Or step with ', h('kbd', {}, 'Space'), ' and only fix the proposals that are wrong.')));
    return;
  }
  const cls = Object.keys(T.meta.classes || {}).map(Number);
  if (!cls.includes(t.cls)) cls.push(t.cls);
  const sel = h('select', { onchange: (e) => T.setClass(tid, Number(e.target.value)), title: 'Class of this track' },
    cls.sort((a, b) => a - b).map((k) => h('option', { value: k, selected: k === t.cls }, T.className(k))));
  const a = t.runs[0][0], b = t.runs[t.runs.length - 1][1];
  const d = T.activeDetail;
  const nKeys = d ? d.keys.filter((k) => !(k[1] & 1)).length : t.keys;
  const nInterp = d ? d.interp.length : null;
  const btn = (label, key, fn, cls2 = '') => h('button', { onclick: fn, class: cls2, title: key }, h('span', {}, label), h('kbd', {}, key));
  c.append(
    h('div', { class: 'trackhead' }, h('span', { class: 'sw', style: { background: idColor(tid) } }), h('span', { class: 'tid' }, `#${tid}`), h('div', { class: 'spacer' }), sel),
    h('div', { class: 'row', style: { marginTop: '8px' } },
      h('button', { onclick: () => T.go(a), title: 'first frame' }, `⇤ ${a}`),
      h('button', { onclick: () => T.jumpKey(-1), title: 'previous keyframe (Alt+←)' }, '‹ key'),
      h('button', { onclick: () => T.jumpKey(1), title: 'next keyframe (Alt+→)' }, 'key ›'),
      h('button', { onclick: () => T.go(b), title: 'last frame' }, `${b} ⇥`)),
    h('div', { class: 'stat', style: { marginTop: '8px' } }, h('span', {}, 'boxes'), h('b', {}, `${t.n}`)),
    h('div', { class: 'stat' }, h('span', {}, 'keyframes'), h('b', {}, `${nKeys}`)),
    nInterp != null ? h('div', { class: 'stat' }, h('span', {}, 'interpolated (no detection)'), h('b', {}, `${nInterp}`)) : '',
    t.runs.length > 1 ? h('div', { class: 'stat' }, h('span', {}, 'gaps'), h('b', {}, `${t.runs.length - 1}`)) : '',
    h('div', { class: 'grid2', style: { marginTop: '10px' } },
      btn('Swap ids…', 'X', () => T.startPending('swap')),
      btn('Merge…', 'M', () => T.startPending('merge')),
      btn('Split here', 'K', () => T.split()),
      btn('Change id', 'E', () => T.editId()),
      btn('Delete box', 'Del', () => T.deleteBox()),
      btn('Absent', 'O', () => T.outside()),
      btn('Ends here', '⇧Del', () => T.endHere()),
      btn('Fill gaps', 'G', () => T.fillGaps()),
    ),
    h('div', { class: 'row', style: { marginTop: '8px' } }, h('button', { class: 'danger', onclick: () => T.deleteTrack(), style: { width: '100%' } }, 'Delete track (Ctrl+Del)')),
  );
}

// ------------------------------------------------------------------ review
function buildReview(T) {
  const tab = $('#tab-review');
  tab.innerHTML = '';
  T.ui.issueList = h('div', { class: 'list' });
  T.ui.issueFilter = new Set(['jump', 'fragment', 'unassigned', 'interp', 'short']);
  const types = [['jump', 'jumps'], ['fragment', 'fragments'], ['unassigned', 'missed'], ['interp', 'interp'], ['short', 'short']];
  T.ui.issueChips = h('div', { class: 'row' }, types.map(([k, label]) => {
    const cb = h('input', { type: 'checkbox', checked: true, onchange: (e) => { e.target.checked ? T.ui.issueFilter.add(k) : T.ui.issueFilter.delete(k); renderIssues(T); } });
    return h('label', { class: 'tog' }, cb, h('span', { class: `issue-t-${k}` }, label));
  }));
  tab.append(
    h('div', { class: 'card' },
      h('h4', {}, 'Review'),
      h('div', { class: 'help' }, h('p', {}, 'Checks for likely mistakes: ', h('b', {}, 'jumps'), ' (ID switch inside a track), ', h('b', {}, 'fragments'),
        ' (a track ends and another starts nearby: merge?), ', h('b', {}, 'missed'), ' objects (detections that stay unassigned), long ',
        h('b', {}, 'interp'), 'olations and ', h('b', {}, 'short'), ' tracks. ', h('kbd', {}, '['), ' ', h('kbd', {}, ']'), ' walk through them.')),
      h('div', { class: 'row' }, h('button', { class: 'primary', onclick: () => refreshIssues(T) }, 'Run checks'), T.ui.issueChips)),
    T.ui.issueList);
  T.ui.issueList.append(h('div', { class: 'empty' }, 'Run the checks when you have tracks.'));
}

export async function refreshIssues(T) {
  T.ui.issueList.innerHTML = '';
  T.ui.issueList.append(h('div', { class: 'empty' }, 'checking ...'));
  try {
    const r = await get(`${T.base}/issues`);
    T.issues = r.issues;
    T.issueIdx = -1;
    renderIssues(T);
  } catch (e) { toast(e.message, 'err'); }
}

function shownIssues(T) { return T.issues.filter((i) => T.ui.issueFilter.has(i.type)); }

function renderIssues(T) {
  const list = T.ui.issueList;
  list.innerHTML = '';
  const items = shownIssues(T);
  $('#issueCount').textContent = T.issues.length ? String(T.issues.length) : '';
  if (!items.length) { list.append(h('div', { class: 'empty' }, T.issues.length ? 'nothing of the selected kinds' : 'No issues found 🎉')); return; }
  items.forEach((it, i) => {
    const extra = it.type === 'fragment'
      ? h('button', { onclick: async (e) => { e.stopPropagation(); T.setActive(it.tracks[0]); await T.mergeIn(it.tracks[1]); refreshIssues(T); } }, `merge into #${it.tracks[0]}`)
      : '';
    list.append(h('div', { class: `issue${i === T.issueIdx ? ' on' : ''}`, onclick: () => openIssue(T, i) },
      h('span', { class: `t ${it.type}` }, it.type === 'unassigned' ? 'missed' : it.type, h('div', { class: 'muted mono' }, `f ${it.frame}`)),
      h('span', { class: 'm' }, it.msg, extra)));
  });
}

function openIssue(T, i) {
  const items = shownIssues(T);
  const it = items[i];
  if (!it) return;
  T.issueIdx = i;
  renderIssues(T);
  const on = T.ui.issueList.querySelector('.issue.on');
  if (on) on.scrollIntoView({ block: 'nearest' });
  if (it.tracks.length) T.setActive(it.type === 'fragment' ? it.tracks[0] : it.tracks[0]);
  T.go(it.type === 'jump' ? it.frame - 1 : it.frame);
}

export function nextIssue(T, dir) {
  const n = shownIssues(T).length;
  if (!n) { toast('No review issues (run the checks in the Review tab)'); return; }
  openIssue(T, clamp(T.issueIdx + dir, 0, n - 1));
}

// -------------------------------------------------------------------- data
// The Video card alone: refreshed after edits without rebuilding the controls
// below it (a rebuild would drop a slider out from under the mouse).
function videoCard(T) {
  const m = T.meta;
  const st = m.stats || {};
  const pct = st.progress != null ? Math.round(st.progress * 100) : null;
  return h('div', { class: 'card', id: 'videoCard' },
    h('h4', {}, 'Video'),
    h('div', { class: 'stat' }, h('span', {}, 'frames'), h('b', {}, `${m.n_frames} @ ${m.fps.toFixed(2)} fps`)),
    h('div', { class: 'stat' }, h('span', {}, 'size'), h('b', {}, `${m.width}×${m.height}`)),
    h('div', { class: 'stat' }, h('span', {}, 'boxes file'), h('b', { title: m.boxes || '' }, m.boxes ? m.boxes.split('/').pop() : 'none')),
    h('div', { class: 'stat' }, h('span', {}, 'input'), h('b', {}, `${m.n_dets} boxes`)),
    h('div', { class: 'stat', title: m.layout.base_reason || '' }, h('span', {}, 'format'),
      h('b', {}, `${m.layout.sep === ',' ? 'comma' : 'space'} separated · frames from ${m.layout.frame_base}`)),
    h('div', { class: 'stat' }, h('span', {}, 'tracks'), h('b', {}, `${st.tracks ?? '-'}`)),
    h('div', { class: 'stat' }, h('span', {}, 'detections with an id'), h('b', {}, pct != null ? `${st.assigned} / ${st.dets} (${pct}%)` : '-')),
    pct != null ? h('div', { class: 'progress' }, h('i', { style: { width: `${pct}%` } })) : '',
    (m.warnings || []).map((w) => h('div', { class: 'help', style: { color: '#f5a524' } }, '⚠ ', w)),
  );
}

export function renderStats(T) {
  const old = document.getElementById('videoCard');
  if (old) old.replaceWith(videoCard(T));
}

export function renderData(T) {
  const tab = $('#tab-data');
  tab.innerHTML = '';
  const m = T.meta;
  const s = T.app.settings;
  tab.append(videoCard(T));

  // classes
  const classes = Object.entries(m.classes || {}).map(([k, n]) => [Number(k), n]).sort((a, b) => a[0] - b[0]);
  const sel = new Set(s.classes || []);
  const names = Object.assign({}, s.class_names || {});
  const saveNames = debounce(() => T.app.saveSettings({ class_names: names }), 500);
  tab.append(h('div', { class: 'card' },
    h('h4', {}, 'Classes to annotate'),
    h('div', { class: 'help' }, h('p', {}, 'Unchecked classes are hidden and ignored (none checked = all). Names are optional.')),
    classes.length ? classes.map(([c, n]) => h('div', { class: 'classrow' },
      h('input', { type: 'checkbox', checked: sel.has(c), onchange: (e) => {
        e.target.checked ? sel.add(c) : sel.delete(c);
        T.app.saveSettings({ classes: [...sel] });
      } }),
      h('span', { class: 'mono' }, String(c)),
      h('input', { type: 'text', value: names[String(c)] || '', placeholder: 'name', onkeydown: (e) => e.stopPropagation(),
        oninput: (e) => { names[String(c)] = e.target.value; saveNames(); } }),
      h('span', { class: 'cnt' }, String(n)))) : h('div', { class: 'muted' }, 'no class column'),
  ));

  // matching
  const [clo, chi] = m.conf_range || [0, 1];
  const span = chi - clo || 1;
  const conf = s.min_conf ?? clo;
  // applied live while dragging (rate limited); the last value always wins
  const liveConf = debounce((v) => T.app.saveSettings({ min_conf: v }), 120);
  const confVal = h('input', { type: 'number', value: +conf.toFixed(3), step: span > 5 ? 1 : 0.01, onkeydown: (e) => e.stopPropagation(),
    onchange: (e) => { confSl.value = e.target.value; T.app.saveSettings({ min_conf: Number(e.target.value) }); } });
  const confSl = h('input', { type: 'range', min: clo, max: chi, step: span / 200, value: conf,
    oninput: (e) => { const v = +e.target.value; confVal.value = v.toFixed(span > 5 ? 1 : 3); liveConf(+confVal.value); } });
  const num = (key, val, min, title) => h('input', { type: 'number', value: val, min, title, onkeydown: (e) => e.stopPropagation(),
    onchange: (e) => T.app.saveSettings({ [key]: Number(e.target.value) }) });
  tab.append(h('div', { class: 'card' },
    h('h4', {}, 'Detections'),
    h('div', { class: 'help' }, 'Minimum confidence (lower boxes are ignored everywhere):'),
    h('div', { class: 'slider' }, confSl, confVal),
    h('div', { class: 'row' }, h('span', { class: 'muted', style: { flex: 1 } }, 'keyframe step'), num('step', s.step, 1, 'frames per A / D / Space')),
    h('div', { class: 'row' }, h('span', { class: 'muted', style: { flex: 1 } }, 'max interpolation gap'), num('max_gap', s.max_gap, 2, 'keys further apart than this are not filled')),
  ));

  // automation
  tab.append(h('div', { class: 'card' },
    h('h4', {}, 'Automation'),
    h('div', { class: 'help' }, h('p', {}, 'Start from tracklets instead of from scratch: a cautious IoU tracker links free detections inside the range (it cuts rather than guesses; merge fragments in Review).')),
    h('div', { class: 'row' },
      h('button', { onclick: async () => { const ids = await T.cmd('bootstrap'); toast(`${ids.length} tracklets created`, 'ok'); } }, 'Build tracklets'),
      h('button', { onclick: async () => { const n = await T.cmd('fill_gaps', {}); toast(`${n} gaps filled`); }, title: 'interpolate every gap of every track' }, 'Fill all gaps')),
  ));

  // display
  const tog = (key, label, kbd) => h('label', { class: 'tog' }, h('input', { type: 'checkbox', checked: T.opts[key],
    onchange: (e) => { T.opts[key] = e.target.checked; T._saveOpts(); T.viewer.render(); if (key === 'props') T.refreshProps(); } }), label, kbd ? h('kbd', {}, kbd) : '');
  tab.append(h('div', { class: 'card' },
    h('h4', {}, 'Display'),
    h('div', { class: 'grid2', style: { display: 'grid' } },
      tog('labels', 'labels', 'L'), tog('free', 'free boxes', 'H'), tog('trails', 'trails', 'T'),
      tog('follow', 'follow track', 'F'), tog('props', 'proposals'), tog('stopNew', 'stop at new objects')),
  ));
}
