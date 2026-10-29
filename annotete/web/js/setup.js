// Project setup: pick videos + box files, preview the pairing and classes,
// choose mode / classes, create (or re-open) a project.
import { get, post } from './api.js';
import { h, toast, store, debounce } from './util.js';

export function renderSetup(root, project, onReady) {
  root.innerHTML = '';
  const last = store.get('setup', {});
  const pathInput = (id, value, placeholder) => {
    const dl = h('datalist', { id: `${id}-dl` });
    const inp = h('input', { type: 'text', id, value: value || '', placeholder, list: `${id}-dl`, autocomplete: 'off', spellcheck: 'false' });
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
  const [videos, vdl] = pathInput('sVideos', last.videos, '/data/videos   or   /data/videos/cam1.mp4');
  const [boxes, bdl] = pathInput('sBoxes', last.boxes, '/data/dets   or   /data/dets/cam1.txt   (optional)');
  const [proj, pdl] = pathInput('sProject', last.project, 'folder for the project (annotations are saved here)');
  const format = h('select', {}, h('option', { value: 'auto' }, 'auto detect'),
    h('option', { value: 'txt' }, 'txt: frame id x y w h conf class quality … (0-based)'),
    h('option', { value: 'mot' }, 'MOTChallenge: comma separated, 1-based'),
    h('option', { value: 'custom' }, 'custom columns …'));
  const custom = h('input', { type: 'text', value: 'frame=0,id=1,x=2,y=3,w=4,h=5,conf=6,cls=7,quality=8', class: 'hidden' });
  format.addEventListener('change', () => custom.classList.toggle('hidden', format.value !== 'custom'));
  let mode = 'auto';
  const modeSeg = h('div', { class: 'seg' }, [['auto', 'auto'], ['detections', 'detections → assign ids'], ['tracks', 'tracker output → correct']].map(([k, label]) =>
    h('button', { 'data-k': k, class: k === mode ? 'on' : '', onclick: (e) => { mode = k; for (const b of modeSeg.children) b.classList.toggle('on', b.dataset.k === k); } }, label)));
  const step = h('input', { type: 'number', value: last.step || 10, min: 1 });
  const minConf = h('input', { type: 'number', value: last.minConf ?? '', step: 'any', placeholder: 'none' });
  const preview = h('div', {});
  const previewCard = h('div', { class: 'card hidden' }, preview);
  let classSel = null;

  const fmt = () => (format.value === 'custom' ? custom.value : format.value);

  async function scan() {
    preview.innerHTML = '';
    previewCard.classList.remove('hidden');
    if (!videos.value.trim()) { toast('Give the videos path', 'err'); return null; }
    preview.append(h('p', { class: 'muted' }, 'scanning …'));
    try {
      const r = await post('/api/scan', { videos: videos.value.trim(), boxes: boxes.value.trim(), format: fmt() });
      preview.innerHTML = '';
      const rows = r.items.map((e) => h('tr', {}, h('td', {}, e.name), h('td', { class: 'mono' }, e.media.split('/').slice(-2).join('/')),
        h('td', { class: 'mono', style: { color: e.boxes ? '' : '#f5a524' } }, e.boxes ? e.boxes.split('/').slice(-2).join('/') : 'no box file')));
      preview.append(h('h4', {}, `${r.items.length} video${r.items.length > 1 ? 's' : ''}`),
        h('div', { style: { maxHeight: '220px', overflow: 'auto' } }, h('table', {}, h('tr', {}, h('th', {}, 'name'), h('th', {}, 'video'), h('th', {}, 'boxes')), rows)));
      for (const w of r.warnings) preview.append(h('div', { class: 'warn' }, '⚠ ', w));
      if (r.preview) {
        const pv = r.preview;
        preview.append(h('h4', { style: { marginTop: '14px' } }, `First box file: ${pv.n} boxes, layout "${pv.layout.name}", frames ${pv.layout.frame_base}-based, confidence ${pv.conf[0]}–${pv.conf[1]}`),
          h('pre', {}, pv.lines.join('\n')));
        if (r.looks_like_tracks != null) {
          preview.append(h('p', { class: 'muted' }, r.looks_like_tracks
            ? 'The id column repeats across frames: this looks like tracker output (auto → correct tracks).'
            : 'The id column is unique per box: these look like detections (auto → assign ids).'));
        }
        const cls = Object.entries(r.classes);
        if (cls.length) {
          const prev = new Set(last.classes || []);
          classSel = new Map();
          preview.append(h('h4', { style: { marginTop: '14px' } }, 'Classes to annotate (none checked = all)'),
            h('div', { class: 'classes' }, cls.map(([c, n]) => {
              const cb = h('input', { type: 'checkbox', checked: prev.has(Number(c)) });
              classSel.set(Number(c), cb);
              return h('label', { class: 'tog' }, cb, `class ${c}`, h('span', { class: 'muted' }, `(${n})`));
            })));
        }
      }
      if (!proj.value.trim()) {
        const base = videos.value.trim().replace(/\/+$/, '');  // sibling folder: <videos>_annotete
        proj.value = `${base.replace(/\.[^/.]+$/, '')}_annotete`;
      }
      return r;
    } catch (e) {
      preview.innerHTML = '';
      preview.append(h('div', { class: 'warn' }, '✖ ', e.message));
      return null;
    }
  }

  async function create() {
    if (!proj.value.trim()) { await scan(); if (!proj.value.trim()) return; }
    const classes = classSel ? [...classSel].filter(([, cb]) => cb.checked).map(([c]) => c) : [];
    const settings = { mode, format: fmt(), step: Number(step.value) || 10, classes,
      min_conf: minConf.value === '' ? null : Number(minConf.value) };
    store.set('setup', { videos: videos.value, boxes: boxes.value, project: proj.value, step: step.value, minConf: minConf.value, classes });
    try {
      const r = await post('/api/project', { root: proj.value.trim(), videos: videos.value.trim(), boxes: boxes.value.trim(), settings });
      for (const w of r.warnings || []) toast(w, '', 5000);
      onReady(r.state);
    } catch (e) { toast(e.message, 'err', 6000); }
  }

  async function openExisting() {
    try {
      const st = await post('/api/open', { root: proj.value.trim() });
      onReady(st);
    } catch (e) { toast(e.message, 'err', 6000); }
  }

  const fld = (label, el, note) => h('div', { class: 'fld' }, h('label', {}, label), el, note ? h('div', { class: 'note' }, note) : '');
  root.append(h('div', { class: 'box' },
    h('h1', {}, 'anno', h('b', {}, 'TETE')),
    h('div', { class: 'sub' }, 'Point to your videos and box files. Everything stays on this machine.'),
    h('div', { class: 'card' },
      fld('Videos', videos, 'a video file, a folder of videos, or image-sequence folders (MOT style seq/img1 works)'),
      fld('Box files', boxes, 'one file for one video, or a folder with files named like the videos (cam1.txt, cam1_dets.txt, cam1/det/det.txt)'),
      fld('Format', h('div', {}, format, custom)),
      fld('Mode', modeSeg),
      h('div', { class: 'row' }, h('button', { onclick: scan }, 'Scan'), h('span', { class: 'muted' }, 'check the pairing and pick classes')),
      vdl, bdl, pdl),
    previewCard,
    h('div', { class: 'card' },
      fld('Project folder', proj, 'created if missing; if it already has a project.json, that project is opened'),
      fld('Keyframe step', step, 'frames per jump (A / D / Space); change any time'),
      fld('Min confidence', minConf, 'ignore weaker boxes (can be changed later)'),
      h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
        h('button', { onclick: openExisting }, 'Open existing project'),
        h('button', { class: 'primary', onclick: create }, 'Create project & start'))),
    project ? h('p', { class: 'muted' }, 'Current project: ', h('a', { href: '#', onclick: (e) => { e.preventDefault(); onReady(null); } }, project.root)) : '',
  ));
}
