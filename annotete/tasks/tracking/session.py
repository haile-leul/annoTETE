"""One opened item of a tracking project: frames + detections + annotation,
plus the command / query / export entry points used by the server and CLI."""
import json
import os
import shutil
import threading

import numpy as np

from ...formats import BoxWriter, Detections, Layout, mot_line
from ...project import atomic_json
from ...sources import FrameServer, open_source
from ... import trim
from .annotation import EditError, TrackingAnnotation
from .model import AUTO_INTERP

SETTING_KEYS = ('step', 'max_gap', 'classes', 'min_conf')
KIND_NAMES = {0: 'key', 1: 'manual', 2: 'auto', 3: 'interp'}


class TrackingSession:
  task = 'tracking'

  def __init__(self, project, item, frames=True):
    self.project = project
    self.item = item
    self.lock = threading.RLock()
    self._save_lock = threading.Lock()
    self.warnings = []
    self.source = open_source(item['media'])
    self.frames = FrameServer(self.source) if frames else None
    if item.get('boxes'):
      self.dets = project.load_boxes(item)
      self.warnings += self.dets.warnings
      self.warnings += sanity_warnings(self.dets, self.source)
    else:
      self.dets = Detections.empty(Layout.from_spec(project.settings.get('format') or 'txt'))
    s = project.settings
    self.ann = TrackingAnnotation(self.dets, self.source.n_frames, self.source.width, self.source.height,
                                  {k: s.get(k) for k in SETTING_KEYS})
    path = project.ann_path(item)
    mode = s.get('mode') or 'auto'
    if mode == 'auto':
      mode = 'tracks' if self.dets.ids_look_like_tracks() else 'detections'
    self.mode = mode
    if os.path.exists(path):
      with open(path) as f:
        data = json.load(f)
      self.warnings += self.ann.load_json(data)
      self.mode = data.get('mode', mode)
    elif mode == 'tracks':
      self.ann.import_tracks()
      self.ann.dirty = True

  # ------------------------------------------------------------------ info
  @property
  def range(self):
    r = self.item.get('range')
    n = self.ann.n_frames
    if not r:
      return [0, n - 1]
    return [max(0, min(int(r[0]), n - 1)), max(0, min(int(r[1]), n - 1))]

  def meta(self):
    d = self.dets
    lo, hi = d.conf_range()
    return dict(name=self.item['name'], media=self.item['media'], boxes=self.item.get('boxes'), kind=self.source.kind,
                n_frames=self.ann.n_frames, fps=self.source.fps, width=self.source.width, height=self.source.height,
                mode=self.mode, classes=d.classes(), conf_range=[lo, hi], n_dets=len(d), range=self.range,
                full_range=self.item.get('range') is None, status=self.item.get('status', 'todo'),
                warnings=self.warnings, rev=self.ann.rev, next_id=self.ann.next_id, stats=self.ann.stats(*self.range),
                layout=d.layout.to_dict(), undo=len(self.ann._undo), redo=len(self.ann._redo))

  # --------------------------------------------------------------- commands
  def command(self, op, a):
    A = self.ann
    f = lambda: int(a['frame'])
    tid = lambda k='track': (int(a[k]) if a.get(k) is not None else None)
    ops = {
      'assign': lambda: A.assign(f(), a['det'], tid()),
      'new_tracks': lambda: A.new_tracks(f(), a.get('dets')),
      'set_box': lambda: A.set_box(f(), a['box'], tid(), a.get('cls')),
      'delete_box': lambda: A.delete_box(f(), tid()),
      'outside': lambda: A.set_outside(f(), tid()),
      'delete_track': lambda: A.delete_track(tid(), int(a['frame']) if a.get('from_frame') else None),
      'swap': lambda: A.swap(f(), tid('a'), tid('b')),
      'merge': lambda: A.merge(tid('into'), tid('other')),
      'split': lambda: A.split(f(), tid()),
      'relabel': lambda: A.relabel(tid(), int(a['new_id'])),
      'set_class': lambda: A.set_class(tid(), int(a['cls'])),
      'fill_gaps': lambda: A.fill_gaps(tid(), a.get('max_len')),
      'accept': lambda: A.accept(f(), a.get('tracks'), bool(a.get('only_sure'))),
      'advance': lambda: A.advance(f(), self.range[1], a.get('step'), int(a.get('max_steps', 1000))),
      'bootstrap': lambda: A.bootstrap(*self.range),
      'undo': A.undo,
      'redo': A.redo,
    }
    if op not in ops:
      raise EditError(f'unknown command {op}')
    with self.lock:
      rev0 = A.rev
      A.last_dirty = None
      result = ops[op]()
      changed = A.rev != rev0
      return dict(result=result, rev=A.rev, changed=changed, dirty=A.last_dirty if changed else None,
                  next_id=A.next_id, undo=len(A._undo), redo=len(A._redo))

  def configure(self, settings):
    with self.lock:
      self.ann.configure(**{k: settings[k] for k in SETTING_KEYS if k in settings})

  # ----------------------------------------------------------------- queries
  def frame_data(self, a, b):
    with self.lock:
      return self.ann.frame_data(a, b)

  def proposals(self, f):
    with self.lock:
      props, lost = self.ann.proposals(f)
      return dict(frame=f, proposals=props, lost=lost)

  def tracks(self):
    with self.lock:
      return self.ann.tracks_summary()

  def track(self, tid):
    with self.lock:
      return self.ann.track_detail(tid)

  def coverage(self, bins):
    with self.lock:
      return self.ann.coverage(*self.range, bins=bins)

  def issues(self):
    with self.lock:
      return self.ann.issues(*self.range, fps=self.source.fps)

  # -------------------------------------------------------------- persistence
  def save(self, force=False):
    with self.lock:  # snapshot under the lock, write without it (big files take a while)
      if not (self.ann.dirty or force):
        return False
      data = self.ann.to_json()
      data.update(item=self.item['name'], mode=self.mode, media=self.item['media'], boxes=self.item.get('boxes'))
      self.ann.dirty = False
      self.item['stats'] = self.ann.stats(*self.range)
    with self._save_lock:
      try:
        atomic_json(self.project.ann_path(self.item), data)
      except Exception:
        self.ann.dirty = True  # retry on the next autosave
        raise
    self.project.save()
    return True

  def close(self):
    self.save()
    if self.frames:
      self.frames.close()
    else:
      self.source.close()

  # ------------------------------------------------------------------ export
  def export(self, out_dir, trimmed=False, formats=('txt',), interp=True, log=print):
    """Write annotations (and with `trimmed`: the trimmed video + input boxes)
    into out_dir. Frames are renumbered from the start of the range when
    trimmed. Returns a list of warnings."""
    name = self.item['name']
    a, b = self.range if trimmed else (0, self.ann.n_frames - 1)
    off = a if trimmed else 0
    warnings = []
    with self.lock:
      rows = self.ann.export_rows(a, b, interp)
      cls_of = {t.id: t.cls for t in self.ann.tracks.values()}
    if 'txt' in formats:
      w = BoxWriter(self.dets)
      lines = []
      for f, tid, box, det, kind, cls in rows:
        if det >= 0:
          lines.append(w.det_line(det, frame=f - off if trimmed else None, track_id=tid))
        else:
          lines.append(w.box_line(f - off, tid, box, cls))
      _write_lines(os.path.join(out_dir, 'annotations', name + '.txt'), lines)
    if 'mot' in formats:
      _write_lines(os.path.join(out_dir, 'annotations_mot', name + '.txt'),
                   [mot_line(f - off, tid, box, cls) for f, tid, box, det, kind, cls in rows])
    if 'json' in formats:
      tracks = {}
      for f, tid, box, det, kind, cls in rows:
        tracks.setdefault(tid, []).append([f - off] + [round(float(v), 2) for v in box] + [int(det), KIND_NAMES[kind]])
      data = dict(item=name, source=self.item['media'], frame_offset=off, n_frames=b - a + 1,
                  width=self.source.width, height=self.source.height, fps=self.source.fps,
                  box_format='x,y,w,h (top-left)', row_format='frame,x,y,w,h,det_index,kind',
                  tracks=[dict(id=tid, cls=cls_of.get(tid, -1), boxes=v) for tid, v in sorted(tracks.items())])
      p = os.path.join(out_dir, 'annotations_json', name + '.json')
      os.makedirs(os.path.dirname(p), exist_ok=True)
      with open(p, 'w') as fh:
        json.dump(data, fh)
    if trimmed:
      full = (a, b) == (0, self.ann.n_frames - 1)
      vdir = os.path.join(out_dir, 'videos')
      if full and self.source.kind == 'video':
        os.makedirs(vdir, exist_ok=True)
        shutil.copy2(self.item['media'], os.path.join(vdir, name + os.path.splitext(self.item['media'])[1]))
      else:
        log(f'{name}: trimming frames {a}-{b}')
        _, warn = trim.trim_media(self.item['media'], vdir, name, a, b, self.source.kind)
        if warn:
          warnings.append(warn)
      if self.item.get('boxes'):
        trim.trim_boxes(self.dets, os.path.join(out_dir, 'detections', name + '.txt'), a, b)
    n_interp = sum(1 for r in rows if r[4] == AUTO_INTERP)
    log(f'{name}: {len(rows)} boxes ({n_interp} interpolated) in {len({r[1] for r in rows})} tracks')
    return warnings


def sanity_warnings(dets, source):
  """Catch the usual mix-ups: wrong video/box pairing, other resolution, frame base."""
  out = []
  if not len(dets):
    return out
  name = os.path.basename(dets.path or '')
  b = dets.box
  inside_w = np.clip(np.minimum(b[:, 0] + b[:, 2], source.width) - np.maximum(b[:, 0], 0), 0, None)
  inside_h = np.clip(np.minimum(b[:, 1] + b[:, 3], source.height) - np.maximum(b[:, 1], 0), 0, None)
  outside = float(np.mean(inside_w * inside_h < 0.5 * b[:, 2] * b[:, 3]))
  if outside > 0.2:
    out.append(f'{name}: {outside:.0%} of the boxes lie outside the {source.width}x{source.height} frame '
               '(wrong video/box pairing, or boxes for another resolution?)')
  last = int(dets.frame.max())
  if last >= source.n_frames:
    n = int((dets.frame >= source.n_frames).sum())
    out.append(f'{name}: {n} box(es) are on frames after the end of the video (last box frame {last}, video has '
               f'{source.n_frames} frames): check the pairing / frame base')
  return out


def _write_lines(path, lines):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + ('\n' if lines else ''))
