"""Text box files: one box per line, whitespace or comma separated.

A `Layout` says which column holds the frame, id, box, confidence, class and
quality. Every other column ("extras") is kept verbatim and written back on
export, so nothing in the input is lost.

Default layout (`txt`, the detector dump format):
  frame id x y w h confidence class quality [extras ...]       frame 0-based
`mot` is MOTChallenge: comma separated, 1-based frames, same first 9 columns.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter

import numpy as np

FIELDS = ('frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'quality')
BASE_COLS = dict(frame=0, id=1, x=2, y=3, w=4, h=5, conf=6, cls=7, quality=8)
PRESETS = {
  'txt': dict(cols=BASE_COLS, frame_base=0, sep=' ', box='tlwh'),
  'mot': dict(cols=BASE_COLS, frame_base=1, sep=',', box='tlwh'),
}
BOX_MODES = ('tlwh', 'xyxy', 'cxcywh')


class Layout:
  def __init__(self, cols=None, frame_base=0, sep=' ', box='tlwh', name='custom'):
    self.cols = {k: int(v) for k, v in (cols or BASE_COLS).items() if v is not None and int(v) >= 0}
    for k in ('frame', 'x', 'y', 'w', 'h'):
      if k not in self.cols:
        raise ValueError(f'layout needs a "{k}" column')
    if box not in BOX_MODES:
      raise ValueError(f'box mode must be one of {BOX_MODES}')
    self.frame_base = int(frame_base)
    self.sep = sep
    self.box = box
    self.name = name

  @classmethod
  def from_spec(cls, spec, path=None):
    """`spec`: preset name ('txt', 'mot', 'auto'), a dict (as written by
    to_dict) or a column string like 'frame=0,id=1,x=2,y=3,w=4,h=5,conf=6,cls=7'."""
    if isinstance(spec, Layout):
      return spec
    if isinstance(spec, dict):
      return cls(spec.get('cols'), spec.get('frame_base', 0), spec.get('sep', ' '), spec.get('box', 'tlwh'),
                 spec.get('name', 'custom'))
    spec = (spec or 'auto').strip()
    if spec == 'auto':
      spec = sniff_preset(path) if path else 'txt'
    if spec in PRESETS:
      return cls(name=spec, **PRESETS[spec])
    cols = {}
    for part in spec.split(','):
      k, _, v = part.partition('=')
      k = k.strip()
      if k not in FIELDS:
        raise ValueError(f'unknown column "{k}" in layout "{spec}" (known: {", ".join(FIELDS)})')
      cols[k] = int(v)
    return cls(cols)

  def to_dict(self):
    return dict(name=self.name, cols=self.cols, frame_base=self.frame_base, sep=self.sep, box=self.box)

  def min_cols(self):
    return max(self.cols.values()) + 1


def sniff_preset(path):
  """'mot' if the file is comma separated, else 'txt'."""
  if path and os.path.isfile(path):
    with open(path) as f:
      for line in f:
        s = line.strip()
        if s and not s.startswith('#'):
          return 'mot' if ',' in s else 'txt'
  return 'txt'


def file_signature(path):
  h = hashlib.sha1()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return f'{os.path.getsize(path)}:{h.hexdigest()[:16]}'


def _split(s):
  return s.replace(',', ' ').split()


class Detections:
  """All boxes of one input file, sorted by frame (stable, so indices are
  reproducible for an unchanged file). Index `i` refers to row i everywhere."""

  def __init__(self, frame, box, conf, cls, quality, src_id, lines, layout, ncols, int_coords, path=None,
               signature=None, warnings=()):
    order = np.argsort(frame, kind='stable')
    self.frame = np.asarray(frame, np.int64)[order]
    self.box = np.asarray(box, np.float64).reshape(-1, 4)[order]
    self.conf = np.asarray(conf, np.float64)[order]
    self.cls = np.asarray(cls, np.int64)[order]
    self.quality = np.asarray(quality, np.float64)[order]
    self.src_id = np.asarray(src_id, np.int64)[order]
    self.lines = [lines[i] for i in order]
    self.layout = layout
    self.ncols = ncols
    self.int_coords = int_coords
    self.path = path
    self.signature = signature
    self.warnings = list(warnings)

  @classmethod
  def empty(cls, layout=None):
    return cls([], np.zeros((0, 4)), [], [], [], [], [], layout or Layout.from_spec('txt'), 9, True)

  def __len__(self):
    return len(self.frame)

  def frame_slice(self, f):
    a = int(np.searchsorted(self.frame, f, 'left'))
    b = int(np.searchsorted(self.frame, f, 'right'))
    return a, b

  def classes(self):
    """{class: count}"""
    vals, counts = np.unique(self.cls, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}

  def conf_range(self):
    if not len(self):
      return 0.0, 1.0
    return float(self.conf.min()), float(self.conf.max())

  def ids_look_like_tracks(self):
    """True if the id column repeats across frames (tracker output) rather than
    being unique per box (detector output)."""
    if 'id' not in self.layout.cols or not len(self):
      return False
    ids = self.src_id[self.src_id >= 0]
    if not len(ids):
      return False
    return len(np.unique(ids)) <= 0.5 * len(ids)


def load_boxes(path, layout='auto'):
  layout = Layout.from_spec(layout, path)
  c = layout.cols
  need = layout.min_cols()
  frame, box, conf, cls, quality, src_id, lines = [], [], [], [], [], [], []
  ncols = Counter()
  int_coords = True
  bad = 0
  first_bad = None
  with open(path) as f:
    for lineno, raw in enumerate(f, 1):
      s = raw.strip()
      if not s or s.startswith('#'):
        continue
      v = _split(s)
      if len(v) < min(need, max(c['frame'], c['x'], c['y'], c['w'], c['h']) + 1):
        bad += 1
        first_bad = first_bad or lineno
        continue
      try:
        fr = int(round(float(v[c['frame']])))
        b = [float(v[c[k]]) for k in ('x', 'y', 'w', 'h')]
      except ValueError:
        if lineno > 1 or bad:  # a header line is fine, other garbage is counted
          bad += 1
          first_bad = first_bad or lineno
        continue
      if layout.box == 'xyxy':
        b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
      elif layout.box == 'cxcywh':
        b = [b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3]]
      if b[2] <= 0 or b[3] <= 0:
        bad += 1
        first_bad = first_bad or lineno
        continue
      if int_coords:
        int_coords = all(_is_int(v[c[k]]) for k in ('x', 'y', 'w', 'h'))
      frame.append(fr - layout.frame_base)
      box.append(b)
      conf.append(_num(v, c.get('conf'), 1.0))
      cls.append(int(_num(v, c.get('cls'), -1)))
      quality.append(_num(v, c.get('quality'), -1.0))
      src_id.append(int(_num(v, c.get('id'), -1)))
      lines.append(s)
      ncols[len(v)] += 1
  warnings = []
  if bad:
    warnings.append(f'{os.path.basename(path)}: skipped {bad} malformed line(s) (first at line {first_bad})')
  dets = Detections(frame, np.array(box, np.float64).reshape(-1, 4), conf, cls, quality, src_id, lines, layout,
                    ncols.most_common(1)[0][0] if ncols else layout.min_cols(), int_coords, path,
                    file_signature(path), warnings)
  if len(dets) and dets.frame.min() < 0:
    warnings.append(f'{os.path.basename(path)}: frames below the frame base {layout.frame_base} were found; '
                    'check the format (0- or 1-based frames)')
  return dets


def _is_int(tok):
  return tok.lstrip('-').isdigit()


def _num(v, col, default):
  if col is None or col >= len(v):
    return default
  try:
    return float(v[col])
  except ValueError:
    return default


def _fmt_num(x, as_int):
  if as_int:
    return str(int(round(x)))
  s = f'{x:.2f}'.rstrip('0').rstrip('.')
  return s if s not in ('-0', '') else '0'


class BoxWriter:
  """Writes boxes in the layout of an input `Detections` file. Boxes that come
  from a detection reuse its original line (only the id / frame columns are
  replaced); other boxes (interpolated / hand drawn) get -1 in every column
  that has no value."""

  def __init__(self, dets):
    self.layout = dets.layout
    self.ncols = max(dets.ncols, self.layout.min_cols())
    self.int_coords = dets.int_coords
    self.lines = dets.lines
    self.comma = self.layout.sep == ',' or (dets.lines and ',' in dets.lines[0])

  def _join(self, tokens):
    return ','.join(tokens) if self.comma else ' '.join(tokens)

  def det_line(self, det, frame=None, track_id=None):
    """Original line of detection `det`; optionally with a new frame (video
    frame index, base is added) and/or id."""
    tok = _split(self.lines[det])
    c = self.layout.cols
    if frame is not None:
      tok[c['frame']] = str(int(frame) + self.layout.frame_base)
    if track_id is not None and 'id' in c:
      tok[c['id']] = str(int(track_id))
    return self._join(tok)

  def box_line(self, frame, track_id, box, cls=-1, conf=-1.0, quality=-1.0):
    tok = ['-1'] * self.ncols
    c = self.layout.cols
    x, y, w, h = box
    if self.layout.box == 'xyxy':
      vals = (x, y, x + w, y + h)
    elif self.layout.box == 'cxcywh':
      vals = (x + w / 2, y + h / 2, w, h)
    else:
      vals = (x, y, w, h)
    tok[c['frame']] = str(int(frame) + self.layout.frame_base)
    for k, val in zip(('x', 'y', 'w', 'h'), vals):
      tok[c[k]] = _fmt_num(val, self.int_coords)
    if 'id' in c:
      tok[c['id']] = str(int(track_id))
    if 'conf' in c:
      tok[c['conf']] = _fmt_num(conf, False)
    if 'cls' in c:
      tok[c['cls']] = str(int(cls))
    if 'quality' in c:
      tok[c['quality']] = _fmt_num(quality, False)
    return self._join(tok)


def mot_line(frame, track_id, box, cls=-1, visibility=1.0):
  """MOTChallenge ground truth line (1-based frame, conf flag 1)."""
  x, y, w, h = box
  return f'{int(frame) + 1},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{int(cls) if cls >= 0 else 1},{visibility:g}'
