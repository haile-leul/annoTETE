"""Tracks are keyframes + automatic fill.

A track stores
  keys  {frame: (x, y, w, h, det, flags)}  boxes the user (or an imported
        tracker) fixed. det is the detection index or -1.
  auto  {frame: (x, y, w, h, det)}         boxes between two keys, found by
        interpolation + snapping to detections (see fill.py).

The box of a track at frame f is keys[f] (unless it is an OUTSIDE key), else
auto[f], else none. An OUTSIDE key means "not visible from here until the
next key" (occlusion / out of view), so nothing is filled after it.
"""
import bisect

OUTSIDE = 1    # object absent from this key until the next key
MANUAL = 2     # drawn / edited by hand
IMPORTED = 4   # came from an input tracking file
FORCE_FILL = 8  # fill to the next key even if the gap is larger than max_gap

# kinds of a resolved box, as sent to the UI
KEY_DET, KEY_MANUAL, AUTO_DET, AUTO_INTERP = 0, 1, 2, 3


class Track:
  __slots__ = ('id', 'cls', 'keys', 'auto', 'attrs', '_kf', '_runs', '_ser')

  def __init__(self, tid, cls=-1, attrs=None):
    self.id = int(tid)
    self.cls = int(cls)
    self.keys = {}
    self.auto = {}
    self.attrs = dict(attrs or {})
    self._kf = None
    self._runs = None
    self._ser = None

  def copy(self):
    t = Track(self.id, self.cls, self.attrs)
    t.keys = dict(self.keys)
    t.auto = dict(self.auto)
    return t

  def changed(self):
    self._kf = None
    self._runs = None
    self._ser = None

  # --- queries -------------------------------------------------------------
  def key_frames(self):
    if self._kf is None:
      self._kf = sorted(self.keys)
    return self._kf

  def visible_keys(self):
    return [f for f in self.key_frames() if not self.keys[f][5] & OUTSIDE]

  def box(self, f):
    """(x, y, w, h, det, kind) or None"""
    k = self.keys.get(f)
    if k is not None:
      if k[5] & OUTSIDE:
        return None
      return (k[0], k[1], k[2], k[3], k[4], KEY_MANUAL if k[4] < 0 else KEY_DET)
    a = self.auto.get(f)
    if a is not None:
      return (a[0], a[1], a[2], a[3], a[4], AUTO_DET if a[4] >= 0 else AUTO_INTERP)
    return None

  def has_box(self, f):
    k = self.keys.get(f)
    if k is not None:
      return not k[5] & OUTSIDE
    return f in self.auto

  def prev_key(self, f, visible=False):
    """Last key frame < f (or None)."""
    kf = self.key_frames()
    i = bisect.bisect_left(kf, f) - 1
    while i >= 0:
      if not visible or not self.keys[kf[i]][5] & OUTSIDE:
        return kf[i]
      i -= 1
    return None

  def next_key(self, f, visible=False):
    """First key frame > f (or None)."""
    kf = self.key_frames()
    i = bisect.bisect_right(kf, f)
    while i < len(kf):
      if not visible or not self.keys[kf[i]][5] & OUTSIDE:
        return kf[i]
      i += 1
    return None

  def runs(self):
    """Visible frame ranges [[a, b], ...] (inclusive), cached."""
    if self._runs is None:
      frames = sorted([f for f, k in self.keys.items() if not k[5] & OUTSIDE] + list(self.auto))
      runs = []
      for f in frames:
        if runs and f == runs[-1][1] + 1:
          runs[-1][1] = f
        else:
          runs.append([f, f])
      self._runs = runs
    return self._runs

  def start(self):
    r = self.runs()
    return r[0][0] if r else None

  def end(self):
    r = self.runs()
    return r[-1][1] if r else None

  def n_boxes(self):
    return sum(b - a + 1 for a, b in self.runs())

  def last_box_before(self, f):
    """(frame, box) of the last visible box at a frame < f."""
    for a, b in reversed(self.runs()):
      if a < f:
        g = min(b, f - 1)
        return g, self.box(g)
    return None, None

  def dets(self):
    """(frame, det) of every box that is linked to a detection."""
    for f, k in self.keys.items():
      if k[4] >= 0:
        yield f, k[4]
    for f, a in self.auto.items():
      if a[4] >= 0:
        yield f, a[4]

  # --- persistence -----------------------------------------------------------
  def to_json(self):
    if self._ser is None:  # boxes are serialized once per change (saves of big projects stay fast)
      r = lambda v: round(float(v), 2)
      self._ser = ([[f, r(k[0]), r(k[1]), r(k[2]), r(k[3]), int(k[4]), int(k[5])] for f, k in sorted(self.keys.items())],
                   [[f, r(a[0]), r(a[1]), r(a[2]), r(a[3]), int(a[4])] for f, a in sorted(self.auto.items())])
    return dict(id=self.id, cls=self.cls, attrs=self.attrs, keys=self._ser[0], auto=self._ser[1])

  @classmethod
  def from_json(cls, d):
    t = cls(d['id'], d.get('cls', -1), d.get('attrs'))
    for f, x, y, w, h, det, flags in d.get('keys', []):
      t.keys[int(f)] = (x, y, w, h, int(det), int(flags))
    for f, x, y, w, h, det in d.get('auto', []):
      t.auto[int(f)] = (x, y, w, h, int(det))
    return t
