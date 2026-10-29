"""Tracking annotation of one video: tracks + the detection pool + edit commands.

Every edit goes through the small set of `_put_* / _clear_*` helpers, which
keep the detection ownership index consistent (a detection belongs to at most
one track) and record undo snapshots. Public commands are transactions: they
either complete or are rolled back, and one command is one undo step.
"""
import bisect
from contextlib import contextmanager

import numpy as np

from ...geometry import iou, lerp
from . import propose, review
from .fill import fill_segment
from .model import (AUTO_INTERP, FORCE_FILL, IMPORTED, MANUAL, OUTSIDE, Track)

DEFAULT_SETTINGS = dict(
  step=10,        # keyframe step (frames) used by proposals / auto-advance
  max_gap=300,    # never fill between keys further apart than this (frames)
  classes=[],     # classes to annotate ([] = all)
  min_conf=None,  # ignore detections below this confidence (None = keep all)
)
UNDO_LIMIT = 300


class EditError(ValueError):
  pass


class TrackingAnnotation:
  def __init__(self, dets, n_frames, width, height, settings=None):
    self.dets = dets
    self.n_frames = int(n_frames)
    self.width, self.height = width, height
    self.settings = dict(DEFAULT_SETTINGS)
    self.tracks = {}
    self.owner = np.full(len(dets), -1, np.int64)
    self.owner_key = np.zeros(len(dets), bool)
    self.next_id = 1
    self.rev = 0
    self.dirty = False
    self._undo, self._redo = [], []
    self._txn = None
    self._pending = {}
    self.configure(**(settings or {}))

  # ---------------------------------------------------------------- settings
  def configure(self, **kw):
    for k, v in kw.items():
      if k in DEFAULT_SETTINGS:
        self.settings[k] = v
    self.settings['step'] = max(1, int(self.settings['step'] or 1))
    self.settings['max_gap'] = max(2, int(self.settings['max_gap'] or 2))
    self.settings['classes'] = sorted(int(c) for c in (self.settings['classes'] or []))
    d = self.dets
    ok = np.ones(len(d), bool)
    if self.settings['classes']:
      ok &= np.isin(d.cls, self.settings['classes']) | (d.cls < 0)
    if self.settings['min_conf'] is not None:
      ok &= d.conf >= float(self.settings['min_conf'])
    self.allowed = ok
    self.rev += 1

  def track_visible(self, t):
    cl = self.settings['classes']
    return not cl or t.cls < 0 or t.cls in cl

  def free_dets(self, t, min_conf=None):
    """(indices, boxes (N, 4)) of usable, unassigned detections at frame t."""
    a, b = self.dets.frame_slice(t)
    if a == b:
      return np.zeros(0, np.int64), np.zeros((0, 4))
    m = self.allowed[a:b] & (self.owner[a:b] < 0)
    if min_conf is not None:
      m &= self.dets.conf[a:b] >= min_conf
    idx = np.nonzero(m)[0] + a
    return idx, self.dets.box[idx]

  # -------------------------------------------------------------- ownership
  def _claim(self, d, tid, is_key, f):
    if self.dets.frame[d] != f:
      raise EditError(f'detection {d} is on frame {self.dets.frame[d]}, not {f}')
    o = int(self.owner[d])
    if o >= 0 and o != tid and o in self.tracks:
      other = self.tracks[o]
      self._clear_frame(other, f)
      self._schedule(other, f, f)
    self.owner[d] = tid
    self.owner_key[d] = is_key

  def _release(self, d, tid):
    if d >= 0 and self.owner[d] == tid:
      self.owner[d] = -1
      self.owner_key[d] = False

  # ------------------------------------------------------------ transactions
  @contextmanager
  def _edit(self, label):
    if self._txn is not None:  # nested command: part of the outer transaction
      yield
      return
    self._txn = {}
    nid0 = self.next_id
    try:
      yield
      self._flush()
    except Exception:
      self._restore(self._txn)
      self.next_id = nid0
      self._txn = None
      self._pending.clear()
      raise
    txn, self._txn = self._txn, None
    self._pending.clear()
    if txn:
      self._undo.append((label, txn, nid0, self.next_id))
      del self._undo[:-UNDO_LIMIT]
      self._redo.clear()
      self._changed(txn)

  def _changed(self, snap):
    self.rev += 1
    self.dirty = True
    lo, hi = None, None
    for tid, old in snap.items():
      for t in (old, self.tracks.get(tid)):
        if t is not None and t.runs():
          lo = t.start() if lo is None else min(lo, t.start())
          hi = t.end() if hi is None else max(hi, t.end())
    self.last_dirty = [lo, hi] if lo is not None else None

  def _touch(self, tid):
    if self._txn is not None and tid not in self._txn:
      t = self.tracks.get(tid)
      self._txn[tid] = t.copy() if t is not None else None

  def _restore(self, snap):
    """Put the tracks in `snap` back; returns the inverse snapshot."""
    inv = {}
    for tid in snap:
      cur = self.tracks.get(tid)
      inv[tid] = cur.copy() if cur is not None else None
      if cur is not None:
        for _, d in cur.dets():
          self._release(d, tid)
        del self.tracks[tid]
    for tid, saved in snap.items():
      if saved is not None:
        t = saved.copy()
        self.tracks[tid] = t
        for f, k in t.keys.items():
          if k[4] >= 0:
            self.owner[k[4]] = tid
            self.owner_key[k[4]] = True
        for f, a in t.auto.items():
          if a[4] >= 0:
            self.owner[a[4]] = tid
            self.owner_key[a[4]] = False
    return inv

  def undo(self):
    if not self._undo:
      return None
    label, snap, nid0, nid1 = self._undo.pop()
    inv = self._restore(snap)
    self.next_id = nid0
    self._redo.append((label, inv, nid0, nid1))
    self._changed(snap)
    return label

  def redo(self):
    if not self._redo:
      return None
    label, snap, nid0, nid1 = self._redo.pop()
    inv = self._restore(snap)
    self.next_id = nid1
    self._undo.append((label, inv, nid0, nid1))
    self._changed(snap)
    return label

  # ------------------------------------------------------ low level editing
  def _new_track(self, cls=-1, tid=None):
    if tid is None:
      tid = self.next_id
    tid = int(tid)
    if tid <= 0:
      raise EditError('track ids must be positive')
    if tid in self.tracks:
      raise EditError(f'track {tid} already exists')
    self._touch(tid)
    t = Track(tid, cls)
    self.tracks[tid] = t
    self.next_id = max(self.next_id, tid + 1)
    return t

  def _drop_track(self, t):
    self._touch(t.id)
    for _, d in t.dets():
      self._release(d, t.id)
    self.tracks.pop(t.id, None)
    self._pending.pop(t.id, None)

  def _clear_frame(self, t, f):
    k = t.keys.get(f)
    a = t.auto.get(f)
    if k is None and a is None:
      return
    self._touch(t.id)
    if k is not None:
      del t.keys[f]
      self._release(k[4], t.id)
    if a is not None:
      del t.auto[f]
      self._release(a[4], t.id)
    t.changed()

  def _put_key(self, t, f, box, det=-1, flags=0):
    if not 0 <= f < self.n_frames:
      raise EditError(f'frame {f} is outside the video (0..{self.n_frames - 1})')
    self._touch(t.id)
    self._clear_frame(t, f)
    if flags & OUTSIDE:
      det = -1
    t.keys[f] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(det), int(flags))
    t.changed()
    if det >= 0:
      self._claim(int(det), t.id, True, f)

  def _put_auto(self, t, f, box, det):
    self._touch(t.id)
    old = t.auto.get(f)
    if old is not None:
      self._release(old[4], t.id)
    t.auto[f] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(det))
    t.changed()
    if det >= 0:
      self._claim(int(det), t.id, False, f)

  def _clear_autos(self, t, a, b):
    if not t.auto:
      return
    self._touch(t.id)
    frames = [f for f in t.auto if a <= f <= b] if b - a > len(t.auto) else [f for f in range(a, b + 1) if f in t.auto]
    for f in frames:
      self._release(t.auto.pop(f)[4], t.id)
    if frames:
      t.changed()

  def _promote(self, t, f):
    """Turn the automatic box at f into a key (so edits at f +- 1 do not move it)."""
    a = t.auto.get(f)
    if a is None:
      return
    self._touch(t.id)
    del t.auto[f]
    t.keys[f] = (a[0], a[1], a[2], a[3], a[4], 0)
    if a[4] >= 0:
      self.owner_key[a[4]] = True
    t.changed()

  def _take_from(self, t, f):
    """Remove and return t's keys and autos at frames >= f."""
    self._promote(t, f - 1)
    self._promote(t, f)
    self._touch(t.id)
    keys = {g: k for g, k in t.keys.items() if g >= f}
    autos = {g: a for g, a in t.auto.items() if g >= f}
    for g in keys:
      del t.keys[g]
    for g in autos:
      del t.auto[g]
    for k in list(keys.values()) + list(autos.values()):
      self._release(k[4], t.id)
    t.changed()
    return keys, autos

  def _give(self, t, keys, autos):
    for g, k in keys.items():
      self._put_key(t, g, k[:4], k[4], k[5])
    for g, a in autos.items():
      if g not in t.keys:
        self._put_auto(t, g, a[:4], a[4])

  # ------------------------------------------------------------------- fill
  def _schedule(self, t, lo, hi):
    p = self._pending.get(t.id)
    self._pending[t.id] = (lo, hi) if p is None else (min(p[0], lo), max(p[1], hi))

  def _flush(self):
    n = 0
    while self._pending and n < 500:
      tid, (lo, hi) = self._pending.popitem()
      t = self.tracks.get(tid)
      if t is not None:
        self._update_fill(t, lo, hi)
      n += 1

  def _normalize(self, t):
    """Drop OUTSIDE keys that do not separate two visible keys; drop empty tracks."""
    kf = t.key_frames()
    vis = [f for f in kf if not t.keys[f][5] & OUTSIDE]
    if not vis:
      self._drop_track(t)
      return False
    redundant = []
    prev_out = False
    for f in kf:
      out = bool(t.keys[f][5] & OUTSIDE)
      if out and (f < vis[0] or f > vis[-1] or prev_out):
        redundant.append(f)
      prev_out = out
    for f in redundant:
      self._clear_frame(t, f)
    return True

  def _valid_pair(self, t, k0, k1):
    a, b = t.keys[k0], t.keys[k1]
    if k1 - k0 < 2 or a[5] & OUTSIDE or b[5] & OUTSIDE:
      return False
    return k1 - k0 <= self.settings['max_gap'] or bool(a[5] & FORCE_FILL)

  def _update_fill(self, t, lo, hi):
    """Recompute automatic boxes of every key segment touching frames lo..hi."""
    if not self._normalize(t):
      return
    kf = t.key_frames()
    i0 = bisect.bisect_left(kf, lo) - 1   # last key < lo
    i1 = bisect.bisect_right(kf, hi)      # first key > hi
    a = kf[i0] if i0 >= 0 else lo
    b = kf[i1] if i1 < len(kf) else hi
    self._clear_autos(t, min(a, lo), max(b, hi))
    for j in range(max(i0, 0), min(i1, len(kf) - 1)):
      if self._valid_pair(t, kf[j], kf[j + 1]):
        self._fill_pair(t, kf[j], kf[j + 1])

  def _fill_pair(self, t, k0, k1):
    b0, b1 = t.keys[k0][:4], t.keys[k1][:4]
    d = self.dets
    excluded = set()

    def candidates(f):
      a, b = d.frame_slice(f)
      for i in range(a, b):
        if not self.allowed[i] or i in excluded:
          continue
        o = self.owner[i]
        if o >= 0 and o != t.id and (self.owner_key[i] or strict):
          continue
        yield i, d.box[i], d.cls[i]

    strict = False
    for attempt in range(6):
      strict = attempt == 5
      path = fill_segment(k0, b0, k1, b1, candidates, t.cls)
      losers = [v[4] for f, v in path.items()
                if v[4] >= 0 and self.owner[v[4]] not in (-1, t.id) and not self._wins(t, f, v[4], k0, b0, k1, b1)]
      if not losers:
        break
      excluded.update(losers)
    for f, v in path.items():
      self._put_auto(t, f, v[:4], v[4])

  def _prior(self, t, f):
    a, b = t.prev_key(f, True), t.next_key(f, True)
    if a is None or b is None:
      bx = t.box(f)
      return bx[:4] if bx else None
    return lerp(t.keys[a][:4], t.keys[b][:4], (f - a) / (b - a))

  def _wins(self, t, f, det, k0, b0, k1, b1):
    """Should track t take det (currently auto-assigned to another track)?"""
    box = self.dets.box[det]
    mine = iou(box, lerp(b0, b1, (f - k0) / (k1 - k0)))
    other = self.tracks.get(int(self.owner[det]))
    theirs = 0.0
    if other is not None:
      p = self._prior(other, f)
      theirs = iou(box, p) if p is not None else 0.0
    return mine > theirs + 0.05

  # --------------------------------------------------------------- commands
  def _track(self, tid):
    t = self.tracks.get(int(tid)) if tid is not None else None
    if t is None:
      raise EditError(f'no track {tid}')
    return t

  def _det(self, det, f):
    det = int(det)
    if not 0 <= det < len(self.dets):
      raise EditError(f'no detection {det}')
    if self.dets.frame[det] != f:
      raise EditError(f'detection {det} is on frame {self.dets.frame[det]}, not {f}')
    return det

  def assign(self, f, det, tid=None):
    """Detection `det` (on frame f) becomes a key of track tid (new track if
    tid is None or does not exist). Returns the track id."""
    det = self._det(det, f)
    with self._edit('assign'):
      t = self.tracks.get(int(tid)) if tid is not None else None
      if t is None:
        t = self._new_track(self.dets.cls[det], tid)
      elif t.cls < 0:
        self._touch(t.id)
        t.cls = int(self.dets.cls[det])
      self._put_key(t, f, self.dets.box[det], det)
      self._schedule(t, f, f)
    return t.id

  def new_tracks(self, f, dets=None):
    """A new track for each detection (default: every free detection at f)."""
    if dets is None:
      dets = self.free_dets(f)[0].tolist()
    ids = []
    with self._edit('new tracks'):
      for det in dets:
        det = self._det(det, f)
        if self.owner[det] >= 0:
          continue
        t = self._new_track(self.dets.cls[det])
        self._put_key(t, f, self.dets.box[det], det)
        self._schedule(t, f, f)
        ids.append(t.id)
    return ids

  def set_box(self, f, box, tid=None, cls=None):
    """Hand drawn / edited box: a manual key."""
    box = [float(v) for v in box]
    if box[2] < 1 or box[3] < 1:
      raise EditError('box too small')
    with self._edit('set box'):
      t = self.tracks.get(int(tid)) if tid is not None else None
      if t is None:
        t = self._new_track(-1 if cls is None else cls, tid)
      self._put_key(t, f, box, -1, MANUAL)
      self._schedule(t, f, f)
    return t.id

  def delete_box(self, f, tid):
    """Remove the box at f. A key is removed (the frame is re-interpolated if it
    lies between two other keys); an automatic box becomes a one frame gap."""
    t = self._track(tid)
    with self._edit('delete box'):
      k = t.keys.get(f)
      if k is not None and not k[5] & OUTSIDE:
        self._clear_frame(t, f)
        self._schedule(t, f, f)
      elif f in t.auto:
        self._promote(t, f - 1)
        self._promote(t, f + 1)
        a = t.auto[f]
        self._put_key(t, f, a[:4], -1, OUTSIDE)
        self._schedule(t, f, f)

  def set_outside(self, f, tid):
    """The object is not visible from f until the track's next key."""
    t = self._track(tid)
    with self._edit('absent'):
      b = t.box(f)
      if b is None:
        b = t.keys.get(f) or (0, 0, 1, 1)
      self._promote(t, f - 1)
      self._put_key(t, f, b[:4], -1, OUTSIDE)
      self._schedule(t, f, f)

  def delete_track(self, tid, f=None):
    """Delete the whole track, or only its frames >= f ("the track ends here")."""
    t = self._track(tid)
    with self._edit('delete track'):
      if f is None:
        self._drop_track(t)
      else:
        self._take_from(t, f)
        self._normalize(t)

  def swap(self, f, a, b):
    """Swap the identities of tracks a and b from frame f on (ID switch fix)."""
    ta = self._track(a)
    with self._edit('swap'):
      tb = self.tracks.get(int(b))
      if tb is None:
        tb = self._new_track(ta.cls, b)
      ka, aa = self._take_from(ta, f)
      kb, ab = self._take_from(tb, f)
      self._give(ta, kb, ab)
      self._give(tb, ka, aa)
      for t in (ta, tb):
        if t.id in self.tracks:
          self._schedule(t, f - 1, f)
    return tb.id

  def merge(self, into, other):
    """Track `other` becomes part of track `into`. Where both have a box,
    `into` keeps its own unless it is only interpolated there."""
    ta, tb = self._track(into), self._track(other)
    if ta is tb:
      return ta.id
    with self._edit('merge'):
      keys = dict(tb.keys)
      self._drop_track(tb)
      for f, k in sorted(keys.items()):
        bx = ta.box(f)
        if f in ta.keys or (bx is not None and bx[5] != AUTO_INTERP):
          continue
        if k[4] >= 0 and self.owner[k[4]] >= 0:
          k = (k[0], k[1], k[2], k[3], -1, k[5] | MANUAL)
        self._put_key(ta, f, k[:4], k[4], k[5] & ~IMPORTED)
      if ta.cls < 0:
        ta.cls = tb.cls
      if keys:
        self._schedule(ta, min(keys), max(keys))
    return ta.id

  def split(self, f, tid):
    """Frames >= f of the track become a new track. Returns the new id."""
    t = self._track(tid)
    with self._edit('split'):
      keys, autos = self._take_from(t, f)
      if not keys and not autos:
        raise EditError(f'track {tid} has nothing at or after frame {f}')
      n = self._new_track(t.cls)
      self._give(n, keys, autos)
      for x in (t, n):
        if x.id in self.tracks:
          self._schedule(x, f - 1, f)
    return n.id

  def relabel(self, tid, new_id):
    """Change a track id; if new_id exists the two tracks are merged."""
    new_id = int(new_id)
    t = self._track(tid)
    if new_id == t.id:
      return new_id
    if new_id in self.tracks:
      return self.merge(new_id, t.id)
    with self._edit('relabel'):
      keys, autos = dict(t.keys), dict(t.auto)
      cls, attrs = t.cls, t.attrs
      self._drop_track(t)
      n = self._new_track(cls, new_id)
      n.attrs = dict(attrs)
      self._give(n, keys, autos)
    return new_id

  def set_class(self, tid, cls):
    t = self._track(tid)
    with self._edit('class'):
      self._touch(t.id)
      t.cls = int(cls)

  def fill_gaps(self, tid=None, max_len=None):
    """Remove 'absent' gaps (e.g. imported tracker gaps) so they get filled."""
    tracks = [self._track(tid)] if tid is not None else list(self.tracks.values())
    n = 0
    with self._edit('fill gaps'):
      for t in tracks:
        for f in [f for f in t.key_frames() if t.keys[f][5] & OUTSIDE]:
          p, q = t.prev_key(f, True), t.next_key(f, True)
          if p is None or q is None or (max_len is not None and q - p - 1 > max_len):
            continue
          self._clear_frame(t, f)
          if q - p > self.settings['max_gap']:
            k = t.keys[p]
            t.keys[p] = k[:5] + (k[5] | FORCE_FILL,)
          self._schedule(t, p, q)
          n += 1
    return n

  # ------------------------------------------------------------- proposals
  def proposals(self, f, lookback=None):
    """Where tracks that were alive shortly before f probably are at f.
    Returns (proposals, lost):
      proposals [{track, det, frame, box, pred, sure, iou, coast}]: frame is f, or
        an earlier frame when the detection at f is missing (coast)
      lost [{track, box, exited}]: no detection found near the prediction"""
    step = self.settings['step']
    lookback = lookback or max(2 * step, 25)
    starts = {}
    for t in self.tracks.values():
      if not self.track_visible(t) or t.has_box(f) or t.next_key(f) is not None:
        continue
      g, box = t.last_box_before(f)
      if g is None or f - g > lookback:
        continue
      starts[t.id] = (g, box, propose.velocity(t, g))
    res = propose.propagate(self, starts, f, max_coast=max(10, step))
    props, lost = [], []
    for tid, r in sorted(res.items()):
      if r['det'] >= 0:
        props.append(dict(track=tid, det=r['det'], frame=r['frame'], box=r['box'], pred=r['pred'], sure=r['sure'],
                          iou=round(r['iou'], 3), coast=r['coast']))
      else:
        lost.append(dict(track=tid, box=r['pred'], exited=propose.exited(r['pred'], self.width, self.height)))
    return props, lost

  def _accept_one(self, p):
    if self.owner[p['det']] >= 0 or p['track'] not in self.tracks:
      return False
    t = self.tracks[p['track']]
    self._put_key(t, p['frame'], self.dets.box[p['det']], p['det'])
    self._schedule(t, p['frame'], p['frame'])
    return True

  def accept(self, f, tids=None, only_sure=False):
    """Accept the proposals at f (all, or only for the given tracks)."""
    props, _ = self.proposals(f)
    n = 0
    with self._edit('accept'):
      for p in props:
        if (tids is not None and p['track'] not in tids) or (only_sure and not p['sure']):
          continue
        if self._accept_one(p):
          n += 1
    return n

  def advance(self, f, end=None, step=None, max_steps=1000):
    """Auto-advance: step keyframe by keyframe, accepting sure proposals, and
    stop where a human is needed (unsure / lost track, new object, end)."""
    step = int(step or self.settings['step'])
    end = self.n_frames - 1 if end is None else min(int(end), self.n_frames - 1)
    accepted, steps, reason = 0, 0, 'end'
    with self._edit('auto advance'):
      cur = f
      for _ in range(max_steps):
        nxt = min(cur + step, end)
        if nxt <= cur:
          reason = 'end'
          break
        alive = {t.id for t in self.tracks.values()
                 if t.runs() and t.end() is not None and cur - step < t.end() <= cur and t.next_key(cur) is None}
        props, lost = self.proposals(nxt)
        unsure = [p for p in props if not p['sure']]
        for p in props:
          if p['sure'] and self._accept_one(p):
            accepted += 1
        self._flush()
        lost_alive = [x for x in lost if x['track'] in alive and not x['exited']]
        new = propose.new_objects(self, cur, nxt, {p['det'] for p in props})
        steps += 1
        cur = nxt
        if unsure or lost_alive or new:
          reason = ', '.join(r for r, c in (('unsure proposal', unsure), ('track lost', lost_alive), ('new object', new)) if c)
          break
      else:
        reason = 'step limit'
    return dict(frame=cur, reason=reason, accepted=accepted, steps=steps)

  def bootstrap(self, a=0, b=None):
    """Create tracklets from the free detections with a simple IoU tracker."""
    b = self.n_frames - 1 if b is None else b
    chains = propose.bootstrap(self, a, b, iou_min=0.5, max_coast=max(3, self.settings['step'] // 2), min_len=3)
    ids = []
    with self._edit('bootstrap'):
      for chain in chains:
        cls = np.bincount(np.maximum(self.dets.cls[[d for _, d in chain]], 0)).argmax()
        t = self._new_track(int(cls))
        for f, d in chain:
          self._put_key(t, f, self.dets.box[d], d)
        self._schedule(t, chain[0][0], chain[-1][0])
        ids.append(t.id)
    return ids

  # ----------------------------------------------------------------- import
  def import_tracks(self):
    """Tracks from the id column of the input (tracker output). Gaps in a track
    are kept as gaps ('absent'); use fill_gaps to interpolate them."""
    d = self.dets
    order = np.lexsort((-d.conf, d.frame, d.src_id))
    with self._edit('import'):
      cur, prev_f = None, None
      for i in order:
        sid = int(d.src_id[i])
        if sid < 0:
          continue
        f = int(d.frame[i])
        if cur is None or cur.id != sid:
          cur = self.tracks.get(sid) or self._new_track(int(d.cls[i]), sid)
          prev_f = None
        if f == prev_f or not 0 <= f < self.n_frames:
          continue  # duplicate id in a frame: keep the most confident
        if prev_f is not None and f > prev_f + 1:
          pk = cur.keys[prev_f]
          cur.keys[prev_f + 1] = (pk[0], pk[1], pk[2], pk[3], -1, OUTSIDE)
        cur.keys[f] = (float(d.box[i, 0]), float(d.box[i, 1]), float(d.box[i, 2]), float(d.box[i, 3]), int(i), IMPORTED)
        self.owner[i] = sid
        self.owner_key[i] = True
        prev_f = f
      for t in self.tracks.values():
        t.changed()
    self._undo.clear()

  # ----------------------------------------------------------------- queries
  def frame_data(self, a, b):
    a, b = max(0, a), min(self.n_frames - 1, b)
    d = self.dets
    tracks = [t for t in self.tracks.values() if self.track_visible(t) and t.runs() and t.start() <= b and t.end() >= a]
    r = lambda v: round(float(v), 1)
    out = {}
    for f in range(a, b + 1):
      lo, hi = d.frame_slice(f)
      free = [[i, r(d.box[i, 0]), r(d.box[i, 1]), r(d.box[i, 2]), r(d.box[i, 3]), round(float(d.conf[i]), 3), int(d.cls[i])]
              for i in range(lo, hi) if self.allowed[i] and self.owner[i] < 0]
      tb = []
      for t in tracks:
        bx = t.box(f)
        if bx is not None:
          tb.append([t.id, r(bx[0]), r(bx[1]), r(bx[2]), r(bx[3]), bx[5], int(bx[4])])
      out[f] = dict(d=free, t=tb)
    return out

  def tracks_summary(self):
    out = []
    for t in sorted(self.tracks.values(), key=lambda t: (t.start() if t.start() is not None else 0, t.id)):
      if not t.runs():
        continue
      out.append(dict(id=t.id, cls=t.cls, runs=t.runs(), n=t.n_boxes(), keys=len(t.keys), visible=self.track_visible(t)))
    return out

  def track_detail(self, tid):
    t = self._track(tid)
    return dict(id=t.id, cls=t.cls, runs=t.runs(), n=t.n_boxes(),
                keys=[[f, int(t.keys[f][5])] for f in t.key_frames()],
                interp=[f for f, a in t.auto.items() if a[4] < 0])

  def coverage(self, a, b, bins=1000):
    """Free (unassigned, usable) detections per frame, max-pooled into bins."""
    a, b = max(0, a), min(self.n_frames - 1, b)
    m = self.allowed & (self.owner < 0) & (self.dets.frame >= a) & (self.dets.frame <= b)
    counts = np.bincount(self.dets.frame[m] - a, minlength=b - a + 1)[:b - a + 1]
    n = len(counts)
    bins = max(1, min(bins, n))
    edges = np.linspace(0, n, bins + 1).astype(int)
    pooled = [int(counts[edges[i]:max(edges[i + 1], edges[i] + 1)].max()) for i in range(bins)]
    return dict(start=a, end=b, bins=pooled)

  def issues(self, a, b, fps):
    return review.find_issues(self, a, b, fps)

  def stats(self, a=0, b=None):
    b = self.n_frames - 1 if b is None else b
    m = self.allowed & (self.dets.frame >= a) & (self.dets.frame <= b)
    total = int(m.sum())
    done = int((m & (self.owner >= 0)).sum())
    return dict(tracks=sum(1 for t in self.tracks.values() if self.track_visible(t)), dets=total, assigned=done,
                progress=round(done / total, 4) if total else None)

  def export_rows(self, a=0, b=None, interp=True):
    """(frame, track_id, box, det, kind, cls) sorted by frame then id."""
    b = self.n_frames - 1 if b is None else b
    rows = []
    for t in self.tracks.values():
      if not self.track_visible(t):
        continue
      for ra, rb in t.runs():
        for f in range(max(ra, a), min(rb, b) + 1):
          bx = t.box(f)
          if bx is None or (not interp and bx[5] == AUTO_INTERP):
            continue
          rows.append((f, t.id, bx[:4], bx[4], bx[5], t.cls))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows

  # ------------------------------------------------------------- persistence
  def to_json(self):
    return dict(version=1, task='tracking', next_id=self.next_id, det_signature=self.dets.signature,
                tracks=[t.to_json() for t in sorted(self.tracks.values(), key=lambda t: t.id)])

  def load_json(self, data):
    """Restore saved tracks. Detection links that do not fit the current
    detection file (changed file) are dropped, the boxes are kept."""
    warnings = []
    relink = data.get('det_signature') not in (None, self.dets.signature)
    if relink:
      warnings.append('the detection file changed since this annotation was saved; boxes are kept but '
                      'detection links were re-checked')
    self.tracks.clear()
    self.owner[:] = -1
    self.owner_key[:] = False
    bad = 0
    for td in data.get('tracks', []):
      t = Track.from_json(td)
      for store, is_key in ((t.keys, True), (t.auto, False)):
        for f, v in list(store.items()):
          det = v[4]
          if det < 0:
            continue
          ok = det < len(self.dets) and self.dets.frame[det] == f and self.owner[det] < 0
          if ok and relink:
            ok = iou(self.dets.box[det], v[:4]) > 0.99
          if ok:
            self.owner[det] = t.id
            self.owner_key[det] = is_key
          else:
            bad += 1
            store[f] = v[:4] + (-1,) + ((v[5] | MANUAL,) if is_key else ())
      t.changed()
      self.tracks[t.id] = t
    self.next_id = max([int(data.get('next_id', 1))] + [t + 1 for t in self.tracks])
    if bad:
      warnings.append(f'{bad} box(es) lost their detection link')
    self._undo.clear()
    self._redo.clear()
    self.rev += 1
    return warnings
