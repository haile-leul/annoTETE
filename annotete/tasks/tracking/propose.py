"""Forward propagation of tracks over free detections.

Used for
  - proposals: when you step from one keyframe to the next, every track that
    was alive is followed through the frames in between (greedy IoU matching
    with a constant velocity prediction, all tracks at once so two tracks never
    take the same box) and the detection it lands on is proposed.
  - the bootstrap tracker that builds tracklets for a whole video.
"""
import numpy as np

from ...geometry import greedy_match, iou_matrix

PROPOSE_IOU = 0.3   # min IoU(prediction, detection) to continue a chain
SURE_IOU = 0.5      # a proposal is "sure" if it lands with this IoU ...
SURE_CHAIN = 0.4    # ... every link of the chain had at least this IoU ...
SURE_MARGIN = 0.25  # ... and the runner-up detection is this much worse


def velocity(track, f, span=5):
  """Per-frame velocity (dx, dy, dw, dh) of `track` around its box at f."""
  b = track.box(f)
  if b is None:
    return np.zeros(4)
  for k in range(span, 0, -1):
    a = track.box(f - k)
    if a is not None:
      return (np.array(b[:4]) - np.array(a[:4])) / k
  return np.zeros(4)


def propagate(ann, starts, f_end, max_coast, used=None):
  """starts: {tid: (frame, box, velocity)}, all frames < f_end.
  Returns {tid: result}. A result has det >= 0 when the chain found a
  detection at f_end, or at a later frame than it started from when it was
  only coasting at f_end (frame says where). Otherwise det = -1 (lost) and
  pred is the predicted box at f_end. If `used` is a dict it receives
  {frame: {det, ...}}: the detections the chains went through."""
  st = {}
  for tid, (f0, box, v) in starts.items():
    st[tid] = dict(f0=f0, f=f0, box=np.array(box[:4], np.float64), v=np.array(v, np.float64), min_iou=1.0,
                   lost=False, det=-1, iou=0.0, runner=0.0)
  if not st:
    return {}
  t0 = min(s['f'] for s in st.values()) + 1
  for t in range(t0, f_end + 1):
    act = [tid for tid, s in st.items() if s['f'] < t and not s['lost']]
    if not act:
      continue
    preds = np.array([_predict(st[tid], t) for tid in act])
    idx, boxes = ann.free_dets(t)
    if len(idx):
      m = iou_matrix(preds, boxes)
      for r, c in greedy_match(m, PROPOSE_IOU):
        s = st[act[r]]
        dt = t - s['f']
        nb = boxes[c]
        s['v'] = 0.5 * s['v'] + 0.5 * (nb - s['box']) / dt if np.any(s['v']) else (nb - s['box']) / dt
        s['box'] = nb.copy()
        s['f'] = t
        s['det'] = int(idx[c])
        s['iou'] = float(m[r, c])
        row = np.delete(m[r], c)
        s['runner'] = float(row.max()) if len(row) else 0.0
        s['min_iou'] = min(s['min_iou'], s['iou'])
        if used is not None:
          used.setdefault(t, set()).add(int(idx[c]))
    for tid in act:
      s = st[tid]
      if s['f'] < t and t - s['f'] > max_coast:
        s['lost'] = True
  res = {}
  for tid, s in st.items():
    if s['det'] >= 0 and not s['lost']:
      coast = s['f'] < f_end
      sure = s['iou'] >= SURE_IOU and s['min_iou'] >= SURE_CHAIN and s['runner'] < s['iou'] - SURE_MARGIN
      res[tid] = dict(det=s['det'], frame=s['f'], box=s['box'].tolist(), iou=s['iou'], sure=bool(sure),
                      coast=coast, pred=_predict(s, f_end).tolist())
    else:
      res[tid] = dict(det=-1, frame=f_end, box=None, iou=0.0, sure=False, coast=False, pred=_predict(s, f_end).tolist())
  return res


def _predict(s, t):
  b = s['box'] + s['v'] * (t - s['f'])
  b[2:] = np.maximum(b[2:], 1.0)
  return b


def exited(box, width, height, margin=0.5):
  """True if less than `margin` of the box is inside the image."""
  x, y, w, h = box
  ix = max(0.0, min(x + w, width) - max(x, 0.0))
  iy = max(0.0, min(y + h, height) - max(y, 0.0))
  return ix * iy < margin * w * h


def first_appearance(ann, cur, target, explained, min_len=3):
  """Earliest frame in (cur, target] where an object appears that no track
  explains: chains of free detections (not in `explained` {frame: {det}})
  that start after `cur` and last `min_len` frames or are still there at
  `target`. Detections already free at `cur` are not new (you saw them).
  Returns (frame, [first det of each chain starting there]) or (None, [])."""
  born = {}
  for chain in bootstrap(ann, cur, target, iou_min=0.3, max_coast=2, min_len=1, exclude=explained):
    f0 = chain[0][0]
    if f0 > cur and (len(chain) >= min_len or chain[-1][0] == target):
      born.setdefault(f0, []).append(chain[0][1])
  if not born:
    return None, []
  f = min(born)
  return f, born[f]


def bootstrap(ann, a, b, iou_min=0.5, max_coast=5, min_len=3, min_conf=None, exclude=None):
  """Simple IoU + velocity tracker over free detections in frames a..b
  (minus `exclude` {frame: {det}}).
  Returns a list of tracklets: [[(frame, det), ...], ...]. It prefers to cut a
  track (easy to merge later) over an uncertain link (an ID switch)."""
  active = []   # dicts: f, box, v, dets
  done = []
  for t in range(a, b + 1):
    idx, boxes = ann.free_dets(t, min_conf=min_conf)
    if exclude and exclude.get(t):
      keep = np.array([int(i) not in exclude[t] for i in idx], bool)
      idx, boxes = idx[keep], boxes[keep]
    used = set()
    if active and len(idx):
      preds = np.array([_predict(s, t) for s in active])
      m = iou_matrix(preds, boxes)
      for r, c in greedy_match(m, iou_min):
        s = active[r]
        dt = t - s['f']
        nb = boxes[c]
        s['v'] = 0.5 * s['v'] + 0.5 * (nb - s['box']) / dt if len(s['dets']) > 1 else (nb - s['box']) / dt
        s['box'] = nb.copy()
        s['f'] = t
        s['dets'].append((t, int(idx[c])))
        used.add(c)
    keep = []
    for s in active:
      (done if t - s['f'] > max_coast else keep).append(s)
    active = keep
    for c in range(len(idx)):
      if c not in used:
        active.append(dict(f=t, box=boxes[c].copy(), v=np.zeros(4), dets=[(t, int(idx[c]))]))
  done.extend(active)
  return [s['dets'] for s in done if len(s['dets']) >= min_len]
