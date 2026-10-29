"""Quality checks that point you at the frames most likely to need a fix."""
import numpy as np

from ...geometry import center_dist, iou
from .propose import bootstrap, velocity


def find_issues(ann, a, b, fps=25.0, max_items=300):
  out = []
  tracks = [t for t in ann.tracks.values() if ann.track_visible(t) and t.runs()]
  out += _fragments(ann, tracks, gap=int(max(2 * fps, ann.settings['step'] * 3)))
  out += _jumps(tracks)
  out += _short(tracks, min_len=max(3, int(fps // 5)))
  out += _long_interp(tracks, min_len=max(10, int(fps)))
  out += _unassigned(ann, a, b, min_len=max(3, int(fps // 5)))
  out = [i for i in out if a <= i['frame'] <= b]
  order = {'jump': 0, 'fragment': 1, 'unassigned': 2, 'interp': 3, 'short': 4}
  out.sort(key=lambda i: (order[i['type']], i['frame']))
  return out[:max_items]


def _fragments(ann, tracks, gap):
  """Track A ends shortly before track B starts close to where A was heading."""
  ends = [(t.end(), t) for t in tracks]
  starts = sorted(((t.start(), t) for t in tracks), key=lambda x: x[0])
  out = []
  for e, ta in ends:
    eb = ta.box(e)
    v = velocity(ta, e)
    best = None
    for s, tb in starts:
      if tb.id == ta.id or s <= e:
        continue
      if s - e > gap:
        break
      if ta.cls >= 0 and tb.cls >= 0 and ta.cls != tb.cls:
        continue
      sb = tb.box(s)
      pred = np.array(eb[:4]) + v * (s - e)
      pred[2:] = np.maximum(pred[2:], 1)
      d = center_dist(pred, sb) / max(sb[2], sb[3], 1)
      score = max(iou(pred, sb), iou(eb, sb))
      if score > 0.1 or d < 1.0:
        cand = (d - score, tb, s)
        if best is None or cand[0] < best[0]:
          best = cand
    if best:
      _, tb, s = best
      out.append(dict(type='fragment', frame=s, tracks=[ta.id, tb.id],
                      msg=f'#{tb.id} starts {s - e} frame(s) after #{ta.id} ends, nearby: same object?'))
  return out


def _jumps(tracks):
  """Sudden box jumps inside a track: usually an ID switch."""
  out = []
  for t in tracks:
    for a, b in t.runs():
      prev = t.box(a)
      for f in range(a + 1, b + 1):
        cur = t.box(f)
        if iou(prev, cur) < 0.1 and center_dist(prev, cur) > 0.7 * max(cur[2], cur[3], prev[2], prev[3]):
          out.append(dict(type='jump', frame=f, tracks=[t.id], msg=f'#{t.id} jumps between frame {f - 1} and {f}'))
        prev = cur
  return out


def _short(tracks, min_len):
  return [dict(type='short', frame=t.start(), tracks=[t.id], msg=f'#{t.id} is only {t.n_boxes()} frame(s) long')
          for t in tracks if t.n_boxes() < min_len]


def _long_interp(tracks, min_len):
  """Long runs without any detection support (occlusion? wrong path?)."""
  out = []
  for t in tracks:
    runs = []  # [first, last] of consecutive interpolated frames
    for f in sorted(t.auto):
      if t.auto[f][4] >= 0:
        continue
      if runs and runs[-1][1] == f - 1:
        runs[-1][1] = f
      else:
        runs.append([f, f])
    for f0, f1 in runs:
      if f1 - f0 + 1 >= min_len:
        out.append(dict(type='interp', frame=f0, tracks=[t.id],
                        msg=f'#{t.id}: {f1 - f0 + 1} interpolated frames with no detection'))
  return out


def _unassigned(ann, a, b, min_len):
  """Free detections that persist over several frames: a missed object."""
  out = []
  for chain in bootstrap(ann, a, b, iou_min=0.4, max_coast=2, min_len=min_len):
    f0, f1 = chain[0][0], chain[-1][0]
    out.append(dict(type='unassigned', frame=f0, tracks=[], det=chain[0][1],
                    msg=f'unassigned object for {len(chain)} frame(s) ({f0}-{f1})'))
  return out
