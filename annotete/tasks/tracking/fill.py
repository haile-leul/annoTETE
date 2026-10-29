"""Filling a track between two keyframes.

Linear interpolation between the keys gives a prior box per frame. Each frame
then picks either one detection near the prior or "no detection" (the box
stays interpolated). The choice is made jointly over the whole segment with a
Viterbi pass, so a smooth chain of detections wins over the detection that
merely sits closest to a (possibly curved) straight line, and a distractor
for one frame does not break the chain.

  cost(state)            = W_PRIOR * (1 - IoU(det, prior)) + class penalty   | MISS for no detection
  cost(prev -> current)  = W_TRANS * (1 - IoU(prev box, current box))
"""
from ...geometry import center_dist, iou, lerp

W_PRIOR = 0.5
W_TRANS = 1.0
MISS = 0.75
CLASS_PENALTY = 0.3
GATE = 1.0         # candidate if centre distance <= GATE * max(w, h) of the prior ...
SIZE_RATIO = 2.0   # ... and its area is within this factor of the prior


def gated(prior, box):
  if iou(prior, box) > 0.1:
    return True
  if center_dist(prior, box) > GATE * max(prior[2], prior[3]):
    return False
  ra = (box[2] * box[3]) / max(prior[2] * prior[3], 1e-9)
  return 1 / SIZE_RATIO ** 2 <= ra <= SIZE_RATIO ** 2


def fill_segment(t0, box0, t1, box1, candidates, track_cls=-1):
  """Boxes for frames t0+1 .. t1-1.

  candidates(t) -> iterable of (det, box, cls).
  Returns {t: (x, y, w, h, det)} with det = -1 for interpolated frames."""
  n = t1 - t0 - 1
  if n <= 0:
    return {}
  priors = [lerp(box0, box1, (k + 1) / (t1 - t0)) for k in range(n)]
  states = []  # per frame: [(det, box, cost)], index 0 = miss
  for k in range(n):
    p = priors[k]
    st = [(-1, p, MISS)]
    for det, b, cls in candidates(t0 + 1 + k):
      b = tuple(float(v) for v in b)
      if not gated(p, b):
        continue
      c = W_PRIOR * (1 - iou(b, p))
      if track_cls >= 0 and cls >= 0 and cls != track_cls:
        c += CLASS_PENALTY
      st.append((int(det), b, c))
    states.append(st)

  # forward pass
  cost = [W_TRANS * (1 - iou(box0, s[1])) + s[2] for s in states[0]]
  back = []
  for k in range(1, n):
    prev = states[k - 1]
    cur_cost, cur_back = [], []
    for s in states[k]:
      best, arg = None, 0
      for j, ps in enumerate(prev):
        c = cost[j] + W_TRANS * (1 - iou(ps[1], s[1]))
        if best is None or c < best:
          best, arg = c, j
      cur_cost.append(best + s[2])
      cur_back.append(arg)
    cost = cur_cost
    back.append(cur_back)
  final = [c + W_TRANS * (1 - iou(s[1], box1)) for c, s in zip(cost, states[-1])]
  j = min(range(len(final)), key=final.__getitem__)
  path = [j]
  for k in range(n - 2, -1, -1):
    j = back[k][j]
    path.append(j)
  path.reverse()

  chosen = [states[k][path[k]] for k in range(n)]
  # frames without a detection: interpolate between the nearest chosen boxes
  # (more accurate than the straight key-to-key line when the path curves)
  anchors = [(t0, box0)] + [(t0 + 1 + k, s[1]) for k, s in enumerate(chosen) if s[0] >= 0] + [(t1, box1)]
  out = {}
  ai = 0
  for k, s in enumerate(chosen):
    t = t0 + 1 + k
    if s[0] >= 0:
      out[t] = (*s[1], s[0])
      continue
    while anchors[ai + 1][0] < t:
      ai += 1
    (ta, ba), (tb, bb) = anchors[ai], anchors[ai + 1]
    out[t] = (*lerp(ba, bb, (t - ta) / (tb - ta)), -1)
  return out
