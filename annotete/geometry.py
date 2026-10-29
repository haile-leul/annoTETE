"""Box helpers. Boxes are (x, y, w, h) with (x, y) the top-left corner."""
import numpy as np


def iou(a, b):
  ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
  iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
  if ix <= 0 or iy <= 0:
    return 0.0
  inter = ix * iy
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def iou_many(a, boxes):
  """IoU of box `a` with every row of `boxes` (N, 4)."""
  boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
  ix = np.minimum(a[0] + a[2], boxes[:, 0] + boxes[:, 2]) - np.maximum(a[0], boxes[:, 0])
  iy = np.minimum(a[1] + a[3], boxes[:, 1] + boxes[:, 3]) - np.maximum(a[1], boxes[:, 1])
  inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
  return inter / (a[2] * a[3] + boxes[:, 2] * boxes[:, 3] - inter + 1e-9)


def iou_matrix(A, B):
  A = np.asarray(A, dtype=np.float64).reshape(-1, 4)
  B = np.asarray(B, dtype=np.float64).reshape(-1, 4)
  if not len(A) or not len(B):
    return np.zeros((len(A), len(B)))
  ix = np.minimum(A[:, None, 0] + A[:, None, 2], B[None, :, 0] + B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0])
  iy = np.minimum(A[:, None, 1] + A[:, None, 3], B[None, :, 1] + B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1])
  inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
  area_a = A[:, 2] * A[:, 3]
  area_b = B[:, 2] * B[:, 3]
  return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def lerp(a, b, t):
  return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def center_dist(a, b):
  return ((a[0] + a[2] / 2 - b[0] - b[2] / 2) ** 2 + (a[1] + a[3] / 2 - b[1] - b[3] / 2) ** 2) ** 0.5


def greedy_match(score, min_score):
  """Greedy one-to-one matching on a score matrix (higher is better).
  Returns [(row, col), ...] with score >= min_score."""
  pairs = []
  if score.size == 0:
    return pairs
  order = np.argsort(-score, axis=None)
  used_r, used_c = set(), set()
  for flat in order:
    r, c = divmod(int(flat), score.shape[1])
    if score[r, c] < min_score:
      break
    if r in used_r or c in used_c:
      continue
    used_r.add(r)
    used_c.add(c)
    pairs.append((r, c))
  return pairs
