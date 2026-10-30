"""Optimal assignment (Hungarian / linear sum assignment).

Uses scipy when it is installed (identical results to TrackEval); otherwise a
numpy implementation of the same shortest augmenting path algorithm
(Crouse 2016, as in scipy) so evaluATE needs nothing beyond numpy.
"""
import numpy as np

try:
  from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover - depends on the machine
  _scipy_lsa = None


def linear_sum_assignment(cost):
  """(rows, cols) minimising cost[rows, cols].sum(); like scipy's."""
  cost = np.asarray(cost, dtype=np.float64)
  if cost.size == 0:
    return np.zeros(0, int), np.zeros(0, int)
  if _scipy_lsa is not None:
    return _scipy_lsa(cost)
  return _lsa(cost)


def _lsa(cost):
  transposed = cost.shape[0] > cost.shape[1]
  if transposed:
    cost = cost.T
  nr, nc = cost.shape
  if not np.all(np.isfinite(cost)):
    raise ValueError('cost matrix has non finite entries')
  u = np.zeros(nr)
  v = np.zeros(nc)
  col4row = np.full(nr, -1)
  row4col = np.full(nc, -1)
  for cur in range(nr):
    shortest = np.full(nc, np.inf)
    path = np.full(nc, -1)
    sr = np.zeros(nr, bool)
    sc = np.zeros(nc, bool)
    remaining = np.arange(nc)
    min_val = 0.0
    i = cur
    sink = -1
    while sink < 0:
      sr[i] = True
      r = min_val + cost[i, remaining] - u[i] - v[remaining]
      better = r < shortest[remaining]
      shortest[remaining[better]] = r[better]
      path[remaining[better]] = i
      sh = shortest[remaining]
      lowest = sh.min()
      ties = remaining[sh == lowest]
      free = ties[row4col[ties] < 0]
      j = free[-1] if len(free) else ties[0]
      min_val = lowest
      sc[j] = True
      remaining = remaining[remaining != j]
      if row4col[j] < 0:
        sink = j
      else:
        i = row4col[j]
    u[cur] += min_val
    others = sr.copy()
    others[cur] = False
    u[others] += min_val - shortest[col4row[others]]
    v[sc] -= min_val - shortest[sc]
    j = sink
    while True:
      i = path[j]
      row4col[j] = i
      col4row[i], j = j, col4row[i]
      if i == cur:
        break
  rows = np.arange(nr)
  if transposed:
    order = np.argsort(col4row)
    return col4row[order], rows[order]
  return rows, col4row
