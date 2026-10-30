"""Tracking metrics, re-implemented from TrackEval
(https://github.com/JonathonLuiten/TrackEval, trackeval/metrics/*.py) so the
numbers are the same as the reference code (verified in tests/test_evaluate.py).

Every metric works on one sequence `data` dict, as TrackEval's preprocessed
data:
  num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets
  gt_ids[t], tracker_ids[t]      int arrays, ids relabelled to 0..n-1
  similarity_scores[t]           IoU matrix (len(gt_ids[t]), len(tracker_ids[t]))
and `combine(results)` merges per sequence results like TrackEval's
combine_sequences.
"""
import numpy as np

from .lap import linear_sum_assignment

EPS = np.finfo('float').eps
ALPHAS = np.arange(0.05, 0.99, 0.05)


# ---------------------------------------------------------------------- HOTA
HOTA_INT = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
HOTA_ARR = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
HOTA_FLOAT = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']


def hota(data):
  n = len(ALPHAS)
  res = {f: np.zeros(n) for f in HOTA_ARR + HOTA_INT}
  for f in HOTA_FLOAT:
    res[f] = 0
  if data['num_tracker_dets'] == 0:
    res['HOTA_FN'] = data['num_gt_dets'] * np.ones(n)
    res['LocA'] = np.ones(n)
    res['LocA(0)'] = 1.0
    return res
  if data['num_gt_dets'] == 0:
    res['HOTA_FP'] = data['num_tracker_dets'] * np.ones(n)
    res['LocA'] = np.ones(n)
    res['LocA(0)'] = 1.0
    return res

  potential = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
  gt_id_count = np.zeros((data['num_gt_ids'], 1))
  tr_id_count = np.zeros((1, data['num_tracker_ids']))
  for gt_t, tr_t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']):
    denom = sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] - sim
    sim_iou = np.zeros_like(sim)
    mask = denom > 0 + EPS
    sim_iou[mask] = sim[mask] / denom[mask]
    potential[gt_t[:, np.newaxis], tr_t[np.newaxis, :]] += sim_iou
    gt_id_count[gt_t] += 1
    tr_id_count[0, tr_t] += 1

  global_score = potential / (gt_id_count + tr_id_count - potential)
  matches = [np.zeros_like(potential) for _ in ALPHAS]
  for gt_t, tr_t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']):
    if len(gt_t) == 0:
      res['HOTA_FP'] += len(tr_t)
      continue
    if len(tr_t) == 0:
      res['HOTA_FN'] += len(gt_t)
      continue
    score = global_score[gt_t[:, np.newaxis], tr_t[np.newaxis, :]] * sim
    rows, cols = linear_sum_assignment(-score)
    for a, alpha in enumerate(ALPHAS):
      ok = sim[rows, cols] >= alpha - EPS
      r, c = rows[ok], cols[ok]
      k = len(r)
      res['HOTA_TP'][a] += k
      res['HOTA_FN'][a] += len(gt_t) - k
      res['HOTA_FP'][a] += len(tr_t) - k
      if k:
        res['LocA'][a] += sum(sim[r, c])
        matches[a][gt_t[r], tr_t[c]] += 1

  for a in range(n):
    mc = matches[a]
    ass_a = mc / np.maximum(1, gt_id_count + tr_id_count - mc)
    res['AssA'][a] = np.sum(mc * ass_a) / np.maximum(1, res['HOTA_TP'][a])
    ass_re = mc / np.maximum(1, gt_id_count)
    res['AssRe'][a] = np.sum(mc * ass_re) / np.maximum(1, res['HOTA_TP'][a])
    ass_pr = mc / np.maximum(1, tr_id_count)
    res['AssPr'][a] = np.sum(mc * ass_pr) / np.maximum(1, res['HOTA_TP'][a])
  res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
  return _hota_final(res)


def _hota_final(res):
  res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
  res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
  res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
  res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
  res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])
  res['HOTA(0)'] = res['HOTA'][0]
  res['LocA(0)'] = res['LocA'][0]
  res['HOTALocA(0)'] = res['HOTA(0)'] * res['LocA(0)']
  return res


def hota_combine(all_res):
  res = {f: sum(r[f] for r in all_res) for f in HOTA_INT}
  for f in ('AssRe', 'AssPr', 'AssA'):
    res[f] = sum(r[f] * r['HOTA_TP'] for r in all_res) / np.maximum(1.0, res['HOTA_TP'])
  loca = sum(r['LocA'] * r['HOTA_TP'] for r in all_res)
  res['LocA'] = np.maximum(1e-10, loca) / np.maximum(1e-10, res['HOTA_TP'])
  return _hota_final(res)


# --------------------------------------------------------------------- CLEAR
CLEAR_INT = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'CLR_Frames']
CLEAR_FLOAT = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_F1', 'FP_per_frame',
               'MOTAL', 'MOTP_sum']


def clear_matches(data, threshold=0.5):
  """The CLEAR MOT matching, frame by frame (keeps the previous frame's
  pairs when possible, then maximises IoU). Yields (t, rows, cols) with
  rows / cols indices into gt_ids[t] / tracker_ids[t]."""
  prev_ts = np.nan * np.zeros(data['num_gt_ids'])
  for t, (gt_t, tr_t, sim) in enumerate(zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores'])):
    if len(gt_t) == 0 or len(tr_t) == 0:  # (as TrackEval: the previous pairs are kept)
      yield t, np.zeros(0, int), np.zeros(0, int)
      continue
    score = (tr_t[np.newaxis, :] == prev_ts[gt_t[:, np.newaxis]])
    score = 1000 * score + sim
    score[sim < threshold - EPS] = 0
    rows, cols = linear_sum_assignment(-score)
    ok = score[rows, cols] > 0 + EPS
    rows, cols = rows[ok], cols[ok]
    prev_ts[:] = np.nan
    prev_ts[gt_t[rows]] = tr_t[cols]
    yield t, rows, cols


def clear(data, threshold=0.5):
  res = {f: 0 for f in CLEAR_INT + CLEAR_FLOAT}
  if data['num_tracker_dets'] == 0:
    res['CLR_FN'] = data['num_gt_dets']
    res['ML'] = data['num_gt_ids']
    res['MLR'] = 1.0
    return res
  if data['num_gt_dets'] == 0:
    res['CLR_FP'] = data['num_tracker_dets']
    res['MLR'] = 1.0
    return res
  n = data['num_gt_ids']
  gt_id_count = np.zeros(n)
  gt_matched = np.zeros(n)
  gt_frag = np.zeros(n)
  prev_tr = np.nan * np.zeros(n)        # last tracker id each gt had (any time before)
  prev_ts = np.nan * np.zeros(n)        # tracker id in the previous timestep
  for t, rows, cols in clear_matches(data, threshold):
    gt_t, tr_t, sim = data['gt_ids'][t], data['tracker_ids'][t], data['similarity_scores'][t]
    if len(gt_t) == 0:
      res['CLR_FP'] += len(tr_t)
      continue  # (TrackEval does not reset the previous timestep here)
    if len(tr_t) == 0:
      res['CLR_FN'] += len(gt_t)
      gt_id_count[gt_t] += 1
      continue
    mg, mt = gt_t[rows], tr_t[cols]
    prev = prev_tr[mg]
    res['IDSW'] += np.sum(~np.isnan(prev) & (mt != prev))
    gt_id_count[gt_t] += 1
    gt_matched[mg] += 1
    not_prev = np.isnan(prev_ts)
    prev_tr[mg] = mt
    prev_ts[:] = np.nan
    prev_ts[mg] = mt
    gt_frag += not_prev & ~np.isnan(prev_ts)
    k = len(mg)
    res['CLR_TP'] += k
    res['CLR_FN'] += len(gt_t) - k
    res['CLR_FP'] += len(tr_t) - k
    if k:
      res['MOTP_sum'] += sum(sim[rows, cols])
  ratio = gt_matched[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
  res['MT'] = int(np.sum(ratio > 0.8))
  res['PT'] = int(np.sum(ratio >= 0.2)) - res['MT']
  res['ML'] = n - res['MT'] - res['PT']
  res['Frag'] = int(np.sum(gt_frag[gt_frag > 0] - 1))
  res['CLR_Frames'] = data['num_timesteps']
  res['IDSW'] = int(res['IDSW'])
  return _clear_final(res)


def _clear_final(res):
  n_ids = res['MT'] + res['ML'] + res['PT']
  res['MTR'] = res['MT'] / max(1.0, n_ids)
  res['MLR'] = res['ML'] / max(1.0, n_ids)
  res['PTR'] = res['PT'] / max(1.0, n_ids)
  res['CLR_Re'] = res['CLR_TP'] / max(1.0, res['CLR_TP'] + res['CLR_FN'])
  res['CLR_Pr'] = res['CLR_TP'] / max(1.0, res['CLR_TP'] + res['CLR_FP'])
  res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / max(1.0, res['CLR_TP'] + res['CLR_FN'])
  res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / max(1.0, res['CLR_TP'] + res['CLR_FN'])
  res['MOTP'] = res['MOTP_sum'] / max(1.0, res['CLR_TP'])
  res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / max(1.0, res['CLR_TP'] + res['CLR_FN'])
  res['CLR_F1'] = res['CLR_TP'] / max(1.0, res['CLR_TP'] + 0.5 * res['CLR_FN'] + 0.5 * res['CLR_FP'])
  res['FP_per_frame'] = res['CLR_FP'] / max(1.0, res['CLR_Frames'])
  log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
  res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - log_idsw) / max(1.0, res['CLR_TP'] + res['CLR_FN'])
  return res


def clear_combine(all_res):
  res = {f: sum(r[f] for r in all_res) for f in CLEAR_INT + ['MOTP_sum']}
  return _clear_final(res)


# ------------------------------------------------------------------ Identity
ID_INT = ['IDTP', 'IDFN', 'IDFP']
ID_FLOAT = ['IDF1', 'IDR', 'IDP']


def identity(data, threshold=0.5):
  res = {f: 0 for f in ID_INT + ID_FLOAT}
  if data['num_tracker_dets'] == 0:
    res['IDFN'] = data['num_gt_dets']
    return _id_final(res)
  if data['num_gt_dets'] == 0:
    res['IDFP'] = data['num_tracker_dets']
    return _id_final(res)
  ng, nt = data['num_gt_ids'], data['num_tracker_ids']
  potential = np.zeros((ng, nt))
  gt_id_count = np.zeros(ng)
  tr_id_count = np.zeros(nt)
  for gt_t, tr_t, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']):
    mg, mt = np.nonzero(sim >= threshold)
    potential[gt_t[mg], tr_t[mt]] += 1
    gt_id_count[gt_t] += 1
    tr_id_count[tr_t] += 1
  fp = np.zeros((ng + nt, ng + nt))
  fn = np.zeros((ng + nt, ng + nt))
  fp[ng:, :nt] = 1e10
  fn[:ng, nt:] = 1e10
  for g in range(ng):
    fn[g, :nt] = gt_id_count[g]
    fn[g, nt + g] = gt_id_count[g]
  for k in range(nt):
    fp[:ng, k] = tr_id_count[k]
    fp[k + ng, k] = tr_id_count[k]
  fn[:ng, :nt] -= potential
  fp[:ng, :nt] -= potential
  rows, cols = linear_sum_assignment(fn + fp)
  res['IDFN'] = int(fn[rows, cols].sum())
  res['IDFP'] = int(fp[rows, cols].sum())
  res['IDTP'] = int(gt_id_count.sum() - res['IDFN'])
  return _id_final(res)


def _id_final(res):
  res['IDR'] = res['IDTP'] / max(1.0, res['IDTP'] + res['IDFN'])
  res['IDP'] = res['IDTP'] / max(1.0, res['IDTP'] + res['IDFP'])
  res['IDF1'] = res['IDTP'] / max(1.0, res['IDTP'] + 0.5 * res['IDFP'] + 0.5 * res['IDFN'])
  return res


def identity_combine(all_res):
  return _id_final({f: sum(r[f] for r in all_res) for f in ID_INT})


# --------------------------------------------------------------------- Count
COUNT_INT = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs', 'Frames']


def count(data):
  return dict(Dets=data['num_tracker_dets'], GT_Dets=data['num_gt_dets'], IDs=data['num_tracker_ids'],
              GT_IDs=data['num_gt_ids'], Frames=data['num_timesteps'])


def count_combine(all_res):
  return {f: sum(r[f] for r in all_res) for f in COUNT_INT}


# ------------------------------------------------------------------- all four
def evaluate_sequence(data):
  out = {}
  out.update(hota(data))
  out.update(clear(data))
  out.update(identity(data))
  out.update(count(data))
  return out


def combine(seq_results):
  out = {}
  rs = list(seq_results)
  out.update(hota_combine(rs))
  out.update(clear_combine(rs))
  out.update(identity_combine(rs))
  out.update(count_combine(rs))
  return out


SUMMARY = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'IDF1', 'IDR', 'IDP', 'MOTA', 'MOTP',
           'IDSW', 'Frag', 'MT', 'PT', 'ML', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDTP', 'IDFN', 'IDFP', 'Dets', 'GT_Dets',
           'IDs', 'GT_IDs', 'Frames']


def summary(res):
  """JSON friendly: HOTA-family arrays averaged over the alphas (as TrackEval's
  summary), plus the per-alpha curves."""
  out = {}
  for k in SUMMARY:
    v = res[k]
    out[k] = float(np.mean(v)) if isinstance(v, np.ndarray) else (int(v) if isinstance(v, (int, np.integer)) else float(v))
  out['curves'] = {k: [round(float(x), 5) for x in res[k]] for k in ('HOTA', 'DetA', 'AssA')}
  return out
