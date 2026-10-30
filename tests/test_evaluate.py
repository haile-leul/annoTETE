import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth_eval import make_eval_data  # noqa: E402

from annotete.evaluate import lap, metrics  # noqa: E402
from annotete.evaluate.run import Evaluation  # noqa: E402


def test_assignment_is_optimal():
  rng = np.random.default_rng(0)
  for k in range(300):
    c = rng.random(rng.integers(1, 9, 2)) if k % 2 else rng.integers(0, 3, rng.integers(1, 9, 2)).astype(float)
    r, cc = lap._lsa(c)
    # brute force on the smaller side
    import itertools
    nr, nc = c.shape
    if nr <= nc:
      best = min(sum(c[i, p[i]] for i in range(nr)) for p in itertools.permutations(range(nc), nr))
    else:
      best = min(sum(c[p[j], j] for j in range(nc)) for p in itertools.permutations(range(nr), nc))
    assert abs(c[r, cc].sum() - best) < 1e-9 and len(r) == min(c.shape)


def _seq(gts, trs):
  """gts / trs: per frame lists of (id, box)."""
  from annotete.geometry import iou_matrix
  gid = sorted({i for f in gts for i, _ in f})
  tid = sorted({i for f in trs for i, _ in f})
  return dict(num_timesteps=len(gts), num_gt_ids=len(gid), num_tracker_ids=len(tid),
              num_gt_dets=sum(map(len, gts)), num_tracker_dets=sum(map(len, trs)),
              gt_ids=[np.array([gid.index(i) for i, _ in f], int) for f in gts],
              tracker_ids=[np.array([tid.index(i) for i, _ in f], int) for f in trs],
              similarity_scores=[iou_matrix([b for _, b in g], [b for _, b in t]).reshape(len(g), len(t))
                                 for g, t in zip(gts, trs)])


def test_metrics_known_cases():
  box = (0, 0, 10, 10)
  far = (100, 100, 10, 10)
  # perfect tracking
  r = metrics.evaluate_sequence(_seq([[(1, box)]] * 10, [[(7, box)]] * 10))
  assert np.allclose(r['HOTA'], 1) and r['MOTA'] == 1 and r['IDF1'] == 1 and r['IDSW'] == 0
  # one id switch half way
  r = metrics.evaluate_sequence(_seq([[(1, box)]] * 10, [[(7, box)]] * 5 + [[(8, box)]] * 5))
  assert r['IDSW'] == 1 and r['MOTA'] == pytest.approx(0.9) and r['IDF1'] == pytest.approx(0.5)
  assert np.mean(r['AssA']) == pytest.approx(0.5) and np.mean(r['DetA']) == pytest.approx(1)
  # a false track and a miss
  r = metrics.evaluate_sequence(_seq([[(1, box)]] * 4, [[(7, far)]] * 4))
  assert r['CLR_FP'] == 4 and r['CLR_FN'] == 4 and r['MOTA'] == -1 and r['ML'] == 1


def test_matches_trackeval_if_available():
  """Exact parity with the original TrackEval code (run when it is installed)."""
  if not hasattr(np, 'float'):
    np.float, np.int = float, int  # TrackEval uses the removed numpy aliases
  tm = pytest.importorskip('trackeval.metrics')
  rng = np.random.default_rng(3)
  for _ in range(5):
    gts, trs = [], []
    for t in range(40):
      g = [(o, (10 * o + t, 5 * o, 20, 30)) for o in range(4) if rng.random() > 0.1]
      k = [((o + (t > 20 and o < 2)) % 9, tuple(np.array(b) + rng.normal(0, 2, 4))) for o, b in g if rng.random() > 0.1]
      gts.append(g)
      trs.append(k)
    data = _seq(gts, trs)
    mine = metrics.evaluate_sequence(data)
    for m in (tm.HOTA(), tm.CLEAR({'PRINT_CONFIG': False}), tm.Identity({'PRINT_CONFIG': False})):
      for k, v in m.eval_sequence(data).items():
        assert np.allclose(np.asarray(v, float), np.asarray(mine[k], float)), k


def test_evaluation_end_to_end(tmp_path):
  exp, a, b = make_eval_data(str(tmp_path))
  ev = Evaluation(exp, [('new', b), ('old', a)], log=lambda m: None)  # b has no swap: the improvement
  s = ev.run()
  new, old = s['results']
  # frames: the trackers ran on the full video (offset 10); found automatically
  assert 'video frame + 10' in new['seqs']['cam']['align']
  assert old['combined']['all']['IDSW'] == 3 and new['combined']['all']['IDSW'] == 1
  assert new['combined']['all']['HOTA'] > old['combined']['all']['HOTA']
  # detector-visible only: the 10 interpolated boxes the trackers miss do not count
  assert new['combined']['all']['CLR_FN'] == 10 and new['combined']['det']['CLR_FN'] == 0
  d = new['combined']['detector']
  assert (d['gt'], d['gt_det'], d['gt_interp'], d['fn_nodet'], d['fn_det']) == (360, 350, 10, 10, 0)
  ev_old = {e['type']: e for e in ev.detail('cam', 1)['events']}
  assert set(ev_old) == {'swap', 'new_id', 'lost', 'false'}
  assert ev_old['swap']['frame'] == 60 and ev_old['swap']['diff'] == 'fixed'
  assert ev_old['lost']['frame'] == 40 and ev_old['lost']['det_frames'] == 0
  assert ev_old['false']['frame'] == 80 and ev_old['false']['end'] == 99
  assert {e['type'] for e in ev.detail('cam', 0)['events']} == {'new_id', 'lost', 'false'}
  assert new['seqs']['cam']['fixed'] == 1 and new['seqs']['cam']['new'] == 0
  out = ev.save_report()
  assert os.path.exists(os.path.join(out, 'new_vs_old.csv'))
