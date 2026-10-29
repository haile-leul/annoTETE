import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_scene  # noqa: E402
from synth_mot import make_tracker_output  # noqa: E402

from annotete.formats import load_boxes  # noqa: E402
from annotete.geometry import iou, iou_many  # noqa: E402
from annotete.tasks.tracking.annotation import EditError, TrackingAnnotation  # noqa: E402
from annotete.tasks.tracking.fill import fill_segment  # noqa: E402
from annotete.tasks.tracking.model import OUTSIDE  # noqa: E402


def write(tmp_path, lines, name='d.txt'):
  p = tmp_path / name
  p.write_text('\n'.join(lines) + '\n')
  return str(p)


def check_consistent(a):
  """Ownership index == what the tracks say; every det used at most once."""
  seen = {}
  for t in a.tracks.values():
    assert t.visible_keys(), f'track {t.id} without visible keys'
    for f, k in t.keys.items():
      if k[4] >= 0:
        assert not k[5] & OUTSIDE
        assert a.dets.frame[k[4]] == f
        assert k[4] not in seen, f'det {k[4]} used twice'
        seen[k[4]] = (t.id, True)
    for f, au in t.auto.items():
      assert f not in t.keys
      if au[4] >= 0:
        assert a.dets.frame[au[4]] == f
        assert au[4] not in seen, f'det {au[4]} used twice'
        seen[au[4]] = (t.id, False)
  owned = np.nonzero(a.owner >= 0)[0]
  assert set(owned.tolist()) == set(seen)
  for d, (tid, key) in seen.items():
    assert a.owner[d] == tid and bool(a.owner_key[d]) == key


def snapshot(a):
  r = lambda v: tuple(round(x, 2) if isinstance(x, float) else x for x in v)
  return {t.id: ({f: r(k) for f, k in t.keys.items()}, {f: r(b) for f, b in t.auto.items()}, t.cls) for t in a.tracks.values()}


@pytest.fixture
def scene(tmp_path):
  gt, lines = make_scene(200, 5, seed=1, miss=0.1)
  return gt, load_boxes(write(tmp_path, lines))


def test_fill_snaps_to_chain_through_distractor():
  # object moves on a curve; a distractor sits on the straight key-to-key line
  path = {t: (100 + 4 * t, 100 + 0.08 * (t - 10) ** 2, 40, 60) for t in range(21)}
  dets = {t: [(t, path[t])] for t in range(1, 20)}
  dets[10].append((100, (100 + 40, 100 + 8 * 10 - 2, 40, 60)))
  del dets[7]  # missed detection
  out = fill_segment(0, path[0], 20, path[20], lambda t: [(d, b, -1) for d, b in dets.get(t, [])])
  for t in range(1, 20):
    if t == 7:
      assert out[t][4] == -1 and iou(out[t][:4], path[7]) > 0.7
    else:
      assert out[t][4] == t, (t, out[t])


def test_assign_and_interpolate(scene):
  gt, d = scene
  a = TrackingAnnotation(d, 200, 640, 480)
  f0, f1 = 0, 20
  det0 = int(d.frame_slice(f0)[0])
  tid = a.assign(f0, det0, None)
  o = max(gt, key=lambda o: iou(gt[o][0], d.box[det0]))
  lo, hi = d.frame_slice(f1)
  det1 = lo + int(np.argmax(iou_many(gt[o][f1], d.box[lo:hi])))
  assert a.assign(f1, det1, tid) == tid
  t = a.tracks[tid]
  assert t.start() == 0 and t.end() == 20 and t.n_boxes() == 21
  for f in range(21):
    assert iou(t.box(f)[:4], gt[o][f]) > 0.6
  check_consistent(a)


def test_oracle_annotator_is_accurate(tmp_path):
  gt, lines = make_scene(300, 8, seed=2, miss=0.15)
  d = load_boxes(write(tmp_path, lines))
  a = TrackingAnnotation(d, 300, 640, 480, dict(step=20, min_conf=50))

  def gt_det(o, f):
    lo, hi = d.frame_slice(f)
    if lo == hi:
      return -1
    ious = iou_many(gt[o][f], d.box[lo:hi])
    j = int(ious.argmax())
    return lo + j if ious[j] > 0.6 else -1

  ids = a.new_tracks(0)
  t2o = {tid: max(gt, key=lambda o: iou(gt[o][0], a.tracks[tid].box(0)[:4])) for tid in ids}
  r = a.advance(0)
  clicks = 0
  while r['frame'] < 299:
    f = r['frame']
    for tid, o in t2o.items():
      want = gt_det(o, f)
      if want >= 0 and a.owner[want] != tid:
        props, _ = a.proposals(f)
        if not any(p['track'] == tid and p['det'] == want and p['frame'] == f for p in props):
          a.assign(f, want, tid)
          clicks += 1
    a.accept(f)
    r = a.advance(f)
  check_consistent(a)
  wrong = total = 0
  for tid, o in t2o.items():
    t = a.tracks[tid]
    for f in range(300):
      b = t.box(f)
      if b is not None:
        total += 1
        wrong += iou(gt[o][f], b[:4]) < 0.5
  assert total > 0.95 * 300 * len(ids)
  assert wrong / total < 0.01
  assert clicks < 60  # vs ~1500 boxes


def test_fuzz_commands_undo_redo(scene):
  _, d = scene
  a = TrackingAnnotation(d, 200, 640, 480, dict(step=10))
  rng = random.Random(0)
  history = [snapshot(a)]
  for i in range(400):
    f = rng.randrange(200)
    tids = list(a.tracks)
    lo, hi = d.frame_slice(f)
    op = rng.choice(['assign', 'assign', 'assign', 'new', 'box', 'del', 'out', 'end', 'swap', 'merge', 'split',
                     'relabel', 'accept', 'advance', 'gaps', 'undo', 'redo', 'dtrack'])
    try:
      if op == 'assign' and hi > lo:
        a.assign(f, rng.randrange(lo, hi), rng.choice(tids + [None]) if tids else None)
      elif op == 'new':
        a.new_tracks(f)
      elif op == 'box':
        a.set_box(f, [rng.uniform(0, 500), rng.uniform(0, 400), 30, 40], rng.choice(tids + [None]) if tids else None)
      elif op == 'del' and tids:
        a.delete_box(f, rng.choice(tids))
      elif op == 'out' and tids:
        a.set_outside(f, rng.choice(tids))
      elif op == 'end' and tids:
        a.delete_track(rng.choice(tids), f)
      elif op == 'dtrack' and tids:
        a.delete_track(rng.choice(tids))
      elif op == 'swap' and len(tids) > 1:
        x, y = rng.sample(tids, 2)
        a.swap(f, x, y)
      elif op == 'merge' and len(tids) > 1:
        x, y = rng.sample(tids, 2)
        a.merge(x, y)
      elif op == 'split' and tids:
        a.split(f, rng.choice(tids))
      elif op == 'relabel' and tids:
        a.relabel(rng.choice(tids), rng.randrange(1, 30))
      elif op == 'accept':
        a.accept(f)
      elif op == 'advance':
        a.advance(f, max_steps=5)
      elif op == 'gaps':
        a.fill_gaps()
      elif op == 'undo':
        a.undo()
      elif op == 'redo':
        a.redo()
    except EditError:
      pass
    check_consistent(a)
  # undo everything back to the start, then redo all
  states = []
  while a._undo:
    states.append(snapshot(a))
    a.undo()
    check_consistent(a)
  assert snapshot(a) == history[0]
  for s in reversed(states):
    a.redo()
    check_consistent(a)
    assert snapshot(a) == s


def test_failed_command_rolls_back(scene):
  _, d = scene
  a = TrackingAnnotation(d, 200, 640, 480)
  a.new_tracks(0)
  before = snapshot(a)
  lo, _ = d.frame_slice(5)
  with pytest.raises(EditError):
    a.assign(6, lo, 1)  # wrong frame for this detection
  assert snapshot(a) == before
  check_consistent(a)


def test_import_and_fix_tracker_output(tmp_path):
  gt, lines = make_tracker_output(150, 4, switch_at=60, frag_at=80)
  d = load_boxes(write(tmp_path, lines))
  assert d.ids_look_like_tracks()
  a = TrackingAnnotation(d, 150, 640, 480)
  a.import_tracks()
  check_consistent(a)
  assert sorted(a.tracks) == [1, 2, 3, 4, 99]
  kinds = {i['type'] for i in a.issues(0, 149, 25)}
  assert {'jump', 'fragment'} <= kinds
  a.swap(60, 1, 2)
  a.merge(3, 99)
  check_consistent(a)
  assert sorted(a.tracks) == [1, 2, 3, 4]
  assert not [i for i in a.issues(0, 149, 25) if i['type'] in ('jump', 'fragment')]
  for tid, o in ((1, 0), (2, 1), (3, 2)):
    t = a.tracks[tid]
    assert t.n_boxes() == 150
    assert all(iou(t.box(f)[:4], gt[o][f]) > 0.9 for f in range(150))
  # split + delete forward + outside
  n = a.split(100, 4)
  assert a.tracks[4].end() == 99 and a.tracks[n].start() == 100
  a.delete_track(n, 120)
  assert a.tracks[n].end() == 119
  a.set_outside(110, n)
  assert a.tracks[n].box(110) is None and a.tracks[n].box(109) is not None
  check_consistent(a)


def test_gaps_are_kept_on_import_and_fillable(tmp_path):
  lines = [f'{f} 7 {100 + f} 100 40 60 90 1 0.9' for f in range(30) if not 10 <= f < 15]
  d = load_boxes(write(tmp_path, lines))
  a = TrackingAnnotation(d, 30, 640, 480)
  a.import_tracks()
  t = a.tracks[7]
  assert t.runs() == [[0, 9], [15, 29]]
  assert a.fill_gaps(7) == 1
  assert t.runs() == [[0, 29]]
  assert t.box(12)[4] == -1 and abs(t.box(12)[0] - 112) < 1e-6


def test_save_load_roundtrip(scene):
  _, d = scene
  a = TrackingAnnotation(d, 200, 640, 480)
  a.new_tracks(0)
  a.advance(0)
  data = a.to_json()
  b = TrackingAnnotation(d, 200, 640, 480)
  assert b.load_json(data) == []
  assert snapshot(a) == snapshot(b)
  check_consistent(b)
  # a changed detection file: links are re-checked, boxes kept
  data['det_signature'] = 'other'
  bad = data['tracks'][0]['keys'][0]
  bad[1] += 50  # no longer matches its detection
  c = TrackingAnnotation(d, 200, 640, 480)
  w = c.load_json(data)
  assert w and c.tracks[data['tracks'][0]['id']].keys[bad[0]][4] == -1
  check_consistent(c)


def test_class_filter_and_conf(scene):
  _, d = scene
  a = TrackingAnnotation(d, 200, 640, 480, dict(classes=[5], min_conf=50))
  idx, _ = a.free_dets(3)
  assert all(d.cls[i] == 5 and d.conf[i] >= 50 for i in idx)
  ids = a.new_tracks(3)
  assert all(a.tracks[t].cls == 5 for t in ids)


def test_bootstrap_tracklets(scene):
  gt, d = scene
  a = TrackingAnnotation(d, 200, 640, 480, dict(min_conf=50))
  ids = a.bootstrap(0, 199)
  check_consistent(a)
  assert len(ids) >= 5
  for tid in ids:  # each tracklet follows one object
    t = a.tracks[tid]
    objs = {max(gt, key=lambda o: iou(gt[o][f], t.box(f)[:4])) for f in range(t.start(), t.end() + 1) if t.box(f)}
    assert len(objs) == 1


def test_step_stops_where_a_new_object_appears(tmp_path):
  lines = []
  for f in range(60):
    lines.append(f'{f} 0 {100 + 2 * f} 100 40 60 90 1 0.9')      # object A, from frame 0
    lines.append(f'{f} 0 {400 - 2 * f} 300 40 60 90 1 0.9')      # object B, from frame 0
    lines.append(f'{f} 0 500 50 30 30 90 1 0.9')                 # static box you chose to ignore
    if f >= 13:
      lines.append(f'{f} 0 {50 + f} 380 40 60 90 1 0.9')         # object C appears at 13
  lines.append('15 0 300 10 20 20 90 1 0.9')                    # one-frame flicker
  d = load_boxes(write(tmp_path, lines))
  a = TrackingAnnotation(d, 60, 640, 480, dict(step=10))
  lo, _ = d.frame_slice(0)
  a.new_tracks(0, [lo, lo + 1])
  assert a.step_target(0) == dict(frame=10, new=[], reason='step')
  a.accept(10)
  st = a.step_target(10)
  assert st['frame'] == 13 and st['reason'] == 'new object'
  assert d.box[st['new'][0]].tolist() == [63, 380, 40, 60]
  # auto-advance stops there too, with A and B already carried to 13
  assert len(a.new_tracks(13)) == 2  # C + the static box; A and B are proposed for tracks 1, 2
  a.accept(13)
  assert a.step_target(13)['frame'] == 23
  r = a.advance(13)
  assert r['frame'] == 59 and r['reason'] == 'end'
  check_consistent(a)


def test_advance_stops_at_first_appearance(tmp_path):
  lines = [f'{f} 0 {100 + 2 * f} 100 40 60 90 1 0.9' for f in range(40)]
  lines += [f'{f} 0 300 300 40 60 90 1 0.9' for f in range(27, 40)]
  d = load_boxes(write(tmp_path, lines))
  a = TrackingAnnotation(d, 40, 640, 480, dict(step=10))
  a.new_tracks(0)
  r = a.advance(0)
  assert r['frame'] == 27 and r['reason'] == 'new object'
  assert a.tracks[1].end() == 27  # the existing track was carried to the stop frame
