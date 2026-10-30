"""Ground truth (an annoTETE export folder) and tracker outputs -> per video
sequences ready for the metrics.

Ground truth: <export>/annotations_mot/<video>.txt (MOTChallenge gt). If
<export>/annotations_json/<video>.json is there too, it tells which boxes had
a detection behind them (kind key / auto) and which did not (interp /
manual): that enables the "detector-visible only" evaluation. The JSON alone
is also enough.

Tracker output: a file per video, named like the video (cam1.txt,
cam1_tracks.txt, cam1/...), any column layout annoTETE reads (space / comma,
0- / 1-based). Its frames are aligned to the ground truth automatically: the
tracker may have run on the exported (trimmed) clip or on the full video.
"""
import json
import os

import numpy as np

from ..formats import load_boxes
from ..geometry import greedy_match, iou_matrix
from ..project import pair_boxes

DET_KINDS = ('key', 'auto')        # ground truth boxes the detector had
NODET_KINDS = ('interp', 'manual')  # ground truth boxes without a detection
KIND_CODE = {'key': 0, 'manual': 1, 'auto': 2, 'interp': 3}
VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm')


class Sequence:
  """One video: per frame ground truth and tracker boxes (original ids)."""

  def __init__(self, name, n_frames):
    self.name = name
    self.n = n_frames
    self.gt = [[] for _ in range(n_frames)]   # (id, box, kind or None, cls)
    self.trk = [[] for _ in range(n_frames)]  # (id, box)
    self.fps = 25.0
    self.width = self.height = None
    self.video = None
    self.video_offset = 0
    self.frame_offset = 0
    self.has_kinds = False
    self.shift = 0
    self.align_note = ''

  def gt_classes(self):
    return {c for frame in self.gt for _, _, _, c in frame}


# --------------------------------------------------------------- ground truth
def list_sequences(export_dir):
  names = set()
  for sub, ext in (('annotations_mot', '.txt'), ('annotations_json', '.json')):
    d = os.path.join(export_dir, sub)
    if os.path.isdir(d):
      names |= {n[:-len(ext)] for n in os.listdir(d) if n.endswith(ext)}
  return sorted(names)


def load_ground_truth(export_dir, name):
  mot = os.path.join(export_dir, 'annotations_mot', name + '.txt')
  js = os.path.join(export_dir, 'annotations_json', name + '.json')
  meta, kinds, json_rows = {}, {}, []
  if os.path.exists(js):
    with open(js) as f:
      meta = json.load(f)
    for t in meta.get('tracks', []):
      for fr, x, y, w, h, det, kind in t['boxes']:
        kinds[(int(fr), int(t['id']))] = kind
        json_rows.append((int(fr), int(t['id']), (x, y, w, h), kind, int(t.get('cls', -1))))
  rows = []
  if os.path.exists(mot):
    for line in open(mot):
      v = line.strip().replace(',', ' ').split()
      if len(v) < 6:
        continue
      fr, tid = int(float(v[0])) - 1, int(float(v[1]))
      box = tuple(float(x) for x in v[2:6])
      cls = int(float(v[7])) if len(v) > 7 else -1
      rows.append((fr, tid, box, kinds.get((fr, tid)), cls))
  else:
    rows = json_rows
  if not rows and not meta:
    raise FileNotFoundError(f'no ground truth for {name} in {export_dir}')
  n = int(meta.get('n_frames') or 0) or (max(r[0] for r in rows) + 1 if rows else 0)
  video = _find_video(export_dir, name)
  if video is None and meta.get('source') and os.path.exists(meta['source']):
    video = meta['source']
  seq = Sequence(name, n)
  seq.frame_offset = int(meta.get('frame_offset') or 0)
  seq.video = video
  # an exported (trimmed) clip starts at the range start; the source video does not
  seq.video_offset = 0 if video and os.path.dirname(video) == os.path.join(export_dir, 'videos') else seq.frame_offset
  seq.fps = float(meta.get('fps') or 25.0)
  seq.width, seq.height = meta.get('width'), meta.get('height')
  seq.has_kinds = bool(kinds)
  for fr, tid, box, kind, cls in rows:
    if 0 <= fr < n:
      seq.gt[fr].append((tid, box, kind, cls))
  return seq


def _find_video(export_dir, name):
  d = os.path.join(export_dir, 'videos')
  for ext in VIDEO_EXTS:
    p = os.path.join(d, name + ext)
    if os.path.exists(p):
      return p
  return None


# -------------------------------------------------------------------- tracker
def pair_tracker_files(tracker_path, names):
  """{sequence name: tracker file}"""
  fake = [os.path.join('/', n + '.mp4') for n in names]
  if os.path.isfile(tracker_path) and len(names) != 1:
    raise ValueError('a single tracker file needs a single video; give a folder of tracker files')
  pairs, _ = pair_boxes(fake, tracker_path)
  return {n: pairs[m] for n, m in zip(names, fake) if pairs[m]}


def load_tracker(seq, path, shift='auto', classes='gt'):
  """Put the tracker boxes of `path` into seq.trk. Returns warnings."""
  warnings = []
  d = load_boxes(path, 'auto')
  keep = np.ones(len(d), bool)
  gt_cls = {c for c in seq.gt_classes() if c >= 0}
  if classes == 'gt' and gt_cls and (d.cls >= 0).any():
    keep = np.isin(d.cls, sorted(gt_cls))
    if not keep.any():
      warnings.append(f'{seq.name}: no tracker box has an annotated class {sorted(gt_cls)}; all classes used')
      keep[:] = True
  idx = np.nonzero(keep)[0]
  frames, boxes, ids = d.frame[idx], d.box[idx], d.src_id[idx]
  if shift == 'auto':
    shift, note = _best_shift(seq, frames, boxes, d.layout.frame_base)
    seq.align_note = note
  else:
    seq.align_note = f'tracker frame = video frame + {int(shift)} (given)'
  seq.shift = int(shift)
  out = 0
  for f, b, i in zip(frames - seq.shift, boxes, ids):
    if 0 <= f < seq.n:
      seq.trk[int(f)].append((int(i), tuple(float(v) for v in b)))
    else:
      out += 1
  if out:
    warnings.append(f'{seq.name}: {out} tracker box(es) are outside the evaluated frames and were ignored')
  return warnings


def _best_shift(seq, frames, boxes, base):
  """Try the plausible frame alignments and keep the one where the tracker
  boxes fit the ground truth best."""
  fo = seq.frame_offset
  cands = []
  for s in (0, -1, 1, fo, fo - 1, fo + 1):
    if s not in cands:
      cands.append(s)
  order = np.argsort(frames, kind='stable')
  frames, boxes = frames[order], boxes[order]
  gt_frames = [t for t in range(seq.n) if seq.gt[t]]
  sample = gt_frames[::max(1, len(gt_frames) // 150)]
  scores = []
  for s in cands:
    hits = 0
    for t in sample:
      a, b = np.searchsorted(frames, t + s, 'left'), np.searchsorted(frames, t + s, 'right')
      if a == b:
        continue
      m = iou_matrix([g[1] for g in seq.gt[t]], boxes[a:b])
      hits += len(greedy_match(m, 0.5))
    scores.append(hits)
  best = int(np.argmax(scores))
  s = cands[best]
  where = 'the full video' if fo and s in (fo, fo - 1, fo + 1) else 'the exported clip'
  note = f'tracker frame = video frame + {s} (auto: {scores[best]} of the sampled boxes line up; tracker ran on {where})'
  if scores[best] == 0:
    note = 'no tracker box lines up with the ground truth in any alignment: wrong files or classes?'
  return s, note


# ------------------------------------------------------- data for the metrics
def sequence_data(seq, variant='all', match_iou=0.5):
  """TrackEval style data dict for `seq`. variant 'det': ground truth boxes
  without a detection behind them are ignore regions (tracker boxes on them
  are neither TP nor FP, like TrackEval's distractor handling)."""
  gt_ids, tr_ids, sims, gt_keep_all, tr_keep_all = [], [], [], [], []
  for t in range(seq.n):
    g, k = seq.gt[t], seq.trk[t]
    sim = iou_matrix([x[1] for x in g], [x[1] for x in k]).reshape(len(g), len(k))
    gkeep = np.ones(len(g), bool)
    tkeep = np.ones(len(k), bool)
    if variant == 'det':
      ign = np.array([x[2] in NODET_KINDS for x in g], bool)
      if ign.any() and len(k):
        from .lap import linear_sum_assignment
        ms = sim.copy()
        ms[ms < match_iou - np.finfo('float').eps] = 0
        r, c = linear_sum_assignment(-ms)
        ok = ms[r, c] > np.finfo('float').eps
        r, c = r[ok], c[ok]
        tkeep[c[ign[r]]] = False
      gkeep = ~ign
    gt_ids.append(np.array([x[0] for x, m in zip(g, gkeep) if m], int))
    tr_ids.append(np.array([x[0] for x, m in zip(k, tkeep) if m], int))
    sims.append(sim[gkeep][:, tkeep])
    gt_keep_all.append(gkeep)
    tr_keep_all.append(tkeep)
  ug = np.unique(np.concatenate(gt_ids)) if gt_ids else np.zeros(0, int)
  ut = np.unique(np.concatenate(tr_ids)) if tr_ids else np.zeros(0, int)
  gmap = {int(v): i for i, v in enumerate(ug)}
  tmap = {int(v): i for i, v in enumerate(ut)}
  data = dict(num_timesteps=seq.n, num_gt_ids=len(ug), num_tracker_ids=len(ut),
              num_gt_dets=int(sum(len(x) for x in gt_ids)), num_tracker_dets=int(sum(len(x) for x in tr_ids)),
              gt_ids=[np.array([gmap[int(v)] for v in x], int) for x in gt_ids],
              tracker_ids=[np.array([tmap[int(v)] for v in x], int) for x in tr_ids],
              similarity_scores=sims)
  data['_gt_orig'], data['_tr_orig'] = ug, ut
  data['_gt_keep'], data['_tr_keep'] = gt_keep_all, tr_keep_all
  return data
