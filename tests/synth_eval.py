"""An annoTETE export + tracker outputs with known mistakes, for evaluATE tests."""
import json
import os

import numpy as np


def make_eval_data(root, n_frames=120, offset=10):
  """Export folder with one trimmed video 'cam' (frames offset.. of a longer
  source), GT for 3 objects; object 2 is interpolated (no detection) in
  frames 40-49. Two tracker outputs on the *full* video (frames shifted):
    A: ids of objects 0 and 1 swap at 60, object 2 is dropped at 40-49 and
       comes back with a new id, a ghost track at 80-99.
    B: same but without the swap."""
  rng = np.random.default_rng(0)
  gt = {}
  for o in range(3):
    x, y = 60 + 150 * o, 80 + 60 * o
    gt[o] = {f: (x + 2.0 * f, y + 0.5 * f, 50.0, 80.0) for f in range(n_frames)}
  exp = os.path.join(root, 'export')
  for d in ('annotations_mot', 'annotations_json'):
    os.makedirs(os.path.join(exp, d), exist_ok=True)
  mot, tracks = [], {}
  for o in range(3):
    for f in range(n_frames):
      b = gt[o][f]
      kind = 'interp' if o == 2 and 40 <= f < 50 else 'key' if f % 10 == 0 else 'auto'
      mot.append(f'{f + 1},{o + 1},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},1,5,1')
      tracks.setdefault(o + 1, []).append([f, *b, -1 if kind == 'interp' else 7, kind])
  open(os.path.join(exp, 'annotations_mot', 'cam.txt'), 'w').write('\n'.join(mot) + '\n')
  json.dump(dict(item='cam', source='', frame_offset=offset, n_frames=n_frames, width=640, height=480, fps=25,
                 tracks=[dict(id=k, cls=5, boxes=v) for k, v in tracks.items()]),
            open(os.path.join(exp, 'annotations_json', 'cam.json'), 'w'))

  def tracker(name, swap):
    lines = []
    for f in range(n_frames):
      for o in range(3):
        tid = o + 1
        if swap and f >= 60 and o in (0, 1):
          tid = 2 - o
        if o == 2 and 40 <= f < 50:
          continue  # detector had nothing: tracker drops it
        if o == 2 and f >= 50:
          tid = 9
        x, y, w, h = (v + rng.normal(0, 1.0) for v in gt[o][f])
        lines.append(f'{f + offset} {tid} {x:.1f} {y:.1f} {w:.1f} {h:.1f} 0.9 5 0.8')
      if 80 <= f < 100:
        lines.append(f'{f + offset} 50 500 400 40 40 0.5 5 0.8')
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'cam_tracks.txt'), 'w').write('\n'.join(lines) + '\n')
    return d

  return exp, tracker('trackA', True), tracker('trackB', False)
