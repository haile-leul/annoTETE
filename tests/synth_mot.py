"""Tracker-like output with known mistakes, for tests."""
from synth import make_scene


def make_tracker_output(n_frames=150, n_obj=4, seed=0, switch_at=60, frag_at=80):
  """Lines in the txt layout with ids from a 'tracker' that
  - swaps the ids of objects 0 and 1 from `switch_at` on
  - loses object 2 for 3 frames at `frag_at` and gives it a new id (99) after."""
  gt, _ = make_scene(n_frames, n_obj, seed=seed, miss=0, distractors=0)
  lines = []
  for f in range(n_frames):
    for o in range(n_obj):
      tid = o + 1
      if f >= switch_at and o in (0, 1):
        tid = 2 - o  # 0 -> 2, 1 -> 1 ... swap ids 1 and 2
      if o == 2:
        if frag_at <= f < frag_at + 3:
          continue
        if f >= frag_at + 3:
          tid = 99
      x, y, w, h = gt[o][f]
      lines.append(f'{f} {tid} {x:.1f} {y:.1f} {w:.1f} {h:.1f} 90.0 5 0.8 -1 -9999 -9999 -9999 -9999')
  return gt, lines
