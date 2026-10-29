"""Synthetic data for tests: moving boxes + noisy detections, optional video."""
import os

import numpy as np


def make_scene(n_frames=120, n_obj=4, seed=0, miss=0.1, distractors=0.05, W=640, H=480):
  """Returns (gt {obj: {frame: box}}, det_lines [str]) in the default txt layout."""
  rng = np.random.default_rng(seed)
  gt = {}
  for o in range(n_obj):
    x, y = rng.uniform(50, W - 150), rng.uniform(50, H - 150)
    vx, vy = rng.uniform(-2, 2), rng.uniform(-1.5, 1.5)
    w, h = rng.uniform(40, 80), rng.uniform(60, 110)
    ax = rng.uniform(-0.02, 0.02)
    gt[o] = {}
    for f in range(n_frames):
      gt[o][f] = (x, y, w, h)
      vx += ax
      x = float(np.clip(x + vx, 0, W - w))
      y = float(np.clip(y + vy, 0, H - h))
  lines, did = [], 0
  for f in range(n_frames):
    for o in range(n_obj):
      if rng.random() < miss:
        continue
      x, y, w, h = gt[o][f]
      j = rng.normal(0, 1.5, 4)
      lines.append(f'{f} {did} {int(x + j[0])} {int(y + j[1])} {int(w + j[2])} {int(h + j[3])} '
                   f'{rng.uniform(60, 99):.4f} {5 if o % 2 == 0 else 1} {rng.uniform(0.3, 0.95):.3f} -1 -9999 -9999 -9999 -9999')
      did += 1
    if rng.random() < distractors:
      lines.append(f'{f} {did} {int(rng.uniform(0, W - 30))} {int(rng.uniform(0, H - 30))} 20 25 22.3 1 0.2 -1 -9999 -9999 -9999 -9999')
      did += 1
  return gt, lines


def write_video(path, gt, n_frames, W=640, H=480, fps=25):
  import cv2
  vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
  colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
  for f in range(n_frames):
    img = np.full((H, W, 3), 40, np.uint8)
    for o, boxes in gt.items():
      x, y, w, h = (int(v) for v in boxes[f])
      cv2.rectangle(img, (x, y), (x + w, y + h), colors[o % len(colors)], -1)
    cv2.putText(img, str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vw.write(img)
  vw.release()


def make_dataset(root, n_videos=2, n_frames=120, n_obj=4):
  os.makedirs(os.path.join(root, 'videos'), exist_ok=True)
  os.makedirs(os.path.join(root, 'dets'), exist_ok=True)
  out = []
  for v in range(n_videos):
    gt, lines = make_scene(n_frames, n_obj, seed=v)
    name = f'cam{v}'
    write_video(os.path.join(root, 'videos', name + '.mp4'), gt, n_frames)
    with open(os.path.join(root, 'dets', name + '.txt'), 'w') as f:
      f.write('\n'.join(lines) + '\n')
    out.append((name, gt))
  return out
