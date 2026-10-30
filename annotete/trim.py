"""Frame exact trimming of videos / image folders and box files."""
import os
import shutil
import subprocess

import cv2

from .formats import canonical_line


def has_ffmpeg():
  return shutil.which('ffmpeg') is not None


def trim_media(src, dst_dir, name, a, b, kind='video'):
  """Frames a..b (inclusive) of `src` into dst_dir. Returns (path, warning|None)."""
  os.makedirs(dst_dir, exist_ok=True)
  if kind == 'images':
    from .sources import ImageFolder
    seq = ImageFolder(src)
    out = os.path.join(dst_dir, name)
    os.makedirs(out, exist_ok=True)
    for k, i in enumerate(range(a, b + 1)):
      ext = os.path.splitext(seq.files[i])[1]
      shutil.copy2(os.path.join(src, seq.files[i]), os.path.join(out, f'{k:06d}{ext}'))
    return out, None
  dst = os.path.join(dst_dir, name + '.mp4')
  n = b - a + 1
  if has_ffmpeg():
    err = _ffmpeg_trim(src, dst, a, b)
    if err is None:
      got = _count(dst)
      if got == n:
        return dst, None
      warn = f'{name}: ffmpeg wrote {got} frames instead of {n}; re-encoding with OpenCV'
    else:
      warn = f'{name}: ffmpeg failed ({err.strip().splitlines()[-1] if err.strip() else "?"}); using OpenCV'
  else:
    warn = None
  _opencv_trim(src, dst, a, b)
  got = _count(dst)
  if got != n:
    warn = (warn + '; ' if warn else '') + f'{name}: trimmed video has {got} frames, expected {n}'
  return dst, warn


def _ffmpeg_trim(src, dst, a, b):
  # the trim filter counts decoded frames, so it is exact like our frame index
  base = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', src, '-vf',
          f'trim=start_frame={a}:end_frame={b + 1},setpts=PTS-STARTPTS', '-an']
  enc = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', dst]
  last = None
  for sync in (['-fps_mode', 'passthrough'], ['-vsync', '0']):  # new / old ffmpeg spelling
    r = subprocess.run(base + sync + enc, capture_output=True, text=True)
    if r.returncode == 0:
      return None
    last = r.stderr
  return last or 'unknown error'


def _opencv_trim(src, dst, a, b):
  cap = cv2.VideoCapture(src)
  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
  w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
  cap.set(cv2.CAP_PROP_POS_FRAMES, a)
  for _ in range(a, b + 1):
    ok, img = cap.read()
    if not ok:
      break
    vw.write(img)
  vw.release()
  cap.release()


def _count(path):
  cap = cv2.VideoCapture(path)
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  cap.release()
  return n


def trim_boxes(dets, dst, a, b):
  """Input detections with frame in a..b, renumbered so a is frame 0, written
  in the detector dump layout (space separated, 0-based:
  frame id x y w h conf class quality extras) whatever the input layout was."""
  lo = int(dets.frame.searchsorted(a, 'left'))
  hi = int(dets.frame.searchsorted(b, 'right'))
  os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
  with open(dst, 'w') as f:
    for i in range(lo, hi):
      f.write(canonical_line(dets, i, int(dets.frame[i]) - a) + '\n')
  return hi - lo
