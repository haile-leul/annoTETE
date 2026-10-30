"""Videos with the annotations drawn on them (for review / sharing)."""
import colorsys
import os
import shutil
import subprocess

import cv2

from .sources import open_source


def id_color(tid):
  """BGR color of a track id; the same golden-angle palette as the web UI."""
  h = (tid * 137.508) % 360 / 360
  light = (52 + (tid % 3) * 6) / 100
  r, g, b = colorsys.hls_to_rgb(h, light, 0.78)
  return int(b * 255), int(g * 255), int(r * 255)


def _dashed_rect(img, p0, p1, color, thick, dash):
  (x0, y0), (x1, y1) = p0, p1
  for a, b, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
    for s in range(a, b, 2 * dash):
      e = min(s + dash, b)
      cv2.line(img, (s, fixed) if horiz else (fixed, s), (e, fixed) if horiz else (fixed, e), color, thick)


def draw_boxes(img, boxes, class_names=None, frame_label=None):
  """boxes: [(track_id, (x, y, w, h), interpolated, cls)]. Interpolated boxes
  (no detection behind them) are dashed."""
  H, W = img.shape[:2]
  thick = max(2, round(max(W, H) / 900))
  scale = max(0.45, max(W, H) / 2600)
  font = cv2.FONT_HERSHEY_SIMPLEX
  for tid, (x, y, w, h), interp, cls in boxes:
    col = id_color(tid)
    p0, p1 = (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h)))
    if interp:
      _dashed_rect(img, p0, p1, col, thick, max(4, 4 * thick))
    else:
      cv2.rectangle(img, p0, p1, col, thick)
    label = str(tid)
    name = (class_names or {}).get(str(cls))
    if name:
      label += f' {name}'
    (tw, th), base = cv2.getTextSize(label, font, scale, max(1, thick // 2))
    ty = p0[1] - 4 if p0[1] - th - base - 4 >= 0 else p0[1] + th + base + 4
    cv2.rectangle(img, (p0[0], ty - th - base), (p0[0] + tw + 6, ty + base // 2), col, -1)
    cv2.putText(img, label, (p0[0] + 3, ty - base // 2), font, scale, (15, 15, 15), max(1, thick // 2), cv2.LINE_AA)
  if frame_label is not None:  # bottom-left: cameras usually burn their own timestamp in at the top
    org = (10, H - 10)
    cv2.putText(img, str(frame_label), org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(frame_label), org, font, scale, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
  return img


class _Writer:
  """H.264 through ffmpeg when available (small files that play everywhere),
  else OpenCV's mp4v."""

  def __init__(self, path, w, h, fps):
    self.proc = self.vw = None
    if shutil.which('ffmpeg'):
      cmd = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
             '-r', f'{fps:g}', '-i', '-', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
             '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', path]
      self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

  def write(self, img):
    if self.proc:
      self.proc.stdin.write(img.tobytes())
    else:
      self.vw.write(img)

  def close(self):
    if self.proc:
      self.proc.stdin.close()
      err = self.proc.stderr.read().decode(errors='replace')
      if self.proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed: {err.strip()[-300:]}')
    else:
      self.vw.release()


def render_video(media, dst, a, b, rows, class_names=None, frame_offset=0, log=None):
  """Frames a..b of `media` with `rows` (frame, track_id, box, det, kind, cls;
  kind 3 = interpolated) drawn on them, written to `dst` (.mp4). The frame
  number printed in the corner is frame - frame_offset (as in the exported
  annotation files)."""
  from .tasks.tracking.model import AUTO_INTERP
  by_frame = {}
  for f, tid, box, det, kind, cls in rows:
    by_frame.setdefault(f, []).append((tid, box, kind == AUTO_INTERP, cls))
  src = open_source(media)
  os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
  writer = None
  n = 0
  try:
    for f in range(a, b + 1):
      img = src.read(f)
      if img is None:
        break
      if writer is None:
        writer = _Writer(dst, img.shape[1], img.shape[0], src.fps)
      writer.write(draw_boxes(img, by_frame.get(f, []), class_names, f - frame_offset))
      n += 1
      if log and n % 500 == 0:
        log(f'  drawn {n}/{b - a + 1} frames')
  finally:
    if writer is not None:
      writer.close()
    src.close()
  return n
