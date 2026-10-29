import threading

import cv2

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg', '.ts', '.mts', '.wmv', '.flv',
              '.h264', '.h265', '.hevc')

# Reading forward this many frames is cheaper than a seek (which decodes from
# the previous key frame anyway).
FORWARD_READ = 90


class VideoSource:
  kind = 'video'

  def __init__(self, path):
    self.path = path
    self._lock = threading.Lock()
    self._cap = cv2.VideoCapture(path)
    if not self._cap.isOpened():
      raise IOError(f'cannot open video {path}')
    self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self._pos = 0  # index of the frame the next cap.read() returns
    if self.n_frames <= 0:  # some containers do not store a frame count: count once
      n = 0
      while self._cap.grab():
        n += 1
      self.n_frames = n
      self._seek(0)

  def _seek(self, i):
    self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    self._pos = i

  def read(self, i):
    """Frame i as a BGR image, or None past the real end of the video (the
    header frame count can be a few frames too large; it is corrected here)."""
    with self._lock:
      if i < 0 or i >= self.n_frames:
        return None
      if not (self._pos <= i < self._pos + FORWARD_READ):
        self._seek(i)
      while self._pos < i:
        if not self._cap.grab():
          break
        self._pos += 1
      ok, img = self._cap.read()
      if not ok:
        if i > 0 and i >= self.n_frames - 5:
          self.n_frames = i  # truncated / over-reported count
        self._seek(min(i, max(self.n_frames - 1, 0)))
        return None
      self._pos = i + 1
      return img

  def close(self):
    with self._lock:
      self._cap.release()
