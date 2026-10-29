import threading
from collections import OrderedDict

import cv2


class FrameServer:
  """JPEG frames of one source, with a byte-bounded LRU cache and a background
  thread that decodes frames the UI is likely to ask for next."""

  def __init__(self, source, cache_mb=384, quality=85):
    self.source = source
    self.quality = quality
    self._cache = OrderedDict()  # (frame, max_w) -> jpeg bytes
    self._bytes = 0
    self._budget = cache_mb << 20
    self._lock = threading.Lock()
    self._wanted = []
    self._wake = threading.Condition()
    self._closed = False
    self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
    self._thread.start()

  def _size(self, max_w):
    w = self.source.width
    if not max_w or max_w >= w:
      return 0  # native
    return int(max_w)

  def get(self, i, max_w=0):
    key = (int(i), self._size(max_w))
    with self._lock:
      data = self._cache.get(key)
      if data is not None:
        self._cache.move_to_end(key)
        return data
    return self._decode(key)

  def _decode(self, key):
    i, w = key
    img = self.source.read(i)
    if img is None:
      return None
    if w:
      h = int(round(img.shape[0] * w / img.shape[1]))
      img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
    if not ok:
      return None
    data = buf.tobytes()
    with self._lock:
      if key not in self._cache:
        self._cache[key] = data
        self._bytes += len(data)
        while self._bytes > self._budget and len(self._cache) > 1:
          _, old = self._cache.popitem(last=False)
          self._bytes -= len(old)
    return data

  def prefetch(self, frames, max_w=0):
    """Replace the prefetch wish list (most important first)."""
    w = self._size(max_w)
    with self._wake:
      self._wanted = [(int(f), w) for f in frames if 0 <= f < self.source.n_frames]
      self._wake.notify()

  def _prefetch_loop(self):
    while True:
      with self._wake:
        while not self._wanted and not self._closed:
          self._wake.wait()
        if self._closed:
          return
        key = self._wanted.pop(0)
      with self._lock:
        cached = key in self._cache
      if not cached:
        try:
          self._decode(key)
        except Exception:  # never let a bad frame kill the thread
          pass

  def close(self):
    with self._wake:
      self._closed = True
      self._wake.notify()
    self.source.close()
