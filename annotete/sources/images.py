import os
import re

import cv2

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _natural_key(name):
  return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


class ImageFolder:
  """A folder of images, one per frame, in natural sort order (img2 < img10)."""
  kind = 'images'

  def __init__(self, path, fps=25.0):
    self.path = path
    self.files = sorted((n for n in os.listdir(path) if os.path.splitext(n)[1].lower() in IMAGE_EXTS),
                        key=_natural_key)
    if not self.files:
      raise IOError(f'no images in {path}')
    self.n_frames = len(self.files)
    self.fps = fps
    first = cv2.imread(os.path.join(path, self.files[0]))
    if first is None:
      raise IOError(f'cannot read {self.files[0]}')
    self.height, self.width = first.shape[:2]

  def read(self, i):
    if i < 0 or i >= self.n_frames:
      return None
    return cv2.imread(os.path.join(self.path, self.files[i]))

  def close(self):
    pass
