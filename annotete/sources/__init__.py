"""Frame sources: anything that can give frame `i` of a sequence as an image.

  VideoSource   video files (OpenCV decoder, sequential read-ahead)
  ImageFolder   a folder of images (one image per frame, sorted by name)

`open_source(path)` picks the right one. A new data type (e.g. a point cloud
sequence or a multi-camera rig) is a new class with the same interface:
`n_frames, fps, width, height, read(i) -> BGR ndarray, close()`.
"""
import os

from .images import IMAGE_EXTS, ImageFolder
from .video import VIDEO_EXTS, VideoSource
from .cache import FrameServer


def is_video(path):
  return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_image_folder(path):
  return os.path.isdir(path) and any(os.path.splitext(n)[1].lower() in IMAGE_EXTS for n in os.listdir(path))


def open_source(path):
  if os.path.isdir(path):
    return ImageFolder(path)
  return VideoSource(path)


__all__ = ['FrameServer', 'ImageFolder', 'VideoSource', 'IMAGE_EXTS', 'VIDEO_EXTS', 'is_video', 'is_image_folder',
           'open_source']
