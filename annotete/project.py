"""A project: a set of items (video + optional box file) with settings and
autosaved annotations.

  <project>/project.json            settings + item list (paths, ranges, status)
  <project>/annotations/<item>.json one annotation file per item
"""
import json
import os
import re
import threading
import time

from .formats import Layout, load_boxes
from .sources import IMAGE_EXTS, VIDEO_EXTS, is_image_folder

BOX_EXTS = ('.txt', '.csv', '.det', '.mot')


def _stem(path):
  return os.path.splitext(os.path.basename(path.rstrip('/')))[0]


def find_media(path):
  """Videos (or image-sequence folders) at `path`: a file, or a folder of them."""
  path = os.path.abspath(os.path.expanduser(path))
  if os.path.isfile(path):
    return [path]
  if not os.path.isdir(path):
    raise FileNotFoundError(f'no such file or folder: {path}')
  if is_image_folder(path) and not any(os.path.splitext(n)[1].lower() in VIDEO_EXTS for n in os.listdir(path)):
    return [path]  # the folder itself is one image sequence
  out = []
  for n in sorted(os.listdir(path)):
    p = os.path.join(path, n)
    if os.path.isfile(p) and os.path.splitext(n)[1].lower() in VIDEO_EXTS:
      out.append(p)
    elif os.path.isdir(p):
      img1 = os.path.join(p, 'img1')  # MOTChallenge sequence folder
      if os.path.isdir(img1) and is_image_folder(img1):
        out.append(img1)
      elif is_image_folder(p):
        out.append(p)
  return out


def media_name(path):
  """Item name: file stem, or the sequence folder name for MOT style seq/img1."""
  if os.path.basename(path.rstrip('/')) == 'img1':
    return os.path.basename(os.path.dirname(path.rstrip('/')))
  return _stem(path)


def pair_boxes(media, boxes_path):
  """Match each media item with a box file. `boxes_path` is a single file
  (only valid with a single video) or a folder. Matching, per video stem S:
  S.txt, then a file whose stem starts with S + separator (S_dets.txt), then
  MOT style S/det/det.txt or S/gt/gt.txt. Returns ({media: path|None}, warnings)."""
  warnings = []
  out = {m: None for m in media}
  if not boxes_path:
    return out, warnings
  boxes_path = os.path.abspath(os.path.expanduser(boxes_path))
  if os.path.isfile(boxes_path):
    if len(media) != 1:
      raise ValueError('a single box file needs a single video; give a folder of box files for a folder of videos')
    out[media[0]] = boxes_path
    return out, warnings
  if not os.path.isdir(boxes_path):
    raise FileNotFoundError(f'no such file or folder: {boxes_path}')
  files = [n for n in os.listdir(boxes_path)
           if os.path.isfile(os.path.join(boxes_path, n)) and os.path.splitext(n)[1].lower() in BOX_EXTS]
  stems = {_stem(n): n for n in files}
  for m in media:
    s = media_name(m)
    hit = None
    if s in stems:
      hit = stems[s]
    else:
      pref = sorted((n for st, n in stems.items() if re.match(re.escape(s) + r'[._\-]', st)), key=len)
      if len(pref) > 1:
        warnings.append(f'{s}: several box files match ({", ".join(pref)}), using {pref[0]}')
      hit = pref[0] if pref else None
    if hit:
      out[m] = os.path.join(boxes_path, hit)
      continue
    for sub in ('det/det.txt', 'gt/gt.txt'):
      p = os.path.join(boxes_path, s, sub)
      if os.path.isfile(p):
        out[m] = p
        break
    # MOT sequence folders also carry their own det/ gt/ next to img1
    if out[m] is None and m.rstrip('/').endswith('img1'):
      for sub in ('det/det.txt', 'gt/gt.txt'):
        p = os.path.join(os.path.dirname(m.rstrip('/')), sub)
        if os.path.isfile(p):
          out[m] = p
          break
    if out[m] is None:
      warnings.append(f'{s}: no box file found')
  return out, warnings


def scan(videos, boxes=None):
  """Preview of what a project would contain."""
  media = find_media(videos)
  if not media:
    raise FileNotFoundError(f'no videos or image folders found in {videos}')
  pairs, warnings = pair_boxes(media, boxes)
  return [dict(name=media_name(m), media=m, boxes=pairs[m]) for m in media], warnings


DEFAULT_SETTINGS = dict(task='tracking', mode='auto', format='auto', step=10, max_gap=300, classes=[], min_conf=None,
                        class_names={})


class Project:
  def __init__(self, root, config):
    self.root = os.path.abspath(root)
    self.config = config
    self.lock = threading.RLock()
    os.makedirs(self.ann_dir, exist_ok=True)

  @property
  def ann_dir(self):
    return os.path.join(self.root, 'annotations')

  @property
  def items(self):
    return self.config['items']

  @property
  def settings(self):
    return self.config['settings']

  @classmethod
  def create(cls, root, videos, boxes=None, **settings):
    entries, warnings = scan(videos, boxes)
    names = set()
    for e in entries:  # unique item names
      base, k = e['name'], 1
      while e['name'] in names:
        k += 1
        e['name'] = f'{base}_{k}'
      names.add(e['name'])
      e.update(range=None, status='todo', stats=None)
    s = dict(DEFAULT_SETTINGS)
    s.update({k: v for k, v in settings.items() if v is not None})
    config = dict(version=1, videos=os.path.abspath(videos), boxes=os.path.abspath(boxes) if boxes else None,
                  settings=s, items=entries, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    p = cls(root, config)
    p.save()
    return p, warnings

  @classmethod
  def open(cls, root):
    with open(os.path.join(root, 'project.json')) as f:
      config = json.load(f)
    s = dict(DEFAULT_SETTINGS)
    s.update(config.get('settings', {}))
    config['settings'] = s
    return cls(root, config)

  @staticmethod
  def exists(root):
    return os.path.isfile(os.path.join(root, 'project.json'))

  def save(self):
    with self.lock:
      atomic_json(os.path.join(self.root, 'project.json'), self.config)

  def item(self, name):
    for it in self.items:
      if it['name'] == name:
        return it
    raise KeyError(f'no item {name}')

  def layout(self, item):
    return Layout.from_spec(self.settings.get('format') or 'auto', item.get('boxes'))

  def load_boxes(self, item):
    return load_boxes(item['boxes'], self.layout(item))

  def ann_path(self, item):
    return os.path.join(self.ann_dir, item['name'] + '.json')


def atomic_json(path, data, backup_every=600):
  """Write JSON via a temp file + rename (never a half written file). Keeps a
  `.bak` copy of the previous version, refreshed at most every `backup_every` s."""
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
    f.flush()
    os.fsync(f.fileno())
  bak = path + '.bak'
  if os.path.exists(path) and (not os.path.exists(bak) or time.time() - os.path.getmtime(bak) > backup_every):
    try:
      os.replace(path, bak)
    except OSError:
      pass
  os.replace(tmp, path)


def parse_time(s, fps):
  """'120' -> frame 120, '12.5s' -> seconds, '1:30' / '1:02:03.5' -> h:m:s."""
  s = str(s).strip()
  if s.endswith('s'):
    return int(round(float(s[:-1]) * fps))
  if ':' in s:
    sec = 0.0
    for part in s.split(':'):
      sec = sec * 60 + float(part)
    return int(round(sec * fps))
  return int(float(s))


__all__ = ['Project', 'scan', 'find_media', 'pair_boxes', 'parse_time', 'atomic_json', 'IMAGE_EXTS']
