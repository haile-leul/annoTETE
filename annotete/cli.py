"""Command line.

  annotete [serve] --videos V [--dets D] [options]   start the annotator (browser UI)
  annotete serve --project P                         re-open a project
  annotete serve                                     setup page in the browser
  annotete trim --videos V [--dets D] --out O --start 1:00 --duration 30s
  annotete export --project P --out O [--trim] [--formats txt,mot,json]
  annotete info --videos V [--dets D]                pairing + classes overview
"""
import argparse
import os
import sys

from . import __version__
from .formats import load_boxes
from .project import Project, find_media, media_name, pair_boxes, parse_time, scan


def _classes(s):
  return [int(c) for c in s.replace(' ', '').split(',') if c] if s else None


def _class_names(s):
  """'1=person,5=car' -> {'1': 'person', '5': 'car'}"""
  out = {}
  for part in (s or '').split(','):
    if '=' in part:
      k, v = part.split('=', 1)
      out[str(int(k))] = v.strip()
  return out or None


def default_project_dir(videos):
  name = media_name(os.path.abspath(videos)) or 'project'
  return os.path.join(os.getcwd(), 'annotete_projects', name)


def cmd_serve(a):
  from .server import App, serve
  project = None
  if a.videos:  # the paths given now always win; an existing project only adds its saved annotations
    root = a.project or default_project_dir(a.videos)
    existed = Project.exists(root)
    project, warnings = Project.open_or_create(root, a.videos, a.dets, mode=a.mode, format=a.format,
                                               frame_base=a.frame_base, step=a.step,
                                               classes=_classes(a.classes), min_conf=a.min_conf,
                                               class_names=_class_names(a.class_names), max_gap=a.max_gap)
    for w in warnings:
      print('note:', w)
    print(f'{"updated" if existed else "created"} project {root} ({len(project.items)} item(s), '
          f'{sum(1 for i in project.items if i["boxes"])} with box files)')
  elif a.project and Project.exists(a.project):
    project = Project.open(a.project)
    print(f'opened project {project.root} ({len(project.items)} item(s))')
  elif a.project:
    sys.exit(f'no project in {a.project}; give --videos to create one')
  serve(App(project), a.host, a.port, not a.no_browser)


def cmd_trim(a):
  """Batch trim without a project: same range for every video."""
  from . import trim
  from .sources import open_source
  media = find_media(a.videos)
  pairs, warnings = pair_boxes(media, a.dets)
  for w in warnings:
    print('warning:', w)
  for m in media:
    name = media_name(m)
    src = open_source(m)
    fps, n, kind = src.fps, src.n_frames, src.kind
    src.close()
    s = parse_time(a.start, fps)
    e = s + parse_time(a.duration, fps) - 1 if a.duration else (parse_time(a.end, fps) if a.end else n - 1)
    s, e = max(0, min(s, n - 1)), max(0, min(e, n - 1))
    print(f'{name}: frames {s}-{e} of {n} ({(e - s + 1) / fps:.1f}s)')
    _, warn = trim.trim_media(m, os.path.join(a.out, 'videos'), name, s, e, kind)
    if warn:
      print('warning:', warn)
    if pairs[m]:
      dets = load_boxes(pairs[m], a.format, a.frame_base, n)
      k = trim.trim_boxes(dets, os.path.join(a.out, 'detections', name + '.txt'), s, e)
      print(f'  {k} boxes')
  print(f'done: {os.path.abspath(a.out)}')


def cmd_export(a):
  from .tasks import session_class
  p = Project.open(a.project)
  out = a.out or os.path.join(p.root, 'export')
  formats = tuple(x.strip() for x in a.formats.split(',') if x.strip())
  for item in p.items:
    if a.items and item['name'] not in a.items.split(','):
      continue
    if not a.trim and not os.path.exists(p.ann_path(item)) and p.settings.get('mode') != 'tracks':
      print(f'{item["name"]}: not annotated yet, skipped')
      continue
    s = session_class(p.settings.get('task', 'tracking'))(p, item, frames=False)
    for w in s.export(out, a.trim, formats, not a.no_interp, video=a.video):
      print('warning:', w)
    s.source.close()
  print(f'done: {os.path.abspath(out)}')


def cmd_info(a):
  entries, warnings = scan(a.videos, a.dets)
  from .sources import open_source
  for e in entries:
    src = open_source(e['media'])
    n = src.n_frames
    line = f'{e["name"]}: {n} frames {src.width}x{src.height} @ {src.fps:.2f} fps'
    src.close()
    if e['boxes']:
      d = load_boxes(e['boxes'], a.format, a.frame_base, n)
      kind = 'tracks' if d.ids_look_like_tracks() else 'detections'
      sep = 'comma' if d.layout.sep == ',' else 'space'
      line += (f' | {os.path.basename(e["boxes"])}: {len(d)} boxes ({kind}, {sep} separated, frames '
               f'{d.layout.frame_base}-based: {d.layout.base_reason}), '
               f'classes {d.classes()}, conf {d.conf_range()[0]:g}..{d.conf_range()[1]:g}')
      warnings += d.warnings
    print(line)
  for w in warnings:
    print('warning:', w)


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  if not argv or argv[0].startswith('-') and argv[0] not in ('-h', '--help', '--version'):
    argv = ['serve'] + argv
  ap = argparse.ArgumentParser(prog='annotete', description='annoTETE annotation tool')
  ap.add_argument('--version', action='version', version=__version__)
  sub = ap.add_subparsers(dest='cmd', required=True)

  def data_args(p, need_videos=True):
    p.add_argument('--videos', required=need_videos, help='a video, an image folder, or a folder of videos')
    p.add_argument('--dets', help='box file (single video) or folder of box files named like the videos')
    p.add_argument('--format', default=None,
                   help='box file columns: auto (frame id x y w h conf class quality ...) | '
                        'columns e.g. "frame=0,id=1,x=2,y=3,w=4,h=5,conf=6,cls=7". '
                        'Space or comma separation is detected from the file.')
    p.add_argument('--frame-base', default=None, choices=['auto', '0', '1'],
                   help='first frame number in the box files (auto: 0 if a box is on frame 0, 1 for MOT style files)')

  p = sub.add_parser('serve', help='start the annotator')
  data_args(p, False)
  p.add_argument('--project', help='project folder (default: ./annotete_projects/<videos name>)')
  p.add_argument('--mode', default=None, choices=['auto', 'detections', 'tracks'],
                 help='detections: assign ids to boxes; tracks: correct the ids of a tracker output '
                      '(auto: tracks if ids repeat across frames)')
  p.add_argument('--classes', help='classes to annotate, e.g. "1,5" (default all)')
  p.add_argument('--class-names', help='e.g. "1=person,5=car"')
  p.add_argument('--min-conf', type=float, help='ignore boxes below this confidence')
  p.add_argument('--step', type=int, help='keyframe step in frames (default 10)')
  p.add_argument('--max-gap', type=int, help='do not interpolate between keys further apart (default 300)')
  p.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to allow other machines (no authentication!)')
  p.add_argument('--port', type=int, default=8765)
  p.add_argument('--no-browser', action='store_true')
  p.set_defaults(fn=cmd_serve)

  p = sub.add_parser('trim', help='cut the same time range out of every video (+ its box file)')
  data_args(p)
  p.add_argument('--out', required=True)
  p.add_argument('--start', default='0', help='frame (120), seconds (12.5s) or [h:]m:s (1:30)')
  g = p.add_mutually_exclusive_group()
  g.add_argument('--end', help='last frame / time (inclusive)')
  g.add_argument('--duration', help='length, e.g. 30s or 750 (frames)')
  p.set_defaults(fn=cmd_trim)

  p = sub.add_parser('export', help='write annotations of a project')
  p.add_argument('--project', required=True)
  p.add_argument('--out', help='output folder (default: <project>/export)')
  p.add_argument('--trim', action='store_true', help='only the selected range of each video, renumbered from 0, '
                                                    'with the trimmed video and input boxes')
  p.add_argument('--formats', default='txt', help='comma list of txt (input layout), mot, json')
  p.add_argument('--no-interp', action='store_true', help='leave out interpolated boxes (no detection behind them)')
  p.add_argument('--video', action='store_true',
                 help='also write videos_annotated/<video>.mp4 with the boxes and ids drawn on the frames')
  p.add_argument('--items', help='comma list of item names (default all)')
  p.set_defaults(fn=cmd_export)

  p = sub.add_parser('info', help='show how videos and box files pair up')
  data_args(p)
  p.set_defaults(fn=cmd_info)

  a = ap.parse_args(argv)
  a.fn(a)


if __name__ == '__main__':
  main()
