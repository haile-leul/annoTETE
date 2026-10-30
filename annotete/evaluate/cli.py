"""evaluATE command line.

  python -m evaluate                                 open evaluATE in the browser
  python -m evaluate --gt EXPORT --tracker DIR       ... and run it right away
  python -m evaluate --gt EXPORT --tracker DIR --tracker2 OLD_DIR    compare two
  python -m evaluate --gt EXPORT --tracker DIR --report               print, no UI
"""
import argparse
import os
import sys
from urllib.parse import urlencode

from .run import Evaluation


def _table(summary):
  cols = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN']
  print(f'{"":28}' + ''.join(f'{c:>8}' for c in cols))
  for r in summary['results']:
    for v, label in (('all', 'all annotations'), ('det', 'detector-visible only')):
      res = r['combined'].get(v)
      if not res:
        continue
      vals = ''.join(f'{res[c] * 100:8.2f}' if isinstance(res[c], float) else f'{res[c]:8d}' for c in cols)
      print(f'{(r["name"] + " " + label)[:28]:28}{vals}')
    d = r['combined']['detector']
    if d['gt']:
      print(f'  detector had a box for {d["gt_det"] / d["gt"]:.1%} of the {d["gt"]} annotated boxes; '
            f'missed boxes: {d["fn"]} = {d["fn_nodet"]} without detection + {d["fn_det"]} dropped by the tracker')


def main(argv=None):
  ap = argparse.ArgumentParser(prog='evaluate', description='evaluATE: evaluate tracker outputs on annoTETE exports')
  ap.add_argument('--gt', help='annoTETE export folder (with annotations_mot/ and annotations_json/)')
  ap.add_argument('--tracker', help='tracker output: a folder of files named like the videos (or one file)')
  ap.add_argument('--name', help='name of the tracker (default: folder name)')
  ap.add_argument('--tracker2', help='a second tracker output to compare with (e.g. the previous version)')
  ap.add_argument('--name2')
  ap.add_argument('--shift', default='auto', help='tracker frame = video frame + SHIFT (default: detected)')
  ap.add_argument('--all-classes', action='store_true', help="keep tracker boxes of every class (default: the "
                                                             "annotated classes only)")
  ap.add_argument('--report', action='store_true', help='print the scores and save the report, no browser UI')
  ap.add_argument('--project', help='also open this annoTETE project')
  ap.add_argument('--host', default='127.0.0.1')
  ap.add_argument('--port', type=int, default=8765)
  ap.add_argument('--no-browser', action='store_true')
  a = ap.parse_args(argv)
  trackers = [(a.name, a.tracker)] + ([(a.name2, a.tracker2)] if a.tracker2 else [])
  if a.report:
    if not (a.gt and a.tracker):
      sys.exit('--report needs --gt and --tracker')
    ev = Evaluation(a.gt, trackers, a.shift, 'all' if a.all_classes else 'gt', log=lambda m: None)
    s = ev.run()
    for w in s['warnings']:
      print('warning:', w)
    _table(s)
    print('report:', ev.save_report())
    return
  from ..project import Project
  from ..server import App, serve
  project = Project.open(a.project) if a.project and Project.exists(a.project) else None
  page = 'evaluate'
  if a.gt or a.tracker:
    q = {k: v for k, v in dict(gt=a.gt and os.path.abspath(a.gt), a=a.tracker and os.path.abspath(a.tracker), an=a.name,
                                 b=a.tracker2 and os.path.abspath(a.tracker2), bn=a.name2).items() if v}
    if a.gt and a.tracker:
      q['run'] = 1
    page += '?' + urlencode(q)
  serve(App(project), a.host, a.port, not a.no_browser, page)


if __name__ == '__main__':
  main()
