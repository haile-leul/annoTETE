"""evaluATE routes on the annoTETE server (page: /evaluate)."""
import os
import threading

from ..sources import FrameServer, open_source
from .run import Evaluation

MAX_KEPT = 6


def _evals(app):
  if not hasattr(app, 'evals'):
    app.evals = {}           # id -> Evaluation (the last few)
    app.eval_frames = {}     # video path -> FrameServer
    app.eval_lock = threading.Lock()
  return app.evals


def r_defaults(h, q):
  """Default folders: the open annoTETE project's export folder."""
  p = h.app.project
  h._send(200, dict(gt=os.path.join(p.root, 'export') if p else ''))


def r_run(h, q):
  b = h._body()
  trackers = [(t.get('name') or '', t['path']) for t in b.get('trackers', []) if t.get('path')]
  if not b.get('gt') or not trackers:
    raise ValueError('give the export folder and at least one tracker output')
  if len(trackers) > 2:
    raise ValueError('compare at most two tracker outputs')
  shift = b.get('shift', 'auto')
  shift = 'auto' if shift in (None, '', 'auto') else int(shift)

  def work(job):
    ev = Evaluation(b['gt'], trackers, shift, b.get('classes', 'gt'), log=job['log'].append)
    ev.run()
    try:
      job['report'] = ev.save_report()
    except OSError as e:  # read only export folder: results still shown
      ev.warnings.append(f'report not saved: {e}')
    evals = _evals(h.app)
    with h.app.eval_lock:
      evals[ev.id] = ev
      while len(evals) > MAX_KEPT:
        evals.pop(next(iter(evals)))
    job['result'] = ev.id

  h._send(200, h.app.start_job(work))


def _get(h, rid):
  ev = _evals(h.app).get(rid)
  if ev is None:
    raise KeyError('this evaluation is no longer in memory (server restarted?): run it again')
  return ev


def r_summary(h, q, rid):
  h._send(200, _get(h, rid).summary)


def r_detail(h, q, rid, seq, ti):
  h._send(200, _get(h, rid).detail(seq, int(ti)))


def _frames(h, rid, seq):
  ev = _get(h, rid)
  s = ev.details[(seq, 0)]['seq']
  if not s.video:
    raise FileNotFoundError(f'no video found for {seq} (export with "trimmed", or keep the source video)')
  servers = h.app.eval_frames
  with h.app.eval_lock:
    fs = servers.get(s.video)
    if fs is None:
      fs = servers[s.video] = FrameServer(open_source(s.video), cache_mb=192)
  return fs, s


def r_video_meta(h, q, rid, seq):
  fs, s = _frames(h, rid, seq)
  src = fs.source
  h._send(200, dict(width=src.width, height=src.height, n_frames=s.n, fps=src.fps or s.fps,
                    media_key=f'{rid}{abs(hash(s.video)) % 10 ** 8}'))


def r_frame(h, q, rid, seq, f):
  fs, s = _frames(h, rid, seq)
  data = fs.get(int(f) + s.video_offset, int(q.get('w') or 0))
  if data is None:
    return h._send(404, dict(error=f'frame {f} could not be read'))
  h._send(200, data, 'image/jpeg', {'Cache-Control': 'private, max-age=86400'})


R = r'/api/eval/(\w+)'
ROUTES = [
  ('GET', r'/api/eval/defaults', r_defaults),
  ('POST', r'/api/eval/run', r_run),
  ('GET', R + r'/summary', r_summary),
  ('GET', R + r'/seq/([^/]+)/(\d+)', r_detail),
  ('GET', R + r'/video/([^/]+)/meta', r_video_meta),
  ('GET', R + r'/video/([^/]+)/frame/(\d+)', r_frame),
]
