"""Local web server: static UI + JSON API. Standard library only.

One project is open at a time. Opened items (sessions) are kept in memory
(a few at a time) and autosaved in the background every few seconds.
"""
import gzip
import json
import mimetypes
import os
import re
import threading
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .formats import load_boxes
from .project import Project, parse_time, scan
from .sources import open_source
from .tasks import session_class
from .tasks.tracking.annotation import EditError

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MAX_OPEN = 3
AUTOSAVE_S = 3.0


class App:
  def __init__(self, project=None, log=print):
    self.project = project
    self.sessions = OrderedDict()
    self.lock = threading.RLock()
    self.jobs = {}
    self.log = log
    self._stop = threading.Event()
    threading.Thread(target=self._autosave, daemon=True).start()

  # ---------------------------------------------------------------- sessions
  def session(self, name):
    with self.lock:
      if self.project is None:
        raise EditError('no project is open')
      s = self.sessions.get(name)
      if s is not None:
        self.sessions.move_to_end(name)
        return s
      item = self.project.item(name)
      s = session_class(self.project.settings.get('task', 'tracking'))(self.project, item)
      item['fps'], item['n_frames'] = s.source.fps, s.ann.n_frames
      self.sessions[name] = s
      while len(self.sessions) > MAX_OPEN:
        _, old = self.sessions.popitem(last=False)
        old.close()
      return s

  def save_all(self, force=False):
    with self.lock:
      sessions = list(self.sessions.values())
    n = 0
    for s in sessions:
      try:
        n += bool(s.save(force))
      except Exception:
        self.log('autosave failed:\n' + traceback.format_exc())
    return n

  def _autosave(self):
    while not self._stop.wait(AUTOSAVE_S):
      self.save_all()

  def set_project(self, project):
    with self.lock:
      self.save_all()
      for s in self.sessions.values():
        s.close()
      self.sessions.clear()
      self.project = project

  def shutdown(self):
    self._stop.set()
    with self.lock:
      for s in self.sessions.values():
        try:
          s.close()
        except Exception:
          traceback.print_exc()

  # -------------------------------------------------------------------- state
  def state(self):
    p = self.project
    if p is None:
      return dict(version=__version__, project=None)
    return dict(version=__version__, project=dict(root=p.root, videos=p.config.get('videos'), boxes=p.config.get('boxes'),
                                                  settings=p.settings, items=p.items))

  # --------------------------------------------------------------------- jobs
  def start_job(self, fn, *args):
    jid = uuid.uuid4().hex[:8]
    job = dict(id=jid, done=False, error=None, log=[], warnings=[], progress=0.0)
    self.jobs[jid] = job

    def run():
      try:
        fn(job, *args)
      except Exception as e:
        job['error'] = f'{type(e).__name__}: {e}'
        self.log(traceback.format_exc())
      job['done'] = True

    threading.Thread(target=run, daemon=True).start()
    return job

  def export_job(self, job, out, names, trimmed, formats, interp):
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)
    self.save_all()
    for k, name in enumerate(names):
      job['log'].append(f'[{k + 1}/{len(names)}] {name}')
      s = self.session(name)
      if not trimmed and not s.ann.tracks and not os.path.exists(self.project.ann_path(s.item)):
        job['log'].append(f'{name}: not annotated yet, skipped')
        continue
      job['warnings'] += s.export(out, trimmed, formats, interp, log=job['log'].append)
      job['progress'] = (k + 1) / len(names)
    job['log'].append(f'done: {out}')
    job['out'] = out


class Handler(BaseHTTPRequestHandler):
  app = None
  protocol_version = 'HTTP/1.1'

  def log_message(self, fmt, *args):  # quiet: only errors are printed
    pass

  # ------------------------------------------------------------------ helpers
  def _send(self, code, body, ctype='application/json', headers=None):
    if isinstance(body, (dict, list)) or body is None:
      body = json.dumps(body, separators=(',', ':')).encode()
    elif isinstance(body, str):
      body = body.encode()
    if ctype.startswith('application/json') and len(body) > 2048 and 'gzip' in self.headers.get('Accept-Encoding', ''):
      body = gzip.compress(body, 5)
      headers = dict(headers or {}, **{'Content-Encoding': 'gzip'})
    self.send_response(code)
    self.send_header('Content-Type', ctype)
    self.send_header('Content-Length', str(len(body)))
    for k, v in (headers or {}).items():
      self.send_header(k, v)
    self.end_headers()
    try:
      self.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
      pass

  def _body(self):
    n = int(self.headers.get('Content-Length') or 0)
    return json.loads(self.rfile.read(n) or b'{}') if n else {}

  def do_GET(self):
    self._dispatch('GET')

  def do_POST(self):
    self._dispatch('POST')

  def _dispatch(self, method):
    url = urlparse(self.path)
    q = {k: v[-1] for k, v in parse_qs(url.query).items()}
    path = unquote(url.path)
    try:
      for m, rx, fn in ROUTES:
        if m != method:
          continue
        mt = rx.fullmatch(path)
        if mt:
          return fn(self, q, *mt.groups())
      if method == 'GET':
        return self._static(path)
      self._send(404, dict(error='not found'))
    except (EditError, ValueError, KeyError, FileNotFoundError, IOError) as e:
      msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
      self._send(400, dict(error=msg))
    except (BrokenPipeError, ConnectionResetError):
      pass
    except Exception as e:
      self.app.log(traceback.format_exc())
      self._send(500, dict(error=f'{type(e).__name__}: {e}'))

  def _static(self, path):
    if path in ('', '/'):
      path = '/index.html'
    full = os.path.normpath(os.path.join(WEB, path.lstrip('/')))
    if not full.startswith(WEB) or not os.path.isfile(full):
      return self._send(404, dict(error='not found'))
    ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
    if full.endswith('.js'):
      ctype = 'text/javascript'
    with open(full, 'rb') as f:
      self._send(200, f.read(), ctype, {'Cache-Control': 'no-cache'})

  # ------------------------------------------------------------------- routes
  def r_state(self, q):
    self._send(200, self.app.state())

  def r_ls(self, q):
    """Directory listing for path autocompletion in the setup form."""
    p = os.path.expanduser(q.get('path') or '~')
    d = p if os.path.isdir(p) else os.path.dirname(p) or '.'
    out = []
    if os.path.isdir(d):
      for n in sorted(os.listdir(d))[:500]:
        if n.startswith('.'):
          continue
        full = os.path.join(d, n)
        out.append(dict(name=n, path=full, dir=os.path.isdir(full)))
    self._send(200, dict(dir=os.path.abspath(d), entries=out))

  def r_scan(self, q):
    b = self._body()
    entries, warnings = scan(b['videos'], b.get('boxes') or None)
    classes, preview = {}, None
    looks_tracks = None
    first = next((e for e in entries if e['boxes']), None)
    if first:
      src = open_source(first['media'])
      n_frames = src.n_frames
      src.close()
      dets = load_boxes(first['boxes'], b.get('format') or 'auto', b.get('frame_base') or 'auto', n_frames)
      classes = dets.classes()
      looks_tracks = dets.ids_look_like_tracks()
      warnings += dets.warnings
      preview = dict(file=first['boxes'], layout=dets.layout.to_dict(), n=len(dets), conf=dets.conf_range(),
                     lines=dets.lines[:5], n_frames=n_frames,
                     frames=[int(dets.frame.min()), int(dets.frame.max())] if len(dets) else None)
    self._send(200, dict(items=entries, warnings=warnings, classes=classes, looks_like_tracks=looks_tracks,
                         preview=preview))

  def r_create(self, q):
    b = self._body()
    root = os.path.abspath(os.path.expanduser(b['root']))
    scan(b['videos'], b.get('boxes') or None)  # bad paths fail here, before anything is closed
    self.app.set_project(None)  # save + close the open videos before their project changes
    p, warnings = Project.open_or_create(root, b['videos'], b.get('boxes') or None, **(b.get('settings') or {}))
    self.app.set_project(p)
    self._send(200, dict(state=self.app.state(), warnings=warnings))

  def r_open(self, q):
    root = os.path.abspath(os.path.expanduser(self._body()['root']))
    if not Project.exists(root):
      raise FileNotFoundError(f'no project.json in {root}')
    self.app.set_project(Project.open(root))
    self._send(200, self.app.state())

  def r_settings(self, q):
    b = self._body()
    p = self.app.project
    with p.lock:
      for k, v in b.items():
        if k in ('step', 'max_gap', 'classes', 'min_conf', 'class_names', 'format', 'frame_base', 'mode'):
          p.settings[k] = v
      p.save()
    with self.app.lock:
      for s in self.app.sessions.values():
        s.configure(p.settings)
    self._send(200, self.app.state())

  def r_meta(self, q, name):
    self._send(200, self.app.session(name).meta())

  def r_frame(self, q, name, f):
    s = self.app.session(name)
    w = int(q.get('w') or 0)
    data = s.frames.get(int(f), w)
    if q.get('next'):
      s.frames.prefetch([int(x) for x in q['next'].split(',') if x.strip().lstrip('-').isdigit()], w)
    if data is None:
      return self._send(404, dict(error=f'frame {f} could not be read'))
    self._send(200, data, 'image/jpeg', {'Cache-Control': 'private, max-age=86400'})

  def r_boxes(self, q, name):
    s = self.app.session(name)
    a, b = int(q['a']), int(q['b'])
    if b - a > 2000:
      raise ValueError('ask for at most 2000 frames at once')
    self._send(200, dict(rev=s.ann.rev, frames=s.frame_data(a, b)))

  def r_proposals(self, q, name):
    self._send(200, self.app.session(name).proposals(int(q['f'])))

  def r_step(self, q, name):
    s = self.app.session(name)
    self._send(200, s.step_target(int(q['f']), int(q['step']) if q.get('step') else None))

  def r_tracks(self, q, name):
    s = self.app.session(name)
    self._send(200, dict(rev=s.ann.rev, tracks=s.tracks()))

  def r_track(self, q, name, tid):
    self._send(200, self.app.session(name).track(int(tid)))

  def r_coverage(self, q, name):
    self._send(200, self.app.session(name).coverage(int(q.get('bins') or 1000)))

  def r_issues(self, q, name):
    self._send(200, dict(issues=self.app.session(name).issues()))

  def r_cmd(self, q, name):
    b = self._body()
    s = self.app.session(name)
    self._send(200, s.command(b.pop('op'), b))

  def r_range(self, q, name):
    b = self._body()
    s = self.app.session(name)
    r = b.get('range')
    if r is not None:
      a, e = int(r[0]), int(r[1])
      if e < a:
        a, e = e, a
      r = [max(0, a), min(s.ann.n_frames - 1, e)]
    s.item['range'] = r
    self.app.project.save()
    self._send(200, s.meta())

  def r_status(self, q, name):
    item = self.app.project.item(name)
    item['status'] = self._body()['status']
    self.app.project.save()
    self._send(200, dict(ok=True))

  def r_ranges(self, q):
    """Same range (times or frames) for every item, clipped to each video."""
    b = self._body()
    p = self.app.project
    for item in p.items:
      if b.get('clear'):
        item['range'] = None
        continue
      if 'fps' not in item:
        src = open_source(item['media'])
        item['fps'], item['n_frames'] = src.fps, src.n_frames
        src.close()
      fps, n = item['fps'], item['n_frames']
      a = parse_time(b.get('start') or 0, fps)
      if b.get('duration'):
        e = a + parse_time(b['duration'], fps) - 1
      elif b.get('end'):
        e = parse_time(b['end'], fps)
      else:
        e = n - 1
      a, e = max(0, min(a, n - 1)), max(0, min(e, n - 1))
      item['range'] = [a, e] if (a, e) != (0, n - 1) else None
    p.save()
    self._send(200, self.app.state())

  def r_save(self, q):
    self._send(200, dict(saved=self.app.save_all(force=True)))

  def r_export(self, q):
    b = self._body()
    names = b.get('items') or [it['name'] for it in self.app.project.items]
    job = self.app.start_job(self.app.export_job, b['out'], names, bool(b.get('trimmed')),
                             tuple(b.get('formats') or ('txt',)), b.get('interp', True))
    self._send(200, job)

  def r_job(self, q, jid):
    job = self.app.jobs.get(jid)
    if job is None:
      raise KeyError(f'no job {jid}')
    self._send(200, job)


ITEM = r'/api/item/([^/]+)'
ROUTES = [(m, re.compile(p), fn) for m, p, fn in [
  ('GET', r'/api/state', Handler.r_state),
  ('GET', r'/api/ls', Handler.r_ls),
  ('POST', r'/api/scan', Handler.r_scan),
  ('POST', r'/api/project', Handler.r_create),
  ('POST', r'/api/open', Handler.r_open),
  ('POST', r'/api/settings', Handler.r_settings),
  ('POST', r'/api/ranges', Handler.r_ranges),
  ('POST', r'/api/save', Handler.r_save),
  ('POST', r'/api/export', Handler.r_export),
  ('GET', r'/api/jobs/(\w+)', Handler.r_job),
  ('GET', ITEM + r'/meta', Handler.r_meta),
  ('GET', ITEM + r'/frame/(\d+)', Handler.r_frame),
  ('GET', ITEM + r'/boxes', Handler.r_boxes),
  ('GET', ITEM + r'/proposals', Handler.r_proposals),
  ('GET', ITEM + r'/step', Handler.r_step),
  ('GET', ITEM + r'/tracks', Handler.r_tracks),
  ('GET', ITEM + r'/track/(\d+)', Handler.r_track),
  ('GET', ITEM + r'/coverage', Handler.r_coverage),
  ('GET', ITEM + r'/issues', Handler.r_issues),
  ('POST', ITEM + r'/cmd', Handler.r_cmd),
  ('POST', ITEM + r'/range', Handler.r_range),
  ('POST', ITEM + r'/status', Handler.r_status),
]]


def serve(app, host='127.0.0.1', port=8765, open_browser=True):
  Handler.app = app
  httpd = None
  for p in range(port, port + 20):
    try:
      httpd = ThreadingHTTPServer((host, p), Handler)
      port = p
      break
    except OSError:
      continue
  if httpd is None:
    raise OSError(f'no free port in {port}..{port + 19}')
  httpd.daemon_threads = True
  url = f'http://{"localhost" if host in ("127.0.0.1", "0.0.0.0") else host}:{port}/'
  app.log(f'annoTETE {__version__} running at {url}  (Ctrl+C to stop; work is autosaved)')
  if open_browser:
    threading.Timer(0.5, _open, (url,)).start()
  try:
    httpd.serve_forever(poll_interval=0.5)
  except KeyboardInterrupt:
    pass
  finally:
    app.log('saving ...')
    app.shutdown()
    httpd.server_close()
  return url


def _open(url):
  try:
    import webbrowser
    webbrowser.open(url)
  except Exception:
    pass
