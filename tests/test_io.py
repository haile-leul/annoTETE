import json
import os
import sys
import threading
import urllib.request

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_dataset  # noqa: E402

from annotete.formats import BoxWriter, Layout, load_boxes  # noqa: E402
from annotete.project import Project, pair_boxes, parse_time  # noqa: E402
from annotete.tasks.tracking.session import TrackingSession  # noqa: E402

EXAMPLE = """0 16 408 662 169 162 71.6309 5 0.625 -1 -9999 -9999 -9999 -9999
0 41 850 668 558 382 94.043 5 0.942 -1 -9999 -9999 -9999 -9999
0 43 3219 811 12 17 22.3267 1 0.213 -1 -9999 -9999 -9999 -9999
1 80 405 661 169 161 71.8262 5 0.609 -1 -9999 -9999 -9999 -9999
1 116 3219 811 12 17 24.7803 1 0.206 -1 -9999 -9999 -9999 -9999
2 162 408 664 167 161 73.5352 5 0.624 -1 -9999 -9999 -9999 -9999
"""


def test_user_format(tmp_path):
  p = tmp_path / 'cam.txt'
  p.write_text(EXAMPLE)
  d = load_boxes(str(p))
  assert d.layout.sep == ' ' and d.layout.frame_base == 0
  assert len(d) == 6 and d.classes() == {1: 2, 5: 4}
  assert d.frame.tolist() == [0, 0, 0, 1, 1, 2]
  assert d.box[0].tolist() == [408, 662, 169, 162]
  assert d.conf[0] == pytest.approx(71.6309) and d.quality[0] == pytest.approx(0.625)
  assert not d.ids_look_like_tracks()
  w = BoxWriter(d)
  # a detection keeps every original column, only the id (and frame) change
  assert w.det_line(0, track_id=7) == '0 7 408 662 169 162 71.6309 5 0.625 -1 -9999 -9999 -9999 -9999'
  assert w.det_line(3, frame=0, track_id=7).startswith('0 7 405 661')
  # interpolated / drawn boxes: same columns, -1 where there is no value
  assert w.box_line(4, 7, (10.4, 20.6, 30, 40), cls=5) == '4 7 10 21 30 40 -1 5 -1 -1 -1 -1 -1 -1'


def test_mot_and_custom_layouts(tmp_path):
  p = tmp_path / 'gt.txt'
  p.write_text('1,3,10.5,20,30,40,1,1,0.8\n2,3,11.5,20,30,40,1,1,0.9\n')
  d = load_boxes(str(p))
  assert d.layout.sep == ',' and d.layout.frame_base == 1 and d.frame.tolist() == [0, 1] and d.ids_look_like_tracks()
  assert BoxWriter(d).det_line(1, track_id=9) == '2,9,11.5,20,30,40,1,1,0.9'
  q = tmp_path / 'xyxy.txt'
  q.write_text('# frame x1 y1 x2 y2 score\n5 10 20 40 60 0.9\n')
  lay = Layout.from_spec('frame=0,x=1,y=2,w=3,h=4,conf=5')
  lay.box = 'xyxy'
  d = load_boxes(str(q), lay)
  assert d.box[0].tolist() == [10, 20, 30, 40] and d.cls[0] == -1


def test_separator_and_frame_base_detection(tmp_path):
  space = tmp_path / 'space.txt'
  space.write_text(EXAMPLE)
  comma = tmp_path / 'comma.txt'
  comma.write_text(EXAMPLE.replace(' ', ','))
  a, b = load_boxes(str(space)), load_boxes(str(comma))
  assert (a.layout.sep, b.layout.sep) == (' ', ',')
  assert a.layout.frame_base == b.layout.frame_base == 0  # a box is on frame 0
  assert a.frame.tolist() == b.frame.tolist() and np.array_equal(a.box, b.box)
  assert a.classes() == b.classes() and a.conf.tolist() == b.conf.tolist()
  # written back with the file's own separator
  assert BoxWriter(b).det_line(0, track_id=7) == '0,7,408,662,169,162,71.6309,5,0.625,-1,-9999,-9999,-9999,-9999'
  assert BoxWriter(b).box_line(4, 7, (10, 20, 30, 40), cls=5).startswith('4,7,10,20,30,40,-1,5,-1')
  # comma + tab mixes and stray spaces after commas
  mixed = tmp_path / 'mixed.txt'
  mixed.write_text('0, 1, 10, 20, 30, 40, 0.9, 1, 0.5\n1,\t1, 11, 20, 30, 40, 0.9, 1, 0.5\n')
  m = load_boxes(str(mixed))
  assert m.layout.sep == ',' and m.frame.tolist() == [0, 1] and m.box[1].tolist() == [11, 20, 30, 40]
  # 1-based: last box on frame n_frames (one past a 0-based end)
  late = tmp_path / 'late.txt'
  late.write_text('3 1 10 10 5 5 1 1 1\n50 1 10 10 5 5 1 1 1\n')
  assert load_boxes(str(late), n_frames=50).layout.frame_base == 1
  assert load_boxes(str(late), n_frames=60).layout.frame_base == 0
  # forced either way
  assert load_boxes(str(late), frame_base=1).frame.tolist() == [2, 49]
  assert load_boxes(str(space), frame_base='1').layout.frame_base == 1


def test_trimmed_detections_always_in_dump_layout(tmp_path):
  """Exported detection files: space separated, frame 0 = range start,
  frame id x y w h conf class quality + extras, whatever the input layout."""
  from annotete.trim import trim_boxes
  user = ("0 2 389 331 342 139 66.3086 5 0.587 -1 -9999 -9999 -9999 -9999\n"
          "0 5 25 346 380 143 74.707 5 0.397 -1 -9999 -9999 -9999 -9999\n"
          "1 11 24 350 383 135 76.6113 5 0.376 -1 -9999 -9999 -9999 -9999\n"
          "1 13 373 324 380 139 66.1621 5 0.508 -1 -9999 -9999 -9999 -9999\n"
          "2 18 24 350 384 135 76.6602 5 0.382 -1 -9999 -9999 -9999 -9999\n"
          "2 20 374 323 381 143 67.1387 5 0.505 -1 -9999 -9999 -9999 -9999\n"
          "3 25 24 350 385 136 75.0977 5 0.375 -1 -9999 -9999 -9999 -9999\n")
  expected = [  # frames 1..3, renumbered from 0, every other token untouched
    "0 11 24 350 383 135 76.6113 5 0.376 -1 -9999 -9999 -9999 -9999",
    "0 13 373 324 380 139 66.1621 5 0.508 -1 -9999 -9999 -9999 -9999",
    "1 18 24 350 384 135 76.6602 5 0.382 -1 -9999 -9999 -9999 -9999",
    "1 20 374 323 381 143 67.1387 5 0.505 -1 -9999 -9999 -9999 -9999",
    "2 25 24 350 385 136 75.0977 5 0.375 -1 -9999 -9999 -9999 -9999"]
  for name, text in (('space', user), ('comma', user.replace(' ', ','))):
    src = tmp_path / f'{name}.txt'
    src.write_text(text)
    out = tmp_path / f'{name}_out.txt'
    assert trim_boxes(load_boxes(str(src)), str(out), 1, 3) == 5
    assert out.read_text().splitlines() == expected, name
  # MOTChallenge det file: 1-based, comma, 10 columns
  mot = tmp_path / 'det.txt'
  mot.write_text('1,-1,10.5,20,30,40,0.9,-1,-1,-1\n2,-1,11.5,20,30,40,0.8,-1,-1,-1\n')
  out = tmp_path / 'mot_out.txt'
  trim_boxes(load_boxes(str(mot)), str(out), 0, 1)
  assert out.read_text().splitlines() == ['0 -1 10.5 20 30 40 0.9 -1 -1 -1 -9999 -9999 -9999 -9999',
                                          '1 -1 11.5 20 30 40 0.8 -1 -1 -1 -9999 -9999 -9999 -9999']
  # another column order (x1 y1 x2 y2, no id / class): rebuilt in the dump layout
  xy = tmp_path / 'xyxy.txt'
  xy.write_text('5 10 20 40 60 0.9\n6 11 20 41 60 0.8\n')
  lay = Layout.from_spec('frame=0,x=1,y=2,w=3,h=4,conf=5')
  lay.box = 'xyxy'
  out = tmp_path / 'xy_out.txt'
  trim_boxes(load_boxes(str(xy), lay), str(out), 5, 6)
  assert out.read_text().splitlines() == ['0 -1 10 20 30 40 0.9 -1 -1 -1 -9999 -9999 -9999 -9999',
                                          '1 -1 11 20 30 40 0.8 -1 -1 -1 -9999 -9999 -9999 -9999']


def test_malformed_lines_are_reported(tmp_path):
  p = tmp_path / 'bad.txt'
  p.write_text('0 1 10 10 20 20 0.9 1 0.5\nnot a box\n1 2 10 10 -5 20 0.9 1 0.5\n')
  d = load_boxes(str(p))
  assert len(d) == 1 and d.warnings


def test_parse_time():
  assert parse_time('120', 25) == 120
  assert parse_time('2s', 25) == 50
  assert parse_time('1:30', 10) == 900
  assert parse_time('0:01:00.5', 10) == 605


def test_pairing(tmp_path):
  vids = [str(tmp_path / f'{n}.mp4') for n in ('gate', 'hall', 'yard')]
  dets = tmp_path / 'dets'
  (dets / 'yard' / 'det').mkdir(parents=True)
  (dets / 'gate.txt').write_text('')
  (dets / 'hall_dets.txt').write_text('')
  (dets / 'yard' / 'det' / 'det.txt').write_text('')
  pairs, warnings = pair_boxes(vids, str(dets))
  assert [os.path.relpath(pairs[v], dets) for v in vids] == ['gate.txt', 'hall_dets.txt', 'yard/det/det.txt']
  with pytest.raises(ValueError):
    pair_boxes(vids, str(dets / 'gate.txt'))


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
  root = tmp_path_factory.mktemp('data')
  make_dataset(str(root), n_videos=2, n_frames=60, n_obj=3)
  return root


def test_project_session_export_trim(dataset, tmp_path):
  p, warnings = Project.create(str(tmp_path / 'proj'), str(dataset / 'videos'), str(dataset / 'dets'), step=10)
  assert not warnings and [i['name'] for i in p.items] == ['cam0', 'cam1']
  s = TrackingSession(p, p.items[0], frames=False)
  assert s.mode == 'detections'
  s.command('new_tracks', dict(frame=0))
  f = 0
  for _ in range(20):  # advance, accepting what it stops at, until the end
    r = s.command('advance', dict(frame=f))
    f = r['result']['frame']
    if f == 59:
      break
    s.command('accept', dict(frame=f))
    s.command('new_tracks', dict(frame=f))
  assert f == 59 and s.ann.stats()['progress'] > 0.95
  s.save()
  assert os.path.exists(p.ann_path(p.items[0]))
  # re-open from disk
  p2 = Project.open(p.root)
  s2 = TrackingSession(p2, p2.items[0], frames=False)
  assert s2.tracks() == s.tracks()
  # trimmed export of frames 20..39
  s2.item['range'] = [20, 39]
  out = tmp_path / 'out'
  assert s2.export(str(out), trimmed=True, formats=('txt', 'mot', 'json'), log=lambda m: None) == []
  ann = (out / 'annotations' / 'cam0.txt').read_text().split('\n')
  frames = sorted({int(l.split()[0]) for l in ann if l})
  assert frames[0] == 0 and frames[-1] == 19
  dets = (out / 'detections' / 'cam0.txt').read_text().split('\n')
  assert {int(l.split()[0]) for l in dets if l} == set(range(20))
  js = json.loads((out / 'annotations_json' / 'cam0.json').read_text())
  assert js['frame_offset'] == 20 and js['tracks']
  cap = cv2.VideoCapture(str(out / 'videos' / 'cam0.mp4'))
  assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
  ok, first = cap.read()
  src = cv2.VideoCapture(p.items[0]['media'])
  src.set(cv2.CAP_PROP_POS_FRAMES, 20)
  _, ref = src.read()
  src.set(cv2.CAP_PROP_POS_FRAMES, 21)
  _, nxt = src.read()
  e0 = np.abs(first.astype(int) - ref.astype(int)).mean()
  e1 = np.abs(first.astype(int) - nxt.astype(int)).mean()
  assert e0 < e1  # the first trimmed frame is source frame 20


def test_server_api(dataset, tmp_path):
  from http.server import ThreadingHTTPServer
  from annotete.server import App, Handler
  p, _ = Project.create(str(tmp_path / 'proj'), str(dataset / 'videos'), str(dataset / 'dets'))
  app = App(p, log=lambda m: None)
  Handler.app = app
  srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  base = f'http://127.0.0.1:{srv.server_address[1]}'

  def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as r:
      data = r.read()
      return json.loads(data) if r.headers.get('Content-Type', '').startswith('application/json') else data

  try:
    assert call('/api/state')['project']['items'][0]['name'] == 'cam0'
    assert b'<canvas' in call('/')
    meta = call('/api/item/cam0/meta')
    assert meta['n_frames'] == 60 and meta['mode'] == 'detections'
    jpg = call('/api/item/cam0/frame/5?w=320&next=6,7')
    assert jpg[:2] == b'\xff\xd8'
    r = call('/api/item/cam0/cmd', dict(op='new_tracks', frame=0))
    assert r['changed'] and len(r['result']) == len(call('/api/item/cam0/boxes?a=0&b=0')['frames']['0']['t'])
    props = call('/api/item/cam0/proposals?f=10')
    assert props['proposals']
    call('/api/item/cam0/cmd', dict(op='accept', frame=10))
    accepted = {p['track']: p['frame'] for p in props['proposals']}
    assert all(t['runs'] == [[0, accepted[t['id']]]] for t in call('/api/item/cam0/tracks')['tracks'] if t['id'] in accepted)
    assert call('/api/item/cam0/cmd', dict(op='undo'))['result'] == 'accept'
    assert len(call('/api/item/cam0/coverage?bins=10')['bins']) == 10
    call('/api/item/cam0/range', dict(range=[5, 30]))
    call('/api/ranges', dict(start='0', duration='1s'))
    assert call('/api/state')['project']['items'][1]['range'] == [0, 24]
    job = call('/api/export', dict(out=str(tmp_path / 'exp'), trimmed=True, formats=['txt']))
    for _ in range(200):
      job = call(f'/api/jobs/{job["id"]}')
      if job['done']:
        break
      threading.Event().wait(0.05)
    assert job['done'] and not job['error'], job
    assert (tmp_path / 'exp' / 'videos' / 'cam1.mp4').exists()
    # errors come back as 400 with a message
    with pytest.raises(urllib.error.HTTPError) as e:
      call('/api/item/cam0/cmd', dict(op='assign', frame=3, det=0))
    assert e.value.code == 400
  finally:
    srv.shutdown()
    app.shutdown()


def test_constant_id_column_is_not_tracks(tmp_path):
  p = tmp_path / 'c.txt'
  p.write_text(''.join(f'{f} 0 {10 + f} 10 20 20 0.9 1 0.5\n{f} 0 100 100 20 20 0.9 1 0.5\n' for f in range(20)))
  assert not load_boxes(str(p)).ids_look_like_tracks()
  q = tmp_path / 't.txt'  # tracker output that uses id 0
  q.write_text(''.join(f'{f} 0 {10 + f} 10 20 20 0.9 1 0.5\n{f} 1 100 100 20 20 0.9 1 0.5\n' for f in range(20)))
  d = load_boxes(str(q))
  assert d.ids_look_like_tracks()
  from annotete.tasks.tracking.annotation import TrackingAnnotation
  a = TrackingAnnotation(d, 20, 640, 480)
  a.import_tracks()
  assert sorted(a.tracks) == [0, 1] and a.tracks[0].n_boxes() == 20


def test_reopening_a_project_uses_the_paths_given_now(dataset, tmp_path):
  import shutil
  # first run by mistake: another videos folder (same names), no box files
  wrong = tmp_path / 'drawn'
  shutil.copytree(dataset / 'videos', wrong)
  root = str(tmp_path / 'proj')
  p, _ = Project.open_or_create(root, str(wrong))
  assert all(i['boxes'] is None and i['media'].startswith(str(wrong)) for i in p.items)
  s = TrackingSession(p, p.items[0], frames=False)
  s.save(force=True)  # an (empty) annotation file exists now
  p.items[1]['status'] = 'done'
  p.save()
  # second run with the right folders into the same project folder
  p2, warnings = Project.open_or_create(root, str(dataset / 'videos'), str(dataset / 'dets'), step=15)
  assert all(i['boxes'] and i['media'].startswith(str(dataset / 'videos')) for i in p2.items)
  assert p2.settings['step'] == 15 and p2.items[1]['status'] == 'done'
  assert Project.open(root).items[0]['boxes']  # also on disk
  s2 = TrackingSession(p2, p2.items[0], frames=False)
  assert len(s2.dets) > 0 and s2.mode == 'detections'
  # real annotations carry over by video name
  s2.command('new_tracks', dict(frame=0))
  s2.save()
  p3, _ = Project.open_or_create(root, str(dataset / 'videos'), str(dataset / 'dets'))
  assert p3.settings['step'] == 15  # None settings keep the saved value
  assert TrackingSession(p3, p3.items[0], frames=False).tracks() == s2.tracks()


def test_export_annotated_video(dataset, tmp_path):
  from annotete.render import id_color
  p, _ = Project.create(str(tmp_path / 'proj'), str(dataset / 'videos'), str(dataset / 'dets'))
  s = TrackingSession(p, p.items[0], frames=False)
  ids = s.command('new_tracks', dict(frame=0))['result']
  s.command('accept', dict(frame=10))
  s.item['range'] = [5, 24]
  out = tmp_path / 'out'
  assert s.export(str(out), trimmed=True, formats=(), log=lambda m: None, video=True) == []
  assert not (out / 'annotations').exists()  # only what was asked for
  cap = cv2.VideoCapture(str(out / 'videos_annotated' / 'cam0.mp4'))
  assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
  cap.set(cv2.CAP_PROP_POS_FRAMES, 3)  # = video frame 8, between the keyframes 0 and 10
  ok, img = cap.read()
  assert ok
  t = s.ann.tracks[ids[0]]
  x, y, w, h = (int(round(v)) for v in t.box(8)[:4])
  # the left edge of the box (below the label) is drawn in the track's color
  edge = img[y + h // 2, max(0, x - 1):x + 2].astype(int)
  assert np.abs(edge - np.array(id_color(ids[0]))).sum(axis=1).min() < 90


def test_unread_request_body_does_not_break_the_next_request(dataset, tmp_path):
  """A POST whose body a route ignores (the page's save beacon) must not leak
  into the next request on the same keep-alive connection."""
  import http.client
  from http.server import ThreadingHTTPServer
  from annotete.server import App, Handler
  p, _ = Project.create(str(tmp_path / 'proj'), str(dataset / 'videos'), str(dataset / 'dets'))
  app = App(p, log=lambda m: None)
  Handler.app = app
  srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try:
    c = http.client.HTTPConnection('127.0.0.1', srv.server_address[1])
    c.request('POST', '/api/save', body='{}', headers={'Content-Type': 'text/plain'})
    assert c.getresponse().read() and True
    c.request('GET', '/api/state')
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())['project']
  finally:
    srv.shutdown()
    app.shutdown()
