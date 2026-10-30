"""What happened where: events and per frame match status, from the same
CLEAR matching that the IDSW / Frag numbers come from (so the events and the
counts agree).

Events (all with the frame they happen at):
  swap       two objects exchange their tracker ids (crossing objects)
  transfer   a tracker id jumps from one object to another (hijack)
  new_id     the object gets a new tracker id (track was lost, restarted)
  switch     the object continues under another existing tracker id
  lost       the object stops being tracked for a while (fragmentation);
             says whether the detector had boxes during the gap. Short gaps
             where the detector simply had no box and the same id continues
             are detector flicker: counted (detector['flicker']), not listed
  false      a tracker id that follows nothing for a while (ghost track)
"""
import numpy as np

from .data import KIND_CODE, NODET_KINDS
from .metrics import clear_matches


def analyse(seq, data, threshold=0.5, min_false=None, min_gap=None):
  """seq: Sequence; data: sequence_data(seq, 'all'). Returns dict with
  frames (boxes + status), gt_rows (timeline segments), events, detector."""
  min_false = min_false or max(5, int(seq.fps // 2))
  min_gap = min_gap or max(3, int(round(seq.fps / 5)))
  frames = []
  events = []
  last_tr = {}       # gt -> last tracker id (any time)
  tr_owner = {}      # tracker id -> last gt it followed
  gap_start = {}     # gt -> first unmatched frame of the current gap (after being tracked)
  gap_det = {}       # gt -> frames of the gap where the detector had a box
  fp_run = {}        # tracker id -> [first frame, frames] of the current unmatched run
  seg = {}           # gt -> list of [a, b, value]
  gt_last = {}       # gt -> last frame it exists
  det = dict(gt=0, gt_det=0, gt_interp=0, gt_manual=0, tp=0, tp_nodet=0, fn=0, fn_det=0, fn_nodet=0, flicker=0,
             flicker_frames=0)

  def put(gid, t, val):
    s = seg.setdefault(gid, [])
    if s and s[-1][1] == t - 1 and s[-1][2] == val:
      s[-1][1] = t
    else:
      s.append([t, t, val])

  def close_fp(tid, t):
    run = fp_run.pop(tid, None)
    if run and run[1] >= min_false:
      events.append(dict(type='false', frame=run[0], end=t - 1, tracks=[], trk=[int(tid)],
                         msg=f'tracker id {tid} follows nothing for {run[1]} frames'))

  for t, rows, cols in clear_matches(data, threshold):
    g = seq.gt[t]
    k = seq.trk[t]
    gkeep, tkeep = data['_gt_keep'][t], data['_tr_keep'][t]
    gidx = np.nonzero(gkeep)[0]
    tidx = np.nonzero(tkeep)[0]
    gt_match = {int(gidx[r]): int(tidx[c]) for r, c in zip(rows, cols)}
    tr_match = {v: kk for kk, v in gt_match.items()}
    # --- boxes with status
    fb_g, fb_t = [], []
    for i, (gid, box, kind, cls) in enumerate(g):
      j = gt_match.get(i)
      tid = k[j][0] if j is not None else None
      nodet = kind in NODET_KINDS
      gt_last[gid] = t
      det['gt'] += 1
      det['gt_det'] += kind in ('key', 'auto')
      det['gt_interp'] += kind == 'interp'
      det['gt_manual'] += kind == 'manual'
      if j is None:
        det['fn'] += 1
        det['fn_nodet' if nodet else 'fn_det'] += 1
        put(gid, t, -2 if nodet else -1)
      else:
        det['tp'] += 1
        det['tp_nodet'] += nodet
        put(gid, t, int(tid))
      fb_g.append([int(gid), *[round(v, 1) for v in box], KIND_CODE.get(kind, -1), int(tid) if tid is not None else None])
    for j, (tid, box) in enumerate(k):
      i = tr_match.get(j)
      fb_t.append([int(tid), *[round(v, 1) for v in box], int(g[i][0]) if i is not None else None])
    frames.append(dict(g=fb_g, t=fb_t))

    # --- identity events
    switched = {}
    for i, j in gt_match.items():
      gid, tid = g[i][0], k[j][0]
      prev = last_tr.get(gid)
      if gid in gap_start:  # tracked again after a gap
        a = gap_start.pop(gid)
        n_det = gap_det.pop(gid, 0)
        how = 'same id' if prev == tid else f'id {prev} → {tid}'
        if n_det == 0 and prev == tid and t - a < min_gap:
          det['flicker'] += 1  # the detector blinked, the tracker carried on: not the tracker's fault
          det['flicker_frames'] += t - a
        else:
          events.append(dict(type='lost', frame=a, end=t - 1, tracks=[int(gid)], trk=[int(prev), int(tid)],
                             det_frames=n_det, gap=t - a,
                             msg=f'object {gid} not tracked for {t - a} frame(s) ({_det_note(n_det, t - a)}); back with {how}'))
      if prev is not None and prev != tid:
        switched[gid] = (prev, tid)
      last_tr[gid] = tid
    for gid, (prev, tid) in list(switched.items()):
      if gid not in switched:  # already reported as the other half of a swap
        continue
      other = next((g2 for g2, (p2, t2) in switched.items() if g2 != gid and p2 == tid and t2 == prev), None)
      if other is not None:
        events.append(dict(type='swap', frame=t, tracks=[int(gid), int(other)], trk=[int(prev), int(tid)],
                           msg=f'objects {gid} and {other} swap tracker ids {prev} ↔ {tid}'))
        switched.pop(other)
      elif tid not in tr_owner:
        events.append(dict(type='new_id', frame=t, tracks=[int(gid)], trk=[int(prev), int(tid)],
                           msg=f'object {gid} gets a new tracker id: {prev} → {tid}'))
      elif tr_owner[tid] != gid:
        events.append(dict(type='transfer', frame=t, tracks=[int(gid), int(tr_owner[tid])], trk=[int(prev), int(tid)],
                           msg=f'tracker id {tid} jumps from object {tr_owner[tid]} to object {gid} (which had {prev})'))
      else:
        events.append(dict(type='switch', frame=t, tracks=[int(gid)], trk=[int(prev), int(tid)],
                           msg=f'object {gid} switches back from tracker id {prev} to {tid}'))
    for i, j in gt_match.items():
      tr_owner[k[j][0]] = g[i][0]
    # --- gaps (unmatched gt that were tracked before)
    for i, (gid, box, kind, cls) in enumerate(g):
      if i in gt_match or gid not in last_tr:
        continue
      gap_start.setdefault(gid, t)
      if kind not in NODET_KINDS:
        gap_det[gid] = gap_det.get(gid, 0) + 1
    # --- false tracks
    present = set()
    for j, (tid, box) in enumerate(k):
      present.add(tid)
      if j in tr_match:
        close_fp(tid, t)
      else:
        run = fp_run.setdefault(tid, [t, 0])
        run[1] += 1
    for tid in [x for x in fp_run if x not in present]:
      close_fp(tid, t)
  for tid in list(fp_run):
    close_fp(tid, seq.n)
  for gid, a in gap_start.items():  # never recovered
    n_det = gap_det.get(gid, 0)
    end = gt_last.get(gid, a)
    if n_det == 0 and end - a + 1 < min_gap:
      det['flicker'] += 1
      det['flicker_frames'] += end - a + 1
      continue
    events.append(dict(type='lost', frame=a, end=end, tracks=[int(gid)], trk=[int(last_tr[gid])], det_frames=n_det,
                       gap=end - a + 1, msg=f'object {gid} lost at frame {a} and not tracked again '
                                            f'({_det_note(n_det, end - a + 1)})'))
  events.sort(key=lambda e: (e['frame'], e['type']))
  gt_rows = [dict(id=int(gid), segs=s) for gid, s in sorted(seg.items(), key=lambda kv: (kv[1][0][0], kv[0]))]
  return dict(frames=frames, gt_rows=gt_rows, events=events, detector=det)


def _det_note(n_det, n):
  if not n:
    return ''
  if n_det == 0:
    return 'the detector had no box for it'
  if n_det == n:
    return 'the detector had a box every frame: tracker dropped it'
  return f'the detector had a box in {n_det} of them'


def compare_events(ev_a, ev_b, tol):
  """Mark events of A that B no longer has ('fixed') and events of B that A
  did not have ('new'). Two events match if same type family, a shared object
  and frames within `tol`."""
  fam = lambda e: 'id' if e['type'] in ('swap', 'transfer', 'new_id', 'switch') else e['type']

  def has(ev, e):
    return any(fam(x) == fam(e) and abs(x['frame'] - e['frame']) <= tol and
               (set(x['tracks']) & set(e['tracks']) or (not x['tracks'] and not e['tracks'] and
                                                        abs(x.get('end', x['frame']) - e.get('end', e['frame'])) <= tol))
               for x in ev)
  for e in ev_a:
    e['diff'] = None if has(ev_b, e) else 'fixed'
  for e in ev_b:
    e['diff'] = None if has(ev_a, e) else 'new'
