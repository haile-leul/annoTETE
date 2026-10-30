"""Evaluate one or two tracker outputs against an annoTETE export folder."""
import csv
import json
import os
import time
import uuid

from . import metrics
from .analysis import analyse, compare_events
from .data import list_sequences, load_ground_truth, load_tracker, pair_tracker_files, sequence_data

VARIANTS = ('all', 'det')  # all annotated boxes / only the boxes the detector had
EVENT_TYPES = ('swap', 'transfer', 'new_id', 'switch', 'lost', 'false')


class Evaluation:
  def __init__(self, gt_dir, trackers, shift='auto', classes='gt', log=print):
    """trackers: [(name, path), ...] (one or two)."""
    self.id = uuid.uuid4().hex[:8]
    self.gt_dir = os.path.abspath(os.path.expanduser(gt_dir))
    self.trackers = [(n or os.path.basename(os.path.normpath(p)), os.path.abspath(os.path.expanduser(p)))
                     for n, p in trackers]
    if len({n for n, _ in self.trackers}) < len(self.trackers):
      self.trackers = [(f'{n} ({i + 1})', p) for i, (n, p) in enumerate(self.trackers)]
    self.shift = shift
    self.classes = classes
    self.log = log
    self.warnings = []
    self.seqs = []
    self.details = {}  # (seq, tracker index) -> analysis + metrics
    self.summary = None

  def run(self):
    t0 = time.time()
    if not os.path.isdir(self.gt_dir):
      raise FileNotFoundError(f'no such folder: {self.gt_dir}')
    names = list_sequences(self.gt_dir)
    if not names:
      raise FileNotFoundError(f'{self.gt_dir} has no annotations_mot/ or annotations_json/ (export with MOT or JSON)')
    files = []
    for name, path in self.trackers:
      if not os.path.exists(path):
        raise FileNotFoundError(f'no such tracker output: {path}')
      f = pair_tracker_files(path, names)
      missing = [n for n in names if n not in f]
      if missing:
        self.warnings.append(f'{name}: no tracker output for {len(missing)} video(s) ({", ".join(missing[:6])}'
                             f'{" ..." if len(missing) > 6 else ""}): skipped')
      files.append(f)
    self.seqs = [n for n in names if all(n in f for f in files)]
    if not self.seqs:
      raise ValueError('no video has both ground truth and tracker output (check the file names)')
    results = []
    for ti, (tname, _) in enumerate(self.trackers):
      per_seq, comb = {}, {v: [] for v in VARIANTS}
      det_sum = {}
      for name in self.seqs:
        self.log(f'{tname}: {name}')
        seq = load_ground_truth(self.gt_dir, name)
        self.warnings += [f'{tname}: {w}' for w in load_tracker(seq, files[ti][name], self.shift, self.classes)]
        entry = dict(align=seq.align_note, has_kinds=seq.has_kinds)
        data = sequence_data(seq, 'all')
        res = metrics.evaluate_sequence(data)
        comb['all'].append(res)
        entry['all'] = metrics.summary(res)
        if seq.has_kinds:
          res_det = metrics.evaluate_sequence(sequence_data(seq, 'det'))
          comb['det'].append(res_det)
          entry['det'] = metrics.summary(res_det)
        a = analyse(seq, data)
        entry['detector'] = a['detector']
        for k, v in a['detector'].items():
          det_sum[k] = det_sum.get(k, 0) + v
        entry['events'] = {t: sum(1 for e in a['events'] if e['type'] == t) for t in EVENT_TYPES}
        per_seq[name] = entry
        self.details[(name, ti)] = dict(seq=seq, analysis=a, metrics=entry)
      all_kinds = all(per_seq[n]['has_kinds'] for n in self.seqs)
      combined = dict(all=metrics.summary(metrics.combine(comb['all'])),
                      det=metrics.summary(metrics.combine(comb['det'])) if all_kinds else None,
                      detector=det_sum,
                      events={t: sum(per_seq[n]['events'][t] for n in self.seqs) for t in EVENT_TYPES})
      results.append(dict(name=tname, path=self.trackers[ti][1], combined=combined, seqs=per_seq))
    if len(self.trackers) == 2:
      for name in self.seqs:
        a, b = self.details[(name, 0)], self.details[(name, 1)]
        tol = max(10, int(a['seq'].fps))
        # tracker 0 = current, tracker 1 = previous: previous events gone now are
        # 'fixed', current events the previous one did not have are 'new'
        compare_events(b['analysis']['events'], a['analysis']['events'], tol)
        results[0]['seqs'][name]['new'] = sum(1 for e in a['analysis']['events'] if e.get('diff') == 'new')
        results[0]['seqs'][name]['fixed'] = sum(1 for e in b['analysis']['events'] if e.get('diff') == 'fixed')
    self.summary = dict(id=self.id, gt_dir=self.gt_dir, trackers=[dict(name=n, path=p) for n, p in self.trackers],
                        seqs=self.seqs, warnings=self.warnings, results=results,
                        has_kinds=all(r['seqs'][n]['has_kinds'] for r in results for n in self.seqs),
                        seconds=round(time.time() - t0, 2), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    return self.summary

  def detail(self, name, ti):
    """Everything the video view needs for one video and one tracker."""
    d = self.details[(name, ti)]
    seq, a = d['seq'], d['analysis']
    return dict(name=name, tracker=self.trackers[ti][0], n_frames=seq.n, fps=seq.fps, has_video=bool(seq.video),
                width=seq.width, height=seq.height, align=seq.align_note, frames=a['frames'], gt_rows=a['gt_rows'],
                events=a['events'], metrics=d['metrics'])

  def save_report(self, out_dir=None):
    out_dir = out_dir or os.path.join(self.gt_dir, 'evaluation')
    os.makedirs(out_dir, exist_ok=True)
    stem = '_vs_'.join(n for n, _ in self.trackers).replace('/', '_').replace(' ', '_')
    with open(os.path.join(out_dir, stem + '.json'), 'w') as f:
      json.dump(self.summary, f, indent=1)
    cols = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'MOTP', 'IDSW', 'Frag', 'MT', 'ML', 'CLR_FP', 'CLR_FN', 'IDs',
            'GT_IDs']
    with open(os.path.join(out_dir, stem + '.csv'), 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['tracker', 'video', 'variant'] + cols + ['detector_coverage', 'FN_no_detection', 'FN_tracker_dropped'])
      for r in self.summary['results']:
        rows = [(n, r['seqs'][n]) for n in self.seqs] + [('COMBINED', dict(r['combined'], **{}))]
        for n, e in rows:
          for v in VARIANTS:
            if not e.get(v):
              continue
            d = e['detector']
            cov = d['gt_det'] / d['gt'] if d['gt'] else ''
            w.writerow([r['name'], n, v] + [_fmt(e[v][c]) for c in cols] +
                       [_fmt(cov) if cov != '' else '', d['fn_nodet'], d['fn_det']])
    return out_dir


def _fmt(v):
  return f'{v:.4f}' if isinstance(v, float) else v
