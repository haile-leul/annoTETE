"""evaluATE: how well does a tracker do on the ground truth made with annoTETE?

  metrics.py   HOTA, CLEAR (MOTA, IDSW, Frag, MT/ML), Identity (IDF1), Count:
               a re-implementation of TrackEval (same numbers)
  lap.py       optimal assignment (scipy if installed, else numpy)
  data.py      ground truth from an annoTETE export folder, tracker outputs,
               automatic frame alignment, "detector-visible only" variant
  analysis.py  events (swaps, id transfers, new ids, lost tracks, ghost
               tracks) at the frames they happen
  run.py       evaluate one or two trackers, reports
  api.py       routes of the web UI (annotete/web/evaluate.html)
  cli.py       python -m evaluate / python -m annotete.evaluate
"""
