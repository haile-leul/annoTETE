"""Annotation tasks. A task is a session class bound to one project item:

  Session(project, item)  -> .meta() .command(op, args) .save() .close()
                             .export(out_dir, trimmed, formats, interp)
plus whatever queries its web UI needs (see server.py for the routes).

The matching web UI lives in annotete/web/js/tasks/<name>/. To add a task (e.g.
'boxes' for per-image detection, 'classify' for clip labels), add a package
here, register it below and add its UI module.
"""
from .tracking.session import TrackingSession

TASKS = {'tracking': TrackingSession}


def session_class(name):
  if name not in TASKS:
    raise ValueError(f'unknown task "{name}" (available: {", ".join(TASKS)})')
  return TASKS[name]
