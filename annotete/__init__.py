"""annoTETE: a local, browser-based annotation tool.

Layout (each layer is a small registry so new data / tasks / formats plug in):
  sources/   frame sources (video files, image folders)
  formats/   box file readers / writers (detection lists, MOT, ...)
  tasks/     annotation tasks (tracking, ...): data model + edit commands
  project.py dataset discovery, pairing, settings, persistence
  server.py  stdlib HTTP server + JSON API used by the web UI (web/)
"""
__version__ = '0.1.0'
