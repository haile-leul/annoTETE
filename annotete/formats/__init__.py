"""Box file formats.

`load_boxes(path, layout)` reads any column based text file (see boxtxt.py).
To add a format (e.g. COCO json), add a module exposing a loader that returns
a `Detections` object and register it in READERS.
"""
from .boxtxt import PRESETS, BoxWriter, Detections, Layout, file_signature, load_boxes, mot_line, sniff_preset

READERS = {'txt': load_boxes, 'mot': load_boxes}

__all__ = ['PRESETS', 'READERS', 'BoxWriter', 'Detections', 'Layout', 'file_signature', 'load_boxes', 'mot_line',
           'sniff_preset']
