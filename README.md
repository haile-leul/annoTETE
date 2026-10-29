# annoTETE

A local, browser based annotation tool. The first task is **multi-object tracking**:
give track ids to detections, or clean up a tracker's output, on one video or a
whole folder of them.

* Pure Python + plain JavaScript. Only needs `numpy` and `opencv-python`; the web
  server is the Python standard library, and the UI has no build step.
* Everything stays on your machine; work is autosaved every few seconds.

```bash
pip install -r requirements.txt        # numpy, opencv-python (ffmpeg on PATH is optional, for trimming)
python -m annotete --videos /data/videos --dets /data/dets
```

The browser opens at <http://localhost:8765>. Running `python -m annotete` with no
arguments opens a setup page where you choose the folders and preview how videos
and box files pair up. You can also install the `annotete` command with `pip install -e .`.

---

## The two workflows

### 1. Detections → tracks (assign ids)

You have a detection list per video (ids unique per box). You do **not** have to
click every frame:

1. On the first frame press <kbd>Shift</kbd>+<kbd>N</kbd>: every detection becomes a new track.
   Or click detections one by one (a click on a white box starts a track).
2. Press <kbd>Space</kbd>. The view jumps one **keyframe step** (default 10 frames)
   ahead and shows **proposals**: dashed boxes where each track probably is now.
   They come from following every track through the skipped frames over your
   detections (IoU + velocity, all tracks at once, so two tracks never take the
   same box).
3. Fix only what is wrong, then press <kbd>Space</kbd> again (accept + next step):
   * wrong proposal → click its tag (`7?`) to make track 7 the brush, then click
     the right box. Two clicks.
   * two ids swapped → click one, <kbd>X</kbd>, click the other.
   * new object → <kbd>Shift</kbd>+click it (or <kbd>Shift</kbd>+<kbd>N</kbd> for all free boxes).
   * object left → do nothing: a track simply ends at its last keyframe.
4. <kbd>Shift</kbd>+<kbd>Space</kbd> **auto-advances**: it keeps accepting confident
   proposals step after step and stops only where a human is needed (an unsure
   proposal, a lost track, a new object, or the end of the range).

The frames between two keyframes are filled automatically: the boxes are
interpolated and then **snapped to the real detections** with a Viterbi pass over
the whole segment (a smooth chain of detections wins over the box that merely
sits closest to a straight line; missed detections stay interpolated). Exported
boxes are therefore your exact detector boxes wherever a detection exists.

On a synthetic test (12 crossing objects, 500 frames, ~5.5k boxes, step 15) a
simulated annotator needed **34 clicks and 24 key presses** for 99.7% correct boxes;
with 6 objects it needed no correction at all.

Want a head start? *Data → Build tracklets* runs a cautious IoU tracker over the
free detections (it cuts rather than guesses), then you merge fragments from the
Review tab.

### 2. Tracker output → clean tracks (correct)

If the id column repeats across frames (MOT style tracker output) the file is
imported as tracks (mode `auto`; force with `--mode tracks|detections`).
Play (<kbd>P</kbd>) or scrub the timeline, and fix:

| problem | fix |
|---|---|
| ID switch | select one track, <kbd>X</kbd>, click the other: ids are swapped from this frame on |
| one object split in two tracks | select the first, <kbd>M</kbd>, click the second (or one click in Review) |
| two objects in one track | <kbd>K</kbd> at the frame where it changes: a new id from here on |
| wrong id number | <kbd>E</kbd> (an existing id merges the two) |
| box wrong on a frame | drag it / its handles, or click the right detection with the brush |
| false track | <kbd>Ctrl</kbd>+<kbd>Del</kbd> |
| gap the tracker left | <kbd>G</kbd> fills it (interpolation + snapping) |

The **Review** tab lists the frames most likely to need you: *jumps* (a sudden
box jump inside a track, usually an ID switch), *fragments* (a track ends and
another starts close by: merge with one click), *missed* objects (detections
that stay unassigned over several frames), long *interpolated* runs and very
*short* tracks. <kbd>[</kbd> / <kbd>]</kbd> walk through them.

---

## The screen

```
 top bar   video picker · status (to do / in progress / done) · brush · undo/redo · Export
 canvas    the frame + boxes            sidebar   Tracks · Review · Data
 bottom    transport · frame · keyframe step · range [in out] · timeline
```

* **White dashed** box: free detection (no id yet).
* **Colored** box: a track. A corner flag marks a keyframe; solid = matched
  detection; dashed = interpolated (no detection behind it).
* **Dashed + filled**: proposal (orange `?` = unsure, `(missed)` = the detection
  is missing on this frame and the proposal uses the last frame where it was seen).
* **Brush** (top bar): the active track. Clicking a free detection gives it the
  brush id. <kbd>Esc</kbd> clears the brush (next click = new track); type an id
  (`12` <kbd>Enter</kbd>) to pick any id, existing or new.
* **Timeline**: red strip = frames with unassigned detections left; one row per
  track (click to select + jump); diamonds = keyframes of the active track;
  green ticks = keyframe steps; shaded = outside the range. Wheel zooms, drag scrubs.

Zoom with the wheel (at the cursor), pan by dragging empty space, double click
fits the view or zooms to a box. When zoomed in on a large video the frame is
re-fetched at native resolution, so small objects in 4K stay sharp.

### Shortcuts (<kbd>?</kbd> in the app)

| | |
|---|---|
| <kbd>→</kbd> <kbd>←</kbd> | ±1 frame (<kbd>Shift</kbd> ±10) |
| <kbd>D</kbd> <kbd>A</kbd> | ±1 keyframe step (<kbd>Shift</kbd> ±5 steps) |
| <kbd>Space</kbd> / <kbd>Shift</kbd>+<kbd>Space</kbd> | accept proposals + next step / auto-advance |
| <kbd>Enter</kbd> | accept proposals here |
| click / <kbd>Shift</kbd>+click | give the brush id / new track |
| <kbd>Shift</kbd>+<kbd>N</kbd> | every free detection here → new track |
| <kbd>N</kbd>, <kbd>Esc</kbd> | brush = new id, clear / cancel |
| `12` <kbd>Enter</kbd> | brush = #12 (or give #12 to the hovered detection) |
| <kbd>Tab</kbd> | cycle the active track among boxes on this frame |
| <kbd>X</kbd> <kbd>M</kbd> <kbd>K</kbd> <kbd>E</kbd> | swap from here, merge, split here, change id |
| <kbd>Del</kbd> | delete the box (a keyframe is removed and the frame re-interpolated) |
| <kbd>O</kbd> | absent (occluded / out of view) until the next keyframe |
| <kbd>Shift</kbd>+<kbd>Del</kbd>, <kbd>Ctrl</kbd>+<kbd>Del</kbd> | track ends here, delete track |
| <kbd>G</kbd> | fill gaps of the active track |
| <kbd>B</kbd> or <kbd>Ctrl</kbd>+drag | draw a box (missed detection) |
| <kbd>Alt</kbd>+<kbd>→</kbd> / <kbd>←</kbd> | next / previous keyframe of the active track |
| <kbd>P</kbd>, <kbd>Home</kbd> / <kbd>End</kbd> | play, range start / end |
| <kbd>Z</kbd>, <kbd>F</kbd> | zoom to the active track, follow it |
| <kbd>L</kbd> <kbd>H</kbd> <kbd>T</kbd> | labels, free detections, trails |
| <kbd>[</kbd> <kbd>]</kbd> | previous / next review issue |
| <kbd>Shift</kbd>+<kbd>I</kbd> / <kbd>Shift</kbd>+<kbd>O</kbd> | range in / out at this frame |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>, <kbd>Ctrl</kbd>+<kbd>S</kbd> | undo / redo, save now |

---

## Input

**Videos**: a video file, a folder of videos (mp4, avi, mkv, mov, …), an image
folder (one image per frame), or MOTChallenge style `seq/img1` folders.

**Box files**: one file for a single video, or a folder. For a video `gate.mp4`
the first match wins: `gate.txt`, `gate_<anything>.txt` (e.g. `gate_dets.txt`),
`gate/det/det.txt`, `gate/gt/gt.txt`. Videos without boxes can still be
annotated by drawing.

**Formats** (`--format`, default `auto`):

* `txt`: whitespace separated, 0-based frames:
  `frame id x y w h confidence class quality [extra columns …]`
  (x, y = top-left). This is the detector dump format, e.g.
  `0 16 408 662 169 162 71.6309 5 0.625 -1 -9999 -9999 -9999 -9999`.
  **All extra columns are kept** and written back on export.
* `mot`: MOTChallenge, comma separated, 1-based frames (auto detected by commas).
* custom columns: `--format "frame=0,id=1,x=2,y=3,w=4,h=5,conf=6,cls=7,quality=8"`.

**Classes**: pick which classes to annotate (`--classes 1,5`, at setup, or any
time in *Data*); others are hidden and ignored. Name them with
`--class-names "1=person,5=car"` or in *Data*. A minimum confidence
(`--min-conf`, or the slider) hides weak detections.

The app warns when the boxes do not fit the video (most boxes outside the frame,
or frames past the end): usually a wrong pairing, frame base or resolution.

## Trimming long videos

* **In the app**: set a range per video with <kbd>Shift</kbd>+<kbd>I</kbd> /
  <kbd>Shift</kbd>+<kbd>O</kbd> (or the `[ in` `out ]` buttons), or *all videos…*
  for the same start + duration everywhere (times as frames `2250`, seconds `90s`
  or `m:s` `1:30`). The range is your work area. *Export → trimmed* then writes
  the cut videos, their annotations and their input boxes, renumbered from frame 0.
* **Before annotating**, from the command line (one video or a whole folder):

  ```bash
  python -m annotete trim --videos /data/videos --dets /data/dets --out /data/short --start 1:00 --duration 30s
  python -m annotete --videos /data/short/videos --dets /data/short/detections
  ```

Cutting is frame exact (ffmpeg `trim` filter, re-encoded with x264 CRF 17; without
ffmpeg OpenCV is used) and the output frame count is verified.

## Output

*Export* (or `python -m annotete export --project P --out O [--trim] [--formats txt,mot,json]`):

```
out/annotations/<video>.txt        same layout as the input, track id in the id column
out/annotations_mot/<video>.txt    MOTChallenge gt: frame(1-based),id,x,y,w,h,1,class,1
out/annotations_json/<video>.json  per track: [frame, x, y, w, h, det_index, kind]
out/videos/<video>.mp4             trimmed export only
out/detections/<video>.txt         trimmed export only: the input boxes of the range
```

In the `txt` output a box that comes from a detection is that detection's
original line with the track id put in the id column (confidence, class, quality
and extra columns untouched). Interpolated or hand drawn boxes get `-1` in the
columns that have no value (confidence, quality, extras). Uncheck *include
interpolated boxes* (or `--no-interp`) to export only detection backed boxes.

## Projects and safety

A project is a folder (default `./annotete_projects/<videos name>`, or `--project`):

```
project.json              paths, settings, per video range and status
annotations/<video>.json  keyframes + filled boxes, autosaved (atomic write, plus a .bak copy)
```

Run the same command again (or `--project P`) to continue. Every edit is one undo
step (up to 300); a failing edit is rolled back as a whole. Annotations remember
which detection each box came from and re-check those links if the box file changes.

Working on a remote machine: run it there and tunnel the port,
`ssh -L 8765:localhost:8765 gpu-box`, then open <http://localhost:8765>
(`--host 0.0.0.0` also works but has no authentication).

## Code layout (and how to extend it)

```
annotete/
  cli.py                serve / trim / export / info
  server.py             stdlib HTTP server + JSON API
  project.py            discovery, video ↔ box file pairing, settings, autosave
  sources/              frame sources: video (OpenCV), image folders; JPEG cache + prefetch
  formats/              box file readers / writers (column layouts, MOT)
  trim.py               frame exact cutting of videos and box files
  tasks/
    tracking/
      model.py          Track = keyframes + automatic fill
      fill.py           interpolation + Viterbi snapping to detections
      propose.py        proposals / auto-advance / bootstrap tracker
      review.py         quality checks
      annotation.py     all edit commands, ownership index, undo
      session.py        one opened video: load, save, export
  web/
    index.html, css/, js/  app shell, viewer (zoom/pan), timeline, setup
    js/tasks/tracking/     the tracking UI (interactions, shortcuts, panels)
tests/                   pytest: formats, pairing, fill, commands (+ fuzzed undo/redo), export/trim, API
```

* **New data type** (e.g. a stereo rig): a class in `sources/` with
  `n_frames, fps, width, height, read(i)`.
* **New file format** (e.g. COCO json): a reader in `formats/` returning `Detections`.
* **New task** (e.g. per-frame boxes, attributes, clip labels): a package in `tasks/`
  with a session class (`meta`, `command`, `save`, `export`), registered in
  `tasks/__init__.py`, and its UI module in `annotete/web/js/tasks/<name>/`.

Run the tests with `python -m pytest tests`.

## Tips

* Pick the step from the motion: 10–15 for busy scenes, 30–50 for slow ones.
  You can change it any time (bottom bar).
* Seeking backwards in long-GOP videos is the slowest operation (the decoder
  starts from the previous key frame); frames you visited are cached, and stepping
  forward is always fast.
* The red strip in the timeline shows where unassigned detections remain;
  *Data* shows how many detections have an id.
