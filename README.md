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
   * object back after being lost → right click its box → *Give id …*: a list of
     the tracks that are not on this frame, latest seen first (type to filter,
     <kbd>↑</kbd><kbd>↓</kbd> <kbd>Enter</kbd>, or type any id). Or make its old id
     the brush (type `7` <kbd>Enter</kbd>, click `7` in the track list, or click
     its `7 lost` tag) and click its box. The frames in between are filled if the gap is shorter than
     *max interpolation gap*; mark them absent with <kbd>O</kbd> if it was hidden.
   A forward step (<kbd>Space</kbd> or <kbd>D</kbd>) **stops early** on the first frame
   where an object that no track explains shows up (it must last a few frames, so
   one-frame flickers are ignored, and boxes you left free on the previous
   keyframe do not count). The new box is marked orange `new`, so its track starts
   where it is first seen. Turn this off in *Data → Display → stop at new objects*.
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

Zoom with the wheel (at the cursor) and pan by dragging empty space; the frame
can't be dragged out of view. Clicking a box never pans, even if the mouse slips
a little, and two quick clicks are just two clicks. When part of the frame is
outside the view, a **fit** button appears in the bottom-right corner (or double
click empty space, <kbd>Ctrl</kbd>+<kbd>0</kbd>). *Follow* (<kbd>F</kbd>) only moves
the view when you are zoomed in. When zoomed in on a large video the frame is
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
| <kbd>Z</kbd>, <kbd>F</kbd> | zoom to the active track / back to the whole frame, follow it when zoomed in |
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

**Formats**: one box per line,
`frame id x y w h confidence class quality [extra columns …]` (x, y = top-left),
e.g. `0 16 408 662 169 162 71.6309 5 0.625 -1 -9999 -9999 -9999 -9999`.
MOTChallenge files use the same first columns. **All extra columns are kept**
and written back on export.

* **Spaces or commas** between columns are detected per file (spaces after
  commas and tabs are fine too). Exports use the same separator as the input.
* **Frame numbers** (setup page *Frame numbers*, `--frame-base auto|0|1`):
  *detect* uses 0 if any box is on frame 0, and 1 if the last box is on frame
  N for an N-frame video, or if the file is comma separated and starts at
  frame 1 (MOTChallenge). Otherwise it uses 0. *Scan* on the setup page shows
  what was detected and why, and the *Data* tab shows it too.
* Other column orders: `--format "frame=0,id=1,x=2,y=3,w=4,h=5,conf=6,cls=7,quality=8"`
  (or *custom columns* on the setup page).

**Classes**: pick which classes to annotate (`--classes 1,5`, the checkboxes after
*Scan* on the setup page, or any time in *Data → Classes to annotate*); the other
classes' detections are hidden and ignored, and nothing checked means all classes.
The chip in the top bar shows the current choice. Name them with
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

*Export* (or `python -m annotete export --project P [--out O] [--trim] [--formats txt,mot,json] [--video]`).
The output folder defaults to `<project folder>/export`; a folder you pick in the
dialog is remembered for that project only.

```
out/annotations/<video>.txt        same layout as the input, track id in the id column
out/annotations_mot/<video>.txt    MOTChallenge gt: frame(1-based),id,x,y,w,h,1,class,1
out/annotations_json/<video>.json  per track: [frame, x, y, w, h, det_index, kind]
out/videos/<video>.mp4             trimmed export only
out/detections/<video>.txt         trimmed export only: the input boxes of the range (see below)
out/videos_annotated/<video>.mp4   "annotated video" / --video: boxes + ids drawn on the frames
```

`detections/<video>.txt` (also written by `annotete trim`) always uses the detector
dump layout, whatever the input looked like: space separated, frames renumbered so
the first frame of the range is 0,
`frame id x y w h conf class quality -1 -9999 -9999 -9999 -9999`.
Lines already in that column order keep their values (and extra columns) verbatim,
only the frame changes; comma separated, MOT or custom layouts are rebuilt from the
parsed values and padded with `-1 -9999 -9999 -9999 -9999`.

The annotated video covers the same frames as the rest of the export (the range
when trimmed). Each track has the same color as in the app, with its id (and
class name if set) on a label, interpolated boxes are dashed, and the frame number
in the bottom-left corner matches the exported annotation files. It is written as
H.264 through ffmpeg when available, otherwise with OpenCV.

In the `txt` output a box that comes from a detection is that detection's
original line with the track id put in the id column (confidence, class, quality
and extra columns untouched). Interpolated or hand drawn boxes get `-1` in the
columns that have no value (confidence, quality, extras). Uncheck *include
interpolated boxes* (or `--no-interp`) to export only detection backed boxes.

## evaluATE: how good is my tracker?

evaluATE scores tracker outputs against the ground truth you made with annoTETE,
shows where and why the tracker fails, and compares two tracker versions.

```bash
python -m evaluate                                  # opens the evaluATE page
python -m evaluate --gt my_project/export --tracker runs/v2
python -m evaluate --gt my_project/export --tracker runs/v2 --tracker2 runs/v1    # compare
python -m evaluate --gt my_project/export --tracker runs/v2 --report             # text only
```

You can also click **annoTETE ⇄** at the top left of either page to switch between the
two apps. `python -m annotete evaluate …` does the same as `python -m evaluate`; with
`pip install -e .` the command is `evaluatete`.

**Inputs**
* **Ground truth**: an annoTETE export folder. It uses `annotations_mot/` (ground
  truth), `annotations_json/` (which boxes had a detection behind them) and `videos/`
  (the trimmed clips) when they are there. The export dialog ticks MOT + JSON by
  default; tick *trimmed* to get the clips.
* **Tracker output**: one file per video, named like the video (`cam1.txt`,
  `cam1_tracks.txt`, `cam1/…`), in any layout annoTETE reads (space or comma,
  MOTChallenge, 0- or 1-based frames). The frame alignment is detected: the tracker may
  have run on the exported clip or on the full video (the offset is found and shown).
  Only tracker boxes of the annotated classes are used (option: all).

**Scores**: HOTA (with DetA, AssA, DetRe/Pr, AssRe/Pr, LocA), CLEAR (MOTA, MOTP,
IDSW, Frag, MT/PT/ML), Identity (IDF1, IDR, IDP) and counts. This is a
re-implementation of [TrackEval](https://github.com/JonathonLuiten/TrackEval)'s
metrics (`annotete/evaluate/metrics.py`). On random test sequences it gives exactly the
same numbers as the original code; `tests/test_evaluate.py` checks this whenever
TrackEval is installed. It needs only numpy (scipy is used when present).

**Detector vs tracker**: every score is computed twice:
* *All annotations*: every annotated box counts, including the ones you interpolated or
  drew because the detector had nothing.
* *Detector-visible only*: those boxes become "don't care" regions, like TrackEval's
  distractors. Tracker boxes there are neither right nor wrong. This is the best the
  tracker can do with this detector.

The gap between the two is what the detector's misses cost. The *Detector vs tracker*
panel also splits the boxes the tracker missed into **no detection there** (a detector
problem) and **the detector had a box but the tracker dropped it** (a tracker
problem). It shows how many detector gaps the tracker filled by itself.

**Where it goes wrong**: click a video to see it frame by frame:
* tracker boxes in their id's colour with `T7 → obj 3`, false boxes dashed red, missed
  objects dashed (red = the detector had a box, purple = it had none);
* trails and a per-object timeline coloured by the tracker id that had the object, so
  an id switch is a colour change;
* an event list, each event jumping to its frame and zooming to the objects involved:
  * **swap**: two objects exchange tracker ids;
  * **id jump**: a tracker id moves to another object;
  * **new id**: the object restarts with a new id;
  * **switch**;
  * **lost**: the object is not tracked for a while, and the event says whether the
    detector had boxes during the gap;
  * **ghost**: a tracker id that follows nothing.

  Short gaps where only the detector blinked and the tracker kept the id are counted,
  not listed. Keys: <kbd>←</kbd> <kbd>→</kbd> frame, <kbd>Space</kbd> play,
  <kbd>[</kbd> <kbd>]</kbd> previous / next event, <kbd>Esc</kbd> back.

**Comparing two trackers** (*Compare with* = your previous version): every score shows
the change, events are tagged **fixed** (gone in the new version) or **new**
(regressions), and each video row shows the counts.

Each run also saves a report (JSON + CSV, per video and combined, both variants) in
`<export>/evaluation/`.

## Projects and safety

A project is a folder (default `./annotete_projects/<videos name>`, or `--project`):

```
project.json              paths, settings, per video range and status
annotations/<video>.json  keyframes + filled boxes, autosaved (atomic write, plus a .bak copy)
```

Run the same command again (or `--project P`) to continue. The videos and box
files are always the ones you give now (`--videos`/`--dets`, or the setup page),
even when the project folder already exists. From an existing project only the
saved annotations, ranges and statuses carry over, matched by video name, so a
run with the wrong folders is fixed by simply running again with the right ones.
`--project P` alone re-opens P with the paths it was last given. Every edit is one undo
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
  evaluate/             evaluATE: metrics (TrackEval re-implementation), data + alignment,
                        events, runner, API; page annotete/web/evaluate.html, js/evaluate/
evaluate/               `python -m evaluate` (starts annotete.evaluate)
tests/                   pytest: formats, pairing, fill, commands (+ fuzzed undo/redo), export/trim, API, evaluATE
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
