# Skytrack — YOLO26 + ByteTrack, for phone clips

Pick one target in a video, and follow that target and nothing else — on a CPU,
no GPU. Built for phone footage: tall 1080×1920 clips at 30 fps.

```bash
python skytrack.py "micro talon.mov" --target aircraft
python skytrack.py car_test.mp4 --target car --output tracked.mp4 --csv telemetry.csv
```

Scrub to a frame where the target is clearly visible, press ENTER, drag a box
around it, press ENTER again. Press **R** at any time to draw the box again, **Q**
or **Esc** to quit.

## The pipeline

One target, four pieces, in this order on every frame:

| | What it does |
| --- | --- |
| **YOLO26** | finds every car or aircraft in the frame. `yolo26m.pt`, downloaded on first use |
| **ByteTrack** | gives each one an identity and keeps it through the frames the detector is unsure about, carrying it on a motion prediction when it is not detected at all |
| **Target memory** | remembers what the target looks like, recognises it when it comes back after a gap, and refuses look-alikes that lead by too little |
| **CSRT** | follows the target from frame to frame between detection passes |

The detector runs in a process of its own during a live preview, so the video
plays at its own speed while detection runs behind it. A run writing only
`--output`/`--csv` (`--no-show`) is synchronous and repeatable frame for frame.

What you get per frame: a box with corner brackets, the target's ID, its position
and speed, a motion trail, a prediction arrow, and a magnified inset of the target.
`--csv` writes `frame, time_s, track_id, state, x, y, w, h, cx, cy, speed_px_s` in
the video's own coordinates.

## Why "mobile clip"

A phone clip is tall and large, and on a CPU a detector at full size is the entire
budget. So the detector is given a 640 px square (`--yolo-imgsz`) instead of 1280,
while frames are still worked on at 1080 px tall (`--work-height`) and output and
telemetry stay in the video's own coordinates.

Shrinking the frames as well is the obvious next step, and it does not work: at
720 px the drone clip held its target on a quarter of the frames instead of
almost all of them. The table below measures all three.

Measured with `bench_mobile.py` on the two clips below, headless, one run at a
time so the timings do not share the CPU. `held` counts frames with a box on
something; whether that box is on the right object is a separate question,
answered by eye from the contact sheets the benchmark writes.

_The measured table lands in the commit that follows this one._



Neither clip is in this repository — footage stays local — but both are ordinary
files you can substitute:

* **micro talon** — a fixed-wing VTOL, phone-shot, 1080×1920, 3403 frames: parked
  on a mat, taking off, climbing away until it is a speck against cloud.
* **car test** — a police helicopter chase, 640×360: a silver SUV in heavy
  traffic, with a camera cut at frame 1044 to a wide shot.

## Install

```bash
pip install -r requirements.txt
```

`ultralytics` pulls in `opencv-python`, which shadows `opencv-contrib-python` and
takes the CSRT tracker with it. Repair it afterwards:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-contrib-python
```

## When the target cannot be recognised

After a hard camera cut to a wide shot of look-alike cars, nothing recovers the
right one reliably — measured on the chase clip, the layout memory ranked it first
on 5 of 10 frames, a vehicle re-identification network trained for exactly this on
5 of 10 at that size, and "the car the camera keeps centred" on about 6 of 10. So
the pipeline says so instead of guessing: it shows SEARCHING with the remembered
view, and waits.

That is when you press **R** and draw the box again. The new view joins the memory
alongside the first one, so the target can be recognised from either afterwards.
The run prints the matching `--repick FRAME:X,Y,W,H`, which repeats a hand-corrected
run exactly.

## Options worth knowing

| Flag | Default | For |
| --- | --- | --- |
| `--target car\|aircraft` | `car` | what the detector looks for |
| `--work-height` | 1080 | smaller is faster and loses small targets |
| `--yolo-imgsz` | 640 | detector input size; same trade |
| `--exclude X,Y,W,H` | — | a region the search may never pick from, for a burned-in HUD or logo. Repeatable |
| `--bbox X,Y,W,H --start-frame N` | — | skip the picker and repeat a run exactly |
| `--repick FRAME:X,Y,W,H` | — | make the R-key correction part of the command |
| `--no-show` | off | write `--output`/`--csv` with no window |
| `--no-pace` | off | run as fast as it computes instead of at the video's speed |

## Files

```
skytrack.py      the pipeline: pick the target, then run it
track.py         the tracking loop, lock, search and drawing
bytetrack.py     identities: two-pass association, Kalman motion, camera motion
reid.py          the target's memory, and recognising it when it returns
detector.py      YOLO26 and the contrast detector, scored for ByteTrack
tracker.py       the plain per-track bookkeeping the loop keeps
overlay.py       brackets, labels, trail, arrow, inset
bench_mobile.py  the measurements in this README
```

This is a single-target pipeline lifted out of a larger tracker, keeping only
YOLO26, ByteTrack, the memory and CSRT.
