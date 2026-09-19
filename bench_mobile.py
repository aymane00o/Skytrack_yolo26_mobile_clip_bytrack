"""Measure the mobile-clip settings against full size, on both clips.

    python bench_mobile.py                 # both clips, both settings
    python bench_mobile.py --clips talon --frames 300

The same target is picked on the same frame either way, and each run is scored on:

    speed    frames per second of tracking, start-up (loading the model) excluded
    held     frames with a box on something, of the frames run
    re-locks times the target was lost and taken back

Whether the box is on the right object is not scored - neither clip has a ground
truth - so a contact sheet is written per clip, one row per setting, to be
checked by eye. Runs are sequential, so the timings do not share the CPU.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# name: video, target, first frame, box on it, extra flags
CLIPS = {
    "talon": ("micro talon.mov", "aircraft", 60, "300,985,270,150", []),
    "car": ("car_test.mp4", "car", 300, "291,162,29,28",
            ["--exclude", "0,0,640,50", "--exclude", "0,300,95,60"]),
}
# name: how tall frames are worked on, what the detector is given
SETTINGS = {
    "mobile": ("1080", "640"),     # what skytrack.py ships with
    "small": ("720", "640"),       # half-size frames: faster, loses small targets
    "full": ("1080", "1280"),      # the detector at full size
}


def run(clip, setting, frames, out_csv):
    video, target, start, bbox, extra = CLIPS[clip]
    height, imgsz = SETTINGS[setting]
    cmd = [sys.executable, os.path.join(HERE, "skytrack.py"), video,
           "--target", target, "--start-frame", str(start), "--bbox", bbox,
           "--max-frames", str(frames), "--no-show", "--csv", out_csv,
           "--work-height", height, "--yolo-imgsz", imgsz] + extra
    began = time.perf_counter()
    log = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if log.returncode:
        raise SystemExit(f"{clip}/{setting} failed:\n{log.stderr[-2000:]}")
    return time.perf_counter() - began, log.stdout


def held_boxes(path):
    return {int(r["frame"]): tuple(float(r[k]) for k in "xywh")
            for r in csv.DictReader(open(path)) if r["state"] in ("locked", "reacquired")}


def contact_sheet(video, start, frames, runs, path, tiles=8, width=300):
    cap = cv2.VideoCapture(os.path.join(HERE, video))
    rows = []
    for name, boxes in runs:
        row = []
        for index in np.linspace(start, start + frames - 1, tiles).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if not ok:
                image = np.zeros((360, 640, 3), np.uint8)
            if int(index) in boxes:
                x, y, w, h = (int(v) for v in boxes[int(index)])
                pad = max(image.shape[:2]) // 150
                cv2.rectangle(image, (x - pad, y - pad), (x + w + pad, y + h + pad),
                              (0, 255, 0), max(image.shape[:2]) // 200)
            scale = width / image.shape[1]
            image = cv2.resize(image, (width, int(image.shape[0] * scale)))
            cv2.putText(image, f"{name} {index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
            row.append(image)
        rows.append(np.hstack(row))
    cv2.imwrite(path, np.vstack(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default=",".join(CLIPS))
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--warmup", type=int, default=30,
                   help="frames of a short run timed alone, to take start-up out of the speed")
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    table = ["| clip | settings | speed | held | re-locks |", "| --- | --- | --- | --- | --- |"]
    for clip in a.clips.split(","):
        runs = []
        for setting in a.settings.split(","):
            short, _ = run(clip, setting, a.warmup, os.path.join(a.out, f"{clip}_{setting}_w.csv"))
            out_csv = os.path.join(a.out, f"{clip}_{setting}.csv")
            took, log = run(clip, setting, a.frames, out_csv)
            done = int(re.search(r"Processed (\d+) frames", log).group(1))
            fps = (done - a.warmup) / (took - short) if took - short > 2 else done / took
            boxes = held_boxes(out_csv)
            open(os.path.join(a.out, f"{clip}_{setting}.log"), "w").write(log)
            line = (f"| {clip} | {setting} | {fps:.1f} fps | {len(boxes)}/{done} "
                    f"({100 * len(boxes) / done:.0f}%) | "
                    f"{len(re.findall(r'^Reacquired at frame', log, re.M))} |")
            print(line, flush=True)
            table.append(line)
            runs.append((setting, boxes))
        video, _, start, _, _ = CLIPS[clip]
        contact_sheet(video, start, a.frames, runs, os.path.join(a.out, f"{clip}_sheet.jpg"))

    open(os.path.join(a.out, "results.md"), "w").write("\n".join(table) + "\n")
    print("\n" + "\n".join(table))


if __name__ == "__main__":
    main()
