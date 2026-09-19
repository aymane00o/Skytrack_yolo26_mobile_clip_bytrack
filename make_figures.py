"""Draw a run as an SVG: where the target went, and when it was held.

    python make_figures.py runs/fig_talon.csv docs/talon_run.svg --title "micro talon"

Takes the telemetry a run writes with --csv and draws, at the video's own
proportions: the path the target's centre took, light at the start and solid at
the end, a box every so often so its size is visible, and a strip underneath
with one mark per frame - filled where the target was held, blank where the run
was searching for it. Vector, so it stays sharp at any size.
"""

import argparse
import csv
import os

import cv2

HELD = ("locked", "reacquired")
INK = "#1f2933"          # text and frame: dark, but not black, on either theme
TRACK = "#0b8f5a"        # the path and the held marks
LOST = "#c2410c"         # frames with no box
PAPER = "#f4f6f8"


def read(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"frame": int(r["frame"]), "state": r["state"],
                     "box": tuple(float(r[k]) for k in "xywh"),
                     "centre": (float(r["cx"]), float(r["cy"]))})
    return rows


def video_size(rows, video):
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w and h:
            return w, h
    # Fall back to what the boxes cover.
    w = max(r["box"][0] + r["box"][2] for r in rows)
    h = max(r["box"][1] + r["box"][3] for r in rows)
    return int(w * 1.05), int(h * 1.05)


def svg(rows, size, title, every, ran=0):
    width, height = size
    strip = max(height // 14, 24)          # the timeline under the frame
    pad = max(width // 30, 12)
    total = width + 2 * pad, height + strip + 3 * pad
    held = [r for r in rows if r["state"] in HELD]
    first = rows[0]["frame"]
    # A run writes no row at all for a frame it had nothing to report, so the
    # length of the run is what the timeline covers - not the last row.
    last = max(rows[-1]["frame"], first + ran - 1)
    span = max(last - first, 1)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total[0]} {total[1]}" '
           f'width="{total[0]}" height="{total[1]}" font-family="Segoe UI, Helvetica, sans-serif">',
           f'<rect width="{total[0]}" height="{total[1]}" fill="{PAPER}"/>',
           f'<rect x="{pad}" y="{pad}" width="{width}" height="{height}" fill="#ffffff" '
           f'stroke="{INK}" stroke-opacity="0.25"/>']

    # The path, in segments, so a gap where the target was lost shows as a gap.
    run = []
    for row in rows:
        if row["state"] in HELD:
            run.append(row)
            continue
        if len(run) > 1:
            out.append(path_of(run, pad))
        run = []
    if len(run) > 1:
        out.append(path_of(run, pad))

    # A box every `every` held frames, so the target's size is visible.
    for row in held[::every]:
        x, y, w, h = row["box"]
        out.append(f'<rect x="{x + pad:.1f}" y="{y + pad:.1f}" width="{w:.1f}" height="{h:.1f}" '
                   f'fill="none" stroke="{TRACK}" stroke-opacity="0.55" stroke-width="2"/>')
    if held:
        for row, label, fill in ((held[0], "start", "#ffffff"), (held[-1], "end", TRACK)):
            cx, cy = row["centre"]
            out.append(f'<circle cx="{cx + pad:.1f}" cy="{cy + pad:.1f}" r="{max(width // 120, 4)}" '
                       f'fill="{fill}" stroke="{TRACK}" stroke-width="3"/>')
            out.append(f'<text x="{cx + pad + max(width // 60, 10):.1f}" y="{cy + pad:.1f}" '
                       f'font-size="{max(width // 34, 13)}" fill="{INK}">{label} {row["frame"]}</text>')

    # The timeline: one mark per frame, filled where the target was held.
    top = height + 2 * pad
    out.append(f'<rect x="{pad}" y="{top}" width="{width}" height="{strip}" fill="#ffffff" '
               f'stroke="{INK}" stroke-opacity="0.25"/>')
    step = width / (span + 1)
    out.append(f'<rect x="{pad}" y="{top + 1}" width="{width}" height="{strip - 2}" '
               f'fill="{LOST}" fill-opacity="0.9"/>')
    for row in held:
        x = pad + (row["frame"] - first) * step
        out.append(f'<rect x="{x:.2f}" y="{top + 1}" width="{max(step, 0.6):.2f}" '
                   f'height="{strip - 2}" fill="{TRACK}"/>')

    kept = f"{len(held)} of {span + 1} frames held"
    out.append(f'<text x="{pad}" y="{pad * 0.75:.0f}" font-size="{max(width // 30, 15)}" '
               f'font-weight="600" fill="{INK}">{title}</text>')
    out.append(f'<text x="{pad}" y="{top + strip + pad * 0.9:.0f}" '
               f'font-size="{max(width // 38, 12)}" fill="{INK}" fill-opacity="0.75">'
               f'frames {first}-{last} &#183; {kept} &#183; green = on target, '
               f'orange = searching</text>')
    out.append("</svg>")
    return "\n".join(out)


def path_of(run, pad):
    points = " ".join(f"{r['centre'][0] + pad:.1f},{r['centre'][1] + pad:.1f}" for r in run)
    return (f'<polyline points="{points}" fill="none" stroke="{TRACK}" stroke-width="3" '
            f'stroke-linejoin="round" stroke-linecap="round" stroke-opacity="0.85"/>')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="telemetry written by skytrack.py --csv")
    p.add_argument("out", help="where to write the .svg")
    p.add_argument("--title", default="run")
    p.add_argument("--video", help="the clip it came from, to size the drawing exactly")
    p.add_argument("--every", type=int, default=30, help="draw the box every N held frames")
    p.add_argument("--frames", type=int, default=0,
                   help="how many frames the run covered, when it stopped writing rows before "
                        "the end - the timeline then shows the rest as searching")
    a = p.parse_args()
    rows = read(a.csv)
    if not rows:
        raise SystemExit(f"{a.csv} has no frames")
    open(a.out, "w").write(svg(rows, video_size(rows, a.video), a.title, a.every, a.frames))
    print(f"{a.out}: {len(rows)} frames")


if __name__ == "__main__":
    main()
