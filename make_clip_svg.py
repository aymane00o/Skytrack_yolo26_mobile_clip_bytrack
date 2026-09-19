"""Turn a stretch of a tracked run into an animated SVG of the real footage.

    python make_clip_svg.py "micro talon.mov" runs/fig_talon.csv docs/talon_clip.svg \
        --start 150 --end 560 --step 6 --width 320

Real frames, with the box the run put on the target drawn in, embedded in one
SVG file that plays on a loop - no video player, no GIF encoder, and it renders
in a browser and in a README.

Each frame is a JPEG inside the SVG, so the file is as big as the frames you ask
for: --width and --step (every Nth frame) are the two knobs. The frames are shown
one at a time by SMIL animation, which every current browser plays; a viewer that
ignores animation shows the first frame.
"""

import argparse
import base64
import csv
import os

import cv2

HELD = ("locked", "reacquired")
GREEN = (80, 220, 120)          # BGR, the colour the box is drawn in
ORANGE = (40, 130, 240)


def boxes_from(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r["frame"])] = (tuple(float(r[k]) for k in "xywh"), r["state"])
    return rows


def frames(video, rows, start, end, step, width, quality, label):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    out, size = [], None
    for index in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        found = rows.get(index)
        if found is not None:
            (x, y, w, h), state = found
            colour = GREEN if state in HELD else ORANGE
            thick = max(round(frame.shape[1] / width), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), colour, thick)
        if label:
            scale = frame.shape[1] / 640
            cv2.putText(frame, f"{'TRACKING' if found else 'SEARCHING'}  frame {index}",
                        (int(16 * scale), int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, GREEN if found else ORANGE, max(int(2 * scale), 1),
                        cv2.LINE_AA)
        scaled = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])),
                            interpolation=cv2.INTER_AREA)
        size = (scaled.shape[1], scaled.shape[0])
        ok, buffer = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise SystemExit("Could not encode a frame")
        out.append(base64.b64encode(buffer).decode("ascii"))
    cap.release()
    if not out:
        raise SystemExit("No frames in that range")
    return out, size


def svg(images, size, fps):
    """One <image> per frame, each visible for its own slot of the loop."""
    width, height = size
    each = 1 / fps
    total = len(images) * each
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    for i, data in enumerate(images):
        # Held off-screen until its moment: opacity rather than display, because
        # that is what renderers animate most consistently.
        begin, end = i / len(images), (i + 1) / len(images)
        # discrete: each value holds until the next keyTime, so a frame is on for
        # its own slot of the loop and off on either side of it.
        if i == 0:
            values, times = "1;0;0", f"0;{end:.6f};1"
        else:
            values, times = "0;1;0;0", f"0;{begin:.6f};{end:.6f};1"
        parts.append(
            f'<image x="0" y="0" width="{width}" height="{height}" opacity="{1 if i == 0 else 0}" '
            f'xlink:href="data:image/jpeg;base64,{data}">'
            f'<animate attributeName="opacity" values="{values}" keyTimes="{times}" '
            f'dur="{total:.3f}s" repeatCount="indefinite" calcMode="discrete"/></image>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("csv", help="telemetry from the run, written with --csv")
    p.add_argument("out", help="where to write the .svg")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--step", type=int, default=6, help="use every Nth frame")
    p.add_argument("--width", type=int, default=320, help="width of the embedded frames")
    p.add_argument("--fps", type=float, default=10, help="how fast the SVG plays them back")
    p.add_argument("--quality", type=int, default=62, help="JPEG quality of each frame")
    p.add_argument("--no-label", action="store_true", help="leave out the corner caption")
    a = p.parse_args()

    images, size = frames(a.video, boxes_from(a.csv), a.start, a.end, a.step, a.width,
                          a.quality, not a.no_label)
    open(a.out, "w").write(svg(images, size, a.fps))
    print(f"{a.out}: {len(images)} frames, {size[0]}x{size[1]}, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
