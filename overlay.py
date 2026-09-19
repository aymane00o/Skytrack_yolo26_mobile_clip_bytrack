"""HUD rendering: bracket boxes, telemetry labels, magnified inset."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LOCKED = (60, 60, 255)
TENTATIVE = (0, 200, 255)
COASTING = (200, 120, 255)
VISUAL = (0, 165, 255)
PALETTE = [(60, 60, 255), (80, 200, 80), (255, 180, 60), (255, 90, 200), (60, 230, 230)]


def track_color(track):
    if track.visual:
        return VISUAL
    if track.coasting:
        return COASTING
    if not track.locked:
        return TENTATIVE
    return PALETTE[track.id % len(PALETTE)]


def draw_brackets(frame, bbox, color, arm=None, thickness=2, pad=6):
    x, y, w, h = [int(v) for v in bbox]
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    arm = arm or max(6, min(w, h) // 3)
    for cx, sx in ((x, 1), (x + w, -1)):
        for cy, sy in ((y, 1), (y + h, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * arm, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * arm), color, thickness, cv2.LINE_AA)
    cv2.drawMarker(frame, (x + w // 2, y + h // 2), color, cv2.MARKER_CROSS, 7, 1)


def draw_label(frame, anchor, lines, color, scale=0.4):
    x, y = int(anchor[0]), int(anchor[1])
    pad, line_h = 5, 14
    width = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines) + 2 * pad
    height = line_h * len(lines) + pad

    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw - width - 1))
    y = max(0, min(y, fh - height - 1))

    panel = frame[y:y + height, x:x + width]
    cv2.addWeighted(panel, 0.25, np.full_like(panel, 25), 0.75, 0, panel)
    for i, text in enumerate(lines):
        col = color if i == 0 else (225, 225, 225)
        cv2.putText(frame, text, (x + pad, y + pad + 9 + i * line_h), FONT, scale, col, 1, cv2.LINE_AA)


def draw_prediction(frame, track, color, seconds=0.35):
    if track.speed < 5:
        return
    start = tuple(track.center.astype(int))
    end = tuple((track.center + track.velocity * seconds).astype(int))
    cv2.arrowedLine(frame, start, end, color, 1, cv2.LINE_AA, tipLength=0.25)


def draw_trail(frame, track, color):
    if len(track.trail) < 2:
        return
    pts = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)


def draw_inset(frame, bbox, source=None, size=200, margin=10, context=2.2, min_zoom=3.0):
    """Magnified crop of the primary target, pinned bottom-right.

    Pass the unannotated frame as `source` so the inset shows the target itself
    rather than the HUD already painted over it.

    The crop is sized from the target rather than fixed, so a distant aircraft a
    dozen pixels across is blown up until it can actually be made out, instead of
    staying the same speck it is in the full frame. A target already large needs
    no help and is shown with a little room around it.
    """
    source = frame if source is None else source
    fh, fw = frame.shape[:2]
    size = min(size, fw // 3, fh // 3)
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0

    span = max(w, h, 1.0)
    half = max(span * context / 2.0, 16.0)
    # Never let the crop grow so wide that the target comes out smaller than
    # min_zoom times life size - magnifying it is the point of the panel.
    half = min(half, max(span * 0.75, size / (2.0 * min_zoom)))

    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, fw)), int(min(cy + half, fh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    grow = size > (x1 - x0)
    crop = cv2.resize(source[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_CUBIC if grow else cv2.INTER_AREA)

    px, py = fw - size - margin, fh - size - margin
    frame[py:py + size, px:px + size] = crop
    cv2.rectangle(frame, (px, py), (px + size, py + size), LOCKED, 2)

    sx, sy = size / (x1 - x0), size / (y1 - y0)
    bx, by = px + int((x - x0) * sx), py + int((y - y0) * sy)
    bw, bh = max(int(w * sx), 3), max(int(h * sy), 3)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), LOCKED, 1)
    cv2.putText(frame, f"LIVE x{size / float(x1 - x0):.1f}", (px + 8, py + size - 8),
                FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

def draw_ghost(frame, bbox, center, radius):
    """Where the target was last held, and how wide the search has opened up.

    Drawn while the target is missing, so it is obvious that it is still being
    looked for rather than forgotten.
    """
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), COASTING, 1, cv2.LINE_AA)
    cv2.putText(frame, "LAST SEEN", (x, max(y - 6, 12)), FONT, 0.4, COASTING, 1, cv2.LINE_AA)
    fh, fw = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    if radius < max(fw, fh) * 1.5:
        cv2.circle(frame, (cx, cy), int(radius), COASTING, 1, cv2.LINE_AA)


def draw_hud(frame, fps, track_count, extra=None):
    text = f"fps {fps:.0f}  tracks {track_count}"
    if extra:
        text += f"  {extra}"
    cv2.putText(frame, text, (10, 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_telemetry(frame, track):
    color = track_color(track)
    if track.visual:
        state = "VISUAL"
    elif track.coasting:
        state = "COASTING"
    else:
        state = "LOCKED" if track.locked else "ACQUIRING"
    draw_brackets(frame, track.bbox, color)
    draw_trail(frame, track, color)
    draw_prediction(frame, track, color)
    draw_label(
        frame,
        (track.bbox[0] + track.bbox[2] + 12, track.bbox[1] - 12),
        [
            f"TRK ID: {track.id} [{state}]",
            f"POS: {int(track.center[0])}, {int(track.center[1])}",
            f"SPD: {track.speed:.0f} px/s",
        ],
        color,
    )
