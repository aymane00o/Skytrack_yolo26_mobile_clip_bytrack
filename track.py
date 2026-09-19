"""Sky Tracker - detect and track planes, drones and birds in video.

Auto mode:   python track.py footage.mp4
Lock mode:   python track.py footage.mp4 --lock
Handheld:    python track.py footage.mp4 --mode diff
"""

import argparse
import csv
import time

import cv2
import numpy as np

import overlay
import reid
from bytetrack import ByteMultiTracker, ByteTracker, CameraMotion, Seen, ious
from detector import SkyDetector, YoloDetector
from tracker import MultiTracker, bbox_center


class Telemetry:
    COLUMNS = ["frame", "time_s", "track_id", "state", "x", "y", "w", "h", "cx", "cy", "speed_px_s"]

    def __init__(self, path):
        self.file = open(path, "w", newline="", encoding="utf-8") if path else None
        self.writer = csv.writer(self.file) if self.file else None
        if self.writer:
            self.writer.writerow(self.COLUMNS)

    def row(self, frame_idx, t, track_id, state, bbox, center, speed):
        if not self.writer:
            return
        x, y, w, h = bbox
        self.writer.writerow([
            frame_idx, f"{t:.3f}", track_id, state,
            f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}",
            f"{center[0]:.1f}", f"{center[1]:.1f}", f"{speed:.1f}",
        ])

    def close(self):
        if self.file:
            self.file.close()


def open_video(path, start_frame=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap, fps, size


def make_writer(path, fps, size):
    if not path:
        return None
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video for writing: {path}")
    return writer


class VisualHold:
    """Follows a confirmed target visually once the detector can no longer see it.

    The detector needs the target to stand out from its surroundings, which stops
    being true the moment it descends into trees or lands on rough ground. A
    correlation tracker does not care about that - it follows appearance frame to
    frame - so a target that would otherwise be lost at exactly the interesting
    moment stays tracked. It is only trusted while the patch still matches what
    was handed over, so it cannot quietly wander onto the scenery.
    """

    def __init__(self, min_score, budget):
        self.min_score = min_score
        self.budget = budget
        self.held = {}
        self.last_seen = {}
        self.spent = {}

    def step(self, frame, tracks, limit):
        # Carrying a target visually costs a correlation-tracker update per frame,
        # so only the strongest few are worth it. Without a cap, a cluttered scene
        # with a hundred stale tracks spends all its time following scenery.
        carried = sorted((t for t in tracks if t.coasting and t.locked),
                         key=lambda t: t.hits, reverse=True)[:limit]
        keep = {t.id for t in carried}

        alive = set()
        for track in tracks:
            alive.add(track.id)
            if not track.coasting:
                self.last_seen[track.id] = track.bbox
                self.held.pop(track.id, None)
                self.spent[track.id] = 0
                continue
            if track.id not in keep:
                self.held.pop(track.id, None)
                continue
            if track.id not in self.held:
                self._seed(frame, track)
                continue
            self._follow(frame, track)

        for store in (self.held, self.last_seen, self.spent):
            for tid in [t for t in store if t not in alive]:
                del store[tid]

    def _seed(self, frame, track):
        bbox = self.last_seen.get(track.id)
        if bbox is None:
            return
        x, y, w, h = (int(v) for v in bbox)
        fh, fw = frame.shape[:2]
        if w < 8 or h < 8 or x < 0 or y < 0 or x + w > fw or y + h > fh:
            return
        csrt = create_csrt()
        csrt.init(frame, (x, y, w, h))
        self.held[track.id] = [csrt, patch_signature(frame, (x, y, w, h))]

    def _follow(self, frame, track):
        csrt, template = self.held[track.id]
        ok, bbox = csrt.update(frame)
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        # A box that has left the frame is not a target any more, and reporting
        # its centre would put a negative coordinate in the telemetry.
        if not ok or w <= 0 or h <= 0 or x + w <= 0 or y + h <= 0 or x >= fw or y >= fh:
            del self.held[track.id]
            return
        patch = patch_signature(frame, bbox)
        score = appearance_score(template, patch)
        if patch is None or score < self.min_score:
            del self.held[track.id]
            return
        spent = self.spent.get(track.id, 0) + 1
        if self.budget and spent > self.budget:
            # A hold that never gets confirmed by a fresh detection is probably
            # sitting on scenery, so give it up rather than carry it forever.
            del self.held[track.id]
            return
        self.spent[track.id] = spent
        self.held[track.id][1] = 0.95 * template + 0.05 * patch
        track.adopt(tuple(float(v) for v in bbox))


FOCUS_CONTEXT = 6.0
FOCUS_MIN_CROP = 420


def pick_primary(current_id, candidates):
    """Follow one target and ignore the rest.

    The point is stickiness: whichever target is being followed keeps the frame
    until it dies, even on the frames where some piece of scenery briefly looks
    more convincing. Re-picking every frame by score would hop between whatever
    happens to rank highest, which is exactly the jitter this avoids.
    """
    held = next((trk for trk in candidates if trk.id == current_id), None)
    if held is not None:
        return current_id, [held]
    best = max(candidates, key=lambda trk: trk.hits, default=None)
    return (best.id, [best]) if best is not None else (None, [])


def focus_regions(tracks, size, args):
    """Square search windows around the targets already being followed."""
    width, height = size
    regions = []
    for track in sorted(tracks, key=lambda t: t.hits, reverse=True)[:args.max_tracks]:
        if not track.locked:
            continue
        x, y, w, h = track.bbox
        side = min(max(max(w, h) * FOCUS_CONTEXT, FOCUS_MIN_CROP), min(width, height))
        cx, cy = x + w / 2.0, y + h / 2.0
        x0 = int(min(max(cx - side / 2, 0), width - side))
        y0 = int(min(max(cy - side / 2, 0), height - side))
        regions.append((x0, y0, x0 + int(side), y0 + int(side)))
    return regions


def build_detector(args, size, min_contrast=None, byte_low=None):
    """Construct the detector args.mode asks for, sized to the frame.

    Shared between auto mode and the lock-mode search, so picking a mode with
    `--mode` (sky/mog2/diff/yolo) steers both rather than just the first one -
    the lock-mode search used to be hardcoded to `sky` regardless of what was
    asked for, which made it useless on anything but sky footage: a busy road
    scene has no silhouette to find, so the search would go silent forever.
    `min_contrast` overrides args.min_contrast where a stricter floor is
    wanted (the lock-mode search asks for a higher one).

    `byte_low` opens the detector's floor up so it also reports the detections
    it would normally suppress, which is what ByteTrack's second pass runs on.
    It is given on the same 0-1 scale ByteTrack's own thresholds use, so the
    figure means the same thing whichever detector is behind it.
    """
    merge_gap = args.merge_gap or max(8, size[0] // 40)
    max_area = args.max_area or (size[0] * size[1]) // 10
    if args.mode == "yolo":
        return YoloDetector(
            model=args.yolo_model, imgsz=args.yolo_imgsz, conf=args.yolo_conf,
            classes=tuple(c.strip() for c in args.yolo_classes.split(",") if c.strip()),
            focus_imgsz=args.yolo_focus_imgsz, low_conf=byte_low or 0.0,
        )
    floor = args.min_contrast if min_contrast is None else min_contrast
    return SkyDetector(
        mode=args.mode, min_area=args.min_area, max_area=max_area,
        sensitivity=args.sensitivity, min_contrast=floor,
        merge_gap=merge_gap, max_texture=args.max_texture,
        # A contrast detector has no confidence of its own, so its score is its
        # contrast against the floor a target must clear - see
        # SkyDetector.detect_scored, whose scale this inverts.
        low_contrast=(byte_low * floor * 8.0) if byte_low else 0.0,
    )


def run_auto(args):
    cap, fps, size = open_video(args.video, args.start_frame)
    detector = build_detector(args, size, byte_low=args.byte_low if args.bytetrack else None)
    shared = dict(min_hits=args.min_hits, min_travel=args.min_travel, min_speed=args.min_speed,
                  min_straightness=args.min_straightness, steady_hits=args.steady_hits,
                  bounds=size)
    tracker = ByteMultiTracker(
        high_thresh=args.byte_high, low_thresh=args.byte_low,
        match_thresh=args.byte_match, buffer=args.byte_buffer,
        expand=args.byte_expand, **shared,
    ) if args.bytetrack else MultiTracker(
        max_missed=args.max_missed, max_distance=args.max_distance, **shared,
    )
    focusing = args.mode == "yolo" and not args.no_yolo_focus
    visual_hold = None if args.no_visual_hold else VisualHold(
        args.visual_min_score, args.visual_hold_frames)
    writer = make_writer(args.output, fps, size)
    telemetry = Telemetry(args.csv)

    frame_idx, shown_fps, t_prev = args.start_frame, fps, time.perf_counter()
    primary_id = None
    stop_frame = args.start_frame + args.max_frames if args.max_frames else None
    while True:
        if stop_frame and frame_idx >= stop_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        clean = frame.copy()
        regions = focus_regions(tracker.tracks.values(), size, args) if focusing else None
        # A sweep every few frames is what lets a new target be picked up at all;
        # without it the search would stay wherever the first one was found.
        if regions and frame_idx % args.yolo_sweep == 0:
            regions = None
        if args.bytetrack:
            scored, mask = detector.detect_scored(frame, regions)
            tracks = tracker.update(scored, t)
        else:
            boxes, mask = detector.detect(frame, regions)
            tracks = tracker.update(boxes, t)
        if visual_hold:
            visual_hold.step(clean, tracks, 1 if args.follow_one else args.max_tracks)
        candidates = tracks if args.show_tentative else [trk for trk in tracks if trk.locked]
        # Show what is actually being seen right now ahead of anything coasting,
        # so a stale ghost never takes the slot of the target in view.
        visible = sorted(
            candidates, key=lambda trk: (trk.visual or not trk.coasting, trk.hits), reverse=True
        )[:args.max_tracks]
        if args.follow_one:
            primary_id, visible = pick_primary(primary_id, candidates)

        for trk in visible:
            overlay.draw_telemetry(frame, trk)
            state = ("visual" if trk.visual else
                     "coasting" if trk.coasting else
                     "locked" if trk.locked else "acquiring")
            telemetry.row(frame_idx, t, trk.id, state, trk.bbox, trk.center, trk.speed)

        primary = next((trk for trk in visible if trk.locked and (trk.visual or not trk.coasting)), None)
        if primary:
            overlay.draw_inset(frame, primary.bbox, source=clean, size=args.inset_size)

        now = time.perf_counter()
        shown_fps = 0.9 * shown_fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        overlay.draw_hud(frame, shown_fps, len(visible))

        if writer:
            writer.write(frame)
        if args.show:
            show("Sky Tracker", frame, args.window_height)
            if args.mask and mask is not None:
                show("Detection mask", mask, args.window_height)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    telemetry.close()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames.")


_sized_windows = set()


def show(window, frame, max_height=900):
    """Display a frame in a window scaled to fit the screen.

    Phone footage is commonly 1080x1920, which does not fit on a monitor at all.
    Only the window is scaled - what goes to --output stays full resolution.
    """
    if window not in _sized_windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        height, width = frame.shape[:2]
        scale = min(1.0, max_height / max(height, 1))
        cv2.resizeWindow(window, max(int(width * scale), 160), max(int(height * scale), 120))
        _sized_windows.add(window)
    cv2.imshow(window, frame)


def ints(bbox):
    """Whole-pixel box, which is all OpenCV's trackers accept."""
    return tuple(int(v) for v in bbox)


def create_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


TEMPLATE_SIZE = 48


def patch_signature(frame, bbox):
    """Normalised colour crop of the target, used to tell a real lock from a drift.

    Colour is kept rather than reduced to grey: a dark airframe against a dark
    treeline is nearly the same brightness as what is behind it, and only its
    hue still says which is which.
    """
    x, y, w, h = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, fw), min(y + h, fh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE)).astype(np.float32)


def appearance_score(template, patch):
    if template is None or patch is None:
        return 0.0
    return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0][0])


class LockedTarget:
    """What is remembered about the chosen target, so it can be found again.

    Two templates are kept, and a candidate is matched against both. The target's
    appearance can change completely - an aircraft leaving the ground goes from a
    front-lit coloured body to a dark silhouette against bright sky - so the
    adapting template follows it through gradual change while the original
    survives as a fallback when the adaptation has followed it somewhere wrong.

    A rediscovered target is adopted on probation: what was remembered is set
    aside rather than overwritten, so latching onto the wrong thing can be undone
    and the real target is still recognisable afterwards.
    """

    def __init__(self, frame, bbox):
        self.first = patch_signature(frame, bbox)
        self.template = self.first
        self.bbox = tuple(float(v) for v in bbox)
        self.center = bbox_center(bbox)
        self.velocity = np.zeros(2)
        self.speed = 0.0
        # Size settles far more slowly than position, so it is worth averaging.
        # It is the only scale reference a search has, and a single bad frame
        # should not be allowed to redefine how big the target is.
        self.size_hint = (float(bbox[2]), float(bbox[3]))
        self._saved = None

    def match(self, frame, bbox):
        patch = patch_signature(frame, bbox)
        if patch is None:
            return 0.0, None
        return max(appearance_score(self.template, patch),
                   appearance_score(self.first, patch)), patch

    def confirm(self, bbox, center, patch, dt):
        if dt > 0:
            self.velocity = (center - self.center) / dt
            self.speed = float(np.linalg.norm(self.velocity))
        self.bbox, self.center = tuple(float(v) for v in bbox), center
        self.size_hint = (0.9 * self.size_hint[0] + 0.1 * bbox[2],
                          0.9 * self.size_hint[1] + 0.1 * bbox[3])
        if patch is not None:
            self.template = 0.95 * self.template + 0.05 * patch

    def resemblance(self, frame, bbox):
        """How strongly a patch looks like the target, in either polarity.

        A target picked on the ground and met again as a silhouette against
        bright sky correlates negatively with what was picked: the same shape
        with its contrast inverted. That is evidence of identity, not evidence
        against it. Reading a strong negative as no-match at all is what used to
        lose the aircraft for good the moment it climbed - the detector was
        offering the right box, frame after frame, and it was being thrown away.
        """
        patch = patch_signature(frame, bbox)
        if patch is None:
            return 0.0
        return max(abs(appearance_score(self.template, patch)),
                   abs(appearance_score(self.first, patch)))

    def area_hint(self):
        return max(self.size_hint[0] * self.size_hint[1], 1.0)

    def propose(self, frame, bbox):
        """Move onto a rediscovered box without yet believing it.

        What the target looked like before is set aside rather than thrown away,
        so if this turns out to be cloud rather than aircraft, `revert` puts the
        memory back and the search carries on from what was known before.

        The new appearance does have to take over for the frames in between: a
        target found again as a silhouette does not resemble what was picked on
        the ground, and judging its next frames against the old template would
        report it as lost within half a second - every time, forever.
        """
        self._saved = (self.template, self.bbox, self.center,
                       self.velocity, self.speed, self.size_hint)
        found = patch_signature(frame, bbox)
        if found is not None:
            self.template = found
        self.bbox, self.center = tuple(float(v) for v in bbox), bbox_center(bbox)
        self.velocity, self.speed = np.zeros(2), 0.0
        self.size_hint = (float(bbox[2]), float(bbox[3]))

    def accept(self):
        """The new lock held for long enough: stop being able to take it back."""
        self._saved = None

    def revert(self):
        """The new lock did not hold: go back to what was known before it."""
        if self._saved is None:
            return
        (self.template, self.bbox, self.center,
         self.velocity, self.speed, self.size_hint) = self._saved
        self._saved = None

    def discard_drift(self):
        """The box was found to be sitting on something static, not the target.

        Every confirmed frame spent on it fed the adapting template 5% closer
        to what it looks like, and by the time that is noticed the template
        can have adapted almost entirely away from the real target - so the
        search that follows would be looking for the wrong thing again,
        using an appearance it only just spent this many frames being fooled
        by. `first`, captured once at the original pick, was never touched by
        any of that and is what the search should trust instead.
        """
        self.template = self.first


def touches_edge(box, size, margin=3):
    """Is the box cut off by the frame border?

    Sky, cloud and ground are detected as blobs that run off the edge of the
    picture, while an aircraft the camera is following sits inside it. This is
    the cheapest test that tells the two apart.
    """
    return (box[0] <= margin or box[1] <= margin
            or box[0] + box[2] >= size[0] - margin
            or box[1] + box[3] >= size[1] - margin)


def centred_in(box, region):
    """Does the box's centre fall inside the region?"""
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    return (region[0] <= cx <= region[0] + region[2]
            and region[1] <= cy <= region[1] + region[3])


def search_for(target, boxes, frame, lost_frames, size, min_score,
               min_appearance=0.15, rejected=(), margin=0.0):
    """Look for a lost target among this frame's candidate boxes.

    Scored on how much a candidate looks like the target and how close it is to
    where it should have got to, because after a change of background appearance
    alone is not enough to recognise it by. The search area grows the longer it
    has been missing, so a target that reappears far away is still reachable
    while a fresh loss stays choosy.

    Takes boxes rather than finding them, so the same search works over raw
    detections or over ByteTrack's tracks. Being a track is worth something -
    it rules out a blob that appeared for one frame - but not as much as it
    sounds: cloud is perfectly consistent frame to frame and makes excellent
    tracks, so the caller still has to screen candidates on their own merit.
    """
    if not boxes:
        return None, 0.0, 0

    # Velocity is only worth extrapolating briefly: carried over several seconds
    # it puts the target thousands of pixels outside the frame and rejects every
    # real candidate. Past that the growing radius does the work instead.
    horizon = min(lost_frames, 15) / 30.0
    predicted = target.center + target.velocity * horizon
    predicted = np.clip(predicted, [0, 0], [size[0], size[1]])
    radius = max(size) * 0.15 + max(target.bbox[2], target.bbox[3]) * 3 + lost_frames * 6
    remembered = target.area_hint()
    # Apparent size is only weak evidence and it weakens with time, since an
    # aircraft that fills the frame on the ground is a speck once it has climbed
    # away. It cannot weaken without limit though, or after a long search every
    # blob of cloud in the sky qualifies on size alone.
    tolerance = min(3.0 + lost_frames * 0.5, 8.0)

    scored = []
    for box in boxes:
        if touches_edge(box, size):
            continue
        if any(centred_in(box, region) for region in rejected):
            continue
        if not 1 / tolerance < box[2] * box[3] / remembered < tolerance:
            continue
        distance = float(np.linalg.norm(bbox_center(box) - predicted))
        if distance > radius:
            continue
        # Position on its own is not evidence of identity. Without a floor here,
        # anything at all near where the target should be scores enough to take
        # the lock, and a patch of cloud usually gets there first.
        looks = target.resemblance(frame, box)
        if looks < min_appearance:
            continue
        closeness = 1.0 - distance / radius
        # A correlation of 0.4 against a target whose background has changed
        # completely is already a strong match, so the appearance term is read
        # against that rather than against a perfect 1.0 nothing ever reaches.
        scored.append((0.5 * min(looks / 0.5, 1.0) + 0.5 * closeness, box))

    if not scored:
        return None, 0.0, len(boxes)
    scored.sort(key=lambda pair: -pair[0])
    best_score, best = scored[0]

    # Refuse to guess between look-alikes. On an empty sky the right candidate
    # wins alone, but in traffic a dozen cars are the same size, the same
    # colour and all near where the target should be - and picking the
    # highest of a dozen near-identical scores is a coin toss that ends up
    # following the wrong one with full confidence. Waiting costs a few more
    # frames of searching; guessing costs the rest of the run.
    if margin > 0 and len(scored) > 1 and best_score - scored[1][0] < margin:
        return None, best_score, len(boxes)

    if best_score >= min_score:
        return best, best_score, len(boxes)
    return None, best_score, len(boxes)


def _detect_worker(conn, shm_name, shape, dtype, args, size, byte_low, threads, static):
    """Detect and associate in a process of our own. Runs until told to stop.

    Lives in a separate process rather than a thread because a thread cannot
    help here: measured, a worker holding Python's interpreter lock drops this
    loop from 53 fps to 15, and running the model behaves exactly like one - the
    tracking loop and the detector end up taking turns instead of running at
    once, which is the whole thing this was meant to avoid.
    """
    from multiprocessing import shared_memory
    import numpy as np

    if threads:
        # Left alone, the model spreads over every core it can find and starves
        # the loop it was moved off in the first place - a worker saturating the
        # machine holds playback down almost as effectively as one holding the
        # interpreter lock. It is worth giving up detector speed for: the loop
        # has a frame rate it must hit, and the detector only has to keep up
        # well enough for the buffer to cover the difference.
        try:
            import torch
            torch.set_num_threads(threads)
        except ImportError:
            pass
        import cv2 as _cv2
        _cv2.setNumThreads(threads)

    shm = shared_memory.SharedMemory(name=shm_name)
    frames = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    detector = build_detector(args, size, byte_low=byte_low)
    byte = ByteTracker(high_thresh=args.byte_high, low_thresh=args.byte_low,
                       match_thresh=args.byte_match, buffer=args.byte_buffer,
                       expand=args.byte_expand)
    motion = None if args.no_cmc else CameraMotion(static_regions=static)
    # A model's first inference is several times slower than the rest - measured,
    # the first answer took 51 frames of footage to arrive. Paid here, before the
    # loop is told to start, so the first frames are answered promptly instead.
    detector.detect_scored(np.zeros(shape, dtype=dtype))
    conn.send(("ready", None))
    previous = None
    try:
        while True:
            message = conn.recv()
            if message is None:
                break
            frame_idx = message
            frame = frames.copy()
            scored, _ = detector.detect_scored(frame)
            boxes = [b for b, _ in scored]
            warp = motion.estimate(frame, boxes) if motion else None
            byte.update(boxes, [s for _, s in scored], warp,
                        frames=frame_idx - previous if previous is not None else 1)
            previous = frame_idx
            taken = byte.snapshot(frame, reid.fingerprint if args.reid else None)
            conn.send((frame_idx, [tuple(t) for t in taken]))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        shm.close()


class Detections:
    """Runs the detector and ByteTrack, and says what is being tracked.

    ByteTrack wants a detection pass on every frame, and with `--mode yolo` one
    pass costs a hundred times what the rest of a frame does. Done in line, that
    is not slow tracking - it is a slideshow: the preview drops to the detector's
    own rate, a few frames a second, while the footage it is showing was shot at
    thirty.

    So the loop stops waiting for it. The detector runs in a second process on
    whatever the newest frame is whenever it becomes free, and the loop reads the
    most recent answer that has come back. Tracking, drawing and playback then run
    at the frame rate of the footage. What ByteTrack gives up is seeing every
    frame - it sees roughly every third instead, a few frames behind - and the
    Kalman prediction and the buffered overlap were already built for exactly
    that, since both exist to cover frames where nothing was matched.

    Headless runs stay in line, on purpose: what `--output` and `--csv` contain
    should not depend on how fast the machine happened to be that day.
    """

    def __init__(self, make_detector, byte, threaded, args=None, size=None, byte_low=None,
                 static=(), cmc=True, describe=False):
        self.make_detector = make_detector
        self.describe = describe
        self.static = list(static)
        self.motion = CameraMotion(static_regions=static) if cmc else None
        self.detector = None
        self.byte = byte
        self.snapshot = []
        self.at_frame = None
        self.worker = None
        self.shm = None
        self.waiting = False
        if not threaded:
            self.detector = self.make_detector()
            return
        try:
            self._start(args, size, byte_low)
        except Exception as exc:                       # noqa: BLE001 - any failure is fatal to it
            print(f"Could not start the detector process ({exc}); "
                  f"running it in the tracking loop instead.")
            self.worker = None
        if self.worker is None:
            # Only built when it is this process that has to do the detecting;
            # with a worker the model is loaded there instead, and loading it
            # twice would cost a second start-up and its memory for nothing.
            self.detector = self.make_detector()

    def _start(self, args, size, byte_low):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        import numpy as np

        shape, dtype = (size[1], size[0], 3), np.uint8
        self.shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(shape)) * np.dtype(dtype).itemsize)
        self.buffer = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        self.conn, child = mp.Pipe()
        self.worker = mp.Process(
            target=_detect_worker,
            args=(child, self.shm.name, shape, dtype, args, size, byte_low,
                  getattr(args, "detect_threads", 0), self.static),
            daemon=True)
        self.worker.start()
        child.close()
        # The model is loaded in the child, which takes a couple of seconds. Wait
        # for it here rather than letting the first frames race an unbuilt model -
        # but watch the process while waiting, so a child that dies on startup is
        # noticed at once instead of after the timeout.
        deadline = time.perf_counter() + 120
        while time.perf_counter() < deadline:
            if self.conn.poll(0.1):
                if self.conn.recv()[0] == "ready":
                    return
                break
            if not self.worker.is_alive():
                break
        raise RuntimeError("detector process did not start")

    def step(self, frame_idx, frame):
        """Offer this frame, and return the latest association available."""
        if self.worker is None:
            scored, _ = self.detector.detect_scored(frame)
            boxes = [b for b, _ in scored]
            warp = self.motion.estimate(frame, boxes) if self.motion else None
            self.byte.update(boxes, [s for _, s in scored], warp)
            self.snapshot = self.byte.snapshot(frame, reid.fingerprint if self.describe else None)
            self.at_frame = frame_idx
            return self.snapshot

        while self.conn.poll():
            at, taken = self.conn.recv()
            self.snapshot = [Seen(*row) for row in taken]
            self.at_frame, self.waiting = at, False
        if not self.waiting:
            # Only the newest frame is worth sending. Queueing them would build a
            # backlog and answer questions about where things were a second ago,
            # which is the latency this exists to remove.
            self.buffer[:] = frame
            self.conn.send(frame_idx)
            self.waiting = True
        return self.snapshot

    def close(self):
        if self.worker is not None:
            try:
                self.conn.send(None)
                self.worker.join(timeout=5)
            except Exception:                          # noqa: BLE001
                pass
            if self.worker.is_alive():
                self.worker.terminate()
            self.conn.close()
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()


def usable_box(box, size, min_side=4):
    """Is this box something a correlation tracker can actually be handed?

    ByteTrack's boxes are a Kalman state, not a measurement, so one that has
    been coasting can have drifted partly off the picture or collapsed to
    nothing - and CSRT raises rather than refuses when initialised on either.
    """
    x, y, w, h = box
    return (w >= min_side and h >= min_side and x >= 0 and y >= 0
            and x + w <= size[0] and y + h <= size[1])


def sitting_on(tracks, bbox, min_iou=0.2):
    """The ByteTrack track that is on this box, if one is.

    How the lock and ByteTrack are tied together: whatever track overlaps the
    box the lock has confirmed is the same object, so its ID becomes the lock's
    identity. From then on the question "where did my target go" has an answer
    that survives the target looking wrong for a while, which is the one thing
    an appearance match cannot do.
    """
    best, best_iou = None, min_iou
    for track in tracks:
        score = float(ious([track.tlwh], [bbox])[0][0])
        if score > best_iou:
            best, best_iou = track, score
    return best


def overlaps(a, b, share=0.25):
    """Do two boxes cover enough of each other to be the same thing?"""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter > 0 and inter / max(min(a[2] * a[3], b[2] * b[3]), 1.0) >= share


def resync_box(boxes, frame, bbox, target, size, min_score, max_growth):
    """Check the lock against the detector, and snap the box back onto it.

    Returns (wider, supported). `supported` says whether the detector saw
    anything at all where the box is sitting, which is what catches a lock
    that has slid off the target: a correlation tracker drifts a few pixels a
    frame, each step far too small to trip a jump limit, and the template
    quietly adapts to whatever it is sliding over - so nothing else notices
    until it has been on a patch of road, or a logo, for hundreds of frames.
    A target worth tracking is one the detector can generally see.

    A correlation tracker keeps following the target but is free to shrink onto
    whatever part of it correlates best - after a while an aircraft is being
    reported as a box around its tail fin. A detection of the same object has the
    real extent, so where one overlaps the tracked box, it replaces it.

    The replacement has to still look like the target and be of a believable
    size. Broken cloud comes back as large blobs, and one of those happening to
    contain a distant aircraft was enough to hand the lock over to the cloud -
    after which the aircraft was never seen again.
    """
    centre = bbox_center(bbox)
    area = max(bbox[2] * bbox[3], 1.0)
    supported = any(overlaps(bbox, box) for box in boxes)
    best, best_score = None, min_score
    for box in boxes:
        if touches_edge(box, size):
            continue
        if not 1.0 < box[2] * box[3] / area <= max_growth:
            continue
        if not (box[0] <= centre[0] <= box[0] + box[2]
                and box[1] <= centre[1] <= box[1] + box[3]):
            continue
        looks = target.resemblance(frame, box)
        if looks > best_score:
            best, best_score = box, looks
    return best, supported


def run_lock(args):
    cap, fps, native = open_video(args.video, args.start_frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Video has no frames")

    # A correlation tracker costs roughly the area of what it follows, and phone
    # footage is 1080x1920. Working at half height makes the run about three
    # times faster while changing what is tracked very little - the aircraft is
    # the same object either way - which is the difference between watching the
    # result at the speed of the footage and watching it crawl.
    scale = 1.0
    if args.proc_height and frame.shape[0] > args.proc_height:
        scale = args.proc_height / float(frame.shape[0])

    def prepare(raw):
        if scale == 1.0:
            return raw
        return cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    frame = prepare(frame)
    size = (frame.shape[1], frame.shape[0])

    if args.lock_bbox:
        given = tuple(float(v) for v in args.lock_bbox.split(","))
        if len(given) != 4:
            raise SystemExit("--lock-bbox expects x,y,w,h")
        bbox = tuple(v * scale for v in given)
    else:
        print("Drag a box around the target, then press ENTER. Press C to cancel.")
        # Size the window before selectROI, or it opens at the video's own
        # resolution and portrait phone footage runs off the screen.
        show("Select target", frame, args.window_height)
        bbox = cv2.selectROI("Select target", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select target")
        _sized_windows.discard("Select target")
    if bbox[2] == 0 or bbox[3] == 0:
        raise SystemExit("No target selected")

    # What follows the target between detection passes: OpenCV's CSRT correlation
    # tracker, which runs at about 50 fps on a CPU. It keeps roughly the box it was
    # given, so what keeps it honest is the detector - every pass says whether the
    # target is still under the box, and ByteTrack says which identity that is.
    def new_tracker():
        return create_csrt()

    csrt = new_tracker()
    csrt.init(frame, ints(bbox))
    target = LockedTarget(frame, bbox)
    max_jump = args.lock_max_jump * scale if args.lock_max_jump > 0 else 0.10 * size[0]

    # The memory of the target you picked (reid.py). It needs ByteTrack, whose
    # identities are what evidence is gathered on, and a trained detector: its
    # fingerprint is the object's layout, and a layout match cannot follow an
    # aircraft from a lit airframe on the ground to a silhouette against sky -
    # the contrast is inverted, which is what the older appearance search in
    # this file handles instead, and still does for the contrast modes.
    args.reid = bool(args.bytetrack and args.mode == "yolo"
                     and not args.no_reid and not args.no_reacquire)
    memory = reidentifier = thumbnail = None
    if args.reid:
        memory = reid.TargetMemory()
        memory.pin(frame, bbox)
        reidentifier = reid.Reidentifier(memory, passes=args.reid_passes, margin=args.reid_margin,
                                         min_likeness=args.reid_min_likeness,
                                         sole_passes=args.reid_sole_passes)
        pick = reid._canonical(frame, bbox, side_scale=1.4)
        thumbnail = None if pick is None else cv2.resize(pick, (110, 110))

    # Only built if it might be needed: losing the target is not a dead end, so
    # the detector is what finds it again once it is somewhere findable. Uses
    # whatever --mode was asked for (sky/mog2/diff/yolo) rather than assuming
    # sky, so this also does something useful on footage that is not aircraft
    # against open sky - a car on a road has no silhouette for `sky` mode to
    # find, `diff` mode's motion compensation is what finds it instead.
    # The search is choosier than a general sweep because it is looking for one
    # known object rather than for anything that might be a target: on sky
    # footage a solid airframe reads over 100 on this contrast scale while
    # broken cloud reads 10 to 35, so a floor between the two leaves the
    # aircraft as the only candidate in a sky full of blobs.
    boosted_contrast = max(args.min_contrast, args.search_contrast)

    # With ByteTrack on, one detector serves everything: it runs every frame
    # anyway, so the search and the periodic resync read this frame's detections
    # instead of paying for their own. It is built at the ordinary floor rather
    # than the search's boosted one, because the faint detections the boost is
    # there to reject are exactly what the second association pass lives on -
    # what keeps those from becoming false targets here is having to be
    # consistent with an existing track, which is a much stronger test than
    # standing out in one frame.
    byte = detections = None
    if args.bytetrack and not args.no_reacquire:
        byte = ByteTracker(high_thresh=args.byte_high, low_thresh=args.byte_low,
                           match_thresh=args.byte_match, buffer=args.byte_buffer,
                           expand=args.byte_expand)
        # Only worth a process of its own when the detector is the expensive
        # part. The contrast modes cost a couple of milliseconds a frame - 31.0
        # fps against 29.6 with ByteTrack on - and handing every frame to another
        # process to save that would cost more than it saved.
        offload = args.show and not args.detect_sync and args.mode == "yolo"
        detections = Detections(lambda: build_detector(args, size, byte_low=args.byte_low),
                                byte, threaded=offload,
                                args=args, size=size, byte_low=args.byte_low,
                                static=[tuple(float(v) for v in r.split(","))
                                        for r in args.exclude],
                                cmc=not args.no_cmc, describe=args.reid)
    searcher = None if (args.no_reacquire or byte is not None) else \
        build_detector(args, size, boosted_contrast)

    # Being one of ByteTrack's tracks says a thing is real. It does not say it
    # could be the target, and the difference matters: cloud is about as
    # consistent from frame to frame as anything in this footage, so it forms
    # excellent tracks. Measured, letting every track stand as a search
    # candidate cost the aircraft outright - the lock went to a cloud at frame
    # 2725 of the VTOL clip and never came back. So a candidate still has to
    # clear the bar the search has always used, expressed on the detector's own
    # score scale: --search-contrast for the contrast modes (which is how they
    # are scored), and the ordinary confidence floor for yolo, which is what
    # the search was reading before ByteTrack existed either way.
    search_floor = (args.yolo_conf if args.mode == "yolo"
                    else boosted_contrast / max(args.min_contrast * 8.0, 1e-6))

    writer = make_writer(args.output, fps, size)
    telemetry = Telemetry(args.csv)

    good_bbox, good_center, good_speed = bbox, bbox_center(bbox), 0.0
    lost, lost_frames, candidates, reacquired = False, 0, 0, 0
    score = 0.0
    # A rediscovered target is believed only after it has held together for a
    # while. Until then it can still be handed back, which is what stops one bad
    # match from replacing the memory of the target with a patch of cloud.
    probation, pending, lost_before = 0, None, 0
    rejected = []
    excluded = [tuple(float(v) for v in region.split(",")) for region in args.exclude]
    prev_raw_patch, static_frac, warned_static, stuck_latched = None, 0.0, False, False
    unsupported = 0
    reacquiring = searcher is not None or byte is not None
    # The identity ByteTrack has given the target. Set from whatever track is
    # sitting on the box once the lock is confirmed, and the first thing asked
    # for when the lock goes.
    lock_id, byte_held = None, 0
    # The last detection pass read, identities that failed probation (id -> the
    # frame they are refused until), and whether the lock has been judged to be on
    # something other than the target.
    last_pass, refused, identity_broken, passes = None, {}, False, 0
    # The first view learned after the pick, or after a re-lock, starts the chain
    # of views the memory follows the target through (TargetMemory.follows). It
    # is taken on trust: the lock has not broken since it was placed, so the car
    # under it is the target - and it cannot be checked against the picked view,
    # which by the time a slow detector first answers can already look nothing
    # like it (the car picked diagonally had turned side-on by then).
    seed_chain, learned_at = True, None
    # Set when the lock was given back for being the only candidate in view: see
    # where it is used.
    by_detection = False
    # Where the confirmed box was on recent frames. A detection pass describes the
    # frame it was run on, which with a slow detector can be a second behind, and
    # it is only against the box as it was on that frame that "is this detection
    # my target" has an exact answer - projecting forward is a guess that grows
    # with the delay, and measured under load the lock never bound at all.
    history = {}
    absent = 0
    proc_fps, frame_period = fps, 1.0 / max(fps, 1.0)
    # Frame numbers are the source video's, not the run's, so telemetry taken
    # from a clip's middle still lines up with the footage it came from.
    first_frame = args.start_frame
    frame_idx, weak_frames = first_frame, 0
    prev_t = first_frame / fps
    stop_frame = first_frame + args.max_frames if args.max_frames else None
    # Picks made by hand during the run: frame -> box in the video's coordinates,
    # from --repick, and one waiting from a keypress (the frame it was drawn on and
    # the box). See the R key below.
    scripted = {}
    for entry in args.repick:
        when, _, where = entry.partition(":")
        values = tuple(float(v) for v in where.split(","))
        if not when.strip().isdigit() or len(values) != 4:
            raise SystemExit("--repick expects FRAME:x,y,w,h")
        scripted[int(when)] = tuple(v * scale for v in values)
    waiting_pick = None
    while not (stop_frame and frame_idx >= stop_frame):
        started = time.perf_counter()
        t = frame_idx / fps
        if frame_idx > first_frame:
            ok, raw = cap.read()
            if not ok:
                break
            frame = prepare(raw)

        # The target picked again by hand. After a camera cut to a wide shot of
        # look-alike cars there is nothing to recognise it by - measured on the
        # chase clip, neither the memory, a vehicle re-identification network nor
        # "the car the camera keeps centred" picked the right 20px car reliably -
        # but whoever is watching usually can. Nothing is forgotten: the new view
        # is added to the memory as permanently as the first pick, so the target
        # can be recognised from either view afterwards.
        picked_on, picked = None, None
        if frame_idx in scripted:
            picked_on, picked = frame, scripted.pop(frame_idx)
        elif waiting_pick is not None:
            (picked_on, picked), waiting_pick = waiting_pick, None
        if picked is not None and usable_box(picked, size):
            csrt = new_tracker()
            csrt.init(picked_on, ints(picked))
            target.propose(picked_on, picked)
            target.accept()
            if memory is not None:
                memory.pin(picked_on, picked)
                reidentifier.reset()
                # The corner panel shows the view picked most recently - the one the
                # search can now most easily match - rather than the very first.
                pick = reid._canonical(picked_on, picked, side_scale=1.4)
                if pick is not None:
                    thumbnail = cv2.resize(pick, (110, 110))
            bbox = good_bbox = tuple(float(v) for v in picked)
            good_center, good_speed, prev_t = bbox_center(good_bbox), 0.0, t
            lost, lost_frames, weak_frames, probation, pending = False, 0, 0, 0, None
            prev_raw_patch, static_frac, stuck_latched, unsupported = None, 0.0, False, 0
            lock_id, identity_broken, absent = None, False, 0
            seed_chain, learned_at, by_detection = True, None, False
            rejected, refused = [], {}
            ok = True
            print(f"Target picked by hand at frame {frame_idx}. Repeat with: --repick "
                  f"{frame_idx}:{int(picked[0] / scale)},{int(picked[1] / scale)},"
                  f"{int(picked[2] / scale)},{int(picked[3] / scale)}")

        if frame_idx > first_frame:
            # While the target is lost the correlation tracker has nothing to
            # follow, and running it anyway over a long search was the largest
            # single cost of the whole run.
            ok, bbox = csrt.update(frame) if not lost else (False, bbox)
        clean = frame
        frame = frame.copy()

        seen, raw_seen, byte_boxes, stale = [], [], [], 0
        if detections is not None:
            seen = raw_seen = detections.step(frame_idx, clean)
            # The detector runs behind the footage when it has a process of its
            # own, so what comes back describes where things were a few frames
            # ago. Every box is carried forward by the motion ByteTrack is
            # already estimating, which is what makes a late answer usable as a
            # current one - without it the box is a target-width behind on
            # anything crossing the frame, and nothing matches anything.
            answered = detections.at_frame if detections.at_frame is not None else frame_idx
            stale = max(frame_idx - answered, 0)
            seen = [track._replace(tlwh=track.after(stale)) for track in seen]
            # Everything ByteTrack holds is available for keeping hold of an
            # identity; only what the search would have looked at anyway is
            # available for choosing a new one.
            byte_boxes = [track.tlwh for track in seen
                          if not track.lost and track.score >= search_floor]
        # Evidence is counted once per detection pass. With the detector in its
        # own process the same answer is read for several frames running, and
        # counting it on each would let one pass masquerade as four.
        fresh_pass = (detections is not None and detections.at_frame is not None
                      and detections.at_frame != last_pass)
        if fresh_pass:
            last_pass, passes = detections.at_frame, passes + 1

        if not lost:
            center = bbox_center(bbox)
            jump = float(np.linalg.norm(center - good_center))
            score, patch = target.match(clean, bbox)

            # A correlation tracker can settle onto something that is not the
            # target at all: a burned-in HUD graphic, a timestamp, a logo -
            # anything drawn identically over every frame. That reads as a
            # perfect match forever, so score and jump alone never catch it -
            # it just never reports lost. The box usually straddles a sliver of
            # real, moving background too, so per-frame stillness is noisy -
            # the median of the patch (immune to a changing minority of it)
            # smoothed over time is what actually separates "sitting on a
            # static graphic" from "genuinely tracking something".
            static_alpha = 2.0 / (args.max_static + 1) if args.max_static > 0 else 0.0
            if patch is not None and prev_raw_patch is not None:
                stillness = float(np.median(np.abs(patch - prev_raw_patch)))
                static_frac = (1 - static_alpha) * static_frac + static_alpha * (stillness < args.static_eps)
            else:
                static_frac = 0.0
            prev_raw_patch = patch
            # Hysteresis: a couple of frames with real motion at the box's edge
            # can dip the fraction below the trip point even while genuinely
            # stuck, so coming unstuck needs a clearer sign than tripping did.
            if args.max_static > 0 and static_frac > 0.85:
                if not stuck_latched:
                    # Whatever the template adapted into while this went
                    # unnoticed is exactly what it should not be searched for.
                    target.discard_drift()
                stuck_latched = True
            elif static_frac < 0.4:
                stuck_latched = False
            if stuck_latched and not warned_static:
                print(f"Frame {frame_idx}: box has barely changed for a while - likely stuck "
                      f"on something drawn into the video, not the target")
                warned_static = True
            elif not stuck_latched:
                warned_static = False

            # CSRT reports success even after it has slid onto background, so
            # trust it only while the patch still looks like the target, has
            # not teleported, and is still actually changing frame to frame.
            healthy = (ok and patch is not None and score >= args.lock_min_score
                      and jump <= max_jump and not stuck_latched)
            if (not healthy and by_detection and ok and lock_id is not None
                    and jump <= max_jump
                    and any(s.id == lock_id and not s.lost
                            and ious([s.tlwh], [bbox])[0, 0] >= 0.3 for s in seen)):
                # A lock given back for being the only candidate is judged by the
                # detector still seeing that identity under the box, not by the
                # appearance template: the template was taken as the target came
                # back, against whatever was behind it then, and a small aircraft
                # climbing past trees into sky stops matching it within frames -
                # measured below 0.2 on every frame after the re-lock on the micro
                # talon clip, with the box on the aircraft throughout.
                healthy = True
            if healthy:
                weak_frames = 0
                good_speed = jump / max(t - prev_t, 1e-6) if frame_idx > first_frame else 0.0
                target.confirm(bbox, center, patch, t - prev_t)
                good_bbox, good_center, prev_t = bbox, center, t
                history[frame_idx] = good_bbox
                if len(history) > 300:
                    del history[min(history)]
                if detections is not None:
                    # Re-read the identity on every confirmed frame rather than
                    # binding once: ByteTrack can legitimately retire an ID and
                    # start a new one on the same object, and a lock still held
                    # by the correlation tracker through that should follow it
                    # rather than keep pointing at a number that no longer exists.
                    then = history.get(detections.at_frame)
                    on_it = (sitting_on([t for t in raw_seen if not t.lost], then)
                             if then is not None else
                             sitting_on([t for t in seen if not t.lost], good_bbox))
                    # ...but only onto something that still looks like the target.
                    # A correlation tracker sliding from one car to the one beside
                    # it takes the overlap with it, and following the overlap
                    # blindly is exactly how the identity used to change hands.
                    if on_it is not None and (memory is None or on_it.id == lock_id
                                              or on_it.feature is None
                                              or memory.likeness(on_it.feature)
                                              >= args.reid_min_likeness - 0.2):
                        lock_id = on_it.id
                    if (memory is not None and fresh_pass and on_it is not None
                            and on_it.id == lock_id and on_it.feature is not None):
                        absent = 0
                        # Learn only while the lock is settled, so the memory grows
                        # with the target as it turns and changes size, and never
                        # from a re-lock still on probation. Whether a view is a
                        # continuation of the target or a swap to another car is
                        # decided by the chain in TargetMemory.learn, not by how much
                        # it resembles the picked view - a car that has turned since
                        # it was picked resembles that view very little, measured
                        # at 0.15-0.25, and is still the same car.
                        if probation == 0:
                            # A long gap between learned views also starts a new
                            # chain, provided the lock has held throughout (this code
                            # only runs while it has): the target can turn a long way
                            # in a second, and comparing across that gap would refuse
                            # the very car the lock has been on the whole time.
                            gap = (detections.at_frame - learned_at) if learned_at is not None else 0
                            if (memory.learn(on_it.feature, trusted=seed_chain or gap > 15)
                                    or memory.follows(on_it.feature)):
                                learned_at = detections.at_frame
                            seed_chain = False
                            if passes % 5 == 0:
                                for other in (raw_seen if then is not None else seen):
                                    if (not other.lost and other.id != lock_id
                                            and other.feature is not None
                                            and ious([other.tlwh], [then or good_bbox])[0, 0] < 0.05):
                                        memory.learn_other(other.feature)
                    elif (by_detection and fresh_pass and lock_id is not None
                            and on_it is None
                            and (held := next((s for s in seen if s.id == lock_id
                                               and not s.lost), None)) is not None
                            and usable_box(held.tlwh, size)):
                        # A lock given back for being the only candidate is on a
                        # target that looks nothing like its memory - often because
                        # it is thin and against plain sky, which is also what a
                        # correlation tracker slides off. Measured on the micro talon
                        # clip: the box sank onto the ground within five passes of
                        # every such re-lock while the detector went on seeing the
                        # aircraft above it. So the detector's box is followed.
                        csrt = new_tracker()
                        csrt.init(clean, ints(held.tlwh))
                        bbox = good_bbox = tuple(float(v) for v in held.tlwh)
                        good_center, absent = bbox_center(good_bbox), 0
                        prev_raw_patch, static_frac = None, 0.0
                        history[frame_idx] = good_bbox
                    elif memory is not None and fresh_pass and lock_id is not None:
                        # Nothing the detector can see is where the lock is. A car
                        # the correlation tracker has let slide onto trees or a
                        # bridge looks exactly like this - it still reports a
                        # confident lock, on scenery - and so does a car briefly
                        # hidden. With the target remembered, letting go costs
                        # little in the second case (it is re-identified within a
                        # few passes of coming back), and saves the run in the first.
                        absent += 1
                        if absent >= 2 * args.reid_guard:
                            print(f"Frame {frame_idx}: the detector has not seen anything on "
                                  f"the lock for {absent} passes - letting go and searching")
                            identity_broken, absent = True, 0
            else:
                weak_frames += 1
            # Being stuck on a static graphic is unambiguous - it doesn't need
            # the same grace period as an ordinary uncertain frame, which might
            # just be a moment of real appearance change.
            lost = weak_frames > args.lock_max_weak or stuck_latched or identity_broken
            if identity_broken:
                target.discard_drift()
                lock_id, identity_broken, absent = None, False, 0
            if lost and probation > 0:
                # It fell apart while still on probation, so it was never the
                # target. Put back what was known before and refuse that spot
                # for a while, or the same blob wins the next search too.
                rejected.append((pending, frame_idx + args.reject_frames))
                target.revert()
                good_bbox, good_center = target.bbox, target.center
                # The search picks up where it left off rather than starting
                # over: it has been missing all this time, and the area worth
                # looking in should not shrink back because of a false alarm.
                lost_frames = lost_before + args.reacquire_confirm
                probation, pending = 0, None
                # Whatever identity that box belonged to was not the target's,
                # so stop asking ByteTrack for it; the next confirmed frame
                # binds to whichever track is actually on the target.
                if lock_id is not None:
                    refused[lock_id] = frame_idx + args.reject_frames
                    wrong = next((s for s in seen if s.id == lock_id), None)
                    if (memory is not None and wrong is not None and wrong.feature is not None
                            and memory.likeness(wrong.feature) < args.reid_min_likeness):
                        memory.learn_other(wrong.feature)
                if reidentifier is not None:
                    reidentifier.reset()
                lock_id = None
                print(f"Dropped a bad re-lock at frame {frame_idx}")
        else:
            healthy = False

        rejected = [(region, until) for region, until in rejected if until > frame_idx]

        if lost and reacquiring:
            lost_frames += 1
            found, how = None, ""

            # Ask ByteTrack first, and every frame rather than every --search-every.
            # It has been following this identity the whole time the lock was
            # gone - through the low-confidence detections the search never
            # sees - so if the ID is still alive and being seen right now, the
            # target has effectively never been lost, and no search is needed.
            # Appearance is still checked, because ByteTrack matches on motion
            # alone and can hand one object's ID to another that crosses it.
            held = next((t for t in seen if t.id == lock_id), None) if lock_id else None
            if held is not None and not held.lost:
                # The regions a failed re-lock and the user's --exclude have put
                # out of bounds apply here too. An identity that has settled on
                # a burned-in graphic is still on a burned-in graphic, and
                # without this it would be handed straight back the lock that
                # was just taken off it, every frame, forever.
                barred = excluded + [r for r, _ in rejected]
                if memory is not None and held.feature is not None:
                    looks = memory.likeness(held.feature)
                    enough = args.reid_min_likeness - 0.1
                else:
                    looks, enough = target.resemblance(clean, held.tlwh), args.reacquire_min_appearance
                if (looks >= enough
                        and usable_box(held.tlwh, size)
                        and not any(centred_in(held.tlwh, region) for region in barred)):
                    found, score = held.tlwh, looks
                    how = f"ByteTrack held ID {lock_id} across the gap (match {looks:.2f})"
                    byte_held += 1

            refused = {tid: until for tid, until in refused.items() if until > frame_idx}
            if found is None and reidentifier is not None:
                if fresh_pass:
                    barred = excluded + [r for r, _ in rejected]
                    # No confidence floor here, unlike the older search: these are
                    # tracks ByteTrack has already confirmed, and a returning target
                    # having one low-confidence frame is exactly what its second pass
                    # exists to carry. Measured, a floor dropped the recognised SUV
                    # from the candidates one pass short of the lock, and the run
                    # never took it back.
                    pool = [(s.id, s.tlwh, s.feature) for s in seen
                            if not s.lost and usable_box(s.tlwh, size)
                            and not any(centred_in(s.tlwh, region) for region in barred)]
                    candidates = len(pool)
                    decision = reidentifier.step(pool, refused,
                                                 in_view=sum(not t.lost for t in seen))
                    if decision is not None:
                        tid, found, likeness = decision
                        score, lock_id = likeness, tid
                        how = (f"ID {tid} was the only candidate for {args.reid_sole_passes} "
                               f"passes running (likeness {likeness:.2f})"
                               if reidentifier.alone else
                               f"re-identified as ID {tid}: likeness {likeness:.2f}, "
                               f"led for {args.reid_passes} passes running")
            elif found is None and lost_frames % max(args.search_every, 1) == 0:
                # Candidates are ByteTrack's tracks rather than raw detections,
                # so a blob that has not been consistent with anything for two
                # frames running never becomes something to argue over - but
                # they are still only the ones strong enough for the search to
                # have considered before (see search_floor).
                # Only ByteTrack's boxes need screening: a detector's box came
                # from pixels that exist, while a Kalman state can have coasted
                # off the edge of the picture or collapsed to nothing.
                boxes = ([box for box in byte_boxes if usable_box(box, size)]
                         if byte is not None else searcher.detect(clean)[0])
                found, score, candidates = search_for(
                    target, boxes, clean, lost_frames, size, args.reacquire_min_score,
                    args.reacquire_min_appearance, excluded + [r for r, _ in rejected],
                    args.reacquire_margin)
                how = f"match {score:.2f}"

            if (found is not None and memory is not None
                    and not (reidentifier is not None and reidentifier.alone)):
                # Align the lock with the car on this frame, not with where a
                # detection a few frames old put it (see reid.refine). Not for a
                # target taken back for being the only candidate: it was taken back
                # because it does not look like the memory, so the memory cannot say
                # where on it the box belongs.
                aligned, _ = reid.refine(memory, clean, found)
                if usable_box(aligned, size):
                    found = tuple(float(v) for v in aligned)
            if found is not None:
                csrt = new_tracker()
                csrt.init(clean, ints(found))
                target.propose(clean, found)
                probation, pending = args.reacquire_confirm, found
                lost_before = lost_frames
                bbox = good_bbox = found
                center = good_center = bbox_center(found)
                good_speed, weak_frames, lost, prev_t = 0.0, 0, False, t
                lost_frames, reacquired = 0, reacquired + 1
                prev_raw_patch, static_frac, stuck_latched = None, 0.0, False
                unsupported = 0
                by_detection = reidentifier is not None and reidentifier.alone
                if reidentifier is not None:
                    reidentifier.reset()
                seed_chain, learned_at = True, None
                print(f"Reacquired at frame {frame_idx} ({how})")
        elif not lost:
            lost_frames = 0
            if probation > 0:
                probation -= 1
                if probation == 0:
                    target.accept()
                    pending = None
            if (reacquiring and args.resync_every and healthy
                    and frame_idx % args.resync_every == 0):
                boxes = byte_boxes if byte is not None else searcher.detect(clean)[0]
                wider, supported = resync_box(boxes, clean, good_bbox, target, size,
                                              args.lock_min_score, args.resync_max_growth)
                # The detector is asked, on this same pass, whether it can see
                # anything where the lock is sitting. A few misses are normal -
                # carrying the target through gaps is what lock mode is for -
                # but a long run of them means the box is no longer on it.
                #
                # Not having heard from the detector yet is not a miss. While it
                # starts up there are no detections anywhere in the frame, and
                # counting that as "nothing where the box is" spends the whole
                # budget before the first answer arrives - measured, it dropped
                # a perfectly good lock four resyncs into the run, every time.
                waiting_on_first = detections is not None and detections.at_frame is None
                if not waiting_on_first:
                    unsupported = 0 if supported else unsupported + 1
                if args.max_unsupported and unsupported >= args.max_unsupported:
                    print(f"Frame {frame_idx}: nothing detected where the box is, "
                          f"{unsupported * args.resync_every} frames running - dropping the lock")
                    target.discard_drift()
                    lost, unsupported = True, 0
                # Resizing the box onto a detection only makes sense while that
                # detection is current. Projected forward it is an estimate, and
                # re-seating the correlation tracker on an estimate every resync
                # is worse than leaving it to follow what it can actually see.
                if wider is not None and stale > 1:
                    wider = None
                if wider is not None:
                    csrt = new_tracker()
                    csrt.init(clean, ints(wider))
                    target.confirm(wider, bbox_center(wider), patch_signature(clean, wider), 0)
                    bbox = good_bbox = wider
                    good_center = bbox_center(wider)

        if not lost:
            # While the update is untrustworthy, hold the last confirmed position
            # rather than following the tracker onto the background.
            x, y, w, h = [float(v) for v in good_bbox]
            state = "LOCKED" if healthy else "UNCERTAIN"
            if probation > 0:
                state = "REACQUIRED"
            color = overlay.LOCKED if healthy else overlay.COASTING

            # The reported ID is ByteTrack's where there is one, so the telemetry
            # says which identity the position belongs to rather than always
            # claiming to be target number one - and a run where that number
            # never changes is a run where the target was never confused with
            # anything else.
            track_id = lock_id or 1
            overlay.draw_brackets(frame, (x, y, w, h), color)
            overlay.draw_label(
                frame, (x + w + 12, y - 12),
                [f"TRK ID: {track_id} [{state}]",
                 f"POS: {int(good_center[0] / scale)}, {int(good_center[1] / scale)}",
                 f"SPD: {good_speed / scale:.0f} px/s"],
                color,
            )
            overlay.draw_inset(frame, (x, y, w, h), source=clean, size=args.inset_size)
            telemetry.row(frame_idx, t, track_id, state.lower(),
                          (x / scale, y / scale, w / scale, h / scale),
                          good_center / scale, good_speed / scale)
        elif reacquiring:
            banner = f"SEARCHING  {lost_frames} frames  {candidates} candidates"
            if lock_id is not None and any(t.id == lock_id for t in seen):
                banner = f"SEARCHING  {lost_frames} frames  -  ID {lock_id} still held"
            if reidentifier is not None:
                banner = f"SEARCHING  {lost_frames} frames  {candidates} candidates"
                if reidentifier.best is not None:
                    like, lead = reidentifier.best
                    banner += (f"  best {like:.2f}"
                               f" [{reidentifier.streak}/{args.reid_passes}]")
            cv2.putText(frame, banner, (20, 45), overlay.FONT, 0.7, overlay.COASTING, 2, cv2.LINE_AA)
            if args.show:
                cv2.putText(frame, "press R to pick it again", (20, 70), overlay.FONT, 0.5,
                            overlay.COASTING, 1, cv2.LINE_AA)
            # Show that the target is still remembered, not abandoned: where it
            # was last held, and how far out the search has had to open up.
            overlay.draw_ghost(frame, target.bbox, target.center,
                               max(size) * 0.15 + max(target.bbox[2], target.bbox[3]) * 3
                               + lost_frames * 6)
            if thumbnail is not None:
                # What is being looked for: the target as it was picked. Shown in
                # the corner the LIVE panel uses, which has nothing to show while
                # nothing is locked.
                th, tw = thumbnail.shape[:2]
                fh, fw = frame.shape[:2]
                x0, y0 = fw - tw - 12, fh - th - 12
                if x0 > 0 and y0 > 20:
                    frame[y0:y0 + th, x0:x0 + tw] = thumbnail
                    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), overlay.COASTING, 2)
                    cv2.putText(frame, "MEMORY", (x0, y0 - 6), overlay.FONT, 0.45,
                                overlay.COASTING, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "TARGET LOST", (20, 45), overlay.FONT, 0.7, overlay.LOCKED, 2, cv2.LINE_AA)

        # How far behind the detector is, when it is running in its own process.
        # Worth showing rather than hiding: it is the price of a preview that
        # plays at the speed of the footage, and it is what to look at if the
        # identity stops sticking - a detector many frames behind is answering
        # about where things were, not where they are.
        behind = ""
        if (detections is not None and detections.worker is not None
                and detections.at_frame is not None):
            behind = f"  det -{max(frame_idx - detections.at_frame, 0)}f"
        overlay.draw_hud(frame, proc_fps, 0 if lost else 1,
                         extra=f"match {score:.2f}  "
                               f"{'x%.2f' % scale if scale != 1 else 'full'}{behind}")
        if writer:
            writer.write(frame)
        spent = time.perf_counter() - started
        proc_fps = 0.9 * proc_fps + 0.1 / max(spent, 1e-6)
        if args.show:
            show("Sky Tracker", frame, args.window_height)
            # Wait out whatever is left of this frame's share of a second, so
            # the preview runs at the speed of the footage instead of as fast
            # as the machine happens to manage.
            pause = 1 if args.no_pace else max(int((frame_period - spent) * 1000), 1)
            key = cv2.waitKey(pause) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                # Paused on this frame until the box is drawn; the lock is placed on
                # it before the next frame is tracked.
                print("Drag a box around the target, then press ENTER. C cancels.")
                show("Pick the target again", clean, args.window_height)
                box = cv2.selectROI("Pick the target again", clean, showCrosshair=True,
                                    fromCenter=False)
                cv2.destroyWindow("Pick the target again")
                _sized_windows.discard("Pick the target again")
                if box[2] > 0 and box[3] > 0:
                    waiting_pick = (clean, tuple(float(v) for v in box))
        frame_idx += 1

    cap.release()
    if detections is not None:
        detections.close()
    if writer:
        writer.release()
    telemetry.close()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx - first_frame} frames." +
          (f" Reacquired the target {reacquired} times." if reacquired else "") +
          (f" {byte_held} of those were ByteTrack still holding the ID." if byte_held else ""))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="input video file")
    p.add_argument("--output", help="write annotated mp4 here")
    p.add_argument("--csv", help="write per-frame telemetry here")
    p.add_argument("--lock", action="store_true", help="pick one target by hand and track only that")
    p.add_argument("--lock-bbox", help="lock onto x,y,w,h on the first frame instead of drawing it by hand")
    p.add_argument("--lock-min-score", type=float, default=0.35,
                   help="lock mode: appearance match below this counts the frame as uncertain")
    p.add_argument("--lock-max-weak", type=int, default=12,
                   help="lock mode: uncertain frames tolerated before the target is called lost")
    p.add_argument("--lock-max-jump", type=float, default=0.0,
                   help="lock mode: max px the box may move in one frame; 0 means a tenth of frame width")
    p.add_argument("--no-reacquire", action="store_true",
                   help="lock mode: give up when the target is lost instead of searching for it")
    p.add_argument("--reacquire-min-score", type=float, default=0.30,
                   help="lock mode: how convincing a candidate must be to re-lock onto it. "
                        "Lower reacquires sooner and risks locking onto the wrong thing")
    p.add_argument("--resync-every", type=int, default=15,
                   help="lock mode: how often to snap the box back onto a detection of the "
                        "target, which stops it shrinking onto one part; 0 disables")
    p.add_argument("--mode", choices=["sky", "mog2", "diff", "yolo"], default="sky",
                   help="sky detects a silhouette against sky and needs no motion; mog2 suits a "
                        "fixed camera; diff suits handheld footage over a textured background; "
                        "yolo runs a trained detector, the only mode that sees a target on the "
                        "ground, but it is many times slower")
    p.add_argument("--yolo-model", default="yolov8m.pt",
                   help="yolo mode: weights to use. yolov8n.pt is ~8x faster and much weaker "
                        "on grounded targets")
    p.add_argument("--yolo-imgsz", type=int, default=1280,
                   help="yolo mode: inference size; smaller is faster and misses small targets")
    p.add_argument("--yolo-conf", type=float, default=0.25, help="yolo mode: confidence floor")
    p.add_argument("--no-yolo-focus", action="store_true",
                   help="yolo mode: always search the whole frame, instead of zooming the "
                        "search onto targets already being followed")
    p.add_argument("--yolo-focus-imgsz", type=int, default=640,
                   help="yolo mode: inference size for a focused crop")
    p.add_argument("--yolo-sweep", type=int, default=8,
                   help="yolo mode: search the whole frame every Nth frame, so new targets "
                        "can still appear while focused")
    p.add_argument("--yolo-classes", default="airplane,bird",
                   help="yolo mode: COCO class names to keep, comma separated. Adding 'kite' "
                        "catches more airframes but also fires on ground sheeting")
    p.add_argument("--no-show", dest="show", action="store_false", help="run without a preview window")
    p.add_argument("--mask", action="store_true", help="also show the motion mask (for tuning)")
    p.add_argument("--start-frame", type=int, default=0, help="skip to this frame before starting")
    p.add_argument("--max-frames", type=int, default=0, help="process at most this many frames; 0 means all")
    p.add_argument("--min-area", type=int, default=6, help="smallest blob to consider, px")
    p.add_argument("--max-area", type=int, default=0,
                   help="largest blob to consider, px; 0 scales it to the frame size")
    p.add_argument("--sensitivity", type=int, default=0,
                   help="detection threshold; lower finds fainter targets. 0 picks the default "
                        "for the chosen mode")
    p.add_argument("--merge-gap", type=int, default=0,
                   help="px gap within which blobs are treated as one target; 0 scales it to the "
                        "frame size. Raise it if one aircraft is split across several boxes")
    p.add_argument("--max-texture", type=float, default=20.0,
                   help="reject targets whose surroundings are this rough - keeps trees and "
                        "rooftops out. 0 disables, raise it to keep targets seen against clutter")
    p.add_argument("--min-contrast", type=float, default=8.0,
                   help="grey levels a target must stand out from nearby sky; raise it to reject noise")
    p.add_argument("--max-tracks", type=int, default=4, help="max targets drawn per frame")
    p.add_argument("--max-missed", type=int, default=20, help="frames a target may coast before it is dropped")
    p.add_argument("--max-distance", type=int, default=90, help="max px between prediction and detection to match")
    p.add_argument("--min-hits", type=int, default=3, help="detections needed before a target is confirmed")
    p.add_argument("--min-travel", type=float, default=18.0, help="px a target must cover to be confirmed")
    p.add_argument("--min-speed", type=float, default=15.0,
                   help="px/s a target must reach to be confirmed; raise it to ignore drifting clouds")
    p.add_argument("--min-straightness", type=float, default=0.55,
                   help="0-1 path directness needed to confirm; lower it for erratic targets like birds")
    p.add_argument("--steady-hits", type=int, default=10,
                   help="consecutive detections that confirm a target that barely moves in frame, "
                        "as when the camera pans to follow it")
    p.add_argument("--no-visual-hold", action="store_true",
                   help="do not follow a target visually once the detector loses it")
    p.add_argument("--visual-min-score", type=float, default=0.35,
                   help="appearance match the visual tracker must keep to stay trusted")
    p.add_argument("--visual-hold-frames", type=int, default=60,
                   help="frames a target may be carried visually before a detection must "
                        "confirm it again; 0 means no limit")
    p.add_argument("--follow-one", action="store_true",
                   help="draw and record a single target, held until it dies, instead of every "
                        "confirmed track")
    p.add_argument("--show-tentative", action="store_true", help="also draw unconfirmed candidates")
    p.add_argument("--window-height", type=int, default=900,
                   help="tallest the preview window may be, in px; the saved video is unaffected")
    p.add_argument("--proc-height", type=int, default=1080,
                   help="downscale taller footage to this height before tracking it; the "
                        "preview and telemetry stay in the video's own coordinates. 0 keeps "
                        "full resolution, which on 4K or portrait phone footage is slow")
    p.add_argument("--no-pace", action="store_true",
                   help="play the preview as fast as it processes instead of at the speed "
                        "of the footage")
    p.add_argument("--inset-size", type=int, default=200,
                   help="size of the magnified LIVE panel, in px")
    p.add_argument("--search-contrast", type=float, default=45.0,
                   help="lock mode: contrast a candidate must have to be considered when "
                        "looking for a lost target. Higher than --min-contrast on purpose: "
                        "the search wants one known object, not every possible one. Lower it "
                        "if a faint target is never found again")
    p.add_argument("--search-every", type=int, default=2,
                   help="lock mode: run the search on every Nth frame while the target is "
                        "missing; 1 searches every frame and costs the most")
    p.add_argument("--max-static", type=int, default=0,
                   help="lock mode: roughly how many trailing frames of an unchanging patch "
                        "before it's treated as stuck on something drawn into the video (a "
                        "HUD readout, a logo, a timestamp) rather than the real target - a "
                        "burned-in graphic reads as a perfect match forever, so nothing else "
                        "catches it. Off by default because a camera following an aircraft "
                        "against smooth overcast produces a barely-changing patch too, and "
                        "this would call that stuck. Turn it on (try 45) for footage with "
                        "graphics burned into it")
    p.add_argument("--static-eps", type=float, default=2.0,
                   help="lock mode: median pixel difference below which two frames' patches "
                        "count as unchanged, for --max-static. Real footage - even of "
                        "something motionless - carries more sensor noise than this; raise it "
                        "if a genuinely static target is wrongly called stuck")
    p.add_argument("--reacquire-margin", type=float, default=0.0,
                   help="lock mode: how far the best candidate must beat the runner-up "
                        "before the lock is handed to it. Raise it (try 0.1) in traffic or "
                        "a flock, where several candidates are the same size and colour and "
                        "picking the top score is a coin toss; 0 always takes the best")
    p.add_argument("--max-unsupported", type=int, default=0,
                   help="lock mode: how many resync passes in a row may find nothing where "
                        "the box is before the lock is dropped and the search restarted. "
                        "Catches a tracker that has slid off the target a few pixels a frame, "
                        "too slowly for any jump limit to notice - but only worth turning on "
                        "when the detector can see the target nearly every frame, as yolo can "
                        "on vehicles. Against sky the detector loses an aircraft for long "
                        "stretches quite normally, and carrying it through those gaps is what "
                        "lock mode is for, so 0 (off) is the default. Try 4 with --mode yolo")
    p.add_argument("--exclude", action="append", default=[], metavar="X,Y,W,H",
                   help="lock mode: a screen region the search may never pick a candidate "
                        "from, for a HUD readout, logo or timestamp burned into the footage "
                        "in a place --max-static keeps rediscovering. Repeatable")
    p.add_argument("--reacquire-min-appearance", type=float, default=0.15,
                   help="lock mode: how much a candidate must look like the target before "
                        "position counts at all. 0 re-locks on position alone")
    p.add_argument("--reacquire-confirm", type=int, default=25,
                   help="lock mode: frames a rediscovered target must hold together before "
                        "it is believed and what the target looks like is relearned")
    p.add_argument("--reject-frames", type=int, default=90,
                   help="lock mode: how long a re-lock that fell apart is refused, so the "
                        "same wrong blob does not win the next search")
    p.add_argument("--bytetrack", action="store_true",
                   help="associate detections with ByteTrack: every detection the detector "
                        "makes is used, including the faint ones it would normally suppress, "
                        "and each target keeps a Kalman-predicted identity through the frames "
                        "nothing is seen. In lock mode this is what holds onto the target you "
                        "picked - if its ID survives, it is reacquired the moment it reappears "
                        "instead of being searched for. The cost is that the detector must run "
                        "on every frame rather than occasionally, which with --mode yolo is "
                        "the difference between hundreds of fps and a handful")
    p.add_argument("--byte-high", type=float, default=0.5,
                   help="bytetrack: confidence at or above which a detection is trusted enough "
                        "to match against any track, and to start a new one")
    p.add_argument("--byte-low", type=float, default=0.1,
                   help="bytetrack: confidence below which a detection is ignored entirely. "
                        "Between this and --byte-high a detection is offered only to tracks "
                        "that found nothing better, which is what carries a target through "
                        "being blurred, occluded or too small to score well")
    p.add_argument("--byte-match", type=float, default=0.8,
                   help="bytetrack: how far apart a track's prediction and a detection may be "
                        "and still be matched, as 1 - IoU; 0.8 means they need only overlap "
                        "slightly. Lower it where targets crowd each other")
    p.add_argument("--repick", action="append", default=[], metavar="FRAME:X,Y,W,H",
                   help="lock mode: pick the target again at this frame, as pressing R and "
                        "drawing a box does during a run - which prints the matching "
                        "--repick, so a run corrected by hand can be repeated. Repeatable")
    p.add_argument("--no-reid", action="store_true",
                   help="bytetrack + yolo: do not remember the target's appearance. With it, "
                        "the target you picked is fingerprinted, the memory grows while it is "
                        "tracked, and a lost target is only taken back when one candidate "
                        "clearly looks like it over several detection passes in a row")
    p.add_argument("--reid-min-likeness", type=float, default=0.68,
                   help="reid: how closely a candidate must resemble the remembered target. "
                        "Measured, the target scores 0.70-0.91 when it returns and the best "
                        "wrong car 0.30-0.71")
    p.add_argument("--reid-margin", type=float, default=0.08,
                   help="reid: how far the best candidate must lead the runner-up")
    p.add_argument("--reid-passes", type=int, default=4,
                   help="reid: detection passes running a candidate must win before it is "
                        "given the lock")
    p.add_argument("--reid-sole-passes", type=int, default=8,
                   help="reid: take the target back when one identity is the only candidate "
                        "for this many detection passes, however unlike its memory it looks "
                        "(a lone aircraft seen from a new angle); 0 disables")
    p.add_argument("--reid-guard", type=int, default=8,
                   help="reid: the lock is let go once the detector has seen nothing where it "
                        "is for twice this many passes running - how a lock that has slid onto "
                        "scenery shows itself. Raise it if a briefly hidden target is let go "
                        "too readily")
    p.add_argument("--no-cmc", action="store_true",
                   help="bytetrack: do not compensate for camera motion. Compensation moves "
                        "every prediction with the camera, which is what keeps an identity "
                        "through a pan or a zoom; turn it off only for a camera on a tripod")
    p.add_argument("--detect-threads", type=int, default=0,
                   help="bytetrack: cores the detector process may use; 0 lets it take what "
                        "it likes. Cap it if the preview still stutters - the detector gets "
                        "slower, but the tracking loop keeps the cores it needs to hit the "
                        "frame rate")
    p.add_argument("--detect-sync", action="store_true",
                   help="bytetrack: run the detector in the tracking loop rather than in a "
                        "process of its own. The preview then waits for every detection "
                        "pass, which with --mode yolo drops it from around 35 frames a "
                        "second to 3; in exchange every frame is detected on, and two runs "
                        "of the same clip give identical output. Headless runs (--no-show) "
                        "and the contrast modes already work this way, so this only changes "
                        "what a yolo-mode preview does")
    p.add_argument("--byte-expand", type=float, default=1.0,
                   help="bytetrack: how far each box is inflated by its own size before "
                        "overlaps are measured. Plain IoU cannot match a target that moves "
                        "further than its own width in a frame - which a distant speck "
                        "crossing the sky does constantly, and measured, it was then never "
                        "tracked at all. 1.0 tolerates twice its own size per frame; 0 is the "
                        "ByteTrack paper's own IoU. Lower it where targets crowd each other")
    p.add_argument("--byte-buffer", type=int, default=60,
                   help="bytetrack: frames a target that has stopped being detected keeps its "
                        "identity, coasting on its predicted motion. At 30fps the default is "
                        "two seconds; raise it for longer occlusions, at the risk of the "
                        "prediction having drifted somewhere wrong by the time it returns")
    p.add_argument("--resync-max-growth", type=float, default=4.0,
                   help="lock mode: most a resync may enlarge the box by, as a multiple of "
                        "its area; this is what stops a cloud swallowing the lock")
    return p


def main():
    args = build_parser().parse_args()
    run_lock(args) if (args.lock or args.lock_bbox) else run_auto(args)


if __name__ == "__main__":
    main()
