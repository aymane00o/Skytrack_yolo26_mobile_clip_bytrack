"""ByteTrack: association that keeps a target through the frames it looks worst.

Every detector has a confidence floor, and the frames where a target is about to
be lost are exactly the frames it scores worst on: half behind a pole, motion
blurred, turned edge-on, shrunk to a dozen pixels. Throw those away and the
track breaks precisely when it mattered, and whatever picks the target up
afterwards gets a new identity - which is how "the target it was told to follow"
quietly becomes "some object near where that one was".

ByteTrack (Zhang et al., 2022) keeps them. Detections are split at a confidence
threshold and associated in two passes:

  1. the confident detections, against every track, including ones already lost,
  2. the leftover low-confidence detections, against the tracks pass one did not
     match.

A low-scoring box on its own is usually noise, which is why detectors discard
it. A low-scoring box that lands where an established track predicted it would
be is almost always that track, having a bad frame. The second pass is the whole
idea, and it is what this file is for.

Motion is a Kalman filter over (centre, aspect, height) at constant velocity, so
a track that matches nothing still has a predicted position to be looked for at
when it comes back - for up to `buffer` frames, keeping its ID the whole time.

This is motion-only, as the paper is: it can still hand one car's ID to another
car that crosses it closely enough. Telling apart two things that look alike is
what the appearance memory in track.py is for, and the two are used together.
"""

from typing import NamedTuple

import cv2
import numpy as np

from tracker import Track, bbox_center

INF = float("inf")


# ---------------------------------------------------------------- assignment

def linear_assignment(cost, max_cost):
    """Cheapest one-to-one pairing of rows to columns, refusing pairs over max_cost.

    Greedy nearest-first is what the older tracker in tracker.py does, and it is
    good enough while targets stay far apart - but it takes a locally cheap pair
    even when that forces a much worse one afterwards, which in a close pass
    between two targets is exactly how their IDs get swapped. This is the
    Hungarian method in its shortest-augmenting-path form, which minimises the
    total cost over the whole frame and so has no such failure.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    # The solver below needs at least as many columns as rows; the transposed
    # problem has the same solution, with each pair read the other way round.
    if rows > cols:
        pairs, un_cols, un_rows = linear_assignment(cost.T, max_cost)
        return [(r, c) for c, r in pairs], un_rows, un_cols

    assigned = _hungarian(np.ascontiguousarray(cost, dtype=np.float64))
    matches, unmatched_rows = [], []
    for row, col in enumerate(assigned):
        if col >= 0 and cost[row, col] <= max_cost:
            matches.append((row, int(col)))
        else:
            unmatched_rows.append(row)
    taken = {c for _, c in matches}
    return matches, unmatched_rows, [c for c in range(cols) if c not in taken]


def _hungarian(cost):
    """Column chosen for each row, or -1. Requires rows <= cols."""
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    parent = np.zeros(m + 1, dtype=np.int64)     # which row currently holds each column
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        parent[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = parent[j0]
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (reduced < minv[1:])
            minv[1:][better] = reduced[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(candidates)) + 1
            delta = minv[j1]
            if not np.isfinite(delta):
                break
            u[parent[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if parent[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            parent[j0] = parent[j1]
            j0 = j1

    out = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if parent[j]:
            out[parent[j] - 1] = j - 1
    return out


def ious(a, b):
    """Intersection over union of two sets of x,y,w,h boxes, as a matrix."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax1, ay1 = a[:, 0, None], a[:, 1, None]
    ax2, ay2 = ax1 + a[:, 2, None], ay1 + a[:, 3, None]
    bx1, by1 = b[None, :, 0], b[None, :, 1]
    bx2, by2 = bx1 + b[None, :, 2], by1 + b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (a[:, 2, None] * a[:, 3, None]) + (b[None, :, 2] * b[None, :, 3]) - inter
    return inter / np.maximum(union, 1e-6)


def buffer_boxes(boxes, amount):
    """Grow each box by a share of its own size, on every side."""
    if not amount:
        return boxes
    grown = []
    for x, y, w, h in boxes:
        dx, dy = w * amount, h * amount
        grown.append((x - dx, y - dy, w + 2 * dx, h + 2 * dy))
    return grown


def iou_distance(tracks, boxes, expand=0.0):
    """How badly each track and box fail to overlap, as 1 - IoU.

    Plain IoU has a hard limit that this domain runs straight into: two boxes
    that do not touch score zero no matter how close they are, so a target that
    moves further than its own width between frames can never be matched to
    itself. A 20px speck crossing the frame at 15px a frame is not an unusual
    case here, it is the normal one - and measured, it was never tracked at all:
    every frame started a fresh track which was gone before it could learn a
    velocity that would have predicted the next one.

    So both sides are inflated by a share of their own size first (the buffered
    IoU of Yang et al., 2023). The tolerance then scales with the target instead
    of being capped by it, and once a track has survived two frames its Kalman
    prediction is doing the work anyway. `expand=0` is the plain IoU the
    ByteTrack paper uses.
    """
    return 1.0 - ious(buffer_boxes([t.tlwh for t in tracks], expand),
                      buffer_boxes(boxes, expand))


# ------------------------------------------------------- camera compensation

class CameraMotion:
    """How the camera moved between two frames, as a similarity transform.

    Every prediction ByteTrack makes is in screen coordinates, which quietly
    assumes the camera holds still. Aerial footage does nothing of the kind -
    measured on the police-chase clip, the camera swings up to 36px and zooms
    up to 5.5% in a single frame, while the car it is following moves about 4px.
    A track that is not matched for a few frames during that is predicted where
    the car *was on screen*, which is somewhere else entirely by then, so it is
    never matched again and the identity is gone. BoT-SORT (Aharon et al., 2022)
    fixes this by moving every prediction with the camera, and so does this.

    Estimated from sparse optical flow on a small copy of the frame, with the
    moving objects masked out - they are exactly the points that do not move
    with the camera - and with any burned-in overlay masked out too, since a HUD
    is drawn in the same place however the camera moves and votes for "still".
    """

    def __init__(self, width=320, static_regions=(), border=8):
        self.width = width
        self.static_regions = list(static_regions)
        self.border = border
        self.prev = None
        self.prev_boxes = []

    def estimate(self, frame, boxes=()):
        """2x3 transform taking the previous frame's coordinates to this one's."""
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        h, w = frame.shape[:2]
        k = self.width / float(w)
        small = cv2.resize(frame, (self.width, max(int(h * k), 1)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        prev, prev_boxes = self.prev, self.prev_boxes
        self.prev, self.prev_boxes = gray, list(boxes)
        if prev is None or prev.shape != gray.shape:
            return identity

        mask = np.zeros_like(prev)
        b = self.border
        mask[b:-b, b:-b] = 255
        for x, y, bw, bh in list(prev_boxes) + self.static_regions:
            x0, y0 = int(max(x * k, 0)), int(max(y * k, 0))
            mask[y0:int(np.ceil((y + bh) * k)), x0:int(np.ceil((x + bw) * k))] = 0

        points = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01,
                                         minDistance=6, mask=mask)
        if points is None or len(points) < 12:
            return identity
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, points, None,
                                                    winSize=(21, 21), maxLevel=3)
        good = status.ravel() == 1
        if good.sum() < 12:
            return identity
        matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good],
                                                      method=cv2.RANSAC,
                                                      ransacReprojThreshold=1.5)
        if matrix is None or inliers is None or inliers.sum() < 10:
            return identity
        matrix = matrix.astype(np.float64)
        matrix[:, 2] /= k                      # the shift was measured on the small copy
        return matrix


def warp_track(track, matrix):
    """Move a track's Kalman state with the camera.

    The state is (cx, cy, aspect, height) and their rates. The centre and its
    velocity rotate and scale with the camera and the centre also shifts; the
    height and its rate scale with the zoom; the aspect ratio does not change at
    all, which is why this cannot simply apply the matrix to every pair of
    numbers the way an (x, y, w, h) state would allow.
    """
    if track.mean is None:
        return
    rot = matrix[:, :2]
    zoom = float(np.sqrt(abs(np.linalg.det(rot))))
    t = np.eye(8)
    t[0:2, 0:2] = rot
    t[4:6, 4:6] = rot
    t[3, 3] = zoom
    t[7, 7] = zoom
    track.mean = t @ track.mean
    track.mean[0:2] += matrix[:, 2]
    track.cov = t @ track.cov @ t.T


# -------------------------------------------------------------- motion model

class KalmanFilter:
    """Constant-velocity filter on (cx, cy, aspect, height) and their rates.

    Uncertainty is scaled by the target's height throughout: how far a box can
    plausibly move between two frames, and how precisely a detector places its
    edges, both scale with how big it is on screen. A fixed figure in pixels
    would be far too loose for a distant speck and far too tight for something
    filling the frame.
    """

    def __init__(self):
        self.motion = np.eye(8)
        for i in range(4):
            self.motion[i, i + 4] = 1.0
        self.observe = np.eye(4, 8)
        self.pos_weight = 1.0 / 20
        self.vel_weight = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = np.array([
            2 * self.pos_weight * h, 2 * self.pos_weight * h, 1e-2, 2 * self.pos_weight * h,
            10 * self.vel_weight * h, 10 * self.vel_weight * h, 1e-5, 10 * self.vel_weight * h,
        ])
        return mean, np.diag(std ** 2)

    def predict(self, mean, cov):
        h = mean[3]
        std = np.array([
            self.pos_weight * h, self.pos_weight * h, 1e-2, self.pos_weight * h,
            self.vel_weight * h, self.vel_weight * h, 1e-5, self.vel_weight * h,
        ])
        mean = self.motion @ mean
        cov = self.motion @ cov @ self.motion.T + np.diag(std ** 2)
        return mean, cov

    def project(self, mean, cov):
        h = mean[3]
        std = np.array([self.pos_weight * h, self.pos_weight * h, 1e-1, self.pos_weight * h])
        projected = self.observe @ cov @ self.observe.T + np.diag(std ** 2)
        return self.observe @ mean, projected

    def update(self, mean, cov, measurement):
        projected_mean, projected_cov = self.project(mean, cov)
        # Solving beats inverting: the projected covariance is only 4x4, but it
        # goes near-singular whenever a target holds still for a while, and an
        # explicit inverse of it then produces a gain that throws the state away.
        gain = np.linalg.solve(projected_cov, self.observe @ cov).T
        innovation = measurement - projected_mean
        return mean + gain @ innovation, cov - gain @ projected_cov @ gain.T


TRACKED, LOST, REMOVED = 0, 1, 2


class Seen(NamedTuple):
    """One track as the tracking loop sees it: an identity and where it is.

    `velocity` is the Kalman estimate of how fast the centre is moving, in
    pixels per frame. It is here so a caller reading this a few frames after it
    was made can work out where the target will have got to since - which is
    exactly the position of anything watching a detector that runs behind the
    footage.
    """
    id: int
    tlwh: tuple
    score: float
    lost: bool
    velocity: tuple = (0.0, 0.0)
    # Appearance fingerprint (reid.fingerprint) from the frame this was made on,
    # or None for a lost track, which has nothing current to be fingerprinted.
    feature: object = None

    def after(self, frames):
        """Where this track should be `frames` frames later."""
        if not frames:
            return self.tlwh
        x, y, w, h = self.tlwh
        return (x + self.velocity[0] * frames, y + self.velocity[1] * frames, w, h)


class STrack:
    """One tracked object: a Kalman state, an ID, and how it is doing."""

    _count = 0

    def __init__(self, tlwh, score, kalman):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = float(score)
        self.kalman = kalman
        self.mean = self.cov = None
        self.id = 0
        self.state = TRACKED
        self.activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.hits = 0
        self.tracklet_len = 0
        # The detection this track was last matched to, as the detector drew it.
        self.measured = tuple(float(v) for v in tlwh)

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_ids():
        STrack._count = 0

    @property
    def tlwh(self):
        if self.mean is None:
            return tuple(self._tlwh)
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return (cx - w / 2.0, cy - h / 2.0, w, h)

    @property
    def center(self):
        return bbox_center(self.tlwh)

    @property
    def lost(self):
        return self.state == LOST

    @staticmethod
    def to_xyah(tlwh):
        x, y, w, h = tlwh
        return np.array([x + w / 2.0, y + h / 2.0, w / max(h, 1e-6), h], dtype=np.float64)

    def activate(self, frame_id):
        self.mean, self.cov = self.kalman.initiate(self.to_xyah(self._tlwh))
        self.id = self.next_id()
        self.state = TRACKED
        self.tracklet_len = 0
        self.hits = 1
        # A target seen on the very first frame has nothing earlier to be
        # confirmed against, so it counts immediately; every later one has to be
        # seen twice before it is reported, which is what stops a single frame of
        # detector noise from ever becoming a track.
        self.activated = frame_id <= 1
        self.frame_id = self.start_frame = frame_id

    def predict(self):
        mean = self.mean.copy()
        if self.state != TRACKED:
            # Nothing has confirmed the target's size since it was lost, so hold
            # its height rather than carrying on shrinking or growing at whatever
            # rate it happened to have at the moment it vanished.
            mean[7] = 0
        self.mean, self.cov = self.kalman.predict(mean, self.cov)

    def update(self, box, score, frame_id):
        self.measured = tuple(float(v) for v in box)
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, self.to_xyah(box))
        self.state = TRACKED
        self.activated = True
        self.frame_id = frame_id
        self.score = float(score)
        self.hits += 1
        self.tracklet_len += 1

    def reactivate(self, box, score, frame_id, new_id=False):
        self.measured = tuple(float(v) for v in box)
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, self.to_xyah(box))
        self.state = TRACKED
        self.activated = True
        self.frame_id = frame_id
        self.score = float(score)
        self.hits += 1
        self.tracklet_len = 0
        if new_id:
            self.id = self.next_id()

    def mark_lost(self):
        self.state = LOST

    def mark_removed(self):
        self.state = REMOVED


def _join(a, b):
    seen = {t.id for t in a}
    return list(a) + [t for t in b if t.id not in seen]


def _subtract(a, b):
    drop = {t.id for t in b}
    return [t for t in a if t.id not in drop]


def _drop_duplicates(a, b, threshold=0.15):
    """Two tracks sitting on the same object: keep whichever has existed longer.

    Happens when a track is lost, a new one starts on the same target, and then
    the old one is found again. Left alone the pair fight over the same
    detections every frame and the ID flickers between them.
    """
    if not a or not b:
        return a, b
    overlap = 1.0 - iou_distance(a, [t.tlwh for t in b])
    dup_a, dup_b = set(), set()
    for i, j in zip(*np.where(overlap > 1 - threshold)):
        age_a = a[i].frame_id - a[i].start_frame
        age_b = b[j].frame_id - b[j].start_frame
        if age_a > age_b:
            dup_b.add(int(j))
        else:
            dup_a.add(int(i))
    return ([t for i, t in enumerate(a) if i not in dup_a],
            [t for i, t in enumerate(b) if i not in dup_b])


class ByteTracker:
    """The two-pass association loop.

    `update` takes this frame's boxes and their confidences and returns the
    tracks currently being seen. Tracks that were not matched are kept for
    `buffer` frames, predicted forward, and stay eligible for matching that whole
    time - so a target that goes behind something comes back with the ID it had,
    rather than as a new object.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8,
                 buffer=30, new_thresh=None, expand=1.0):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.new_thresh = high_thresh + 0.1 if new_thresh is None else new_thresh
        self.max_lost = buffer
        self.expand = expand
        self.kalman = KalmanFilter()
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        # Frames of footage the last update covered. The Kalman velocity is per
        # update, and when the detector cannot keep up an update spans several
        # frames - so anything projecting a box forward in frames has to divide by
        # this, or it overshoots by the same factor.
        self.step = 1

    def update(self, boxes, scores, warp=None, frames=1):
        """Advance one frame. Returns the tracks being seen right now.

        `warp` is how the camera moved since the previous update (see
        CameraMotion); every prediction is carried along with it before anything
        is matched, lost tracks included - they are the ones it matters most for.
        """
        self.frame_id += 1
        self.step = max(int(frames), 1)
        boxes = [tuple(float(v) for v in b) for b in boxes]
        scores = [float(s) for s in scores]

        high = [i for i, s in enumerate(scores) if s >= self.high_thresh]
        low = [i for i, s in enumerate(scores) if self.low_thresh <= s < self.high_thresh]

        # A track seen only once is not reported yet, but it still has to compete
        # for detections, or the object it sits on starts a fresh track every
        # frame and never gets confirmed at all.
        unconfirmed = [t for t in self.tracked if not t.activated]
        tracked = [t for t in self.tracked if t.activated]

        pool = _join(tracked, self.lost)
        for track in pool:
            track.predict()
        if warp is not None:
            for track in pool + unconfirmed:
                warp_track(track, warp)

        # Pass one: the confident detections, against everything - including the
        # tracks that are currently lost, which is what lets a target that has
        # been missing for a second simply be recognised when it reappears.
        matches, un_tracks, un_dets = linear_assignment(
            iou_distance(pool, [boxes[i] for i in high], self.expand), self.match_thresh)
        activated, refound = [], []
        for ti, di in matches:
            track, box, score = pool[ti], boxes[high[di]], scores[high[di]]
            if track.state == TRACKED:
                track.update(box, score, self.frame_id)
                activated.append(track)
            else:
                track.reactivate(box, score, self.frame_id)
                refound.append(track)

        # Pass two: the boxes the detector was unsure about, offered only to the
        # tracks pass one left empty-handed. This is what an ordinary tracker
        # throws away, and it is usually the target itself having a bad frame -
        # half occluded, motion blurred, or briefly too small to score well.
        leftover = [pool[i] for i in un_tracks if pool[i].state == TRACKED]
        matches, un_leftover, _ = linear_assignment(
            iou_distance(leftover, [boxes[i] for i in low], self.expand), 0.5)
        for ti, di in matches:
            track, box, score = leftover[ti], boxes[low[di]], scores[low[di]]
            track.update(box, score, self.frame_id)
            activated.append(track)

        lost = []
        for ti in un_leftover:
            track = leftover[ti]
            if track.state == TRACKED:
                track.mark_lost()
                lost.append(track)

        # A track that has been seen exactly once and is not seen again is
        # dropped outright rather than kept, so a detector flicker cannot linger
        # as a ghost with a prediction of its own.
        remaining = [boxes[high[i]] for i in un_dets]
        remaining_scores = [scores[high[i]] for i in un_dets]
        matches, un_unconfirmed, un_remaining = linear_assignment(
            iou_distance(unconfirmed, remaining, self.expand), 0.7)
        removed = []
        for ti, di in matches:
            unconfirmed[ti].update(remaining[di], remaining_scores[di], self.frame_id)
            activated.append(unconfirmed[ti])
        for ti in un_unconfirmed:
            unconfirmed[ti].mark_removed()
            removed.append(unconfirmed[ti])

        for di in un_remaining:
            if remaining_scores[di] < self.new_thresh:
                continue
            track = STrack(remaining[di], remaining_scores[di], self.kalman)
            track.activate(self.frame_id)
            activated.append(track)

        for track in self.lost:
            if self.frame_id - track.frame_id > self.max_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked = _join(_join([t for t in self.tracked if t.state == TRACKED],
                                   activated), refound)
        self.lost = _subtract(_join(_subtract(self.lost, self.tracked), lost), removed)
        self.lost = [t for t in self.lost if t.state == LOST]
        self.removed = (self.removed + removed)[-1000:]
        self.tracked, self.lost = _drop_duplicates(self.tracked, self.lost)
        return [t for t in self.tracked if t.activated]

    def get(self, track_id):
        """A track by ID, tracked or lost, or None once it has been dropped."""
        for track in self.tracked:
            if track.id == track_id:
                return track
        for track in self.lost:
            if track.id == track_id:
                return track
        return None

    def snapshot(self, frame=None, describe=None):
        """A plain, frozen copy of what is being tracked right now.

        Everything the tracking loop needs to read, and nothing that can change
        underneath it. That matters when the detector is running on its own
        thread: the loop reads one of these while the next frame's association
        is already being worked out, and neither can see the other half-done.
        """
        def speed(track):
            if track.mean is None:
                return (0.0, 0.0)
            return (track.mean[4] / self.step, track.mean[5] / self.step)

        def looks(track):
            # Fingerprinted from the detection, not from the Kalman estimate. The
            # estimate is smoothed and lags a moving target - with the detector
            # running every third frame it sat only half on the car it described,
            # measured at IoU 0.46-0.59 - and a layout fingerprint of a crop that
            # is half road does not resemble the car at all. The detection is
            # where the detector actually saw it.
            if not (describe and frame is not None) or track.frame_id != self.frame_id:
                return None
            return describe(frame, track.measured)

        def where(track):
            # A track matched on this update is where the detector saw it; the
            # Kalman estimate is smoothed and lags, and a lock placed on it lands
            # half on the road. Only a track with no measurement this update needs
            # the estimate.
            return track.measured if track.frame_id == self.frame_id else track.tlwh

        return ([Seen(t.id, where(t), t.score, False, speed(t), looks(t))
                 for t in self.tracked if t.activated]
                + [Seen(t.id, t.tlwh, t.score, True, speed(t)) for t in self.lost])


class ByteMultiTracker:
    """ByteTrack behind the same interface as tracker.MultiTracker.

    Auto mode's overlay, telemetry and visual hold all speak in terms of the
    Track objects in tracker.py - the confirmation rules, the trail, coasting -
    and none of that has to change to swap out how detections are assigned. So
    each ByteTrack track is mirrored by one of those, and only the matching
    differs.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, buffer=30,
                 expand=1.0, min_hits=3, min_travel=18.0, min_speed=15.0,
                 min_straightness=0.55, steady_hits=10, bounds=None):
        self.byte = ByteTracker(high_thresh, low_thresh, match_thresh, buffer, expand=expand)
        self.tracks = {}
        self.bounds = bounds
        self.settings = dict(min_hits=min_hits, min_travel=min_travel, min_speed=min_speed,
                             min_straightness=min_straightness, steady_hits=steady_hits)

    def update(self, scored, timestamp):
        """`scored` is this frame's (box, confidence) pairs."""
        seen = self.byte.update([b for b, _ in scored], [s for _, s in scored])

        alive = set()
        for strack in seen:
            alive.add(strack.id)
            track = self.tracks.get(strack.id)
            if track is None:
                self.tracks[strack.id] = Track(strack.id, strack.tlwh, timestamp, **self.settings)
            else:
                track.update(strack.tlwh, timestamp)

        # A track ByteTrack is holding through a gap is not gone - it is coasting
        # on its Kalman prediction, which is exactly what the overlay's coasting
        # state already means.
        for strack in self.byte.lost:
            track = self.tracks.get(strack.id)
            if track is not None and strack.id not in alive:
                alive.add(strack.id)
                track.coast(timestamp)

        for tid in [t for t in self.tracks if t not in alive]:
            del self.tracks[tid]
        if self.bounds is not None:
            width, height = self.bounds
            for tid, track in list(self.tracks.items()):
                x, y = track.center
                if not (0 <= x < width and 0 <= y < height):
                    del self.tracks[tid]
        return list(self.tracks.values())

    def confirmed(self):
        return [t for t in self.tracks.values() if t.locked]
