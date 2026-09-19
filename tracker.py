"""Multi-target tracking with persistent IDs, prediction and coasting."""

import numpy as np


def bbox_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0])


class Track:
    def __init__(self, track_id, bbox, timestamp, min_hits=3, min_travel=18.0, min_speed=15.0,
                 min_straightness=0.55, steady_hits=10):
        self.id = track_id
        self.bbox = bbox
        self.center = bbox_center(bbox)
        self.origin = self.center.copy()
        self.velocity = np.zeros(2)
        self.speed = 0.0
        self.hits = 1
        self.missed = 0
        self.coasting = False
        self.visual = False
        self.last_timestamp = timestamp
        self.trail = [self.center.copy()]
        self.min_hits = min_hits
        self.min_travel = min_travel
        self.min_speed = min_speed
        self.min_straightness = min_straightness
        self.steady_hits = steady_hits
        self._was_locked = False

    @property
    def travel(self):
        return float(np.linalg.norm(self.center - self.origin))

    @property
    def straightness(self):
        """Net displacement over path length: 1.0 is a straight line, ~0 is jitter."""
        if len(self.trail) < 4:
            return 0.0
        pts = np.array(self.trail[-20:])
        path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if path < 1e-6:
            return 0.0
        return float(np.linalg.norm(pts[-1] - pts[0]) / path)

    @property
    def locked(self):
        """Confirmed target.

        A target crossing the frame is confirmed quickly: it covers ground, moves
        at a real rate, and holds a heading, which drifting cloud edges and noise
        do not. But when the operator pans to follow the target it barely moves in
        frame at all, so a track that keeps producing detections frame after frame
        is confirmed on that evidence instead - noise seldom survives that long
        without a gap. Once confirmed a track stays confirmed, otherwise the label
        would flicker every time the target slowed or turned.
        """
        if self._was_locked:
            return True
        moving = (
            self.hits >= self.min_hits
            and self.travel >= self.min_travel
            and self.speed >= self.min_speed
            and self.straightness >= self.min_straightness
        )
        self._was_locked = moving or self.hits >= self.steady_hits
        return self._was_locked

    def predict(self, timestamp):
        dt = max(timestamp - self.last_timestamp, 0.0)
        return self.center + self.velocity * dt

    def update(self, bbox, timestamp):
        new_center = bbox_center(bbox)
        dt = max(timestamp - self.last_timestamp, 1e-6)
        measured_velocity = (new_center - self.center) / dt
        # Smooth velocity so a single noisy box doesn't throw the prediction off.
        self.velocity = 0.6 * self.velocity + 0.4 * measured_velocity if self.hits > 1 else measured_velocity
        self.speed = float(np.linalg.norm(self.velocity))
        self.bbox = bbox
        self.center = new_center
        self.last_timestamp = timestamp
        self.hits += 1
        self.missed = 0
        self.coasting = False
        self.visual = False
        self.trail.append(self.center.copy())
        if len(self.trail) > 40:
            self.trail.pop(0)

    def adopt(self, bbox):
        """Take a position from the visual tracker instead of from a detection.

        Used where the detector cannot work - a target that has descended into
        clutter - so the track keeps following it rather than coasting blind.
        """
        self.bbox = bbox
        self.center = bbox_center(bbox)
        self.missed = 0
        self.visual = True
        self.trail[-1] = self.center.copy()

    def coast(self, timestamp):
        """No observation this frame: keep flying on the last known velocity."""
        self.center = self.predict(timestamp)
        x, y, w, h = self.bbox
        self.bbox = (self.center[0] - w / 2.0, self.center[1] - h / 2.0, w, h)
        self.last_timestamp = timestamp
        self.missed += 1
        self.coasting = True
        self.trail.append(self.center.copy())
        if len(self.trail) > 40:
            self.trail.pop(0)


class MultiTracker:
    """Greedy one-to-one assignment of detections to predicted track positions.

    One-to-one matching is what stops two tracks from claiming the same blob when
    targets pass close to each other.
    """

    def __init__(self, max_missed=20, max_distance=90, min_hits=3, min_travel=18.0, min_speed=15.0,
                 min_straightness=0.55, steady_hits=10, bounds=None):
        self.tracks = {}
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.min_travel = min_travel
        self.min_speed = min_speed
        self.min_straightness = min_straightness
        self.steady_hits = steady_hits
        self.bounds = bounds
        self._next_id = 1

    def _match(self, track_ids, boxes, claimed_boxes, claimed_tracks, timestamp):
        pairs = []
        for tid in track_ids:
            predicted = self.tracks[tid].predict(timestamp)
            for bi, bbox in enumerate(boxes):
                if bi in claimed_boxes:
                    continue
                dist = float(np.linalg.norm(bbox_center(bbox) - predicted))
                if dist <= self.max_distance:
                    pairs.append((dist, tid, bi))
        pairs.sort()

        for _dist, tid, bi in pairs:
            if tid in claimed_tracks or bi in claimed_boxes:
                continue
            self.tracks[tid].update(boxes[bi], timestamp)
            claimed_tracks.add(tid)
            claimed_boxes.add(bi)

    def update(self, boxes, timestamp):
        track_ids = list(self.tracks.keys())
        claimed_tracks, claimed_boxes = set(), set()

        # Established tracks choose first. Otherwise a stale track coasting on a
        # bad velocity can land nearer the detection than the track that has been
        # following the target all along, steal it, and kill the real track.
        live = [t for t in track_ids if not self.tracks[t].coasting]
        confirmed = [t for t in live if self.tracks[t].locked]
        for group in (confirmed, [t for t in live if t not in confirmed],
                      [t for t in track_ids if t not in live]):
            self._match(group, boxes, claimed_boxes, claimed_tracks, timestamp)

        for tid in track_ids:
            if tid not in claimed_tracks:
                self.tracks[tid].coast(timestamp)

        for bi, bbox in enumerate(boxes):
            if bi not in claimed_boxes:
                self.tracks[self._next_id] = Track(
                    self._next_id, bbox, timestamp,
                    min_hits=self.min_hits, min_travel=self.min_travel, min_speed=self.min_speed,
                    min_straightness=self.min_straightness, steady_hits=self.steady_hits,
                )
                self._next_id += 1

        for tid in [t for t, trk in self.tracks.items() if self._finished(trk)]:
            del self.tracks[tid]

        return list(self.tracks.values())

    def _finished(self, track):
        if track.missed > self.max_missed:
            return True
        # A prediction that has left the frame is not a target any longer, and
        # reporting it would put impossible coordinates in the telemetry.
        if self.bounds is not None:
            width, height = self.bounds
            x, y = track.center
            return not (0 <= x < width and 0 <= y < height)
        return False

        return list(self.tracks.values())

    def confirmed(self):
        return [t for t in self.tracks.values() if t.locked]
