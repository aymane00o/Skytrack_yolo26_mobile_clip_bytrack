"""Remembering the target: re-identification for a target that has been lost.

ByteTrack keeps an identity for as long as motion can carry it. Once a target has
been gone longer than that - behind trees, out of shot, through a camera swing -
its track is gone too, and whatever comes back is a new track with a new ID. What
connects the two is appearance, and this file is the memory of what the target
you picked looks like.

Three things about this footage shape it, all measured:

- A single appearance score is not enough. On the police-chase clip, every
  fingerprint tried - the old 48x48 template, the detector's own features,
  colour, shape - was beaten by some other car in the scene more often than not.
  Look-alikes are the normal case in traffic, so an absolute threshold that lets
  the target back in also lets a dozen strangers in.
- So a candidate is judged against the *other cars*, not against a fixed number.
  While the target is locked, every other track in the scene is known not to be
  it, and their fingerprints are kept as negatives. A candidate has to look more
  like the target than like any car it is known not to be - which is the only
  question that tells two silver SUVs apart when neither looks much like the
  thumbnail it was picked from.
- And one good frame is not evidence. A re-lock needs the same track to win,
  clearly, over several detection passes in a row. Waiting costs a second of
  SEARCHING; guessing costs the rest of the run on the wrong car.

The memory is never discarded. However long the target is gone, the search keeps
looking for it, and the view you picked it from is never evicted however many
views are learned after it.
"""

import cv2
import numpy as np

CANONICAL = 96


def _canonical(frame, box, side_scale=1.1):
    """A square crop around the box, resampled to one fixed size.

    Every comparison happens between images of the same size, so a target picked
    at 27px wide and met again at 110px is compared like with like. The crop is
    square and slightly larger than the box so a car turning (and changing its
    box's aspect) is framed the same way throughout.
    """
    x, y, w, h = box
    side = max(w, h) * side_scale
    cx, cy = x + w / 2.0, y + h / 2.0
    fh, fw = frame.shape[:2]
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, fw)), int(min(cy + side / 2, fh))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (CANONICAL, CANONICAL), interpolation=cv2.INTER_AREA)


def fingerprint(frame, box, size=32):
    """An L2-normalised appearance vector for the object in `box`, or None.

    The object's layout at a fixed small size: a zero-mean colour image, so that
    comparing two of them is normalised cross-correlation. Layout is what tells
    two cars of the same colour apart - where the windows, the roof, the rails
    are - and it was measured against the alternatives on the police-chase clip,
    with a verified target, after a 70-frame gap, as which picks the right car
    out of everything in the frame:

        layout at a fixed scale (this)       100%
        the old 48x48 template                88%
        colour histogram + edge directions    33-43%
        the detector's own ROI features       11-25%

    A histogram throws layout away, and every silver car has the same one; the
    detector's features are trained to say "car", not "which car".
    """
    crop = _canonical(frame, box)
    if crop is None:
        return None
    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean(axis=(0, 1))
    vector = small.ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else None


class Gallery:
    """A bounded set of views, kept diverse.

    A new view that is nearly identical to one already kept adds nothing but
    weight, so it is skipped; when the gallery is full, the most redundant
    learned view makes room. Pinned views - the ones from the moment you picked
    the target - are never evicted.
    """

    def __init__(self, capacity, redundant=0.97):
        self.capacity = capacity
        self.redundant = redundant
        self.pinned, self.views = [], []

    def __len__(self):
        return len(self.pinned) + len(self.views)

    def all(self):
        return self.pinned + self.views

    def similarity(self, vector, top=1):
        views = self.all()
        if vector is None or not views:
            return 0.0
        sims = np.sort(np.stack(views) @ vector)[::-1]
        return float(sims[:top].mean())

    def add(self, vector, pin=False):
        if vector is None:
            return False
        if pin:
            self.pinned.append(vector)
            return True
        if self.all() and self.similarity(vector) >= self.redundant:
            return False
        self.views.append(vector)
        if len(self.views) > self.capacity:
            stack = np.stack(self.views)
            crowding = (stack @ stack.T - np.eye(len(self.views))).max(axis=1)
            self.views.pop(int(np.argmax(crowding)))
        return True


class TargetMemory:
    """What the picked target looks like, and what the cars around it look like.

    `score` is the heart of it: how much more a candidate resembles the target
    than it resembles the closest car the target is known not to be. Positive
    means "more like my target than like anything else I have seen"; negative
    means it is a better match for a stranger, however much it also looks like
    the target on its own.
    """

    def __init__(self, capacity=40, negatives=200, continuity=0.6):
        self.target = Gallery(capacity)
        self.others = Gallery(negatives, redundant=0.985)
        self.continuity = continuity
        self.recent = []

    def pin(self, frame, box):
        """Remember the target from the box that was drawn around it.

        A few slightly shifted and rescaled crops, not just the one: the box you
        draw by hand is never exactly where a detector will put its box later,
        and a memory built from a single framing is needlessly brittle to that.
        """
        x, y, w, h = box
        added = []
        for dx, dy, s in ((0, 0, 1.0), (-0.1, 0, 1.0), (0.1, 0, 1.0), (0, -0.1, 1.0),
                          (0, 0.1, 1.0), (0, 0, 0.85), (0, 0, 1.2)):
            nw, nh = w * s, h * s
            view = fingerprint(frame, (x + dx * w + (w - nw) / 2, y + dy * h + (h - nh) / 2, nw, nh))
            if self.target.add(view, pin=True):
                added.append(view)
        # Called again when the target is picked again by hand: those views join the
        # ones already pinned rather than replacing them, and learning carries on
        # from the view just picked.
        if added:
            self.recent = [added[0]]
        return bool(added)

    def follows(self, vector):
        """Is this view a believable next step from the views just learned?

        A tracked target changes gradually - it turns, it grows as the camera
        zooms - and a layout fingerprint cannot compare views across a big change,
        so the memory has to keep learning as it goes. What must not happen is
        learning a *different* car when the lock slides onto one. The two look
        nothing alike step to step, measured on the chase clip with views five
        frames apart:

            the same car                     median 0.89, 5th percentile 0.43
            a jump to the nearest other car  median 0.14, 95th percentile 0.46

        so a view is learned only if it resembles one of the last few learned,
        which follows a turn and refuses a swap.
        """
        if vector is None:
            return False
        return not self.recent or max(float(r @ vector) for r in self.recent) >= self.continuity

    def learn(self, vector, trusted=False):
        """Add a view of the target, if it follows on from the ones before.

        `trusted` skips the continuity check, for a view whose identity is known
        some other way - the first detection of the target after it was picked or
        re-locked, which starts a new chain.

        Views must come from detection boxes, not from the correlation tracker's
        box: the tracker keeps roughly the size it was given while the car turns
        and its real outline doubles in length, and views framed at half the car
        do not match detections of the very same car (measured at 0.29 against
        0.75).
        """
        if vector is None:
            return False
        if not trusted and not self.follows(vector):
            return False
        self.recent = [vector] if trusted else (self.recent + [vector])[-3:]
        return self.target.add(vector)

    def learn_other(self, vector):
        # Something that already looks a lot like the target is not kept as a
        # negative: one mistaken negative that close would veto the target itself.
        if vector is not None and self.target.similarity(vector) < 0.9:
            self.others.add(vector)

    def score(self, vector):
        if vector is None:
            return -1.0
        mine = self.target.similarity(vector, top=2)
        theirs = self.others.similarity(vector, top=1) if len(self.others) else 0.0
        return mine - theirs

    def likeness(self, vector):
        return self.target.similarity(vector, top=2)


class Reidentifier:
    """Decides, pass by pass, whether a lost target has come back and which one it is.

    Thresholds come from measurement on the police-chase clip, with the target
    verified and removed from view for 70-100 frames:

        the target when it returns          likeness 0.70-0.91 (5th-95th pct)
        the best wrong car in the same frame         0.30-0.71
        the target's lead over that car              0.14-0.47, negative on 1 of 81 frames
        cars in wide shots of heavy traffic          0.10-0.65, max 0.82

    So a candidate has to clear `min_likeness`, lead every other candidate by
    `margin`, not look more like a car the target is known not to be, and keep
    doing all three on `passes` detection passes running on the same ByteTrack
    identity. The lone stranger that happens to score 0.82 in a wide shot does
    not hold that for four passes; the target, when it is really back, does.

    A scene holding one object of the target's kind is the other way round:
    there is nothing to confuse it with, and what it lacks is likeness. A VTOL
    picked parked on its mat and seen next from below in flight scored 0.22
    against its memory - measured on the micro talon clip - and was never taken
    back although it was the only aircraft in the sky for 3000 frames. So an
    identity that is the *only* candidate for `sole_passes` passes running is
    taken back too, unless it looks more like something the target is known not
    to be. Among look-alike traffic there is never a sole candidate for long,
    and that rule never gets a say.
    """

    def __init__(self, memory, passes=4, margin=0.08, min_likeness=0.68, stranger_veto=-0.02,
                 sole_passes=8):
        self.memory = memory
        self.passes = passes
        self.sole_passes = sole_passes
        self.sole, self.sole_streak = None, 0
        self.alone = False        # whether the last decision came from being the only candidate
        self.margin = margin
        self.min_likeness = min_likeness
        self.stranger_veto = stranger_veto
        self.leader, self.streak = None, 0
        self.best = None          # (likeness, lead) of the current leader, for the HUD

    def reset(self):
        self.leader, self.streak, self.best = None, 0, None
        self.sole, self.sole_streak, self.alone = None, 0, False

    def step(self, candidates, refused=(), in_view=None):
        """One detection pass. `candidates`: (track_id, box, feature).

        `in_view` counts everything the detector is following this pass, including
        what is too small, excluded or refused to be a candidate: a candidate is
        only alone if nothing else is there at all.

        Returns (track_id, box, likeness) when a candidate has earned the lock,
        otherwise None. `refused` holds IDs that already failed probation.
        """
        ranked = sorted(((self.memory.likeness(v), tid, box, v)
                         for tid, box, v in candidates if v is not None and tid not in refused),
                        key=lambda row: -row[0])
        alone = len(ranked) == 1 and (in_view is None or in_view == 1)
        if self.sole_passes and alone:
            like, tid, box, vector = ranked[0]
            # Not memory.score: with nothing else known, that is plain likeness, and a
            # target seen from a new side has little or none - which is the very case
            # this rule is for. What rules a sole candidate out is looking more like
            # something the target is known not to be than like the target.
            others = self.memory.others
            if not len(others) or others.similarity(vector, top=1) <= like:
                self.sole_streak = self.sole_streak + 1 if tid == self.sole else 1
                self.sole = tid
                if self.sole_streak >= self.sole_passes:
                    self.best, self.alone = (like, 1.0), True
                    return tid, box, like
            else:
                self.sole, self.sole_streak = None, 0
        elif ranked:
            # Anything else in view, even for one pass, and it is no longer alone.
            self.sole, self.sole_streak = None, 0
        if self.leader is not None and all(tid != self.leader for _, tid, _, _ in ranked):
            # The leader was not fingerprinted this pass - a missed detection, a
            # frame without a feature. That is an absence of evidence, not
            # evidence against it, so its streak is held rather than wiped; only
            # another candidate taking the lead convincingly resets it.
            challenger = ranked[0] if ranked else None
            if challenger is None or challenger[0] < self.min_likeness:
                return None
        if not ranked:
            self.reset()
            return None
        like, tid, box, vector = ranked[0]
        lead = like - ranked[1][0] if len(ranked) > 1 else 1.0
        self.best = (like, lead)
        convincing = (like >= self.min_likeness and lead >= self.margin
                      and self.memory.score(vector) > self.stranger_veto)
        if not convincing:
            self.leader, self.streak = None, 0
            return None
        self.streak = self.streak + 1 if tid == self.leader else 1
        self.leader = tid
        if self.streak >= self.passes:
            return tid, box, like
        return None


def refine(memory, frame, box, reach=0.5, steps=5):
    """The position near `box` that looks most like the remembered target.

    A re-identified target's box comes from a detection pass that, with the
    detector running behind the footage, describes a frame or several ago - and
    a correlation tracker started even half a car off keeps following the half
    it was given, measured sliding off a correctly recognised car within a
    second. So before the lock is placed, the neighbourhood is searched on the
    current frame for the framing that best matches the memory.
    """
    x, y, w, h = box
    best, best_like = box, memory.likeness(fingerprint(frame, box))
    for dy in np.linspace(-reach, reach, steps):
        for dx in np.linspace(-reach, reach, steps):
            for scale in (0.85, 1.0, 1.15):
                nw, nh = w * scale, h * scale
                candidate = (x + dx * w + (w - nw) / 2, y + dy * h + (h - nh) / 2, nw, nh)
                like = memory.likeness(fingerprint(frame, candidate))
                if like > best_like:
                    best, best_like = candidate, like
    return best, best_like
