"""Motion-based detection of small airborne objects against sky."""

import cv2
import numpy as np


class YoloDetector:
    """Trained object detector, for targets the contrast detectors cannot isolate.

    A parked or landed aircraft is not the only high-contrast thing on the ground
    and is not moving, so nothing about the image itself marks it out - only
    knowing what an aircraft looks like does. Needs `pip install ultralytics`,
    and is far slower than the other modes.
    """

    def __init__(self, model="yolov8m.pt", imgsz=1280, conf=0.25, classes=("airplane", "bird"),
                 focus_imgsz=640, low_conf=0.0):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                "--mode yolo needs ultralytics: pip install ultralytics\n"
                "(then reinstall opencv-contrib-python, which it downgrades)"
            )
        self.model = YOLO(model)
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf = conf
        self.classes = set(classes)
        self.focus_imgsz = focus_imgsz
        self.low_conf = low_conf

    def detect(self, frame, regions=None):
        """Run the model on the whole frame, or only inside `regions`.

        Searching a crop around a target the tracker is already following beats
        searching the whole frame: the target occupies far more of the pixels the
        model actually sees, so a small, cheap inference resolves it better than a
        large, expensive one. Regions come from the tracker, so this only applies
        once something is being followed.
        """
        scored, mask = self.detect_scored(frame, regions)
        return [box for box, score in scored if score >= self.conf], mask

    def detect_scored(self, frame, regions=None):
        """The same detections, each with the model's confidence in it.

        ByteTrack's second pass is fed by exactly the boxes a detector would
        normally suppress, so the floor is dropped to `low_conf` here. That costs
        nothing extra: it is the same forward pass, with less of its output
        thrown away before it is returned.
        """
        if not regions:
            return self._infer(frame, self.imgsz, 0, 0), None

        boxes = []
        for x0, y0, x1, y1 in regions:
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                boxes.extend(self._infer(crop, self.focus_imgsz, x0, y0))
        return dedupe(boxes), None

    def _infer(self, image, imgsz, off_x, off_y):
        floor = min(self.conf, self.low_conf) if self.low_conf else self.conf
        result = self.model.predict(image, conf=floor, imgsz=imgsz, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            if self.classes and self.names[int(box.cls)] not in self.classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            boxes.append(((x1 + off_x, y1 + off_y, x2 - x1, y2 - y1), float(box.conf)))
        return boxes


def dedupe(scored, overlap=0.5):
    """Drop detections that substantially repeat one already kept.

    Focus crops are allowed to overlap, so the same target can be reported by
    two of them; the tracker must not see it twice. Takes and returns
    (box, confidence) pairs, most confident first, so where two crops disagree
    about the same object the better-resolved reading is the one kept.
    """
    kept = []
    for box, score in sorted(scored, key=lambda pair: -pair[1]):
        x, y, w, h = box
        for (kx, ky, kw, kh), _ in kept:
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            if inter and inter / min(w * h, kw * kh) > overlap:
                break
        else:
            kept.append((box, score))
    return kept


def merge_boxes(boxes, gap, max_union):
    """Union boxes that sit within `gap` px of each other.

    One aircraft rarely gives one blob: wings, fuselage and props each survive
    thresholding separately. Without this every fragment becomes its own track.

    A union that would exceed `max_union` is refused. Merging is transitive, so
    over a textured region like a treeline a chain of fragments would otherwise
    link everything into one frame-sized box and swallow the real target with it.
    """
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        for box in merged:
            x, y, w, h = box
            for i, (ox, oy, ow, oh) in enumerate(out):
                if (x - gap < ox + ow and ox - gap < x + w
                        and y - gap < oy + oh and oy - gap < y + h):
                    nx, ny = min(x, ox), min(y, oy)
                    nw = max(x + w, ox + ow) - nx
                    nh = max(y + h, oy + oh) - ny
                    if nw * nh > max_union:
                        continue
                    out[i] = (nx, ny, nw, nh)
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


class SkyDetector:
    """Finds candidate targets in a frame.

    mode="sky":  silhouette against sky - anything markedly darker or brighter
                 than the smooth sky around it. Needs no motion at all, so it
                 still sees a target that hovers, or one the operator is panning
                 to follow (which sits still in frame and is invisible to any
                 motion-based method).
    mode="mog2": adaptive background model. Best when the camera is on a tripod.
    mode="diff": frame differencing with global-motion compensation, for handheld
                 or panning footage where mog2 flags the whole frame.
    """

    SENSITIVITY_DEFAULTS = {"sky": 38, "mog2": 18, "diff": 18}

    def __init__(self, mode="sky", min_area=6, max_area=8000, sensitivity=0, min_contrast=8.0,
                 merge_gap=14, max_texture=0.0, low_contrast=0.0):
        self.mode = mode
        self.low_contrast = low_contrast
        self.min_area = min_area
        self.max_area = max_area
        self.sensitivity = sensitivity or self.SENSITIVITY_DEFAULTS[mode]
        self.min_contrast = min_contrast
        self.merge_gap = merge_gap
        self.max_texture = max_texture
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Rectangular, not elliptical: this only has to erase structures smaller
        # than itself from a sky estimate, where the shape of the footprint does
        # not matter - and a rectangle is separable, which makes it ~10x cheaper
        # than the disc. It is the hottest operation in the pipeline.
        self.sky_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        self.prev_gray = None
        self.bg = None
        if mode == "mog2":
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=self.sensitivity, detectShadows=False
            )

    def detect(self, frame, regions=None):
        found, mask = self._candidates(frame, self.min_contrast)
        return [box for box, _ in found], mask

    def detect_scored(self, frame, regions=None):
        """The same detections, each with a 0-1 confidence.

        There is no trained score here to use as one, so the evidence the
        detector actually has stands in for it: how far the blob departs from
        the background around it. The scale comes from the floor a real target
        has to clear - against sky an airframe reads over 100 grey levels while
        broken cloud reads 10 to 35, so eight times the floor (64 by default)
        maps the airframe to 1.0 and cloud to around 0.3, putting them on
        opposite sides of the split ByteTrack divides its two passes at.

        The floor itself drops to `low_contrast`, because the faint detections
        below the normal one are precisely what the second pass exists to use.
        """
        found, mask = self._candidates(frame, self.low_contrast or self.min_contrast)
        scale = max(self.min_contrast * 8.0, 1e-6)
        return [(box, min(contrast / scale, 1.0)) for box, contrast in found], mask

    def _candidates(self, frame, min_contrast):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.mode == "sky":
            mask = self._sky_silhouette(gray)
        elif self.mode == "mog2":
            mask = self.bg.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        else:
            mask = self._compensated_diff(gray)

        self.prev_gray = gray
        if mask is None:
            return [], np.zeros_like(gray)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 2]

        boxes = []
        for x, y, w, h in merge_boxes(raw, self.merge_gap, self.max_area):
            if w * h < self.min_area or w * h > self.max_area:
                continue
            if max(w, h) > 6 * max(min(w, h), 1):
                continue
            contrast, texture = self._ring_stats(gray, x, y, w, h)
            if contrast < min_contrast:
                continue
            if self.max_texture and texture > self.max_texture:
                continue
            boxes.append(((x, y, w, h), contrast))
        return boxes, mask

    def _ring_stats(self, gray, x, y, w, h):
        """Contrast against, and roughness of, the band just outside the blob.

        Contrast separates a real silhouette from a noise speckle: compression
        noise flickers without changing local brightness. Roughness says what the
        blob is sitting against - sky stays smooth even when cloudy, while foliage
        and rooftops are full of edges, so a textured surround means the blob is
        part of the scenery rather than an aircraft in front of it.
        """
        pad = 8
        fh, fw = gray.shape
        ox0, oy0 = max(x - pad, 0), max(y - pad, 0)
        ox1, oy1 = min(x + w + pad, fw), min(y + h + pad, fh)
        outer = gray[oy0:oy1, ox0:ox1].astype(np.float32)
        inner = gray[y:y + h, x:x + w].astype(np.float32)
        ring_count = outer.size - inner.size
        if ring_count <= 0 or inner.size == 0:
            return 0.0, 0.0
        ring_mean = (outer.sum() - inner.sum()) / ring_count
        ring_sq = (float((outer ** 2).sum()) - float((inner ** 2).sum())) / ring_count
        ring_std = max(ring_sq - ring_mean * ring_mean, 0.0) ** 0.5

        # Contrast is how far the darkest or brightest part of the blob departs
        # from the sky around it, not how far its average does. A multirotor seen
        # from below is mostly gaps: its bounding box is nearly all sky, and
        # averaging over it hides the airframe completely. On overcast footage
        # the drone scored 3.5 by the mean and was thrown away every frame, while
        # scoring 110 by this measure - three times higher than any cloud in the
        # same picture. Percentiles rather than the true extremes, so one
        # compression artefact cannot invent a target.
        flat = inner.ravel()
        if flat.size > 4096:
            flat = flat[::flat.size // 4096 + 1]
        low, high = np.percentile(flat, (5, 95))
        return max(abs(low - ring_mean), abs(high - ring_mean)), ring_std

    def _sky_silhouette(self, gray):
        """Everything that stands out from the smooth sky behind it.

        The sky is estimated by removing small structures at 1/8 scale - closing
        erases dark blobs, opening erases bright ones - so what survives the
        difference is an object rather than cloud shading, whether or not it moves.
        """
        h, w = gray.shape
        small = cv2.resize(gray, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        sky_without_dark = cv2.morphologyEx(small, cv2.MORPH_CLOSE, self.sky_kernel)
        sky_without_bright = cv2.morphologyEx(small, cv2.MORPH_OPEN, self.sky_kernel)
        size = (w, h)
        darker = cv2.subtract(cv2.resize(sky_without_dark, size, interpolation=cv2.INTER_LINEAR), gray)
        brighter = cv2.subtract(gray, cv2.resize(sky_without_bright, size, interpolation=cv2.INTER_LINEAR))
        _, mask = cv2.threshold(cv2.max(darker, brighter), self.sensitivity, 255, cv2.THRESH_BINARY)
        return mask

    def _compensated_diff(self, gray):
        """Warp the previous frame onto the current one, then difference.

        Cancels camera pan/shake so only independently moving objects survive.
        """
        if self.prev_gray is None:
            return None

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=20
        )
        warped = self.prev_gray
        if prev_pts is not None and len(prev_pts) >= 10:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None
            )
            if curr_pts is not None:
                ok = status.ravel() == 1
                if ok.sum() >= 10:
                    matrix, _ = cv2.estimateAffinePartial2D(
                        prev_pts[ok], curr_pts[ok], method=cv2.RANSAC, ransacReprojThreshold=3
                    )
                    if matrix is not None:
                        warped = cv2.warpAffine(
                            self.prev_gray, matrix, (gray.shape[1], gray.shape[0]),
                            borderMode=cv2.BORDER_REPLICATE,
                        )

        diff = cv2.absdiff(gray, warped)
        _, mask = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)
        # Warping leaves garbage at the frame edge; ignore a border strip.
        border = 12
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        return mask
