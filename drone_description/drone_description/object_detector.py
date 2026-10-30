#!/usr/bin/env python3
"""
object_detector.py

Pluggable detection backends for object_tracker.py, plus an appearance-based
re-identifier. Not wired into object_tracker.py itself yet (see the wiring
note in the module docstring below) -- this file is standalone and
importable on its own.

WHY THIS EXISTS: object_tracker.py's detection was a single inline block of
HSV-threshold + contour code baked directly into _camera_cb. That's fast and
fine for one specific color of box, but it can't do two things a "real"
detector needs to eventually do:
  1. Detect a NAMED OBJECT CLASS (person, car, box, ...) instead of "any
     blob in this hue range" -- which is what a small DNN detector buys you.
  2. Recognize "this is the SAME object I saw before" after it's been out
     of frame for a while, rather than always minting a fresh track id --
     which is what re-identification (appearance matching) buys you.

DETECTION BACKENDS both return a list of Detection(area, x, y, w, h) in the
exact tuple shape/order object_tracker.py's _camera_cb already builds
in-line, sorted largest-first -- so wiring one in is a drop-in replacement
for that block, not a rewrite of the surrounding tracking/matching logic:

    detector = build_detector(backend='auto', ...)
    detections = detector.detect(frame)   # frame is RGB8, HxWx3 uint8

  HSVDetector -- byte-for-byte the same color-threshold + contour logic
  object_tracker.py already had inline (same RGB2HSV assumption per the
  camera's actual pixel format, same open+close morphology, same area
  floor), just factored out so it can sit behind the same interface as
  DNNDetector rather than being irreversibly inline.

  DNNDetector -- MobileNet-SSD via OpenCV's DNN module (no extra ML
  framework dependency beyond opencv-contrib). Needs a Caffe prototxt +
  caffemodel pair (the standard MobileNet-SSD "deploy.prototxt" /
  ".caffemodel" files widely distributed for this exact model -- point
  model_path/config_path at wherever you've downloaded them; this file
  doesn't fetch them itself). If those files aren't present,
  DNNDetector.available is False and build_detector() falls back to HSV
  automatically rather than crashing the tracker node.

RE-IDENTIFICATION (ReIdentifier): a lost track's last known appearance
(an HSV color-histogram "signature" of its bounding-box crop, computed once
when the track is last seen) is kept in a gallery for re_id_timeout_sec
after it disappears. When a brand-new, unmatched detection shows up, its
own histogram is compared (cv2.compareHist, correlation method) against
every gallery entry; a good enough match reclaims the OLD track id instead
of minting a new one. HONEST CAVEAT: this world's patrol boxes are all the
same color (see moving_object_patrol.py / the HSV tuning in
object_tracker.py) -- color-histogram re-ID mostly can't tell BOX_1 from
BOX_2 apart, only "this hue blob is probably one of the boxes I've seen
before" from something that ISN'T that color. Re-ID is far more useful once
DNNDetector is pointed at a scene with visually distinct objects; wiring it
in now means it's ready and does no harm (nothing changes for
same-colored objects, since they all just compare similarly).

WIRING NOTE (next step, not done in this file): object_tracker.py's
_camera_cb would replace its inline HSV block with a call to a
`self.detector.detect(frame)`, and its per-track dict (currently just
{'cx', 'cy', 'last_seen'}) would gain a stored histogram signature, offered
to a ReIdentifier instance on track expiry and checked against on every new,
unmatched detection before minting a fresh id.
"""
import math
from collections import namedtuple

import numpy as np
import cv2


Detection = namedtuple('Detection', ['area', 'x', 'y', 'w', 'h'])


# --------------------------------------------------------------------------
# HSV backend
# --------------------------------------------------------------------------
class HSVDetector:
    """Color-threshold + contour detector. Same logic object_tracker.py had
    inline: RGB2HSV (matches the camera's actual RGB8 pixel format), open+
    close morphology to clean the mask, contours above an area floor,
    sorted largest-first and capped at max_targets.

    SHAPE FILTERING (min_solidity / min_aspect_ratio / max_aspect_ratio):
    color alone can't tell a painted box apart from a same-colored patch of
    ground -- Baylands' dirt/sand terrain overlaps the box's tuned HSV range
    at the margins, so on its own this detector occasionally flags a patch
    of yellowish/tan terrain as a target. Two cheap shape checks, run AFTER
    the area filter, reject most of those without needing to know the
    terrain's exact color:
      - SOLIDITY (contour area / convex-hull area): a painted box's
        silhouette is a clean, essentially convex shape, so its solidity
        sits close to 1.0. A ground/shadow patch picked up by HSV
        thresholding typically has a ragged, partly-concave boundary
        (grass blades, mixed lighting, terrain undulation) and measurably
        lower solidity. Rejecting anything below min_solidity screens most
        of those out.
      - ASPECT RATIO (bbox width / height): the box reads as roughly
        square-ish from most viewing angles. A strip of ground/horizon
        picked up along a distant tan patch tends to be much wider than
        tall (or, rarely, a thin sliver the other way) -- outside
        [min_aspect_ratio, max_aspect_ratio] gets rejected.
    Both are heuristics, not guarantees -- a heavily occluded or
    steeply-angled real box can occasionally trip one of them too, which is
    why the defaults are deliberately lenient (0.85 solidity, 0.35-3.0
    aspect) rather than tight. If ground is still getting flagged, tighten
    these first (raise min_solidity toward 0.9-0.95, narrow the aspect
    bounds) before touching hsv_lower/hsv_upper -- narrowing the color
    range itself risks losing the box under lighting variation, whereas
    shape filtering doesn't depend on getting the color range exactly
    right. If a real box starts getting missed instead, loosen these back
    down; set debug=True (object_tracker.py's debug_detection param) to
    see accepted vs. rejected contours' area/solidity/aspect/extent/
    texture while tuning.

    EXTENT (min_extent): contour area / bounding-rect area (w*h). Cheaper
    than solidity (no hull computation) and catches a slightly different
    failure mode -- a blob whose convex hull happens to be tight (so
    solidity looks fine) but that doesn't actually fill its own bounding
    box, e.g. an L-shaped or notched patch of terrain.

    TEXTURE (max_texture_variance, optional -- None disables it): shape
    alone can't catch a ground patch that happens to be both fairly convex
    AND roughly square (a shadowed dip, a cluster of rocks/dirt clumps).
    What still tells it apart from the box is surface detail: this world's
    box is a single flat-shaded material, so its interior is smooth aside
    from a slow lighting gradient; procedural ground/dirt texture has real
    high-frequency detail (speckle, grain, rock/grass micro-structure) even
    within the same HSV range. This computes the variance of a Laplacian
    (a high-pass filter) over the *filled contour's own interior* -- a
    smooth gradient barely excites a Laplacian (it responds to sharp local
    change, not slow change), so this stays low for the box and high for
    textured terrain, largely independent of overall brightness/hue.
    Disabled by default (None) because the right cutoff is scene/render-
    dependent -- turn on debug=True, note the texture value logged for
    real-box accepts vs. any ground-patch accepts, and set
    max_texture_variance somewhere between the two clusters.
    """

    def __init__(self, hsv_lower, hsv_upper, min_contour_area_px=50,
                 morph_kernel_size=5, max_targets=5,
                 min_solidity=0.85, min_aspect_ratio=0.35, max_aspect_ratio=3.0,
                 min_extent=0.5, max_texture_variance=None,
                 debug=False, logger=None):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_contour_area_px = min_contour_area_px
        self.max_targets = max_targets
        self.min_solidity = min_solidity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent
        self.max_texture_variance = max_texture_variance
        self.debug = debug
        self.logger = logger
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.available = True

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(f'[HSVDetector] {msg}')
        else:
            print(f'[HSVDetector] {msg}')

    def _texture_variance(self, gray, contour, x, y, w, h):
        """Laplacian variance over the filled contour's own interior only
        (not the whole bbox -- a jagged/notched blob's crop would
        otherwise get diluted with excluded background pixels, which
        would skew this toward "smooth" regardless of the blob's real
        texture)."""
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local_mask, [contour - (x, y)], -1, 255, thickness=cv2.FILLED)
        lap = cv2.Laplacian(crop, cv2.CV_64F)
        vals = lap[local_mask == 255]
        return float(vals.var()) if vals.size > 0 else 0.0

    def detect(self, frame_rgb):
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        # Only pay for a grayscale conversion when the texture filter is
        # actually enabled.
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if self.max_texture_variance is not None else None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            # GROUND-PATCH REJECTION -- see class docstring. Ordered
            # cheapest-first so a blob only pays for the next, pricier
            # check once it's cleared the last one.
            aspect = w / h
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                if self.debug:
                    self._log(f'reject aspect={aspect:.2f} area={area:.0f} bbox=({x},{y},{w},{h})')
                continue

            extent = area / (w * h)
            if extent < self.min_extent:
                if self.debug:
                    self._log(f'reject extent={extent:.2f} area={area:.0f} bbox=({x},{y},{w},{h})')
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                if self.debug:
                    self._log(f'reject solidity={solidity:.2f} area={area:.0f} bbox=({x},{y},{w},{h})')
                continue

            texture = None
            if self.max_texture_variance is not None:
                texture = self._texture_variance(gray, c, x, y, w, h)
                if texture > self.max_texture_variance:
                    if self.debug:
                        self._log(
                            f'reject texture={texture:.0f} (limit {self.max_texture_variance:.0f}) '
                            f'area={area:.0f} bbox=({x},{y},{w},{h})'
                        )
                    continue

            if self.debug:
                texture_str = f' texture={texture:.0f}' if texture is not None else ''
                self._log(
                    f'accept area={area:.0f} bbox=({x},{y},{w},{h}) '
                    f'solidity={solidity:.2f} aspect={aspect:.2f} extent={extent:.2f}{texture_str}'
                )

            detections.append(Detection(area, x, y, w, h))
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections[:self.max_targets]


# --------------------------------------------------------------------------
# DNN backend
# --------------------------------------------------------------------------
# Standard 20-class MobileNet-SSD label order (the widely-distributed
# Caffe model this targets was trained on this exact class list).
_MOBILENET_SSD_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor',
]


class DNNDetector:
    """MobileNet-SSD object detector via OpenCV's DNN module. Detects one
    or more NAMED classes (from _MOBILENET_SSD_CLASSES) instead of an HSV
    color range -- point target_classes at whichever labels matter (e.g.
    ['person'] or ['car', 'bus']).

    If model_path/config_path don't point at real files, `available` is
    False and detect() always returns an empty list -- build_detector()
    checks this and falls back to HSVDetector rather than crashing.
    """

    def __init__(self, model_path, config_path, target_classes=None,
                 confidence_threshold=0.5, max_targets=5, input_size=300):
        self.confidence_threshold = confidence_threshold
        self.max_targets = max_targets
        self.input_size = input_size
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = {
                _MOBILENET_SSD_CLASSES.index(c) for c in target_classes
                if c in _MOBILENET_SSD_CLASSES
            }

        self.net = None
        self.available = False
        try:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.available = True
        except Exception as e:
            # Missing/invalid model files, or an OpenCV build without the
            # dnn module's Caffe importer -- either way, fail soft. The
            # caller (build_detector) is responsible for falling back.
            self._load_error = str(e)

    def detect(self, frame_rgb):
        if not self.available:
            return []

        img_h, img_w = frame_rgb.shape[:2]
        # MobileNet-SSD's Caffe deploy net expects BGR input and its
        # standard (127.5, 127.5, 127.5) mean-subtraction / 0.007843 scale
        # (2/255) preprocessing -- this is the network's OWN trained
        # convention, unrelated to the RGB8 the camera itself publishes in
        # (which is why frame_rgb is explicitly converted here rather than
        # assumed).
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            frame_bgr, scalefactor=0.007843,
            size=(self.input_size, self.input_size),
            mean=(127.5, 127.5, 127.5), swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        detections = []
        # raw shape: [1, 1, N, 7] -- each row is
        # [batch_id, class_id, confidence, x1, y1, x2, y2] (normalized 0..1)
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(raw[0, 0, i, 1])
            if self.target_class_ids is not None and class_id not in self.target_class_ids:
                continue
            x1 = raw[0, 0, i, 3] * img_w
            y1 = raw[0, 0, i, 4] * img_h
            x2 = raw[0, 0, i, 5] * img_w
            y2 = raw[0, 0, i, 6] * img_h
            x, y = max(0, int(x1)), max(0, int(y1))
            w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
            if w <= 0 or h <= 0:
                continue
            detections.append(Detection(float(w * h), x, y, w, h))

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections[:self.max_targets]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_detector(backend='hsv', logger=None, **kwargs):
    """backend: 'hsv', 'dnn', or 'auto' (try dnn, silently fall back to hsv
    if the model files aren't available). kwargs are passed to whichever
    detector(s) get constructed -- pass every HSV* and DNN* constructor
    kwarg you might need; unused ones for the backend that doesn't end up
    running are simply ignored below.

    hsv_lower/hsv_upper/min_contour_area_px/morph_kernel_size/max_targets
      -> HSVDetector
    model_path/config_path/target_classes/confidence_threshold/input_size
      -> DNNDetector
    """
    def log(msg):
        if logger is not None:
            logger.info(msg)

    def warn(msg):
        if logger is not None:
            logger.warn(msg)
        else:
            print(f'[object_detector] WARNING: {msg}')

    hsv_kwargs = {
        k: kwargs[k] for k in
        ('hsv_lower', 'hsv_upper', 'min_contour_area_px', 'morph_kernel_size', 'max_targets',
         'min_solidity', 'min_aspect_ratio', 'max_aspect_ratio',
         'min_extent', 'max_texture_variance', 'debug')
        if k in kwargs
    }

    if backend == 'hsv':
        log('object_detector: using HSV backend')
        return HSVDetector(logger=logger, **hsv_kwargs)

    if backend in ('dnn', 'auto'):
        dnn_kwargs = {
            k: kwargs[k] for k in
            ('model_path', 'config_path', 'target_classes', 'confidence_threshold',
             'max_targets', 'input_size')
            if k in kwargs
        }
        if 'model_path' in dnn_kwargs and 'config_path' in dnn_kwargs:
            detector = DNNDetector(**dnn_kwargs)
            if detector.available:
                log('object_detector: using DNN (MobileNet-SSD) backend')
                return detector
            warn(
                f'DNN backend requested but model failed to load '
                f'({getattr(detector, "_load_error", "unknown error")}) -- '
                f'falling back to HSV.'
            )
        elif backend == 'dnn':
            warn('DNN backend requested but model_path/config_path not given -- falling back to HSV.')
        return HSVDetector(logger=logger, **hsv_kwargs)

    raise ValueError(f'unknown detector backend: {backend!r}')


# --------------------------------------------------------------------------
# Re-identification
# --------------------------------------------------------------------------
class ReIdentifier:
    """Appearance-histogram-based re-identification gallery. Offer a
    track's last-seen crop when it's about to be dropped; check brand-new,
    unmatched detections against the gallery before minting a fresh id.

    See the module docstring's HONEST CAVEAT: on same-colored objects this
    mostly narrows "probably one of the boxes" rather than picking a
    specific one -- it's most useful once paired with DNNDetector on a
    scene with visually distinct objects.
    """

    def __init__(self, similarity_threshold=0.7, gallery_timeout_sec=15.0):
        self.similarity_threshold = similarity_threshold
        self.gallery_timeout_sec = gallery_timeout_sec
        # tid -> {'hist': np.ndarray, 'last_seen': monotonic time}
        self.gallery = {}

    @staticmethod
    def compute_histogram(frame_rgb, bbox):
        x, y, w, h = bbox
        img_h, img_w = frame_rgb.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img_w, int(x + w)), min(img_h, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_rgb[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        # Hue + saturation histogram (ignore value/brightness -- it's the
        # most lighting-sensitive channel and adds little to telling two
        # objects apart by color).
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def offer(self, track_id, frame_rgb, bbox, now):
        """Call when a track is about to be dropped (expired from frame),
        so its appearance is remembered for possible reclaim. Computes the
        histogram from frame_rgb/bbox directly -- use offer_hist instead if
        you already have a histogram computed from when the track was last
        actually visible (the frame at expiry time may no longer show it)."""
        hist = self.compute_histogram(frame_rgb, bbox)
        if hist is not None:
            self.gallery[track_id] = {'hist': hist, 'last_seen': now}

    def offer_hist(self, track_id, hist, now):
        """Same as offer(), but takes an already-computed histogram --
        the usual case, since a track's LAST KNOWN appearance has to be
        captured from whichever frame it was still visible in, not the
        (possibly much later) frame where it's finally declared expired."""
        if hist is not None:
            self.gallery[track_id] = {'hist': hist, 'last_seen': now}

    def match(self, frame_rgb, bbox, now):
        """Compare a new, unmatched detection's appearance against the
        gallery. Returns the best-matching old track id (and removes it
        from the gallery, since it's now active again) if similarity
        clears similarity_threshold, else None."""
        self._expire(now)
        hist = self.compute_histogram(frame_rgb, bbox)
        if hist is None or not self.gallery:
            return None

        best_tid, best_score = None, -1.0
        for tid, entry in self.gallery.items():
            score = cv2.compareHist(hist, entry['hist'], cv2.HISTCMP_CORREL)
            if score > best_score:
                best_tid, best_score = tid, score

        if best_tid is not None and best_score >= self.similarity_threshold:
            del self.gallery[best_tid]
            return best_tid
        return None

    def _expire(self, now):
        stale = [tid for tid, e in self.gallery.items()
                 if (now - e['last_seen']) > self.gallery_timeout_sec]
        for tid in stale:
            del self.gallery[tid]
