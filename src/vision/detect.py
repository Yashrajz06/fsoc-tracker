"""Detection: adaptive thresholding, connected components, gating, and the quality flags.

Pipeline position (``docs/DESIGN.md`` section 5.1): the top-hat residual arrives here, is
thresholded adaptively, segmented into candidate blobs, gated on scale-relative geometry, and the
survivors are flagged.

**No fixed intensity thresholds, and no fixed pixel geometry.** The threshold derives from frame
statistics; the area and shape gates derive from the runtime spot scale via
:meth:`~src.config.VisionConfig.resolve_geometry`. 30% of the grade runs on evaluator video whose
brightness and resolution we cannot predict, and absolute constants tuned to our own simulator
are the single most likely way to fail it.

Two kinds of rejection, and the difference is load-bearing
----------------------------------------------------------
This module distinguishes **gates** from **flags**, and they carry opposite downstream policies.
The asymmetry is expressed in the code, not merely documented:

* A **gate failure** (area, circularity) means *this is not the target*. Reject outright. A salt
  impulse or a background structure admitted here would drag the centroid an arbitrary distance,
  and an outlier carries no usable information.

* A **flag** (:attr:`Detection.clipped`, :attr:`Detection.saturated`) means *this is the target,
  and its centroid carries a known, bounded, systematic bias*. Keep it, and inflate the Kalman
  measurement noise in Phase 4. Measured: clipping biases the centroid by up to ~2 px inward with
  no noise present (Phase 1), and saturation by ~0.19 px (Phase 2), against a 10 px tracking
  budget. Discarding a measurement biased by 0.19 px would cost far more than keeping it.

A bounded bias is still information; an outlier is not. See :meth:`Detection.is_reliable` and
:func:`gate_candidates`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import DetectionConfig, ResolvedVisionGeometry
from src.vision.snr import core_saturated_fraction, robust_sigma

#: Robust sigma below which ``mean + k*sigma`` degenerates and a percentile is used instead.
_DEGENERATE_SIGMA: float = 1e-6

__all__ = [
    "Detection",
    "DetectionResult",
    "adaptive_threshold",
    "detect",
    "gate_candidates",
]


@dataclass(frozen=True)
class Detection:
    """One gated candidate blob.

    Attributes:
        x: Blob centroid x in frame coordinates (coarse; refined in :mod:`src.vision.centroid`).
        y: Blob centroid y in frame coordinates.
        area_px: Blob area in pixels.
        peak: Peak residual intensity within the blob.
        flux: Integrated residual intensity. **The ranking statistic, never peak** -- salt
            impulses sit at the 8-bit ceiling and out-rank any beacon dimmer than 255 on peak.
        circularity: ``4*pi*area / perimeter^2``, dimensionless and therefore already scale-free.
        bbox: Bounding box as ``(x0, y0, width, height)``.
        clipped: Bounding box touches the frame edge. The spot is truncated, so its centroid is
            pulled *inward* -- up to ~2 px with no noise present (Phase 1). It looks exactly like
            tracker lag, which is why it is recorded at the moment it is knowable.
        saturated_fraction: Core saturation, per the canonical Phase 2 definition in
            :func:`~src.vision.snr.core_saturated_fraction`.
        saturated: Whether saturation exceeds the reporting threshold.
    """

    x: float
    y: float
    area_px: float
    peak: float
    flux: float
    circularity: float
    bbox: Tuple[int, int, int, int]
    clipped: bool = False
    saturated_fraction: float = 0.0
    saturated: bool = False

    @property
    def is_reliable(self) -> bool:
        """Whether this detection is free of known systematic bias.

        An *unreliable* detection is still a detection: it passed every gate, so it is the
        target. It simply carries a bounded, quantified bias, and Phase 4 responds by inflating
        the Kalman measurement noise rather than by rejecting the measurement. Anything that
        should be rejected never becomes a :class:`Detection` at all.
        """
        return not (self.clipped or self.saturated)


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one detection pass.

    Attributes:
        detections: Surviving candidates, strongest first by flux.
        threshold: Absolute threshold level used, for the trace.
        method: Threshold operator used.
        n_candidates: Blobs found before gating.
        rejected_small: Blobs rejected by the lower area gate.
        rejected_large: Blobs rejected by the upper area gate.
        rejected_shape: Blobs rejected by the circularity gate.
        rejected_faint: Blobs rejected by the peak-significance gate.
    """

    detections: Tuple[Detection, ...] = ()
    threshold: float = 0.0
    method: str = ""
    n_candidates: int = 0
    rejected_small: int = 0
    rejected_large: int = 0
    rejected_shape: int = 0
    rejected_faint: int = 0

    @property
    def best(self) -> Optional[Detection]:
        """The strongest surviving detection, or ``None``."""
        return self.detections[0] if self.detections else None

    @property
    def found(self) -> bool:
        """Whether anything survived gating."""
        return bool(self.detections)


def adaptive_threshold(residual: np.ndarray, config: DetectionConfig) -> float:
    """Compute an absolute threshold level from frame statistics.

    Every operator here is parameter-free or single-parameter, and none is a fixed intensity.

    The default, ``mean_plus_k_sigma`` on the top-hat residual, uses **robust** statistics
    (median and MAD). The mean and standard deviation of a residual containing a bright spot are
    both pulled by the spot itself, which raises the threshold in proportion to signal strength
    -- precisely backwards.

    Args:
        residual: Top-hat residual, 2-D.
        config: Detection configuration selecting the operator.

    Returns:
        An absolute threshold level in the residual's own units.

    Raises:
        ValueError: On an unknown threshold method.
    """
    values = residual.astype(np.float32, copy=False)
    method = config.threshold_method
    median_level = float(np.median(values))
    peak_level = float(values.max())

    if method == "mean_plus_k_sigma":
        median = float(np.median(values))
        # Routed through the shared helper so the quantisation floor cannot be bypassed. The
        # explicit degeneracy check below is retained: on a *normalised* residual the floor is in
        # different units, so a flat residual can still yield a threshold that fails to separate.
        sigma = robust_sigma(values, floor=0.0)
        if sigma <= _DEGENERATE_SIGMA:
            # MAD collapses to zero whenever more than half the residual is identically flat --
            # a noiseless frame, or a large uniform region such as heavy compression produces.
            # `median + k*0` then equals the median, so the mask swallows the whole frame and
            # detection fails outright. Found by measurement on a zero-noise sweep, where every
            # single detection failed. Fall back to a high percentile, which is still derived
            # from frame statistics and never a fixed intensity.
            return _separating_threshold(float(np.percentile(values, config.percentile)),
                                         median_level, peak_level)
        return _separating_threshold(median + config.k_sigma * sigma, median_level, peak_level)

    if method == "percentile":
        return _separating_threshold(float(np.percentile(values, config.percentile)),
                                     median_level, peak_level)

    if method == "otsu":
        # Otsu needs an 8-bit histogram; rescale the residual into that range first.
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return high
        scaled = ((values - low) / (high - low) * 255.0).astype(np.uint8)
        level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return low + (level / 255.0) * (high - low)

    if method == "adaptive_gaussian":
        # Local mean minus a robust global offset; approximates cv2.adaptiveThreshold while
        # staying in float, which the residual requires.
        blurred = cv2.GaussianBlur(values, (0, 0), sigmaX=max(1.0, values.shape[0] / 64.0))
        return _separating_threshold(
            float(np.median(blurred) + config.k_sigma * robust_sigma(values)),
            median_level, peak_level)

    raise ValueError(f"Unknown threshold method: {method!r}")


def _separating_threshold(level: float, median_level: float, peak_level: float) -> float:
    """Guarantee a threshold that actually separates the target from the background.

    Two degenerate cases were found by measurement on flat, low-noise frames, and both make
    detection fail outright rather than degrade:

    * ``mean + k*sigma`` collapses when the robust sigma is zero -- more than half the residual
      identically flat, as a noiseless frame or a large uniform region from heavy compression
      produces. The threshold then equals the median and the mask swallows the whole frame.
    * The percentile fallback collapses when the target is *smaller* than the tail the percentile
      selects. A 100 px beacon in a 25600 px frame is 0.39% of it, below the 0.5% that the 99.5th
      percentile keeps, so the percentile itself lands on background and returns the same
      degenerate answer.

    When the computed level fails to exceed the background, fall back to the midpoint between the
    background median and the residual peak. That is still derived entirely from frame statistics
    -- never a fixed intensity -- and it matches the half-maximum convention used for FWHM
    elsewhere in the project.

    Args:
        level: The operator's computed threshold.
        median_level: Robust background level of the residual.
        peak_level: Maximum residual value.

    Returns:
        A threshold strictly above the background whenever any signal exists.
    """
    if level > median_level or peak_level <= median_level:
        return level
    return median_level + 0.5 * (peak_level - median_level)


def gate_candidates(stats: np.ndarray, geometry: ResolvedVisionGeometry) -> np.ndarray:
    """Return a boolean mask of components passing the scale-relative area gates.

    Gates are *rejections*: a blob failing here is not the target, so it is discarded rather than
    flagged. Contrast with :attr:`Detection.clipped` and :attr:`Detection.saturated`, which mark
    a genuine target whose centroid is merely biased.

    Args:
        stats: OpenCV connected-component stats, excluding the background row.
        geometry: Resolved geometry supplying the area bounds.

    Returns:
        Boolean mask over the supplied components.
    """
    if stats.size == 0:
        return np.zeros(0, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    return (areas >= geometry.min_blob_area_px) & (areas <= geometry.max_blob_area_px)


def _circularity(mask: np.ndarray) -> float:
    """Compute ``4*pi*area / perimeter^2`` for a binary blob mask.

    Args:
        mask: Binary ``uint8`` mask of a single blob.

    Returns:
        Circularity in ``[0, 1]`` for convex shapes, 0.0 when undefined.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * cv2.contourArea(contour) / (perimeter * perimeter))


def detect(residual: np.ndarray, denoised: np.ndarray, config: DetectionConfig,
           geometry: ResolvedVisionGeometry,
           saturation_report_threshold: float = 0.05,
           min_peak_snr: float = 5.0) -> DetectionResult:
    """Threshold, segment, gate and flag a preprocessed frame.

    Args:
        residual: Top-hat residual from :mod:`src.vision.preprocess`, used for detection.
        denoised: Median-filtered frame before background suppression, used for photometry.
            Saturation must be measured here: the top-hat residual has had its pedestal removed,
            so a flat top is no longer at the quantisation ceiling in it.
        config: Detection configuration.
        geometry: Resolved geometry supplying the gates. Never literals.
        saturation_report_threshold: Core saturation above which the flag is raised.
        min_peak_snr: Minimum blob peak, in robust residual sigmas above the residual background,
            for the blob to be accepted at all.

            **This gate is required, and its absence was a real defect.** Thresholding at
            ``median + 3*sigma`` guarantees only that a blob's pixels cleared 3 sigma, and on a
            target-free frame some noise cluster always does: measured on pure Gaussian noise,
            the pipeline returned a confident detection (flux 7.5, area 11 px, circularity 0.46)
            on every single frame. Without a significance floor a target-free frame yields a
            false lock rather than an honest miss, which corrupts loss rate, false-lock rate and
            re-acquisition timing simultaneously. The Tier-0 bootstrap already carries the same
            5-sigma floor for the same reason (:mod:`src.vision.spotscale`), and the two are kept
            consistent deliberately.

    Returns:
        A :class:`DetectionResult`.

    Raises:
        ValueError: If the inputs are not 2-D or differ in shape.
    """
    if residual.ndim != 2:
        raise ValueError(f"Detection requires a 2-D residual, got shape {residual.shape}")
    if denoised.shape != residual.shape:
        raise ValueError(
            f"Residual {residual.shape} and denoised {denoised.shape} frames must match")

    height, width = residual.shape
    level = adaptive_threshold(residual, config)
    mask = (residual >= level).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return DetectionResult(threshold=level, method=config.threshold_method)

    # Peak significance is measured on the DENOISED frame, not the top-hat residual. The
    # residual's noise is rectified by the morphological opening, so it is strongly
    # non-Gaussian and its MAD understates the tail -- a 5-sigma floor on the residual still
    # admitted noise blobs on 12 of 30 target-free frames. The denoised frame retains the
    # pedestal that robust statistics are defined against, and the same criterion there gives
    # the clean separation Tier 0 measured: real beacons at 6-40 sigma, pure noise at 3.4-3.9.
    photometric_median = float(np.median(denoised))
    photometric_sigma = robust_sigma(denoised)
    significance_floor = photometric_median + min_peak_snr * max(photometric_sigma, 1e-9)

    body = stats[1:]
    keep = gate_candidates(body, geometry)
    areas = body[:, cv2.CC_STAT_AREA].astype(np.float64)
    rejected_small = int(np.count_nonzero(areas < geometry.min_blob_area_px))
    rejected_large = int(np.count_nonzero(areas > geometry.max_blob_area_px))

    # Per-label flux, vectorised over every component at once. Ranking must happen before any
    # truncation: component labels run in raster order, so capping by label index would silently
    # discard a target near frame centre on a noisy frame.
    flux_by_label = np.bincount(labels.reshape(-1),
                                weights=np.maximum(residual, 0.0).reshape(-1).astype(np.float64),
                                minlength=count)[1:]

    detections: List[Detection] = []
    rejected_shape = 0
    rejected_faint = 0
    for index in np.nonzero(keep)[0]:
        label = int(index) + 1
        x0, y0, w, h, area = (int(v) for v in body[index])

        if float(denoised[y0:y0 + h, x0:x0 + w].max()) < significance_floor:
            rejected_faint += 1
            continue

        blob = (labels[y0:y0 + h, x0:x0 + w] == label).astype(np.uint8)
        circularity = _circularity(blob)
        if circularity < config.min_circularity:
            rejected_shape += 1
            continue

        patch = residual[y0:y0 + h, x0:x0 + w]
        detections.append(Detection(
            x=float(centroids[label][0]), y=float(centroids[label][1]),
            area_px=float(area), peak=float(patch.max()),
            flux=float(flux_by_label[index]), circularity=circularity,
            bbox=(x0, y0, w, h),
            clipped=bool(x0 <= 0 or y0 <= 0 or (x0 + w) >= width or (y0 + h) >= height),
            saturated_fraction=core_saturated_fraction(denoised[y0:y0 + h, x0:x0 + w]),
            saturated=core_saturated_fraction(
                denoised[y0:y0 + h, x0:x0 + w]) > saturation_report_threshold,
        ))

    detections.sort(key=lambda d: d.flux, reverse=True)
    return DetectionResult(detections=tuple(detections), threshold=level,
                           method=config.threshold_method, n_candidates=count - 1,
                           rejected_small=rejected_small, rejected_large=rejected_large,
                           rejected_shape=rejected_shape, rejected_faint=rejected_faint)
