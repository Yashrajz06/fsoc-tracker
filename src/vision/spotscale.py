"""Spot-scale estimation: the bootstrap that makes the vision pipeline resolution-agnostic.

``VisionConfig.resolve_geometry`` needs the beacon's scale, but the scale can only be measured
from a detection, and detection geometry depends on the scale. This module breaks that circle in
three tiers (``docs/DESIGN.md`` section 5.1.2):

* **Tier 0 (this file, :func:`bootstrap_scale`)** -- a strictly *scale-free* first pass that
  produces a candidate blob and a first FWHM estimate using no scale-dependent geometry at all.
* **Tier 1** -- scale-normalised Laplacian-of-Gaussian confirmation on the Tier-0 region.
* **Tier 2** -- continuous refinement from ROI moments while locked.

Why this matters: Benchmark Performance-2 is 30% of the grade and runs on evaluator video whose
resolution and spot size we cannot predict. Absolute pixel geometry tuned to a 640x480 frame
with a 10 px spot is the single most likely cause of failure there.

Every constant in Tier 0 is scale-free by construction:

* the median kernel is **3** because impulses are one pixel wide -- not because the spot is any
  particular size;
* the threshold is a **percentile**, which is dimensionless and adapts to any brightness range;
* the blob gates are a **sampling floor** and a **fraction of frame dimension**, never absolute
  pixel counts tuned to our own simulator.

The area gate is load-bearing, not hygiene
------------------------------------------
A 3x3 median fails wherever 5 or more of its 9 pixels are corrupted. At 10% independent
salt-and-pepper that probability is about ``8.9e-4``, leaving roughly 273 residual impulse pixels
on a 640x480 frame -- against a 99.9th-percentile selection of about 307 pixels. Without an area
gate the bootstrap would lock onto salt rather than the beacon. Residual impulses survive as
1-2 px specks; the smallest legal beacon is 5 px across (~20 px^2), so the gate separates them
with margin.

**That arithmetic assumes independent impulses, and H.264 violates the assumption.** Compression
smears impulses across transform blocks into correlated multi-pixel artifacts, and the direction
of the risk is unfavourable: artifacts get *larger*, moving out of the 1-2 px class this gate
cleanly rejects and toward the beacon's own size class. The gate is derived here from the
independent case and must be re-validated against H.264 round-tripped video in Phase 6.
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import FWHM_PER_SIGMA
from src.vision.snr import core_saturated_fraction, robust_sigma

__all__ = [
    "BootstrapParams",
    "ScaleCandidate",
    "BootstrapResult",
    "bootstrap_scale",
    "robust_background",
    "LadderParams",
    "Tier1Result",
    "confirm_scale",
    "ProfileClass",
    "ScaleEstimate",
    "estimate_scale",
    "classify_rho",
    "rho_in_agreement",
    "RHO_AGREEMENT_BAND",
    "bootstrap_scale_roi",
    "TrackerParams",
    "TrackerState",
    "ScaleTracker",
    "recommended_recheck_interval_s",
]


@dataclass(frozen=True)
class BootstrapParams:
    """Tier-0 bootstrap parameters, all expressed scale-free.

    Attributes:
        median_kernel: Impulse-rejection kernel size. Fixed at 3 by the width of an impulse, not
            by the size of the spot, which is what keeps it scale-independent.
        percentile: Selection percentile for the threshold. Dimensionless, so it adapts to any
            brightness range without assuming one.
        min_area_px: Absolute lower area gate, rejecting residual impulse specks. This is a
            *sampling* floor rather than a spot-size assumption: below a few pixels a blob
            cannot carry usable sub-pixel information whatever the true scale is.
        max_area_frame_fraction: Upper area gate as a fraction of total frame area. A beacon is
            by definition small relative to the frame; anything larger is background structure.
        min_fwhm_px: Lower clamp on the reported FWHM.
        max_fwhm_frame_fraction: Upper clamp on the reported FWHM, as a fraction of the smaller
            frame dimension. Expressed relative to the frame so it holds at any resolution.
        max_candidates: Cap on blobs examined, bounding worst-case cost on a pathological frame.
        min_peak_snr: Minimum candidate peak, in robust noise sigmas above background, for the
            scale to be reported as confident. Below this the result fails closed with
            ``fwhm_px = None`` so the caller sets ``from_fallback``.

            Measured separation on 640x480 frames: real beacons score 6.1 (peak 45 against
            sigma 15) to 40.0, while pure Gaussian noise peaks at 3.4-3.9. A floor of 5.0 sits
            cleanly between, and returning nothing is strictly better than returning a
            confident wrong scale -- the caller would resolve geometry to it and never find out.
    """

    median_kernel: int = 3
    percentile: float = 99.9
    min_area_px: float = 4.0
    max_area_frame_fraction: float = 0.01
    min_fwhm_px: float = 1.5
    max_fwhm_frame_fraction: float = 0.05
    max_candidates: int = 64
    min_peak_snr: float = 5.0

    def max_area_px(self, frame_shape: Tuple[int, int]) -> float:
        """Upper area gate in pixels for a given frame shape.

        Args:
            frame_shape: ``(height, width)``.

        Returns:
            Maximum permitted blob area in pixels.
        """
        height, width = frame_shape
        return self.max_area_frame_fraction * float(height) * float(width)

    def max_fwhm_px(self, frame_shape: Tuple[int, int]) -> float:
        """Upper FWHM clamp in pixels for a given frame shape.

        Args:
            frame_shape: ``(height, width)``.

        Returns:
            Maximum permitted FWHM in pixels.
        """
        height, width = frame_shape
        return self.max_fwhm_frame_fraction * float(min(height, width))


@dataclass(frozen=True)
class ScaleCandidate:
    """One blob surviving the Tier-0 gates.

    Attributes:
        x: Intensity-weighted centroid x in frame coordinates.
        y: Intensity-weighted centroid y in frame coordinates.
        area_px: Blob area in pixels.
        peak: Peak intensity within the blob, background-subtracted.
        flux: Integrated background-subtracted intensity over the blob. **This, not peak, is the
            ranking statistic.** Salt impulses sit at the 8-bit ceiling, so they out-rank a
            genuine beacon on peak whenever the beacon is dimmer than 255 -- measured: ranking by
            peak made bootstrap fail outright under 10% salt-and-pepper. A beacon spreads
            comparable intensity over many pixels while a surviving impulse occupies one or two,
            so integrated flux separates them decisively.
        fwhm_px: Full width at half maximum estimated from the intensity-weighted second moment.
        bbox: Bounding box as ``(x0, y0, width, height)``.
        saturated_fraction: Fraction of the blob's core pixels at the 8-bit ceiling. **Gates
            scale resolution** -- saturation inflates the half-maximum contour and therefore the
            reported scale, measured up to +173% (see :func:`estimate_scale`).
        touches_edge: Whether the bounding box touches the frame boundary. A clipped blob has a
            biased centroid *and* an underestimated scale, so this is recorded at the moment it
            is knowable rather than re-derived later.
    """

    x: float
    y: float
    area_px: float
    peak: float
    flux: float
    fwhm_px: float
    bbox: Tuple[int, int, int, int]
    saturated_fraction: float = 0.0
    touches_edge: bool = False


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a Tier-0 bootstrap pass.

    Attributes:
        candidate: The strongest surviving blob, or ``None`` when none survived.
        fwhm_px: Estimated FWHM, or ``None`` when there is no confident answer.
        candidates: All surviving blobs, strongest first. Multi-target scenarios need these.
        rejected_impulse_like: Count of blobs rejected by the lower area gate. Watching this
            number is how the H.264 re-validation in Phase 6 will show up: if compression
            artifacts start reaching the beacon's size class, they stop landing here and start
            appearing as candidates instead.
        rejected_too_large: Count of blobs rejected by the upper area gate.
        threshold: The absolute threshold level actually used, for logging.
    """

    candidate: Optional[ScaleCandidate] = None
    fwhm_px: Optional[float] = None
    candidates: Tuple[ScaleCandidate, ...] = ()
    rejected_impulse_like: int = 0
    rejected_too_large: int = 0
    threshold: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Whether the bootstrap produced a usable scale estimate."""
        return self.fwhm_px is not None


def robust_background(frame: np.ndarray) -> Tuple[float, float]:
    """Estimate background level and noise using median and MAD.

    Robust statistics are not a refinement here, they are required. Plain mean and standard
    deviation are both badly inflated by salt-and-pepper impulses: at 10% density the estimate of
    the noise sigma roughly doubles, which would silently halve every SNR we report and push
    every ``mean + k*sigma`` threshold far above the beacon.

    Args:
        frame: Input frame, any numeric dtype.

    Returns:
        ``(median, sigma)`` where sigma is ``1.4826 * MAD``, the consistent estimator of the
        standard deviation for Gaussian data.
    """
    values = frame.astype(np.float32, copy=False).reshape(-1)
    return float(np.median(values)), robust_sigma(values)


def _fwhm_from_half_max_area(residual: np.ndarray, peak_yx: Tuple[int, int],
                             noise_sigma: float) -> float:
    """Estimate FWHM from the area of the half-maximum contour around a given peak.

    For a roughly circular spot the half-maximum contour encloses a disc of diameter FWHM, so
    ``FWHM = 2 * sqrt(area / pi)`` directly by definition.

    **Preferred over a second moment.** A second moment weights by distance squared, so it is
    dominated by the patch's outer pixels -- exactly where signal is weakest and noise strongest.
    Worse, clipping a background-subtracted residual at zero rectifies zero-mean noise into a
    strictly *positive* pedestal, which the ``r^2`` weighting then amplifies in proportion to
    patch area. Measured error of the second-moment estimator ran 21-99% under noise; the
    half-maximum area stays near 6-8% because it only ever counts pixels near the peak.

    The contour is restricted to the connected component containing the seed peak, so an
    isolated noise spike above half maximum cannot join the measurement.

    Args:
        residual: 2-D background-subtracted intensities.
        peak_yx: ``(row, col)`` of the seed peak. **Must come from the flux-ranked candidate
            blob, never from the global argmax of the frame:** salt impulses sit at the 8-bit
            ceiling, so a global argmax lands on salt and the contour collapses to a single
            pixel. Measured, that failure produced 1.13 px against a true 5.89.
        noise_sigma: Robust noise sigma, used to reject peaks that are not real signal.

    Returns:
        Estimated FWHM in pixels, or 0.0 when the peak is not significant above the noise.
    """
    row, col = peak_yx
    if not (0 <= row < residual.shape[0] and 0 <= col < residual.shape[1]):
        return 0.0
    peak = float(residual[row, col])
    if peak <= 3.0 * max(noise_sigma, 1e-6):
        return 0.0

    mask = (residual >= 0.5 * peak).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    seed = int(labels[row, col])
    if seed == 0:
        return 0.0
    area = float(np.count_nonzero(labels == seed))
    return 2.0 * math.sqrt(area / math.pi)


def bootstrap_scale(frame: np.ndarray,
                    params: Optional[BootstrapParams] = None,
                    gate_shape: Optional[Tuple[int, int]] = None) -> BootstrapResult:
    """Estimate the beacon's scale from a frame, using no scale-dependent geometry.

    This is Tier 0. It runs on the first frames of a run, after any loss of lock, and whenever a
    re-check is triggered. It is deliberately conservative: it would rather return no answer
    (leaving :attr:`BootstrapResult.fwhm_px` as ``None``, which drives ``from_fallback``) than a
    confident wrong one.

    Args:
        frame: Input frame, 2-D, any numeric dtype. Not modified.
        params: Bootstrap parameters. Defaults are used when omitted.
        gate_shape: ``(height, width)`` used to size the frame-fraction gates. Defaults to the
            shape of ``frame``. **Must be supplied when running on a cropped ROI**, otherwise the
            gates are computed from the crop and become far too tight: on a 128 px ROI the
            ``max_fwhm`` ceiling collapses to 6.4 px, which rejects a perfectly legitimate 14 px
            beacon. The gates describe "small relative to the scene", and the scene is the full
            frame regardless of how much of it we chose to examine.

    Returns:
        A :class:`BootstrapResult`. When no blob survives the gates, ``candidate`` and
        ``fwhm_px`` are ``None`` and the caller must fall back to configured geometry.

    Raises:
        ValueError: If the frame is not 2-D.
    """
    if frame.ndim != 2:
        raise ValueError(f"Bootstrap requires a 2-D frame, got shape {frame.shape}")
    params = params or BootstrapParams()

    height, width = frame.shape
    gates_shape = gate_shape if gate_shape is not None else frame.shape
    working = frame.astype(np.float32, copy=False)

    # 1. Impulse rejection. Kernel size is set by impulse width, not by the spot, so this stage
    #    is scale-free. OpenCV's median needs uint8 or float32 with an odd kernel.
    if params.median_kernel > 1:
        working = cv2.medianBlur(working, params.median_kernel)

    # 2. Robust background, then a dimensionless percentile threshold. Nothing here assumes a
    #    brightness range, which is what lets it survive an unseen evaluator video.
    background, sigma = robust_background(working)
    level = float(np.percentile(working, params.percentile))
    # Never accept a threshold at or below the background itself: on a frame with no target the
    # percentile lands inside the noise, and thresholding there floods us with spurious blobs.
    threshold = max(level, background + max(sigma, 1e-6))

    mask = (working >= threshold).astype(np.uint8)
    residual = np.maximum(working - background, 0.0)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_area = params.max_area_px(gates_shape)
    max_fwhm = params.max_fwhm_px(gates_shape)
    candidates: List[ScaleCandidate] = []
    rejected_small = 0
    rejected_large = 0

    if count > 1:
        # Per-label integrated flux for *every* component at once. Computing this vectorised is
        # not just an optimisation: it is what lets ranking happen before truncation.
        #
        # An earlier version examined only the first `max_candidates` labels. Connected-component
        # labels are assigned in raster order, so on a noisy frame with hundreds of spurious
        # blobs the beacon -- typically near frame centre -- received a high label index and was
        # never examined at all. Truncating by label index silently discards the target; any cap
        # must be applied after ranking by strength, never before.
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        flux_by_label = np.bincount(labels.reshape(-1),
                                    weights=residual.reshape(-1).astype(np.float64),
                                    minlength=count)[1:]

        keep_small = areas >= params.min_area_px
        keep_large = areas <= max_area
        rejected_small = int(np.count_nonzero(~keep_small))
        rejected_large = int(np.count_nonzero(keep_small & ~keep_large))

        eligible = np.nonzero(keep_small & keep_large)[0]
        # Rank by flux first, then measure only the strongest few. The expensive second-moment
        # work is thereby bounded regardless of how many blobs the noise produced.
        eligible = eligible[np.argsort(flux_by_label[eligible])[::-1][:params.max_candidates]]

        for index in eligible:
            label = int(index) + 1
            x0, y0, w, h, area = stats[label]

            # Pad generously: a high percentile keeps only the spot's core, so a window sized to
            # the thresholded blob would truncate the wings and bias the second moment *low*.
            # Two blob-extents of padding is still scale-free, being relative to the blob itself.
            pad = 2 * max(int(w), int(h))
            px0, py0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            px1, py1 = min(width, int(x0) + int(w) + pad), min(height, int(y0) + int(h) + pad)
            patch = residual[py0:py1, px0:px1]

            core = residual[int(y0):int(y0) + int(h), int(x0):int(x0) + int(w)]
            # Seed the half-maximum contour from this blob's own peak, not the frame's. The blob
            # was selected by integrated flux, which is salt-robust; the frame's global maximum
            # is not.
            local = np.unravel_index(int(np.argmax(core)), core.shape)
            peak_yx = (int(y0) + int(local[0]), int(x0) + int(local[1]))
            fwhm = _fwhm_from_half_max_area(patch, (peak_yx[0] - py0, peak_yx[1] - px0), sigma)
            touches = (int(x0) <= 0 or int(y0) <= 0
                       or (int(x0) + int(w)) >= width or (int(y0) + int(h)) >= height)

            # Saturation over the blob's own core, measured on the *original* frame so the
            # median prefilter cannot mask a flat top.
            raw_core = frame[int(y0):int(y0) + int(h), int(x0):int(x0) + int(w)]
            sat = core_saturated_fraction(raw_core)

            candidates.append(ScaleCandidate(
                x=float(centroids[label][0]), y=float(centroids[label][1]),
                area_px=float(area), peak=float(core.max()),
                flux=float(flux_by_label[index]), fwhm_px=fwhm,
                bbox=(int(x0), int(y0), int(w), int(h)),
                saturated_fraction=sat, touches_edge=bool(touches)))

    # Strongest first by *integrated flux*, never by peak. See ScaleCandidate.flux: salt
    # impulses are pinned at the 8-bit ceiling and out-rank any beacon dimmer than 255 on peak.
    candidates.sort(key=lambda c: c.flux, reverse=True)

    best = candidates[0] if candidates else None
    fwhm: Optional[float] = None
    if (best is not None
            and best.peak >= params.min_peak_snr * max(sigma, 1e-6)
            and params.min_fwhm_px <= best.fwhm_px <= max_fwhm):
        fwhm = best.fwhm_px

    return BootstrapResult(candidate=best, fwhm_px=fwhm, candidates=tuple(candidates),
                           rejected_impulse_like=rejected_small,
                           rejected_too_large=rejected_large, threshold=threshold)


# --------------------------------------------------------------------------------------------
# Tier 1: scale-normalised Laplacian-of-Gaussian confirmation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderParams:
    """Tier-1 scale-space search parameters.

    Attributes:
        sigma_min: Smallest sigma on the ladder, in pixels.
        ratio: Geometric step between rungs. 1.3 is fine enough that the parabola refinement
            below has a well-conditioned three-point neighbourhood, and coarse enough that the
            ladder spans a useful range in ~11 rungs.
        steps: Number of rungs.
        window_px: Half-width of the neighbourhood searched around the Tier-0 seed. Kept small so
            the confirmation is an ROI operation, not a full-frame one.
        min_curvature: Confidence floor. Below this the response peak is too flat for the scale
            to be considered determined.
    """

    sigma_min: float = 0.8
    ratio: float = 1.3
    steps: int = 11
    window_px: int = 6
    min_curvature: float = 0.5

    def sigmas(self) -> np.ndarray:
        """Return the sigma ladder as an array."""
        return self.sigma_min * self.ratio ** np.arange(self.steps, dtype=np.float64)


@dataclass(frozen=True)
class Tier1Result:
    """Outcome of a Tier-1 confirmation pass.

    Attributes:
        sigma_px: Refined characteristic sigma, or ``None`` when undetermined.
        fwhm_px: ``2.355 * sigma_px``, expressed as an equivalent-Gaussian FWHM.
        curvature: Peak sharpness from the parabola fit. **This is the confidence measure**, not
            peak-to-runner-up: measured, the ratio is 1.01-1.06 for correct estimates on real
            spots but 1.12-1.14 for pure noise, i.e. inverted and useless. Curvature reads
            ~1.9-2.4 for real spots and 0.0 for noise.
        at_ladder_edge: Whether the argmax landed on the first or last rung, meaning the true
            scale lies outside the searched range and the argmax is meaningless.
    """

    sigma_px: Optional[float] = None
    fwhm_px: Optional[float] = None
    curvature: float = 0.0
    at_ladder_edge: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether Tier 1 produced a usable scale."""
        return self.fwhm_px is not None


def confirm_scale(frame: np.ndarray, seed_xy: Tuple[int, int],
                  params: Optional[LadderParams] = None) -> Tier1Result:
    """Confirm the spot scale by scale-normalised Laplacian-of-Gaussian, near a seed point.

    Lindeberg scale selection: ``sigma^2 * |laplacian(G_sigma * I)|`` attains a genuine maximum
    over sigma at a blob's characteristic scale.

    **Top-hat cannot be used for this.** Its response is monotonically non-decreasing in kernel
    size once the structuring element exceeds the spot -- a larger element simply passes the
    whole spot plus more background -- so there is no interior maximum to find. Measured on a
    sigma=2.5 spot over a x1.3 ladder, the argmax landed at the largest kernel with a
    peak-to-runner-up ratio of exactly 1.000. Top-hat remains right for background suppression
    in the detection path; it is the wrong operator for scale selection.

    Args:
        frame: Input frame, 2-D. Not modified.
        seed_xy: ``(x, y)`` of the Tier-0 candidate, in frame coordinates. Restricting the search
            to this neighbourhood is what makes the multi-scale pass affordable -- full-frame at
            11 scales on a 2000x2000 video would not be.
        params: Ladder parameters. Defaults are used when omitted.

    Returns:
        A :class:`Tier1Result`. ``sigma_px`` is ``None`` when the argmax sits at a ladder edge or
        the peak is too flat to be trusted.

    Raises:
        ValueError: If the frame is not 2-D.
    """
    if frame.ndim != 2:
        raise ValueError(f"Tier 1 requires a 2-D frame, got shape {frame.shape}")
    params = params or LadderParams()

    height, width = frame.shape
    x, y = int(seed_xy[0]), int(seed_xy[1])
    sigmas = params.sigmas()
    win = params.window_px

    # Crop to a padded ROI *before* running the blur ladder, not after. Blurring the full frame
    # and sampling only the ROI gives identical answers but costs 121 ms on a 2000x2000 frame
    # versus 18 ms on 640x480 -- i.e. it is not ROI-limited at all, which defeats the point of
    # confirming on a region. The pad must cover the largest kernel's reach (3 sigma_max) so the
    # cropped result matches the full-frame one away from the crop border.
    pad = int(math.ceil(3.0 * float(sigmas[-1]))) + win + 2
    cy0, cy1 = max(0, y - pad), min(height, y + pad + 1)
    cx0, cx1 = max(0, x - pad), min(width, x + pad + 1)
    working = frame[cy0:cy1, cx0:cx1].astype(np.float32, copy=False)

    # Seed position within the crop, and the sampling window around it.
    sy, sx = y - cy0, x - cx0
    y0, y1 = max(0, sy - win), min(working.shape[0], sy + win + 1)
    x0, x1 = max(0, sx - win), min(working.shape[1], sx + win + 1)

    response = np.empty(sigmas.size, dtype=np.float64)
    for index, sigma in enumerate(sigmas):
        kernel = int(2 * round(3.0 * sigma) + 1)
        blurred = cv2.GaussianBlur(working, (kernel, kernel), float(sigma))
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        response[index] = (sigma * sigma) * float(np.abs(laplacian[y0:y1, x0:x1]).max())

    peak = int(np.argmax(response))
    if peak == 0 or peak == sigmas.size - 1:
        # The true scale is outside the searched range, so the argmax carries no information.
        return Tier1Result(at_ladder_edge=True)

    # Parabola fit in log-log. Gives sub-rung refinement and the curvature confidence together.
    log_sigma = np.log(sigmas[peak - 1:peak + 2])
    log_response = np.log(np.maximum(response[peak - 1:peak + 2], 1e-12))
    a, b, _ = np.polyfit(log_sigma, log_response, 2)
    if a >= 0:
        return Tier1Result(curvature=0.0)

    curvature = float(-2.0 * a)
    if curvature < params.min_curvature:
        return Tier1Result(curvature=curvature)

    sigma = float(np.exp(-b / (2.0 * a)))
    return Tier1Result(sigma_px=sigma, fwhm_px=FWHM_PER_SIGMA * sigma, curvature=curvature)


# --------------------------------------------------------------------------------------------
# Combining the tiers
# --------------------------------------------------------------------------------------------


class ProfileClass(str, Enum):
    """Profile class suggested by the Tier-1/Tier-0 ratio.

    **Advisory only. This is not a reliable classifier and must not gate anything.**

    ``rho = Tier1_FWHM / Tier0_FWHM`` looked promising: both estimators return a Gaussian's true
    FWHM, so ``rho ~ 1.00`` for a smooth profile, while their shape biases are *opposed* on
    hard-edged profiles, giving ``rho ~ 0.865``. An initial 216-sample sweep (4 sizes, all
    >= 8 px, one brightness) showed the clusters separating with a gap of +0.024.

    **A wider sweep refuted that.** Over 10 sizes from 5 to 24 px, 4 noise levels and 4 seeds:

    ==========  =====  =====  ================
    profile     mean   std    range
    ==========  =====  =====  ================
    gaussian    0.987  0.033  [0.868, 1.054]
    square      0.876  0.029  [0.843, 0.976]
    circle      0.861  0.040  [0.822, 1.072]
    ==========  =====  =====  ================

    The raw gap is **-0.204**: 99 hard-edged samples sit above the Gaussian minimum and 144
    Gaussian samples below the hard-edged maximum. Smoothing does not rescue it -- the gap stays
    at about -0.05 for run means over 1, 5, 10 and 20 frames, because the overlap is driven by
    *systematic variation across spot size*, not by frame-to-frame noise, so averaging cannot
    remove it.

    The class is therefore reported for diagnostics and logging only, never used to decide
    ``from_fallback``.
    """

    SMOOTH = "smooth"
    HARD_EDGED = "hard_edged"
    UNEXPLAINED = "unexplained"


#: Indicative cluster centres, for the advisory label only. Overlapping -- see
#: :class:`ProfileClass`.
SMOOTH_RHO_RANGE: Tuple[float, float] = (0.945, 1.10)
HARD_EDGED_RHO_RANGE: Tuple[float, float] = (0.80, 0.925)

#: Agreement band on ``rho``, as ``(low, high)``. Because profile class cannot be inferred
#: reliably, the band must **absorb** the shape divergence rather than explain it away. Budget,
#: stated explicitly:
#:
#: * shape divergence between the two estimators: up to ~15% (Gaussian mean 0.987 against
#:   hard-edged mean 0.861);
#: * spread from spot size and noise: about +/-11% at 3 sigma (std 0.033-0.040);
#: * total ~26%, so the band is set at +/-30% with a little margin.
#:
#: This is deliberately wide, and that costs sensitivity -- which is why ``from_fallback`` does
#: **not** lean on ``rho``. The sharp gates are the saturation gate, the Tier-1 curvature floor,
#: the ladder-edge check and the Tier-0 peak-significance floor, all of which discriminate
#: cleanly. ``rho`` is the weakest signal of the five and is weighted accordingly.
RHO_AGREEMENT_BAND: Tuple[float, float] = (0.70, 1.30)

#: Saturation fraction above which a frame may not be used to resolve scale. Clipping flattens
#: the spot's top, widening the half-maximum contour and inflating the reported scale: measured
#: +53% at 24-29% saturation, +102% at 37-39%, and +173% at 45%, identically for a 6 px and a
#: 14 px beacon. The ``max_fwhm`` clamp is **not** a safety net -- a 6 px beacon at heavy
#: saturation reports 16.4 px, stays under the clamp, and claims success. Only this explicit
#: gate catches it.
#:
#: Note this is the opposite policy to the Phase 2 saturation finding for *centroiding*, where
#: the bias is ~0.19 px against a 10 px budget and the right response is to inflate ``R`` rather
#: than reject the measurement. Same physical effect, opposite conclusion, because a centroid
#: tolerates a bounded bias whereas a wrong scale poisons every downstream kernel and ROI size.
MAX_SATURATION_FOR_SCALE: float = 0.10


def classify_rho(rho: Optional[float]) -> ProfileClass:
    """Label a Tier1/Tier0 ratio with an advisory profile class.

    **Advisory only** -- the clusters overlap heavily, so this label must not gate behaviour.
    See :class:`ProfileClass` for the measurements.

    Args:
        rho: The ratio, or ``None``.

    Returns:
        The indicative :class:`ProfileClass`.
    """
    if rho is None:
        return ProfileClass.UNEXPLAINED
    if SMOOTH_RHO_RANGE[0] <= rho <= SMOOTH_RHO_RANGE[1]:
        return ProfileClass.SMOOTH
    if HARD_EDGED_RHO_RANGE[0] <= rho <= HARD_EDGED_RHO_RANGE[1]:
        return ProfileClass.HARD_EDGED
    return ProfileClass.UNEXPLAINED


def rho_in_agreement(rho: Optional[float]) -> bool:
    """Whether the two estimators agree within the band.

    Args:
        rho: The Tier1/Tier0 ratio, or ``None``.

    Returns:
        True when ``rho`` lies inside :data:`RHO_AGREEMENT_BAND`.
    """
    if rho is None:
        return False
    return RHO_AGREEMENT_BAND[0] <= rho <= RHO_AGREEMENT_BAND[1]


@dataclass(frozen=True)
class ScaleEstimate:
    """A spot-scale estimate with the full trace needed to explain it.

    Every diagnostic here exists because a bare flag is worth less than the flag plus the trace
    that explains it -- the same reasoning as ``clipped``, ``saturated`` and ``from_fallback``.

    Attributes:
        fwhm_px: The reported scale, or ``None`` when there is no confident answer.
        from_fallback: True when the caller must use configured geometry instead.
        reason: Short machine-readable code for *why* fallback was chosen. Logged per frame.
        rho: Tier1/Tier0 ratio, or ``None`` when either tier failed. **Log this every frame, not
            only when geometry is resolved.** A rho trajectory that walks between clusters is
            diagnosable after the fact; a from_fallback that fires with no history is not.
        profile_class: **Advisory** profile label implied by ``rho``. Not a reliable classifier
            and never used to gate behaviour -- see :class:`ProfileClass`.
        tier0_fwhm_px: Tier-0 half-max-area estimate, for the trace.
        tier1_fwhm_px: Tier-1 LoG estimate, for the trace.
        curvature: Tier-1 peak sharpness, the confidence measure.
        saturated_fraction: Saturation over the candidate blob's core.
        clipped: Whether the candidate touches the frame edge.
    """

    fwhm_px: Optional[float] = None
    from_fallback: bool = True
    reason: str = "no_candidate"
    rho: Optional[float] = None
    profile_class: ProfileClass = ProfileClass.UNEXPLAINED
    tier0_fwhm_px: Optional[float] = None
    tier1_fwhm_px: Optional[float] = None
    curvature: float = 0.0
    saturated_fraction: float = 0.0
    clipped: bool = False


def estimate_scale(frame: np.ndarray,
                   bootstrap_params: Optional[BootstrapParams] = None,
                   ladder_params: Optional[LadderParams] = None,
                   max_saturation: float = MAX_SATURATION_FOR_SCALE) -> ScaleEstimate:
    """Estimate spot scale end to end: Tier 0, Tier 1 confirmation, and the agreement rule.

    Tier 0 supplies the reported scale -- it is the more noise-robust of the two, because the
    half-maximum contour only counts pixels near the peak. Tier 1 is purely the cross-check.

    **Saturation gates the estimate outright.** Clipping flattens the spot's top, which widens
    the half-maximum contour and inflates the reported scale. Measured, for both a 6 px and a
    14 px beacon:

    ==================  ==================
    saturated fraction  scale error
    ==================  ==================
    0.00                2-5%
    0.24-0.29           +53%
    0.37-0.39           +102%
    0.45                +173%
    ==================  ==================

    A 6 px beacon at heavy saturation reports 16.4 px and would otherwise have claimed success:
    the ``max_fwhm`` clamp caught only the 14 px case, and only incidentally, because that spot
    was large enough relative to the frame to trip it. The clamp is not a saturation safety net.

    Note this is the **opposite** policy to the Phase 2 saturation finding for *centroiding*,
    where the bias is ~0.19 px against a 10 px budget and the right response is to inflate ``R``
    rather than reject the measurement. Same physical effect, opposite conclusion, because the
    two uses differ enormously in sensitivity: a centroid tolerates a bounded bias, whereas a
    wrong scale poisons every downstream kernel and ROI size for subsequent frames.

    Args:
        frame: Input frame, 2-D. Not modified.
        bootstrap_params: Tier-0 parameters.
        ladder_params: Tier-1 parameters.
        max_saturation: Saturation fraction above which scale may not be resolved.

    Returns:
        A :class:`ScaleEstimate`, always populated with the diagnostic trace even when it fails.
    """
    tier0 = bootstrap_scale(frame, bootstrap_params)
    if tier0.candidate is None:
        return ScaleEstimate(reason="no_candidate")

    candidate = tier0.candidate
    base = ScaleEstimate(
        tier0_fwhm_px=tier0.fwhm_px,
        saturated_fraction=candidate.saturated_fraction,
        clipped=candidate.touches_edge,
    )

    if candidate.saturated_fraction > max_saturation:
        return replace(base, reason="saturated")
    if tier0.fwhm_px is None:
        return replace(base, reason="tier0_rejected")

    tier1 = confirm_scale(frame, (int(round(candidate.x)), int(round(candidate.y))),
                          ladder_params)
    base = replace(base, tier1_fwhm_px=tier1.fwhm_px, curvature=tier1.curvature)

    if not tier1.succeeded:
        reason = "tier1_ladder_edge" if tier1.at_ladder_edge else "tier1_low_curvature"
        return replace(base, reason=reason)

    assert tier1.fwhm_px is not None
    rho = tier1.fwhm_px / tier0.fwhm_px
    profile = classify_rho(rho)
    base = replace(base, rho=rho, profile_class=profile)

    if not rho_in_agreement(rho):
        return replace(base, reason="rho_out_of_band")

    return replace(base, fwhm_px=tier0.fwhm_px, from_fallback=False, reason="ok")


def bootstrap_scale_roi(frame: np.ndarray, center_xy: Tuple[float, float], size_px: int,
                        params: Optional[BootstrapParams] = None) -> BootstrapResult:
    """Run the Tier-0 bootstrap on a padded window instead of the whole frame.

    Once lock is established we know roughly where the target is, so a periodic re-check has no
    reason to sweep the full canvas. Full-frame Tier 0 is needed only for cold bootstrap and for
    post-loss re-acquisition, where the target's position is genuinely unknown.

    This matters because full-frame Tier 0 is the dominant steady-state cost: 253 ms on a
    2000x2000 frame, against ~1 ms for ROI-cropped Tier 1.

    Candidate coordinates and bounding boxes are translated back into full-frame coordinates, so
    the result is indistinguishable from a full-frame pass apart from what it could see. Note
    that ``touches_edge`` then refers to the **ROI** boundary, not the frame boundary -- a target
    against the ROI edge is a sign the ROI is mispositioned or too small, which is worth knowing
    in its own right.

    Args:
        frame: Full input frame, 2-D. Not modified.
        center_xy: Approximate target position ``(x, y)`` in full-frame coordinates.
        size_px: Side length of the square window to examine.
        params: Tier-0 parameters.

    Returns:
        A :class:`BootstrapResult` with coordinates in full-frame terms.

    Raises:
        ValueError: If the frame is not 2-D or ``size_px`` is not positive.
    """
    if frame.ndim != 2:
        raise ValueError(f"Bootstrap requires a 2-D frame, got shape {frame.shape}")
    if size_px <= 0:
        raise ValueError(f"ROI size must be positive, got {size_px}")

    height, width = frame.shape
    half = size_px // 2
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, cx + half + 1), min(height, cy + half + 1)
    if x0 >= x1 or y0 >= y1:
        return BootstrapResult()

    # Gates are sized from the FULL frame, not the crop -- see bootstrap_scale(gate_shape).
    result = bootstrap_scale(frame[y0:y1, x0:x1], params, gate_shape=frame.shape)
    if not result.candidates:
        return result

    shifted = tuple(
        replace(c, x=c.x + x0, y=c.y + y0,
                bbox=(c.bbox[0] + x0, c.bbox[1] + y0, c.bbox[2], c.bbox[3]))
        for c in result.candidates
    )
    best = shifted[0]
    return replace(result, candidate=best, candidates=shifted,
                   fwhm_px=result.fwhm_px)


# --------------------------------------------------------------------------------------------
# Tier 2: continuous tracking, re-check scheduling and ratchet defence
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerParams:
    """Tier-2 scale-tracking parameters.

    Attributes:
        smoothing_alpha: EMA coefficient for valid measurements. The true scale changes slowly
            while per-frame estimates are noisy, so smoothing stops the whole vision geometry
            breathing frame to frame.
        confirm_m_of_n: ``(M, N)`` agreement required before a scale is adopted for the first
            time. Never let one frame set the scale -- a bad first estimate would otherwise
            poison the entire run.
        agreement_tolerance: Fractional spread within which M-of-N measurements count as
            agreeing.
        deadband_px: Re-resolve geometry only when a resolved integer would move by at least
            this much. Resolved geometry yields *integer* kernel sizes, so an estimate
            oscillating near a boundary would chatter the kernel frame to frame and shift the
            detection statistics underneath the tracker.
        drift_trigger: Fractional drift from the last resolved scale that forces a re-check.
        recheck_interval_s: Seconds between periodic re-checks. See
            :func:`recommended_recheck_interval_s`, which derives this from measured cost rather
            than assuming a round number.
        search_growth: Multiplier applied to the *search* scale per consecutive frame with no
            candidate at all. Underestimating scale is self-reinforcing -- too small a top-hat
            element deletes the beacon, and no detection means no correction -- so the search
            widens rather than narrows when blind.
        max_search_inflation: Ceiling on accumulated search growth.
    """

    smoothing_alpha: float = 0.2
    confirm_m_of_n: Tuple[int, int] = (3, 5)
    agreement_tolerance: float = 0.25
    deadband_px: float = 2.0
    drift_trigger: float = 0.25
    recheck_interval_s: float = 2.0
    search_growth: float = 1.15
    max_search_inflation: float = 3.0


@dataclass(frozen=True)
class TrackerState:
    """The tracker's view after one update.

    Attributes:
        estimated_fwhm_px: Smoothed scale from valid measurements, or ``None`` before adoption.
        resolved_fwhm_px: The scale geometry is currently resolved at. Changes only outside the
            deadband.
        search_fwhm_px: Scale used for *searching* while blind. Transient and never folded back
            into the estimate.
        search_inflation: Current transient search multiplier. 1.0 whenever a candidate is seen.
        adopted: Whether M-of-N confirmation has been satisfied.
        from_fallback: Whether the caller must use configured geometry.
        reason: Machine-readable code for the most recent update.
        rho: Latest Tier1/Tier0 ratio, logged every frame as a diagnostic trace.
        consecutive_blind: Consecutive frames with no candidate at all.
        consecutive_blocked: Consecutive frames with a candidate whose scale could not be
            resolved (saturation, band, curvature). **Counted separately from blind frames**, and
            deliberately does not drive search growth.
    """

    estimated_fwhm_px: Optional[float] = None
    resolved_fwhm_px: Optional[float] = None
    search_fwhm_px: Optional[float] = None
    search_inflation: float = 1.0
    adopted: bool = False
    from_fallback: bool = True
    reason: str = "init"
    rho: Optional[float] = None
    consecutive_blind: int = 0
    consecutive_blocked: int = 0


def recommended_recheck_interval_s(frame_shape: Tuple[int, int],
                                   measured_cost_ms: Optional[float] = None,
                                   duty_budget: float = 0.02,
                                   minimum_s: float = 0.5,
                                   maximum_s: float = 5.0) -> float:
    """Derive a re-check interval from measured cost and a duty-cycle budget.

    A single fixed interval cannot be right for every frame size. Measured full-frame Tier 0
    costs 17 ms at 640x480 and 234 ms at 2000x2000 -- a 14x span. Once locked, however, re-checks
    run Tier 0 on a padded ROI plus ROI-cropped Tier 1, which measures ~2.1-2.5 ms at *every*
    frame size, so the steady-state interval barely needs to vary. The frame-size dependence that
    remains matters for cold bootstrap and post-loss re-acquisition, which are full-frame.

    Args:
        frame_shape: ``(height, width)``.
        measured_cost_ms: Cost of one re-check, if known. Estimated from frame area otherwise.
        duty_budget: Fraction of wall-clock time the re-check may consume.
        minimum_s: Floor, so re-checks never dominate the loop.
        maximum_s: Ceiling, so a scale change is never missed for long.

    Returns:
        Interval in seconds, clamped to ``[minimum_s, maximum_s]``.

    Raises:
        ValueError: If ``duty_budget`` is not in ``(0, 1]``.
    """
    if not 0.0 < duty_budget <= 1.0:
        raise ValueError(f"Duty budget must be in (0, 1], got {duty_budget}")
    if measured_cost_ms is None:
        # ROI re-checks are near frame-size independent; scale gently for decode/mask overheads.
        height, width = frame_shape
        measured_cost_ms = 2.0 + 0.5 * (height * width) / (2000.0 * 2000.0)
    interval = (measured_cost_ms / 1000.0) / duty_budget
    return float(min(maximum_s, max(minimum_s, interval)))


class ScaleTracker:
    """Tracks spot scale over a run, with re-check scheduling and ratchet defence.

    **The ratchet, and why the design separates two scales.** Growing the assumed scale when
    detection fails is necessary -- too small a scale deletes the beacon under the top-hat and
    the failure is self-reinforcing. But growth plus the integer deadband can interact into a
    one-way ratchet: each failure nudges the scale up, the deadband lets accumulated growth take
    effect, and nothing symmetric ever pushes it back down.

    The fix is to keep the two quantities apart:

    * :attr:`TrackerState.estimated_fwhm_px` is **persistent** and is updated *only* from valid
      measurements. Failures never write to it.
    * :attr:`TrackerState.search_fwhm_px` is **transient**. It inflates while blind and is reset
      to the estimate the moment any candidate is seen again.

    Because growth lives only in the transient quantity, a run with intermittent failures at
    constant true scale returns to the correct resolved kernel rather than drifting upward.

    **The second path into the ratchet** is scale resolution being *blocked* rather than absent:
    saturation gating, an out-of-band ``rho``, or low curvature all leave us with a perfectly
    good detection whose scale we decline to trust. A gated stretch looks like sustained failure
    to a naive grow policy. It is not -- we can see the target, we simply cannot measure it -- so
    blocked frames **hold** the current scale and never inflate the search.
    """

    def __init__(self, params: Optional[TrackerParams] = None,
                 fallback_fwhm_px: float = 5.89) -> None:
        """Initialise an unadopted tracker.

        Args:
            params: Tracker parameters.
            fallback_fwhm_px: Configured scale used until a measurement is adopted.
        """
        self.params = params or TrackerParams()
        self.fallback_fwhm_px = float(fallback_fwhm_px)
        self._estimate: Optional[float] = None
        self._resolved: Optional[float] = None
        self._pending: List[float] = []
        self._inflation = 1.0
        self._blind = 0
        self._blocked = 0
        self._last_check_s = -math.inf
        self._last_state = TrackerState(search_fwhm_px=self.fallback_fwhm_px)

    @property
    def state(self) -> TrackerState:
        """The state produced by the most recent update, or the initial state before any.

        Returns the *last emitted* state rather than rebuilding one, so that ``reason`` and
        ``rho`` survive the query. Telemetry reads this every frame, and a state whose reason
        code was replaced by a placeholder on read would lose exactly the attribution the trace
        exists to provide.
        """
        return self._last_state

    def _search_scale(self) -> float:
        """Scale to search at, including any transient inflation."""
        base = self._estimate if self._estimate is not None else self.fallback_fwhm_px
        return base * self._inflation

    def due_for_recheck(self, now_s: float) -> bool:
        """Whether a periodic re-check is due.

        Args:
            now_s: Current run time in seconds.

        Returns:
            True when the configured interval has elapsed.
        """
        return (now_s - self._last_check_s) >= self.params.recheck_interval_s

    def needs_full_frame(self) -> bool:
        """Whether the next check must sweep the full frame rather than an ROI.

        Full-frame is needed only for cold bootstrap and post-loss re-acquisition, where the
        target position is genuinely unknown. Once locked, an ROI re-check is ~2.5 ms at any
        frame size against up to 234 ms full-frame.
        """
        return self._estimate is None or self._blind > 0

    def update(self, estimate: ScaleEstimate, now_s: float) -> TrackerState:
        """Fold one scale estimate into the tracker.

        Args:
            estimate: Result of :func:`estimate_scale` for this frame.
            now_s: Current run time in seconds.

        Returns:
            The new :class:`TrackerState`.
        """
        self._last_check_s = now_s
        params = self.params

        if estimate.reason == "no_candidate":
            # Genuinely blind: widen the transient search. The persistent estimate is untouched.
            self._blind += 1
            self._inflation = min(params.max_search_inflation,
                                  self._inflation * params.search_growth)
            return self._emit("blind", estimate)

        # A candidate exists, so we are not blind -- reset the transient inflation immediately.
        self._blind = 0
        self._inflation = 1.0

        if estimate.from_fallback or estimate.fwhm_px is None:
            # Blocked, not blind: we can see the target but decline to trust its scale. Hold.
            self._blocked += 1
            return self._emit(f"blocked:{estimate.reason}", estimate)

        self._blocked = 0
        measurement = float(estimate.fwhm_px)

        if self._estimate is None:
            # M-of-N confirmation before first adoption.
            self._pending.append(measurement)
            m, n = params.confirm_m_of_n
            if len(self._pending) > n:
                self._pending.pop(0)
            agreeing = [v for v in self._pending
                        if abs(v - measurement) / measurement <= params.agreement_tolerance]
            if len(agreeing) < m:
                return self._emit("confirming", estimate)
            self._estimate = float(np.median(agreeing))
            self._resolved = self._estimate
            self._pending.clear()
            return self._emit("adopted", estimate)

        alpha = params.smoothing_alpha
        self._estimate = (1.0 - alpha) * self._estimate + alpha * measurement

        # Deadband: re-resolve only when the change is large enough to matter downstream.
        assert self._resolved is not None
        if abs(self._estimate - self._resolved) >= params.deadband_px:
            self._resolved = self._estimate
            return self._emit("re_resolved", estimate)
        return self._emit("tracking", estimate)

    def _emit(self, reason: str, estimate: ScaleEstimate) -> TrackerState:
        """Build a state snapshot carrying the per-frame diagnostic trace.

        Args:
            reason: Machine-readable code for this update.
            estimate: The estimate that produced it.

        Returns:
            The :class:`TrackerState`.
        """
        self._last_state = TrackerState(
            estimated_fwhm_px=self._estimate,
            resolved_fwhm_px=self._resolved,
            search_fwhm_px=self._search_scale(),
            search_inflation=self._inflation,
            adopted=self._estimate is not None,
            from_fallback=self._resolved is None,
            reason=reason,
            rho=estimate.rho,
            consecutive_blind=self._blind,
            consecutive_blocked=self._blocked,
        )
        return self._last_state

    def reset(self) -> None:
        """Return the tracker to its initial unadopted state."""
        self._estimate = None
        self._resolved = None
        self._pending.clear()
        self._inflation = 1.0
        self._blind = 0
        self._blocked = 0
        self._last_check_s = -math.inf
        self._last_state = TrackerState(search_fwhm_px=self.fallback_fwhm_px)
