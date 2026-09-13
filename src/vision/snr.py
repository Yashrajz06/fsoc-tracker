"""Signal-to-noise ratio, defined once so the SNR curve is reproducible.

"SNR" is used inconsistently across the centroiding literature -- peak, integrated, and ROI
definitions all appear under the same name -- and the centroid-error-vs-SNR curve is a graded
deliverable. So the definition is settled here rather than at each call site, and the formula
itself (not merely its name) goes on the plot axis, so the curve can be reproduced without the
code. See ``CLAUDE.md`` -> Metric definitions.

**Primary, and the x-axis of the SNR curve:**

.. math::

    \\mathrm{SNR_{aperture}}
      = \\frac{\\sum_{\\text{aperture}} (I - \\mu_{bg})}{\\sigma_{bg}\\sqrt{N_{aperture}}}

with the aperture a disc of radius ``1.5 * FWHM`` centred on the estimated centroid. Chosen as
primary because it is the SNR that the accuracy law ``sigma_x ~ FWHM / (2 * SNR)``
(``docs/DESIGN.md`` section 5.3) is written in terms of; using peak SNR there would make the
quoted law wrong by a spot-shape-dependent factor.

**Secondary, logged alongside:** ``SNR_peak = (I_peak - mu_bg) / sigma_bg``. Detection
thresholding keys on the peak, so this is the natural companion to the lock criterion.

**Background statistics are robust, never mean/std.** ``mu_bg`` is the median and ``sigma_bg`` is
``1.4826 * MAD``, measured over an annulus from ``2.5 * FWHM`` to ``4 * FWHM``. At 10%
salt-and-pepper a naive standard deviation roughly doubles, which would silently *halve* every
reported SNR and make the curve look considerably better than reality.

**Carry ``from_fallback``.** The aperture radius depends on the spot-scale estimate, so an SNR
computed under fallback geometry is not the same measurement as one computed under measured
geometry. A sweep that mixes them is two measurements plotted on one axis, so
:class:`SnrRecord` records the provenance and the sweep must be splittable by it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "robust_sigma",
    "QUANTISATION_SIGMA_LSB",
    "core_saturated_fraction",
    "SnrRecord",
    "SnrApertureParams",
    "measure_snr",
    "annulus_background",
    "SNR_APERTURE_FORMULA",
    "SNR_PEAK_FORMULA",
]

#: The primary definition, written out for the plot axis so the curve is reproducible from the
#: figure alone.
SNR_APERTURE_FORMULA: str = (
    r"$\mathrm{SNR_{ap}} = \sum_{r<1.5\,\mathrm{FWHM}}(I-\mu_{bg})\;/\;"
    r"(\sigma_{bg}\sqrt{N_{ap}})$"
)

#: The secondary definition, for the same reason.
SNR_PEAK_FORMULA: str = r"$\mathrm{SNR_{peak}} = (I_{peak}-\mu_{bg})/\sigma_{bg}$"


@dataclass(frozen=True)
class SnrApertureParams:
    """Aperture and annulus geometry, in units of the spot FWHM.

    Expressed as FWHM multiples rather than pixels so the measurement means the same thing at any
    resolution or spot size.

    Attributes:
        aperture_fwhm_multiple: Aperture radius. 1.5 x FWHM captures essentially all of a
            Gaussian's flux while excluding background that would only add noise.
        annulus_inner_fwhm_multiple: Inner radius of the background annulus. Far enough out that
            spot flux does not contaminate the background estimate.
        annulus_outer_fwhm_multiple: Outer radius of the background annulus.
        min_annulus_pixels: Minimum annulus sample count for the estimate to be trusted.
    """

    aperture_fwhm_multiple: float = 1.5
    annulus_inner_fwhm_multiple: float = 2.5
    annulus_outer_fwhm_multiple: float = 4.0
    min_annulus_pixels: int = 16


@dataclass(frozen=True)
class SnrRecord:
    """One SNR measurement with the provenance needed to interpret it.

    Attributes:
        snr_aperture: Primary SNR. ``None`` when it could not be measured.
        snr_peak: Secondary SNR.
        mu_bg: Robust background level (median over the annulus).
        sigma_bg: Robust background noise (1.4826 x MAD over the annulus).
        aperture_radius_px: Aperture radius actually used.
        aperture_pixels: Number of pixels in the aperture.
        annulus_pixels: Number of pixels in the background annulus.
        fwhm_px: Spot scale the geometry was derived from.
        from_fallback: **Whether that scale came from configured fallback rather than
            measurement.** The aperture radius depends on the scale, so an SNR measured under
            fallback geometry is a different measurement from one under measured geometry; a
            sweep mixing the two is two curves on one axis.
    """

    snr_aperture: Optional[float] = None
    snr_peak: Optional[float] = None
    mu_bg: float = 0.0
    sigma_bg: float = 0.0
    aperture_radius_px: float = 0.0
    aperture_pixels: int = 0
    annulus_pixels: int = 0
    fwhm_px: float = 0.0
    from_fallback: bool = True

    @property
    def valid(self) -> bool:
        """Whether a usable SNR was obtained."""
        return self.snr_aperture is not None


#: Standard deviation of uniform quantisation noise, in least-significant bits: ``1/sqrt(12)``.
#:
#: This is the floor for any robust noise estimate on quantised data. A median absolute deviation
#: of zero does **not** mean there is no noise -- it means the noise is smaller than one grey
#: level, which is a bound, not an absence. Treating it as zero has produced two separate
#: failures in this project, so the floor is applied inside :func:`robust_sigma` and every caller
#: goes through it.
QUANTISATION_SIGMA_LSB: float = 1.0 / math.sqrt(12.0)


def robust_sigma(values: np.ndarray, floor: float = QUANTISATION_SIGMA_LSB) -> float:
    """Estimate noise as ``1.4826 * MAD``, floored at the quantisation limit.

    **Use this rather than computing MAD inline.** The degenerate case -- more than half the
    samples identical, so MAD is exactly zero -- is not exotic: it happens whenever a region is
    flat, which heavy compression produces routinely. It has now bitten twice:

    * ``adaptive_threshold`` collapsed to the median on noiseless frames, so the mask swallowed
      the whole frame and every detection failed.
    * ``annulus_background`` reported ``sigma_bg = 0`` on a compressed 2000x2000 background, so
      SNR became unmeasurable, the adaptive measurement noise jumped to its ceiling, the Kalman
      gain collapsed, and the filter ran open-loop while still accepting every measurement.

    Both were guarded individually and the guard was not propagated; an audit then found four
    unguarded call sites against two guarded. Centralising it is the point.

    Args:
        values: Sample values. Any shape; flattened internally.
        floor: Minimum returned sigma. Defaults to the quantisation noise of 8-bit data.

    Returns:
        The robust standard deviation estimate, never below ``floor``.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return floor
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    return max(1.4826 * mad, floor)


def _radius_grid(shape: Tuple[int, int], center_xy: Tuple[float, float]) -> np.ndarray:
    """Return a grid of distances from a sub-pixel centre.

    Args:
        shape: ``(height, width)``.
        center_xy: Centre as ``(x, y)`` under the pixel-centre convention.

    Returns:
        A ``float32`` array of Euclidean distances in pixels.
    """
    height, width = shape
    ys = np.arange(height, dtype=np.float32)[:, None] - float(center_xy[1])
    xs = np.arange(width, dtype=np.float32)[None, :] - float(center_xy[0])
    return np.sqrt(xs * xs + ys * ys)


def annulus_background(frame: np.ndarray, center_xy: Tuple[float, float],
                       inner_px: float, outer_px: float) -> Tuple[float, float, int]:
    """Estimate background level and noise robustly over an annulus.

    Median and MAD rather than mean and standard deviation. This is not a refinement: at 10%
    salt-and-pepper the naive sigma roughly doubles, which halves every SNR computed from it.

    Args:
        frame: Input frame, 2-D. Photometry should be done on the *denoised* frame, not the
            top-hat residual -- the top-hat removes exactly the pedestal these statistics are
            defined against.
        center_xy: Spot centre as ``(x, y)``.
        inner_px: Inner annulus radius.
        outer_px: Outer annulus radius.

    Returns:
        ``(mu_bg, sigma_bg, n_pixels)``. ``sigma_bg`` is ``1.4826 * MAD``.

    Raises:
        ValueError: If the radii are not ordered and positive.
    """
    if not 0 < inner_px < outer_px:
        raise ValueError(f"Annulus requires 0 < inner < outer, got {inner_px}, {outer_px}")
    radius = _radius_grid(frame.shape, center_xy)
    mask = (radius >= inner_px) & (radius <= outer_px)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0, 0.0, 0
    values = frame.astype(np.float64, copy=False)[mask]
    return float(np.median(values)), robust_sigma(values), count


def measure_snr(frame: np.ndarray, center_xy: Tuple[float, float], fwhm_px: float,
                from_fallback: bool = True,
                params: Optional[SnrApertureParams] = None) -> SnrRecord:
    """Measure aperture and peak SNR at a given position and scale.

    Args:
        frame: Input frame, 2-D. Should be the denoised (median-filtered) frame, not the top-hat
            residual, so that background statistics retain the pedestal they are defined against.
        center_xy: Spot centre as ``(x, y)`` in frame coordinates.
        fwhm_px: Spot scale, setting the aperture and annulus radii.
        from_fallback: Whether ``fwhm_px`` came from configured fallback rather than measurement.
            Carried into the record so a sweep can be split by it.
        params: Aperture geometry. Defaults are used when omitted.

    Returns:
        An :class:`SnrRecord`. ``snr_aperture`` is ``None`` when the annulus holds too few
        samples or the background noise estimate is degenerate.

    Raises:
        ValueError: If the frame is not 2-D or ``fwhm_px`` is not positive.
    """
    if frame.ndim != 2:
        raise ValueError(f"SNR requires a 2-D frame, got shape {frame.shape}")
    if not (fwhm_px > 0 and math.isfinite(fwhm_px)):
        raise ValueError(f"FWHM must be a positive finite number, got {fwhm_px}")
    params = params or SnrApertureParams()

    aperture_radius = params.aperture_fwhm_multiple * fwhm_px
    inner = params.annulus_inner_fwhm_multiple * fwhm_px
    outer = params.annulus_outer_fwhm_multiple * fwhm_px

    mu_bg, sigma_bg, n_annulus = annulus_background(frame, center_xy, inner, outer)
    base = SnrRecord(mu_bg=mu_bg, sigma_bg=sigma_bg, aperture_radius_px=aperture_radius,
                     annulus_pixels=n_annulus, fwhm_px=fwhm_px, from_fallback=from_fallback)

    if n_annulus < params.min_annulus_pixels or sigma_bg <= 0:
        return base

    radius = _radius_grid(frame.shape, center_xy)
    aperture = radius <= aperture_radius
    n_aperture = int(np.count_nonzero(aperture))
    if n_aperture == 0:
        return base

    values = frame.astype(np.float64, copy=False)
    signal = float((values[aperture] - mu_bg).sum())
    snr_aperture = signal / (sigma_bg * math.sqrt(n_aperture))
    snr_peak = (float(values[aperture].max()) - mu_bg) / sigma_bg

    return SnrRecord(snr_aperture=snr_aperture, snr_peak=snr_peak, mu_bg=mu_bg,
                     sigma_bg=sigma_bg, aperture_radius_px=aperture_radius,
                     aperture_pixels=n_aperture, annulus_pixels=n_annulus,
                     fwhm_px=fwhm_px, from_fallback=from_fallback)


def core_saturated_fraction(frame: np.ndarray, ceiling: float = 255.0) -> float:
    """Fraction of a spot's core pixels sitting at the quantisation ceiling.

    **This is the canonical definition, inherited from the Phase 2 saturation study.** Both
    :mod:`src.vision.detect` and :mod:`src.vision.spotscale` use it, so a threshold tuned against
    one means the same thing in the other.

    "Core" is pixels at or above half the *quantised* maximum. That detail matters: once a spot
    saturates, its maximum is pinned at the ceiling, so the half-maximum contour sits at a fixed
    level and the core keeps growing outward into unsaturated wing pixels as the beacon
    brightens. The fraction therefore **plateaus around 0.83 rather than reaching 1.0**, even for
    a grossly over-exposed spot. Do not "fix" that by redefining the core -- downstream
    thresholds are calibrated to this behaviour.

    Args:
        frame: A patch containing the spot, 2-D.
        ceiling: Quantisation ceiling, 255 for 8-bit.

    Returns:
        Fraction in ``[0, 1]``, or 0.0 for an empty patch.
    """
    if frame.size == 0:
        return 0.0
    values = frame.astype(np.float32, copy=False)
    core = values >= 0.5 * float(values.max())
    core_count = int(np.count_nonzero(core))
    if core_count == 0:
        return 0.0
    return float(np.count_nonzero(values[core] >= ceiling) / core_count)
