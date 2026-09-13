"""Sub-pixel centroid estimation.

Classical weighted centroiding is near-optimal for a bright, symmetric blob on a near-uniform
background: localisation error scales as spot-width over sqrt(photons) in the shot-noise limit,
and roughly ``FWHM / (2 * SNR)`` in general (``docs/DESIGN.md`` section 5.3). This is the fast
path that runs on every frame.

Estimators, in increasing order of cost:

* **Centre of gravity (CoG)** -- ``sum(x*I)/sum(I)``. Included for comparison only. Unthresholded
  CoG is badly biased by background and by impulse noise, so it is never the default.
* **Thresholded CoG** -- subtract ``mu + k*sigma`` before summing. Essential, not optional: the
  background pedestal pulls an unthresholded estimate toward the window centre in proportion to
  how much background the window contains.
* **IWCoG** (default) -- iteratively re-centre the window on the current estimate and re-weight.
  Lower variance than plain CoG at low SNR, because each iteration discards background that the
  previous window included off-centre.
* **Gaussian fit** -- optional refinement, unbiased to lower SNR at higher cost.

Background statistics are robust (median, MAD) throughout, for the same reason as everywhere
else: at 10% salt-and-pepper a naive sigma roughly doubles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config import CentroidConfig, ResolvedVisionGeometry
from src.vision.snr import robust_sigma

__all__ = ["CentroidResult", "centroid", "center_of_gravity", "thresholded_cog", "iwcog"]


@dataclass(frozen=True)
class CentroidResult:
    """A sub-pixel centroid estimate.

    Attributes:
        x: Estimated centroid x in frame coordinates.
        y: Estimated centroid y in frame coordinates.
        converged: Whether the iterative estimator settled before its iteration cap.
        iterations: Iterations actually performed.
        window_px: Window size used.
        flux: Total background-subtracted intensity inside the final window.
        method: Estimator used.
    """

    x: float
    y: float
    converged: bool = True
    iterations: int = 1
    window_px: int = 0
    flux: float = 0.0
    method: str = "iwcog"


def _window(frame: np.ndarray, center_xy: Tuple[float, float],
            size_px: int) -> Tuple[np.ndarray, int, int]:
    """Extract a square window clipped to the frame.

    Args:
        frame: Source frame, 2-D.
        center_xy: Window centre as ``(x, y)``.
        size_px: Window side length; forced odd so the window has a defined centre pixel.

    Returns:
        ``(patch, x0, y0)`` where ``(x0, y0)`` is the patch origin in frame coordinates.
    """
    half = max(1, size_px // 2)
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(frame.shape[1], cx + half + 1)
    y1 = min(frame.shape[0], cy + half + 1)
    return frame[y0:y1, x0:x1], x0, y0


def _robust_stats(patch: np.ndarray) -> Tuple[float, float]:
    """Return ``(median, 1.4826 * MAD)`` for a patch."""
    values = patch.astype(np.float64, copy=False)
    return float(np.median(values)), robust_sigma(values)


def center_of_gravity(patch: np.ndarray, x0: float = 0.0,
                      y0: float = 0.0) -> Tuple[float, float, float]:
    """Compute the intensity-weighted centre of gravity of a patch.

    Args:
        patch: 2-D non-negative intensities.
        x0: Patch origin x in frame coordinates.
        y0: Patch origin y in frame coordinates.

    Returns:
        ``(x, y, flux)``. Returns the window centre with zero flux if the patch is empty.
    """
    weights = patch.astype(np.float64, copy=False)
    total = float(weights.sum())
    if total <= 0:
        return (x0 + (patch.shape[1] - 1) / 2.0, y0 + (patch.shape[0] - 1) / 2.0, 0.0)
    cols = np.arange(patch.shape[1], dtype=np.float64)
    rows = np.arange(patch.shape[0], dtype=np.float64)
    cx = float((weights.sum(axis=0) * cols).sum() / total)
    cy = float((weights.sum(axis=1) * rows).sum() / total)
    return cx + x0, cy + y0, total


def thresholded_cog(frame: np.ndarray, center_xy: Tuple[float, float], window_px: int,
                    k_sigma: float = 3.0) -> CentroidResult:
    """Centre of gravity after subtracting a robust background pedestal.

    Args:
        frame: Source frame, 2-D.
        center_xy: Initial estimate as ``(x, y)``.
        window_px: Window side length.
        k_sigma: Pedestal level as ``median + k * sigma``.

    Returns:
        A :class:`CentroidResult`.
    """
    patch, x0, y0 = _window(frame, center_xy, window_px)
    median, sigma = _robust_stats(patch)
    weights = np.maximum(patch.astype(np.float64) - (median + k_sigma * sigma), 0.0)
    cx, cy, flux = center_of_gravity(weights, x0, y0)
    return CentroidResult(x=cx, y=cy, iterations=1, window_px=window_px, flux=flux,
                          method="thresholded_cog")


def iwcog(frame: np.ndarray, center_xy: Tuple[float, float], window_px: int,
          iterations: int = 3, k_sigma: float = 3.0,
          tolerance_px: float = 0.01) -> CentroidResult:
    """Iteratively-weighted centre of gravity.

    Each iteration re-centres the window on the current estimate and recomputes the thresholded
    centroid. Re-centring is what reduces variance: a window offset from the true centre includes
    background on one side and clips signal on the other, which is exactly the asymmetry that
    biases a single-pass estimate.

    Args:
        frame: Source frame, 2-D.
        center_xy: Initial estimate as ``(x, y)``, typically the blob centroid from detection.
        window_px: Window side length.
        iterations: Maximum iterations.
        k_sigma: Pedestal level as ``median + k * sigma``.
        tolerance_px: Convergence threshold on successive estimates.

    Returns:
        A :class:`CentroidResult` with ``converged`` set when successive estimates agreed within
        ``tolerance_px``.

    Raises:
        ValueError: If ``iterations`` is below 1.
    """
    if iterations < 1:
        raise ValueError(f"IWCoG needs at least one iteration, got {iterations}")

    current = (float(center_xy[0]), float(center_xy[1]))
    flux = 0.0
    converged = False
    performed = 0

    for _ in range(iterations):
        performed += 1
        step = thresholded_cog(frame, current, window_px, k_sigma)
        moved = math.hypot(step.x - current[0], step.y - current[1])
        current = (step.x, step.y)
        flux = step.flux
        if moved <= tolerance_px:
            converged = True
            break

    return CentroidResult(x=current[0], y=current[1], converged=converged,
                          iterations=performed, window_px=window_px, flux=flux, method="iwcog")


def centroid(frame: np.ndarray, center_xy: Tuple[float, float], config: CentroidConfig,
             geometry: ResolvedVisionGeometry) -> CentroidResult:
    """Estimate a sub-pixel centroid using the configured estimator.

    Args:
        frame: Source frame, 2-D. Normally the denoised frame or the top-hat residual.
        center_xy: Initial estimate as ``(x, y)`` from detection.
        config: Centroid configuration selecting the estimator.
        geometry: Resolved geometry supplying the window size. Never a literal.

    Returns:
        A :class:`CentroidResult`.

    Raises:
        ValueError: On an unknown estimator, or a frame that is not 2-D.
    """
    if frame.ndim != 2:
        raise ValueError(f"Centroiding requires a 2-D frame, got shape {frame.shape}")

    window = geometry.centroid_window_px
    method = config.method

    if method == "iwcog":
        return iwcog(frame, center_xy, window, config.iterations, config.background_k_sigma)
    if method == "thresholded_cog":
        return thresholded_cog(frame, center_xy, window, config.background_k_sigma)
    if method == "cog":
        patch, x0, y0 = _window(frame, center_xy, window)
        cx, cy, flux = center_of_gravity(patch, x0, y0)
        return CentroidResult(x=cx, y=cy, window_px=window, flux=flux, method="cog")
    if method == "gaussian_fit":
        # Refinement on top of IWCoG: parabolic interpolation of the log-intensity peak, which is
        # exact for a Gaussian profile.
        seed = iwcog(frame, center_xy, window, config.iterations, config.background_k_sigma)
        return _gaussian_refine(frame, seed, config.background_k_sigma)

    raise ValueError(f"Unknown centroid method: {method!r}")


def _gaussian_refine(frame: np.ndarray, seed: CentroidResult,
                     k_sigma: float) -> CentroidResult:
    """Refine a centroid by parabolic interpolation of log intensity.

    For a Gaussian profile the log of the intensity is exactly a parabola, so fitting three
    points about the peak recovers the sub-pixel offset in closed form.

    Args:
        frame: Source frame, 2-D.
        seed: Starting estimate.
        k_sigma: Pedestal level for background subtraction.

    Returns:
        A refined :class:`CentroidResult`, or the seed unchanged if refinement is ill-conditioned.
    """
    patch, x0, y0 = _window(frame, (seed.x, seed.y), seed.window_px)
    median, sigma = _robust_stats(patch)
    values = np.maximum(patch.astype(np.float64) - (median + k_sigma * sigma), 0.0)
    if values.max() <= 0:
        return seed

    row, col = np.unravel_index(int(np.argmax(values)), values.shape)
    if not (0 < row < values.shape[0] - 1 and 0 < col < values.shape[1] - 1):
        return seed

    def offset(a: float, b: float, c: float) -> float:
        """Parabolic vertex offset from three samples straddling the peak."""
        denominator = a - 2.0 * b + c
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denominator, -1.0, 1.0))

    logs = np.log(values + 1e-9)
    dx = offset(logs[row, col - 1], logs[row, col], logs[row, col + 1])
    dy = offset(logs[row - 1, col], logs[row, col], logs[row + 1, col])
    return CentroidResult(x=x0 + col + dx, y=y0 + row + dy, converged=seed.converged,
                          iterations=seed.iterations, window_px=seed.window_px,
                          flux=seed.flux, method="gaussian_fit")
