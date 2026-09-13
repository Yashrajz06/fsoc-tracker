"""Preprocessing: impulse rejection, background suppression, normalisation.

Three stages, in this order, and the order is not arbitrary (``docs/DESIGN.md`` section 5.1):

1. **Median filter** kills salt-and-pepper *before* anything else sees it. Impulses are the
   primary false-positive threat to centroiding: a single salt pixel is a maximum-intensity point
   that survives thresholding and, if it reaches the centroid stage, drags the estimate an
   arbitrary distance. The kernel is 3 because an impulse is one pixel wide -- it is set by the
   noise, not by the spot, which is why it is the one size here that is *not* scale-relative.

2. **White top-hat** suppresses smooth background. **This is background suppression only.** It is
   emphatically not a scale selector: top-hat response is monotonically non-decreasing in kernel
   size once the structuring element exceeds the spot, so it has no interior maximum over scale
   (measured: argmax at the largest kernel with a peak-to-runner-up ratio of exactly 1.000). Scale
   selection is the scale-normalised LoG in :mod:`src.vision.spotscale`.

3. **Per-frame normalisation** rescales intensity. Essential for Mode B, where we cannot assume
   anything about the brightness range of evaluator video.

All geometry comes from :meth:`~src.config.VisionConfig.resolve_geometry`, never from literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import PreprocessConfig, ResolvedVisionGeometry

__all__ = ["PreprocessResult", "preprocess", "median_filter", "top_hat", "normalise"]


@dataclass(frozen=True)
class PreprocessResult:
    """A preprocessed frame plus the intermediates needed to interpret it.

    Attributes:
        residual: The top-hat residual, ``float32``. This is what detection thresholds against.
        denoised: The frame after median filtering but before background suppression. Photometry
            (SNR, saturation) is measured on this rather than on the residual, because the
            top-hat removes the pedestal that background statistics are defined against.
        tophat_kernel_px: Structuring element size actually used, for the trace.
    """

    residual: np.ndarray
    denoised: np.ndarray
    tophat_kernel_px: int = 0


def median_filter(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a median filter to remove impulse noise.

    Args:
        frame: Input frame, 2-D.
        kernel_size: Odd kernel size. 3 is correct for single-pixel impulses.

    Returns:
        A new filtered ``float32`` frame.

    Raises:
        ValueError: If the kernel size is even or non-positive.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Median kernel must be a positive odd integer, got {kernel_size}")
    working = frame.astype(np.float32, copy=False)
    if kernel_size == 1:
        return working.copy()
    return cv2.medianBlur(working, kernel_size)


def top_hat(frame: np.ndarray, kernel_px: int) -> np.ndarray:
    """Apply a white top-hat transform to suppress smooth background.

    The white top-hat is ``I - opening(I)``. An opening with a structuring element larger than the
    spot removes the spot, so the residual retains it while discarding anything broader --
    gradients, airlight, vignetting.

    Args:
        frame: Input frame, 2-D.
        kernel_px: Structuring element side length. Must exceed the spot, which
            :meth:`~src.config.VisionConfig.resolve_geometry` guarantees.

    Returns:
        A new ``float32`` residual.

    Raises:
        ValueError: If the kernel size is even or non-positive.
    """
    if kernel_px < 1 or kernel_px % 2 == 0:
        raise ValueError(f"Top-hat kernel must be a positive odd integer, got {kernel_px}")
    working = frame.astype(np.float32, copy=False)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.morphologyEx(working, cv2.MORPH_TOPHAT, element)


def normalise(frame: np.ndarray) -> np.ndarray:
    """Rescale a frame to ``[0, 1]`` using its own range.

    Never assumes a brightness range, which is what lets the identical pipeline run on evaluator
    video of unknown exposure.

    Args:
        frame: Input frame, 2-D.

    Returns:
        A new ``float32`` frame in ``[0, 1]``. A constant frame maps to all zeros.
    """
    working = frame.astype(np.float32, copy=False)
    low = float(working.min())
    high = float(working.max())
    if high <= low:
        return np.zeros_like(working)
    return (working - low) / (high - low)


def preprocess(frame: np.ndarray, config: PreprocessConfig,
               geometry: ResolvedVisionGeometry) -> PreprocessResult:
    """Run the full preprocessing chain.

    Args:
        frame: Raw input frame, 2-D.
        config: Preprocessing configuration.
        geometry: Resolved vision geometry, supplying the top-hat kernel size. Geometry is always
            resolved from the runtime spot scale, never from literals.

    Returns:
        A :class:`PreprocessResult`.

    Raises:
        ValueError: If the frame is not 2-D.
    """
    if frame.ndim != 2:
        raise ValueError(f"Preprocessing requires a 2-D frame, got shape {frame.shape}")

    denoised = (median_filter(frame, config.median_kernel_size)
                if config.median_filter_enabled else frame.astype(np.float32, copy=True))

    kernel = 0
    if config.tophat_enabled:
        kernel = geometry.tophat_kernel_px
        residual = top_hat(denoised, kernel)
    else:
        residual = denoised.copy()

    if config.normalise_per_frame:
        residual = normalise(residual)

    return PreprocessResult(residual=residual, denoised=denoised, tophat_kernel_px=kernel)
