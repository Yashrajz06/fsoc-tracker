"""The full vision pipeline: one frame in, one measurement out.

Chains preprocessing, detection, centroiding and photometry, with all geometry resolved from the
runtime spot scale. This is the code that must run **byte-for-byte identically** in Mode A and
Mode B -- the structural guarantee that protects Benchmark Performance-2 (``docs/DESIGN.md``
section 8). There is deliberately no mode parameter anywhere in this module.

ROI-limited operation is supported through :func:`process`'s ``roi`` argument: once locked, the
caller passes a window around the Kalman prediction and the pipeline works only inside it,
returning coordinates in full-frame terms regardless. Full-frame processing is for acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config import AppConfig, ResolvedVisionGeometry, VisionConfig
from src.vision.centroid import CentroidResult, centroid
from src.vision.detect import Detection, DetectionResult, detect
from src.vision.preprocess import PreprocessResult, preprocess
from src.vision.snr import SnrRecord, measure_snr

__all__ = ["Measurement", "VisionPipeline"]


@dataclass(frozen=True)
class Measurement:
    """One frame's vision output, with the full trace needed to explain it.

    Attributes:
        found: Whether a target was detected.
        x: Sub-pixel centroid x in **full-frame** coordinates.
        y: Sub-pixel centroid y in full-frame coordinates.
        snr: Photometry record, carrying ``from_fallback`` provenance.
        detection: The gated blob this measurement came from.
        clipped: Blob touches the frame edge; centroid biased inward by up to ~2 px.
        saturated: Blob core is saturated; centroid biased by ~0.19 px.
        from_fallback: Geometry came from configured fallback rather than a measured scale.
        threshold: Detection threshold used, for the trace.
        n_candidates: Blobs found before gating.
        detections: All surviving candidates, strongest first by flux. Exposed so a discriminator
            can re-rank them; the classical ordering is preserved here regardless.
        discriminator_scores: Per-candidate scores when the discriminator ran, else ``None``.
        discriminator_used: Whether the discriminator changed which candidate was selected.
        geometry: The resolved geometry this frame was processed with.

    Note:
        :attr:`clipped` and :attr:`saturated` mark a *real* target whose centroid carries a
        known, bounded bias -- Phase 4 inflates the Kalman ``R`` rather than rejecting it. A blob
        that should be rejected never reaches this class at all; it fails a gate in
        :mod:`src.vision.detect`. A bounded bias is still information; an outlier is not.
    """

    found: bool = False
    x: float = 0.0
    y: float = 0.0
    snr: Optional[SnrRecord] = None
    detection: Optional[Detection] = None
    centroid_result: Optional[CentroidResult] = None
    clipped: bool = False
    saturated: bool = False
    from_fallback: bool = True
    threshold: float = 0.0
    n_candidates: int = 0
    detections: Tuple[Detection, ...] = ()
    discriminator_scores: Optional[Tuple[float, ...]] = None
    discriminator_used: bool = False
    geometry: Optional[ResolvedVisionGeometry] = None

    @property
    def is_reliable(self) -> bool:
        """Whether the measurement is free of known systematic bias."""
        return self.found and not (self.clipped or self.saturated)

    @property
    def position(self) -> Tuple[float, float]:
        """Estimated position as ``(x, y)``."""
        return self.x, self.y


class VisionPipeline:
    """Stateless per-frame vision processing.

    Attributes:
        config: Vision configuration.
    """

    def __init__(self, config: VisionConfig, discriminator: Optional[object] = None) -> None:
        """Initialise the pipeline.

        Args:
            config: Vision configuration supplying stage parameters.
            discriminator: Optional candidate discriminator exposing ``should_run`` and ``rank``
                (see :class:`src.ai.validator.CandidateDiscriminator`). Typed loosely on purpose:
                the vision package must not import the AI package, so that a missing
                ``onnxruntime`` can never break the classical path. ``None`` disables it.
        """
        self.config = config
        self.discriminator = discriminator

    @classmethod
    def from_config(cls, config: AppConfig) -> "VisionPipeline":
        """Build a pipeline from an application configuration.

        Args:
            config: Validated application configuration.

        Returns:
            A configured :class:`VisionPipeline`. The discriminator is attached only when the
            ``ai`` block enables it *and* the model loads; any failure leaves the classical path
            running untouched and is reported by :func:`src.ai.validator.load_discriminator`.
        """
        discriminator = None
        if config.ai.enabled:
            from src.ai.validator import load_discriminator

            discriminator = load_discriminator(config.ai)
        return cls(config.vision, discriminator=discriminator)

    def process(self, frame: np.ndarray, fwhm_px: Optional[float] = None,
                from_fallback: Optional[bool] = None,
                roi: Optional[Tuple[int, int, int, int]] = None) -> Measurement:
        """Process one frame.

        Args:
            frame: Input frame, 2-D, single channel.
            fwhm_px: Measured spot scale, or ``None`` to use configured fallback geometry.
            from_fallback: Override the provenance flag. Defaults to ``fwhm_px is None``.
            roi: Optional ``(x0, y0, width, height)`` window to restrict processing to. Results
                are returned in full-frame coordinates regardless, so a caller cannot accidentally
                mix coordinate systems.

        Returns:
            A :class:`Measurement`.

        Raises:
            ValueError: If the frame is not 2-D or the ROI is degenerate.
        """
        if frame.ndim != 2:
            raise ValueError(f"Pipeline requires a 2-D frame, got shape {frame.shape}")

        offset_x, offset_y = 0, 0
        working = frame
        if roi is not None:
            rx, ry, rw, rh = roi
            if rw <= 0 or rh <= 0:
                raise ValueError(f"ROI must have positive size, got {rw}x{rh}")
            x0, y0 = max(0, rx), max(0, ry)
            x1, y1 = min(frame.shape[1], rx + rw), min(frame.shape[0], ry + rh)
            if x0 >= x1 or y0 >= y1:
                raise ValueError("ROI lies entirely outside the frame")
            working = frame[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0

        fallback = (fwhm_px is None) if from_fallback is None else bool(from_fallback)
        geometry = self.config.resolve_geometry(fwhm_px)

        pre: PreprocessResult = preprocess(working, self.config.preprocess, geometry)
        found: DetectionResult = detect(pre.residual, pre.denoised, self.config.detection,
                                        geometry)
        if not found.found:
            return Measurement(threshold=found.threshold, n_candidates=found.n_candidates,
                               from_fallback=fallback, geometry=geometry,
                               detections=found.detections)

        best = found.best
        assert best is not None

        # Optional CNN discrimination. The classical path has already chosen `best` by flux; the
        # discriminator only re-ranks the candidates it produced, and never localises -- the
        # centroid below is intensity-weighted centre of gravity either way.
        scores: Optional[Tuple[float, ...]] = None
        reranked = False
        if self.discriminator is not None and self.discriminator.should_run(found.detections):
            scores, chosen = self.discriminator.rank(working, found.detections)
            if chosen is not None and chosen is not best:
                best, reranked = chosen, True
        estimate = centroid(pre.denoised, (best.x, best.y), self.config.centroid, geometry)
        snr = measure_snr(pre.denoised, (estimate.x, estimate.y), geometry.fwhm_px,
                          from_fallback=fallback)

        return Measurement(
            found=True,
            x=estimate.x + offset_x,
            y=estimate.y + offset_y,
            snr=snr,
            detection=best,
            centroid_result=estimate,
            clipped=best.clipped,
            saturated=best.saturated,
            from_fallback=fallback,
            threshold=found.threshold,
            n_candidates=found.n_candidates,
            detections=found.detections,
            discriminator_scores=scores,
            discriminator_used=reranked,
            geometry=geometry,
        )
