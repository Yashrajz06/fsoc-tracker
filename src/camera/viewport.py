"""Viewport extraction: cutting the camera's frame out of the world canvas.

This is the seam where the 2000x2000 world becomes a 640x480 detector frame. Two things must
survive the cut intact, and both are tested:

1. **The coordinate convention.** A target at canvas ``(X, Y)`` must appear at frame-local
   ``(X - x0, Y - y0)``. Any inconsistency here shifts ground truth relative to the image and
   would make the centroid error we report meaningless.
2. **Sub-pixel information.** The viewport origin is integer-aligned because a detector reads out
   whole pixels, but the *target* keeps its sub-pixel position within the frame.

Camera jitter is applied here rather than in the noise pipeline, because it is physically a
disturbance of where the camera is looking, not of the image the camera produces. Keeping it at
the extraction point means the rendered scene stays pristine and the jitter shows up exactly as
it would in hardware: as an offset of the readout window. Note the distinction from platform
motion -- jitter is high-frequency and zero-mean, platform motion is a low-frequency bias the
controller must actively reject (``docs/DESIGN.md`` section 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.camera.model import CameraModel
from src.noise.disturbance import JitterModel, JitterParams
from src.sim.canvas import Canvas

__all__ = ["Viewport", "ViewportFrame"]


@dataclass(frozen=True)
class ViewportFrame:
    """An extracted camera frame and the geometry needed to interpret it.

    Attributes:
        frame: 2-D ``uint8`` array of shape ``(height, width)``.
        x0: Canvas x coordinate of the frame's top-left pixel centre, jitter included.
        y0: Canvas y coordinate of the frame's top-left pixel centre, jitter included.
        jitter_x: Horizontal jitter applied to this extraction, in pixels.
        jitter_y: Vertical jitter applied to this extraction, in pixels.
    """

    frame: np.ndarray
    x0: int
    y0: int
    jitter_x: float = 0.0
    jitter_y: float = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        """Frame size as ``(height, width)``."""
        return self.frame.shape[0], self.frame.shape[1]

    @property
    def center(self) -> Tuple[float, float]:
        """Frame centre as ``(x, y)``, the boresight in frame-local coordinates.

        Uses the pixel-centre convention and must agree exactly with ``FrameData.center``.
        """
        height, width = self.shape
        return (width - 1) / 2.0, (height - 1) / 2.0

    def canvas_to_frame(self, x: float, y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to frame-local coordinates.

        Args:
            x: Canvas x coordinate.
            y: Canvas y coordinate.

        Returns:
            The position as ``(x, y)`` relative to this frame's top-left pixel centre.
        """
        return x - self.x0, y - self.y0

    def frame_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert frame-local coordinates to canvas coordinates.

        Args:
            x: Frame-local x coordinate.
            y: Frame-local y coordinate.

        Returns:
            The position as ``(x, y)`` in canvas coordinates.
        """
        return x + self.x0, y + self.y0

    def contains(self, x: float, y: float) -> bool:
        """Report whether a canvas position falls inside this frame.

        Args:
            x: Canvas x coordinate.
            y: Canvas y coordinate.

        Returns:
            True if the position lies within the frame bounds.
        """
        fx, fy = self.canvas_to_frame(x, y)
        height, width = self.shape
        return -0.5 <= fx <= width - 0.5 and -0.5 <= fy <= height - 0.5


class Viewport:
    """Extracts camera frames from a canvas, optionally with camera jitter.

    Attributes:
        camera: The camera model supplying boresight and resolution.
    """

    def __init__(self, camera: CameraModel,
                 jitter_px: float = 0.0,
                 rng: Optional[np.random.Generator] = None,
                 jitter_distribution: str = "gaussian") -> None:
        """Initialise the viewport.

        Args:
            camera: Camera model supplying the boresight and viewport size.
            jitter_px: Jitter magnitude in pixels per frame. For a Gaussian distribution this is
                treated as a 3-sigma bound, so the configured value is a near-maximum excursion
                rather than a typical one -- the specification phrases parameter 23 as a maximum.
            rng: Seeded generator. One is created if omitted.
            jitter_distribution: ``"gaussian"`` or ``"uniform"``.

        Raises:
            ValueError: On a negative jitter magnitude or an unknown distribution.
        """
        self.camera = camera
        self.jitter_px = float(jitter_px)
        self.jitter_distribution = jitter_distribution
        # Delegate to the shared model in src.noise.disturbance rather than reimplementing the
        # distribution here: jitter must behave identically whether it reaches the viewport from
        # this constructor or from the noise pipeline. Validation lives there too.
        self._jitter = JitterModel(
            JitterParams(enabled=jitter_px > 0, max_px_per_frame=jitter_px,
                         distribution=jitter_distribution),
            rng=rng if rng is not None else np.random.default_rng(),
        )

    def _draw_jitter(self) -> Tuple[float, float]:
        """Draw a jitter offset for one frame.

        Returns:
            The offset as ``(dx, dy)`` in pixels, bounded by the configured magnitude.
        """
        return self._jitter.sample()

    def extract(self, canvas: Canvas, apply_jitter: bool = True) -> ViewportFrame:
        """Extract the current camera frame from a canvas.

        Args:
            canvas: The world canvas to read from.
            apply_jitter: Apply camera jitter to the extraction position. Disable for
                deterministic tests and for rendering ground-truth reference frames.

        Returns:
            The extracted :class:`ViewportFrame`.
        """
        width, height = self.camera.config.fov_px
        jitter_x, jitter_y = self._draw_jitter() if apply_jitter else (0.0, 0.0)

        bx, by = self.camera.boresight
        x0 = int(round(bx + jitter_x - (width - 1) / 2.0))
        y0 = int(round(by + jitter_y - (height - 1) / 2.0))

        frame = canvas.extract(x0, y0, width, height)
        return ViewportFrame(frame=frame, x0=x0, y0=y0,
                             jitter_x=jitter_x, jitter_y=jitter_y)
