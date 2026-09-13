"""The virtual world canvas: a large scene buffer allocated once and reused.

The canvas is the 2000x2000 (or larger) world the camera looks at. It is allocated a single time
and refilled in place from a prebuilt background template; nothing here reallocates per frame.
At 30 Hz a 2000x2000 ``uint8`` canvas reallocated every frame would churn 120 MB/s through the
allocator for no benefit, and the resulting garbage-collection pauses show up directly as frame
rate jitter -- which we would then have to explain in the FPS log. See ``CLAUDE.md`` -> Working
conventions.

Coordinate convention: pixel centres at integer indices, origin top-left, tuples ordered
``(x, y)``. NumPy indexing is ``[y, x]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.config import AppConfig, SceneConfig
from src.sim.beacon import BeaconPatch

__all__ = ["Canvas"]


class Canvas:
    """A reusable world-scene buffer.

    Typical per-frame use::

        canvas.clear()
        canvas.composite(patch)
        view = canvas.data

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
    """

    def __init__(self, width: int, height: int, background_level: int = 10,
                 background_gradient: bool = False) -> None:
        """Allocate the canvas and its background template.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            background_level: Uniform background grey level in ``[0, 255]``. When a gradient is
                enabled this is the *mean* level.
            background_gradient: Apply a gentle diagonal brightness gradient. Useful because a
                perfectly uniform background is unrealistically kind to the detector -- it is
                exactly the case top-hat background suppression is meant to handle, so leaving it
                off everywhere would let a regression in that stage go unnoticed.

        Raises:
            ValueError: If the dimensions are not positive or the background level is outside
                the 8-bit range.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Canvas dimensions must be positive, got {width}x{height}")
        if not 0 <= background_level <= 255:
            raise ValueError(f"Background level must be in [0, 255], got {background_level}")

        self.width = int(width)
        self.height = int(height)
        self._background_level = int(background_level)
        self._background_gradient = bool(background_gradient)

        self._background = self._build_background()
        # The working buffer. Allocated once here and never reallocated.
        self._data = self._background.copy()

    @classmethod
    def from_config(cls, config: AppConfig) -> "Canvas":
        """Build a canvas from an application configuration.

        Args:
            config: Validated application configuration.

        Returns:
            A canvas matching ``config.scene``.
        """
        scene: SceneConfig = config.scene
        return cls(scene.width, scene.height, scene.background_level, scene.background_gradient)

    def _build_background(self) -> np.ndarray:
        """Construct the background template that :meth:`clear` restores.

        Returns:
            A ``uint8`` array of shape ``(height, width)``.
        """
        if not self._background_gradient:
            return np.full((self.height, self.width), self._background_level, dtype=np.uint8)

        # A gentle diagonal ramp of +/-50% around the mean level, clipped into range.
        amplitude = self._background_level * 0.5
        ramp_x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        ramp_y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        field = self._background_level + amplitude * 0.5 * (ramp_x[None, :] + ramp_y[:, None])
        return np.clip(field, 0, 255).astype(np.uint8)

    @property
    def data(self) -> np.ndarray:
        """The live canvas buffer as a ``uint8`` array of shape ``(height, width)``.

        This is the working buffer itself, not a copy: callers must not retain it across a
        :meth:`clear`, and must not mutate it unless that is the intent.
        """
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Canvas size as ``(height, width)``."""
        return self.height, self.width

    def clear(self) -> None:
        """Restore the background in place, without allocating.

        ``np.copyto`` writes into the existing buffer, which is the entire reason the background
        template is retained separately.
        """
        np.copyto(self._data, self._background)

    def contains(self, x: float, y: float) -> bool:
        """Report whether a canvas coordinate lies within the canvas bounds.

        Uses the pixel-centre convention: a canvas of width ``W`` spans x from ``-0.5`` to
        ``W - 0.5``.

        Args:
            x: Canvas x coordinate.
            y: Canvas y coordinate.

        Returns:
            True if the point lies inside the canvas.
        """
        return -0.5 <= x <= self.width - 0.5 and -0.5 <= y <= self.height - 0.5

    def composite(self, patch: BeaconPatch, mode: str = "add") -> None:
        """Draw a rendered beacon patch onto the canvas, clipped at the canvas edges.

        Args:
            patch: The rendered beacon patch to draw.
            mode: ``"add"`` (default) accumulates intensity onto the background, which is what
                photons physically do, and saturates at 255. ``"max"`` takes the per-pixel
                maximum, which avoids saturation but is not physical. Note that saturation
                flattens the peak of a bright beacon; because the clipping is symmetric about
                the true centre it does not bias the centroid, but it does reduce the sub-pixel
                information available to the estimator.

        Raises:
            ValueError: On an unknown mode.
        """
        if mode not in ("add", "max"):
            raise ValueError(f"Composite mode must be 'add' or 'max', got {mode!r}")

        ph, pw = patch.shape
        # Intersect the patch rectangle with the canvas rectangle.
        cx0, cy0 = max(0, patch.x0), max(0, patch.y0)
        cx1, cy1 = min(self.width, patch.x0 + pw), min(self.height, patch.y0 + ph)
        if cx0 >= cx1 or cy0 >= cy1:
            return  # Entirely off-canvas.

        px0, py0 = cx0 - patch.x0, cy0 - patch.y0
        source = patch.data[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]
        target = self._data[cy0:cy1, cx0:cx1]

        if mode == "add":
            combined = target.astype(np.float32) + source
        else:
            combined = np.maximum(target.astype(np.float32), source)
        np.copyto(target, np.clip(combined, 0, 255).astype(np.uint8))

    def extract(self, x0: int, y0: int, width: int, height: int,
                fill: Optional[int] = None) -> np.ndarray:
        """Extract a sub-window, padding with the background level where it runs off-canvas.

        Args:
            x0: Left edge of the window, in canvas coordinates.
            y0: Top edge of the window, in canvas coordinates.
            width: Window width in pixels.
            height: Window height in pixels.
            fill: Grey level for out-of-bounds regions. Defaults to the background level, which
                keeps an off-edge viewport statistically similar to an on-canvas one rather than
                introducing a hard black band that adaptive thresholding would key on.

        Returns:
            A new ``uint8`` array of shape ``(height, width)``.

        Raises:
            ValueError: If the requested size is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Extraction size must be positive, got {width}x{height}")
        level = self._background_level if fill is None else int(fill)
        out = np.full((height, width), level, dtype=np.uint8)

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.width, x0 + width), min(self.height, y0 + height)
        if sx0 < sx1 and sy0 < sy1:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self._data[sy0:sy1, sx0:sx1]
        return out
