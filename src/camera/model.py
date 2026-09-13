"""Virtual pan/tilt camera: angular scale, boresight state and mechanical limits.

This module owns the conversion between pixels and degrees, and the enforcement of the gimbal's
slew and acceleration limits. Limits are enforced *here*, inside the model, rather than trusted
to the controller: a controller bug must show up as degraded tracking, never as a camera that
physically cannot exist. See ``docs/DESIGN.md`` section 7.

Key numbers for the default configuration (4 deg x 3 deg over 640x480 px):

* ``0.00625 deg/px`` on both axes;
* a 5 deg/s slew ceiling is ``800 px/s``, i.e. ``26.7 px/frame`` at 30 Hz.

That last figure is the binding feasibility constraint of the whole system -- a target whose
apparent motion exceeds it cannot be kept centred by any controller, however good the vision
pipeline is.

Coordinate convention: pixel centres at integer indices, origin top-left, tuples ordered
``(x, y)``. The boresight is the primary state and is held in **canvas pixel coordinates**;
pan/tilt angles are derived from it relative to the canvas centre.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from src.config import AppConfig, CameraConfig, SceneConfig

__all__ = ["CameraModel"]


class CameraModel:
    """A steerable virtual camera with enforced mechanical limits.

    Attributes:
        config: The camera configuration this model was built from.
        scene: The scene the camera is looking at, used to define the angular origin.
    """

    def __init__(self, config: CameraConfig, scene: SceneConfig,
                 initial_x: Optional[float] = None,
                 initial_y: Optional[float] = None) -> None:
        """Initialise the camera at a boresight position.

        Args:
            config: Camera configuration supplying FOV, resolution and limits.
            scene: Scene configuration, defining the canvas the camera points into.
            initial_x: Initial boresight x in canvas coordinates. Defaults to the configured
                initial position.
            initial_y: Initial boresight y in canvas coordinates.
        """
        self.config = config
        self.scene = scene
        self._home_x, self._home_y = self._resolve_home(initial_x, initial_y)
        self._x = self._home_x
        self._y = self._home_y
        self._pan_rate_deg_s = 0.0
        self._tilt_rate_deg_s = 0.0

    @classmethod
    def from_config(cls, config: AppConfig) -> "CameraModel":
        """Build a camera model from an application configuration.

        Args:
            config: Validated application configuration.

        Returns:
            A camera positioned per ``config.camera.initial_position``.
        """
        return cls(config.camera, config.scene)

    def _resolve_home(self, initial_x: Optional[float],
                      initial_y: Optional[float]) -> Tuple[float, float]:
        """Determine the starting boresight position.

        Args:
            initial_x: Explicit x, or ``None`` to use the configuration.
            initial_y: Explicit y, or ``None`` to use the configuration.

        Returns:
            The starting boresight as ``(x, y)`` in canvas coordinates.
        """
        if initial_x is not None and initial_y is not None:
            return float(initial_x), float(initial_y)
        if (self.config.initial_position == "custom"
                and self.config.initial_pan_px is not None
                and self.config.initial_tilt_px is not None):
            return float(self.config.initial_pan_px), float(self.config.initial_tilt_px)
        # Spec parameter 6: the camera starts at the centre of the screen.
        return self.scene_center

    # -------------------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------------------

    @property
    def scene_center(self) -> Tuple[float, float]:
        """Canvas centre as ``(x, y)``, the zero-angle reference for pan and tilt."""
        return (self.scene.width - 1) / 2.0, (self.scene.height - 1) / 2.0

    @property
    def deg_per_pixel(self) -> Tuple[float, float]:
        """Angular resolution as ``(horizontal, vertical)`` degrees per pixel."""
        return self.config.deg_per_pixel

    @property
    def boresight(self) -> Tuple[float, float]:
        """Current boresight position as ``(x, y)`` in canvas coordinates."""
        return self._x, self._y

    @property
    def pan_deg(self) -> float:
        """Pan angle in degrees, measured from the canvas centre."""
        return (self._x - self.scene_center[0]) * self.deg_per_pixel[0]

    @property
    def tilt_deg(self) -> float:
        """Tilt angle in degrees, measured from the canvas centre."""
        return (self._y - self.scene_center[1]) * self.deg_per_pixel[1]

    @property
    def rates_deg_s(self) -> Tuple[float, float]:
        """The most recently applied ``(pan, tilt)`` rates in degrees per second.

        These are the *clamped* rates actually executed, not the rates requested. Logging the
        executed rate is what makes slew saturation visible in the telemetry rather than
        appearing as unexplained tracking error.
        """
        return self._pan_rate_deg_s, self._tilt_rate_deg_s

    def pixels_to_degrees(self, dx_px: float, dy_px: float) -> Tuple[float, float]:
        """Convert a pixel offset to an angular offset.

        Args:
            dx_px: Horizontal offset in pixels.
            dy_px: Vertical offset in pixels.

        Returns:
            The offset as ``(pan_deg, tilt_deg)``.
        """
        dpp_x, dpp_y = self.deg_per_pixel
        return dx_px * dpp_x, dy_px * dpp_y

    def degrees_to_pixels(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """Convert an angular offset to a pixel offset.

        Args:
            pan_deg: Horizontal angular offset in degrees.
            tilt_deg: Vertical angular offset in degrees.

        Returns:
            The offset as ``(dx_px, dy_px)``.
        """
        dpp_x, dpp_y = self.deg_per_pixel
        return pan_deg / dpp_x, tilt_deg / dpp_y

    @property
    def viewport_origin(self) -> Tuple[int, int]:
        """Top-left pixel of the viewport as ``(x0, y0)``, integer-aligned.

        The viewport is snapped to whole pixels because a detector reads out whole pixels. The
        boresight itself remains continuous, so sub-pixel pointing information is preserved in
        the target's position *within* the frame rather than being lost to the snap.
        """
        width, height = self.config.fov_px
        return (int(round(self._x - (width - 1) / 2.0)),
                int(round(self._y - (height - 1) / 2.0)))

    def canvas_to_frame(self, x: float, y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to frame-local coordinates.

        Args:
            x: Canvas x coordinate.
            y: Canvas y coordinate.

        Returns:
            The position as ``(x, y)`` relative to the viewport's top-left pixel centre.
        """
        x0, y0 = self.viewport_origin
        return x - x0, y - y0

    def frame_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert frame-local coordinates to canvas coordinates.

        Args:
            x: Frame-local x coordinate.
            y: Frame-local y coordinate.

        Returns:
            The position as ``(x, y)`` in canvas coordinates.
        """
        x0, y0 = self.viewport_origin
        return x + x0, y + y0

    def is_visible(self, x: float, y: float, margin_px: float = 0.0) -> bool:
        """Report whether a canvas position falls inside the current viewport.

        Used to classify an acquisition event as in-FOV or search-limited (``docs/DESIGN.md``
        section 7.5).

        Args:
            x: Canvas x coordinate.
            y: Canvas y coordinate.
            margin_px: Inset from the viewport edge. A positive margin requires the point to be
                comfortably inside rather than clipping the border.

        Returns:
            True if the position lies within the viewport.
        """
        fx, fy = self.canvas_to_frame(x, y)
        width, height = self.config.fov_px
        return (margin_px - 0.5 <= fx <= width - 0.5 - margin_px
                and margin_px - 0.5 <= fy <= height - 0.5 - margin_px)

    # -------------------------------------------------------------------------------------
    # Motion
    # -------------------------------------------------------------------------------------

    @property
    def max_px_per_frame(self) -> Tuple[float, float]:
        """Maximum boresight travel per frame as ``(pan, tilt)`` pixels."""
        return self.config.max_px_per_frame

    def clamp_rates(self, pan_rate_deg_s: float,
                    tilt_rate_deg_s: float) -> Tuple[float, float]:
        """Clamp requested angular rates to the configured slew ceiling.

        Args:
            pan_rate_deg_s: Requested pan rate.
            tilt_rate_deg_s: Requested tilt rate.

        Returns:
            The clamped ``(pan, tilt)`` rates.
        """
        max_pan = self.config.max_pan_speed_deg_s
        max_tilt = self.config.max_tilt_speed_deg_s
        return (max(-max_pan, min(max_pan, pan_rate_deg_s)),
                max(-max_tilt, min(max_tilt, tilt_rate_deg_s)))

    def apply_rates(self, pan_rate_deg_s: float, tilt_rate_deg_s: float,
                    dt: float) -> Tuple[float, float]:
        """Slew the camera at the requested rates for ``dt`` seconds.

        Rates are clamped to the slew ceiling, and the *change* in rate is clamped to the
        acceleration limit, so a step command produces a physically achievable ramp rather than
        an instantaneous jump. The boresight is then confined to the canvas.

        Args:
            pan_rate_deg_s: Requested pan rate in degrees per second.
            tilt_rate_deg_s: Requested tilt rate in degrees per second.
            dt: Interval over which the rates apply, in seconds.

        Returns:
            The new boresight as ``(x, y)`` in canvas coordinates.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")

        pan_rate, tilt_rate = self.clamp_rates(pan_rate_deg_s, tilt_rate_deg_s)

        max_delta = self.config.max_acceleration_deg_s2 * dt
        pan_rate = self._pan_rate_deg_s + max(
            -max_delta, min(max_delta, pan_rate - self._pan_rate_deg_s))
        tilt_rate = self._tilt_rate_deg_s + max(
            -max_delta, min(max_delta, tilt_rate - self._tilt_rate_deg_s))

        dpp_x, dpp_y = self.deg_per_pixel
        self._x += (pan_rate * dt) / dpp_x
        self._y += (tilt_rate * dt) / dpp_y
        self._x = max(0.0, min(float(self.scene.width - 1), self._x))
        self._y = max(0.0, min(float(self.scene.height - 1), self._y))

        self._pan_rate_deg_s = pan_rate
        self._tilt_rate_deg_s = tilt_rate
        return self.boresight

    def set_boresight(self, x: float, y: float) -> None:
        """Place the boresight directly, bypassing the slew limits.

        Intended for initialisation and tests only. Using it inside a control loop would silently
        remove the mechanical constraint the whole feasibility analysis rests on.

        Args:
            x: Boresight x in canvas coordinates.
            y: Boresight y in canvas coordinates.
        """
        self._x = max(0.0, min(float(self.scene.width - 1), float(x)))
        self._y = max(0.0, min(float(self.scene.height - 1), float(y)))

    def reset(self) -> None:
        """Return the camera to its initial boresight and zero its rates."""
        self._x, self._y = self._home_x, self._home_y
        self._pan_rate_deg_s = 0.0
        self._tilt_rate_deg_s = 0.0
