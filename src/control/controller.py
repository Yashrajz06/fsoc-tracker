"""Pointing controller: Kalman state in, angular rate command out.

Converts the pointing error into a pan/tilt rate, with PID feedback plus Kalman-velocity
feedforward. The camera model enforces the slew and acceleration limits itself, so a controller
bug degrades tracking rather than producing a camera that could not exist.

**The controller consumes the Kalman-smoothed state, never a raw per-frame centroid.** This is a
hard rule (``docs/DESIGN.md`` section 7.2), not a preference. Camera jitter is specified at up to
+-20 px/frame, zero-mean, changing every frame -- its energy is at the Nyquist edge of a 30 Hz
loop, above the closed-loop bandwidth, so no causal controller can reject it. Feeding raw
centroids to a PID does not attenuate jitter, it *injects* it: the derivative term differentiates
it into large spurious commands and the integral accumulates its excursions. The correct
decomposition is that jitter is absorbed by the filter as measurement noise and appears in the
centroiding-error budget, while platform motion -- a low-frequency bias inside the loop bandwidth
-- is what the integral term exists to reject.

Feedforward is the other half. A feedback-only loop always carries a steady-state phase lag
against a continuously moving reference; commanding the camera to match the target's estimated
velocity cancels it rather than merely correcting the error it produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from src.camera.model import CameraModel
from src.control.pid import Pid, PidParams, PidState

__all__ = ["ControllerParams", "ControlCommand", "PointingController"]


@dataclass(frozen=True)
class ControllerParams:
    """Controller configuration.

    Attributes:
        pid: PID gains and limits, applied identically to both axes.
        feedforward_enabled: Apply Kalman-velocity feedforward.
        feedforward_gain: Scale on the feedforward term. 1.0 exactly cancels the estimated target
            motion.
    """

    pid: PidParams = PidParams()
    feedforward_enabled: bool = True
    feedforward_gain: float = 1.0


@dataclass(frozen=True)
class ControlCommand:
    """One controller output, decomposed for the trace.

    Attributes:
        pan_rate_deg_s: Commanded pan rate.
        tilt_rate_deg_s: Commanded tilt rate.
        pointing_error_px: Distance from the target estimate to the boresight, in pixels. This is
            the spec's "tracking error" under its pointing interpretation.
        saturated: Whether either axis hit the slew ceiling. Slew saturation is a physical limit,
            not a tracking failure, and the two are indistinguishable in a pointing-error plot
            unless this is logged.
        pan: Pan-axis PID decomposition.
        tilt: Tilt-axis PID decomposition.
    """

    pan_rate_deg_s: float = 0.0
    tilt_rate_deg_s: float = 0.0
    pointing_error_px: float = 0.0
    saturated: bool = False
    pan: Optional[PidState] = None
    tilt: Optional[PidState] = None


class PointingController:
    """Two-axis pointing controller driving a :class:`~src.camera.model.CameraModel`.

    Attributes:
        camera: The camera being steered.
        params: Controller configuration.
    """

    def __init__(self, camera: CameraModel,
                 params: Optional[ControllerParams] = None) -> None:
        """Initialise the controller.

        Args:
            camera: Camera model supplying the angular scale and mechanical limits.
            params: Controller configuration.
        """
        self.camera = camera
        self.params = params or ControllerParams()
        limits = self.params.pid
        self._pan = Pid(PidParams(kp=limits.kp, ki=limits.ki, kd=limits.kd,
                                  integral_limit=limits.integral_limit,
                                  output_limit=camera.config.max_pan_speed_deg_s))
        self._tilt = Pid(PidParams(kp=limits.kp, ki=limits.ki, kd=limits.kd,
                                   integral_limit=limits.integral_limit,
                                   output_limit=camera.config.max_tilt_speed_deg_s))

    def step(self, target_xy: Tuple[float, float], dt: float,
             target_velocity_px_s: Tuple[float, float] = (0.0, 0.0)) -> ControlCommand:
        """Compute and apply a rate command for one control interval.

        Args:
            target_xy: **Kalman-smoothed** target position in canvas coordinates. Passing a raw
                per-frame centroid here would inject jitter into the loop.
            dt: Control interval in seconds.
            target_velocity_px_s: Kalman velocity estimate, used for feedforward and for the
                derivative term. Never obtained by differencing measurements.

        Returns:
            The :class:`ControlCommand` that was applied to the camera.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")

        boresight_x, boresight_y = self.camera.boresight
        error_x_px = target_xy[0] - boresight_x
        error_y_px = target_xy[1] - boresight_y
        error_pan_deg, error_tilt_deg = self.camera.pixels_to_degrees(error_x_px, error_y_px)

        # Feedforward uses the target's angular velocity directly.
        velocity_pan_deg_s, velocity_tilt_deg_s = self.camera.pixels_to_degrees(
            target_velocity_px_s[0], target_velocity_px_s[1])

        # The derivative term acts on the rate of change of the *error*, which is the target's
        # angular velocity minus the camera's own. Passing the target velocity alone -- as an
        # earlier version did -- makes ``kd`` a second feedforward with the wrong gain, so the
        # target's motion is counted twice. Measured, that cost a factor of 2-4 in steady-state
        # pointing error on a moving target (8.44 px against 3.62 px at 100 px/s), and it looked
        # like "kd is harmful" rather than like a double-count.
        camera_pan_rate, camera_tilt_rate = self.camera.rates_deg_s
        error_rate_pan = velocity_pan_deg_s - camera_pan_rate
        error_rate_tilt = velocity_tilt_deg_s - camera_tilt_rate

        feedforward_pan = feedforward_tilt = 0.0
        if self.params.feedforward_enabled:
            feedforward_pan = self.params.feedforward_gain * velocity_pan_deg_s
            feedforward_tilt = self.params.feedforward_gain * velocity_tilt_deg_s

        pan = self._pan.step(error_pan_deg, dt, rate_deg_s=error_rate_pan,
                             feedforward_deg_s=feedforward_pan)
        tilt = self._tilt.step(error_tilt_deg, dt, rate_deg_s=error_rate_tilt,
                               feedforward_deg_s=feedforward_tilt)

        self.camera.apply_rates(pan.output, tilt.output, dt)

        return ControlCommand(pan_rate_deg_s=pan.output, tilt_rate_deg_s=tilt.output,
                              pointing_error_px=math.hypot(error_x_px, error_y_px),
                              saturated=pan.saturated or tilt.saturated,
                              pan=pan, tilt=tilt)

    def drive_to(self, waypoint_xy: Tuple[float, float], dt: float) -> ControlCommand:
        """Slew toward an open-loop waypoint, as during a search sweep.

        Args:
            waypoint_xy: Desired boresight position as ``(x, y)``.
            dt: Control interval in seconds.

        Returns:
            The applied :class:`ControlCommand`.
        """
        return self.step(waypoint_xy, dt)

    def reset(self) -> None:
        """Zero both integrators."""
        self._pan.reset()
        self._tilt.reset()
