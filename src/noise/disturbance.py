"""Mechanical disturbances: camera jitter and platform motion.

Both displace the camera boresight, but they are fundamentally different disturbances and the
controller must treat them differently. Getting this distinction wrong is a real failure mode,
so it is worth stating precisely:

* **Camera jitter** (specification parameter 23, up to +/-20 px/frame) is high-frequency and
  zero-mean. Its energy sits at the Nyquist edge of a 30 Hz loop, *above* the closed-loop
  bandwidth. It is therefore unobservable and uncontrollable: no causal controller can reject a
  disturbance that decorrelates between samples. Feeding raw jittered measurements to a PID does
  not attenuate it, it *injects* it. Jitter is absorbed by the Kalman filter as measurement
  noise and shows up in the centroiding-error budget.

* **Platform motion** (specification parameter 25, up to +/-20 px/frame) is a low-frequency bias
  *inside* the loop bandwidth -- the vehicle carrying the terminal drifting. This is precisely
  what the controller, and specifically its integral term, exists to reject. It shows up in the
  pointing-error budget.

Same magnitude limit, opposite treatment. See ``docs/DESIGN.md`` sections 4.4 and 7.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

__all__ = ["JitterParams", "PlatformMotionParams", "JitterModel", "PlatformMotion"]


@dataclass(frozen=True)
class JitterParams:
    """High-frequency zero-mean boresight disturbance.

    Attributes:
        enabled: Whether jitter is active.
        max_px_per_frame: Magnitude bound in pixels. The specification phrases parameter 23 as a
            *maximum*, so for the Gaussian distribution this is treated as a 3-sigma bound and
            the draw is clipped -- a Gaussian tail must never exceed a stated maximum.
        distribution: ``"gaussian"`` or ``"uniform"``.
    """

    enabled: bool = True
    max_px_per_frame: float = 5.0
    distribution: str = "gaussian"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JitterParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.camera_jitter`` block.

        Returns:
            The corresponding parameters.
        """
        return cls(enabled=bool(raw.get("enabled", True)),
                   max_px_per_frame=float(raw.get("max_px_per_frame", 5.0)),
                   distribution=str(raw.get("distribution", "gaussian")))

    @property
    def sigma_px(self) -> float:
        """Effective per-axis standard deviation in pixels.

        Useful as a floor for the Kalman measurement noise: the filter cannot do better than the
        jitter, so claiming a smaller ``R`` would make the validation gate over-tight.
        """
        if not self.enabled or self.max_px_per_frame <= 0:
            return 0.0
        if self.distribution == "uniform":
            return self.max_px_per_frame / math.sqrt(3.0)
        return self.max_px_per_frame / 3.0


@dataclass(frozen=True)
class PlatformMotionParams:
    """Low-frequency boresight drift from the carrying vehicle.

    Attributes:
        enabled: Whether platform motion is active.
        type: ``"linear"`` (mandatory per spec), or ``"circular"``, ``"random"``, ``"spiral"``,
            ``"figure8"``.
        max_px_per_frame: Bound on drift speed, in pixels per frame.
        velocity_x_px_s: Horizontal drift velocity for linear motion.
        velocity_y_px_s: Vertical drift velocity for linear motion.
        amplitude_px: Amplitude for the periodic motion types.
        angular_velocity_rad_s: Rate for the periodic motion types.
        step_sigma_px: Per-step displacement sigma for random drift.
    """

    enabled: bool = False
    type: str = "linear"
    max_px_per_frame: float = 5.0
    velocity_x_px_s: float = 30.0
    velocity_y_px_s: float = 15.0
    amplitude_px: float = 50.0
    angular_velocity_rad_s: float = 0.2
    step_sigma_px: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlatformMotionParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.platform_motion`` block.

        Returns:
            The corresponding parameters.
        """
        return cls(
            enabled=bool(raw.get("enabled", False)),
            type=str(raw.get("type", "linear")),
            max_px_per_frame=float(raw.get("max_px_per_frame", 5.0)),
            velocity_x_px_s=float(raw.get("velocity_x_px_s", 30.0)),
            velocity_y_px_s=float(raw.get("velocity_y_px_s", 15.0)),
            amplitude_px=float(raw.get("amplitude_px", 50.0)),
            angular_velocity_rad_s=float(raw.get("angular_velocity_rad_s", 0.2)),
            step_sigma_px=float(raw.get("step_sigma_px", 1.0)),
        )


class JitterModel:
    """Draws per-frame jitter offsets, bounded by the configured maximum."""

    def __init__(self, params: JitterParams,
                 rng: Optional[np.random.Generator] = None) -> None:
        """Initialise the jitter model.

        Args:
            params: Jitter parameters.
            rng: Seeded generator. One is created if omitted.

        Raises:
            ValueError: On a negative magnitude or an unknown distribution.
        """
        if params.max_px_per_frame < 0:
            raise ValueError(
                f"Jitter magnitude must be non-negative, got {params.max_px_per_frame}")
        if params.distribution not in ("gaussian", "uniform"):
            raise ValueError(f"Unknown jitter distribution: {params.distribution!r}")
        self.params = params
        self._rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> Tuple[float, float]:
        """Draw one jitter offset.

        Returns:
            The offset as ``(dx, dy)`` in pixels, guaranteed within the configured maximum.
        """
        bound = self.params.max_px_per_frame
        if not self.params.enabled or bound <= 0:
            return 0.0, 0.0
        if self.params.distribution == "uniform":
            dx, dy = self._rng.uniform(-bound, bound, size=2)
            return float(dx), float(dy)
        dx, dy = self._rng.normal(0.0, bound / 3.0, size=2)
        return (float(np.clip(dx, -bound, bound)), float(np.clip(dy, -bound, bound)))


class PlatformMotion:
    """Accumulating low-frequency boresight drift.

    Unlike jitter this is *stateful and biased*: the offset accumulates, so the controller must
    actively work against it. That is exactly the point -- it is the disturbance that exercises
    the integral term.
    """

    def __init__(self, params: PlatformMotionParams,
                 rng: Optional[np.random.Generator] = None) -> None:
        """Initialise the drift at zero offset.

        Args:
            params: Platform-motion parameters.
            rng: Seeded generator, used by the random drift type.

        Raises:
            ValueError: On an unknown motion type.
        """
        known = ("linear", "circular", "random", "spiral", "figure8")
        if params.type not in known:
            raise ValueError(
                f"Unknown platform motion type {params.type!r}. Known: {list(known)}")
        self.params = params
        self._rng = rng if rng is not None else np.random.default_rng()
        self._dx = 0.0
        self._dy = 0.0
        self._time = 0.0

    @property
    def offset(self) -> Tuple[float, float]:
        """Accumulated drift as ``(dx, dy)`` in pixels."""
        return self._dx, self._dy

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance the drift by ``dt`` seconds.

        The per-frame displacement is capped at ``max_px_per_frame`` so a configuration cannot
        exceed specification parameter 25 regardless of the velocity settings.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new accumulated offset as ``(dx, dy)`` in pixels.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        if not self.params.enabled or dt == 0:
            self._time += dt
            return self.offset

        previous = (self._dx, self._dy)
        self._time += dt
        p = self.params
        t = self._time

        if p.type == "linear":
            self._dx = p.velocity_x_px_s * t
            self._dy = p.velocity_y_px_s * t
        elif p.type == "circular":
            self._dx = p.amplitude_px * math.cos(p.angular_velocity_rad_s * t)
            self._dy = p.amplitude_px * math.sin(p.angular_velocity_rad_s * t)
        elif p.type == "figure8":
            self._dx = p.amplitude_px * math.sin(p.angular_velocity_rad_s * t)
            self._dy = p.amplitude_px * math.sin(2.0 * p.angular_velocity_rad_s * t)
        elif p.type == "spiral":
            angle = p.angular_velocity_rad_s * t
            radius = p.amplitude_px * angle / (2.0 * math.pi)
            self._dx = radius * math.cos(angle)
            self._dy = radius * math.sin(angle)
        else:  # random
            step = self._rng.normal(0.0, p.step_sigma_px, size=2)
            self._dx += float(step[0])
            self._dy += float(step[1])

        # Enforce the specification's per-frame bound on how fast the platform may drift.
        delta_x = self._dx - previous[0]
        delta_y = self._dy - previous[1]
        magnitude = math.hypot(delta_x, delta_y)
        if magnitude > p.max_px_per_frame > 0:
            scale = p.max_px_per_frame / magnitude
            self._dx = previous[0] + delta_x * scale
            self._dy = previous[1] + delta_y * scale
        return self.offset

    def reset(self) -> None:
        """Return the drift to zero."""
        self._dx = 0.0
        self._dy = 0.0
        self._time = 0.0
