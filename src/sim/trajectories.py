"""Target motion models.

Four motions are mandatory per specification parameter 12 -- straight line, circular,
figure-of-8 and random -- with spiral, sinusoidal and Ornstein-Uhlenbeck available as optional
extras. See ``docs/DESIGN.md`` section 3.

Two kinds of trajectory live here and they are deliberately distinguished:

* **Analytic** trajectories are pure functions of elapsed time. Given a seed-free closed form,
  ``position_at(t)`` is reproducible, order-independent and testable at arbitrary times.
* **Stochastic** trajectories (random walk, Ornstein-Uhlenbeck) evolve by integration and are
  therefore path-dependent: they are reproducible only from a seeded generator advanced in the
  same step sequence. :meth:`Trajectory.position_at` is unavailable for these, by design, rather
  than silently returning something that depends on call history.

Boundary handling uses coordinate folding rather than velocity reflection. Folding a coordinate
through a triangle wave is exactly equivalent to elastic reflection for a constant-velocity
path, but it stays a pure function of position, so it composes with the analytic trajectories
without giving them hidden velocity state. See :func:`apply_boundary`.

Coordinate convention: pixel centres at integer indices, tuples ordered ``(x, y)``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import numpy as np

from src.config import AppConfig, ConfigError

__all__ = [
    "Trajectory",
    "AnalyticTrajectory",
    "LinearTrajectory",
    "CircularTrajectory",
    "Figure8Trajectory",
    "RandomWalkTrajectory",
    "SpiralTrajectory",
    "SinusoidalTrajectory",
    "OrnsteinUhlenbeckTrajectory",
    "apply_boundary",
    "build_trajectory",
    "MANDATORY_MOTIONS",
]

#: Motion types the specification requires (parameter 12).
MANDATORY_MOTIONS: Tuple[str, ...] = ("linear", "circular", "figure8", "random")


def apply_boundary(value: float, low: float, high: float, behaviour: str) -> float:
    """Constrain a coordinate to ``[low, high]`` according to a boundary policy.

    Args:
        value: The unconstrained coordinate.
        low: Lower bound, inclusive.
        high: Upper bound, inclusive.
        behaviour: ``"bounce"``, ``"wrap"`` or ``"clamp"``.

    Returns:
        The constrained coordinate.

    Raises:
        ValueError: On an unknown behaviour or an inverted range.

    Notes:
        ``"bounce"`` folds the coordinate through a triangle wave of period ``2*(high-low)``.
        For constant-velocity motion this is exactly elastic reflection off the boundary, but
        unlike a velocity flip it remains a pure function of position, so it can be applied to
        an analytic trajectory without introducing hidden state that would make the path depend
        on the step sequence used to sample it.
    """
    if high < low:
        raise ValueError(f"Boundary range is inverted: [{low}, {high}]")
    span = high - low
    if span == 0:
        return low

    if behaviour == "clamp":
        return min(high, max(low, value))
    if behaviour == "wrap":
        return low + (value - low) % span
    if behaviour == "bounce":
        folded = (value - low) % (2.0 * span)
        return low + (folded if folded <= span else 2.0 * span - folded)
    raise ValueError(f"Unknown boundary behaviour: {behaviour!r}")


class Trajectory(ABC):
    """Base class for all target motions.

    Attributes:
        boundary: Boundary policy applied to every emitted position.
        bounds: ``(min_x, min_y, max_x, max_y)`` the target is constrained to, or ``None`` for
            unconstrained motion.
    """

    def __init__(self, boundary: str = "bounce",
                 bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Initialise common trajectory state.

        Args:
            boundary: ``"bounce"``, ``"wrap"`` or ``"clamp"``.
            bounds: ``(min_x, min_y, max_x, max_y)``, or ``None`` to leave motion unconstrained.
        """
        self.boundary = boundary
        self.bounds = bounds
        self._time = 0.0

    @property
    def elapsed(self) -> float:
        """Elapsed trajectory time in seconds."""
        return self._time

    @property
    def is_stochastic(self) -> bool:
        """Whether this trajectory is path-dependent and therefore has no closed form."""
        return False

    def constrain(self, x: float, y: float) -> Tuple[float, float]:
        """Apply the configured boundary policy to a position.

        Args:
            x: Unconstrained x coordinate.
            y: Unconstrained y coordinate.

        Returns:
            The constrained position as ``(x, y)``.
        """
        if self.bounds is None:
            return x, y
        min_x, min_y, max_x, max_y = self.bounds
        return (apply_boundary(x, min_x, max_x, self.boundary),
                apply_boundary(y, min_y, max_y, self.boundary))

    def position_at(self, t: float) -> Tuple[float, float]:
        """Return the constrained position at absolute time ``t``.

        Args:
            t: Elapsed time in seconds since the start of the trajectory.

        Returns:
            The position as ``(x, y)``.

        Raises:
            NotImplementedError: For stochastic trajectories, which have no closed form. Ask a
                stochastic trajectory to :meth:`step` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is stochastic and has no closed form; use step() instead"
        )

    @abstractmethod
    def step(self, dt: float) -> Tuple[float, float]:
        """Advance the trajectory by ``dt`` seconds and return the new position.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new constrained position as ``(x, y)``.
        """

    def reset(self) -> None:
        """Return the trajectory to its initial state so a scenario can be re-run."""
        self._time = 0.0


class AnalyticTrajectory(Trajectory):
    """A trajectory that is a closed-form function of elapsed time.

    Subclasses implement :meth:`_evaluate`; stepping is then just time accumulation, which makes
    the path independent of how it is sampled.
    """

    @abstractmethod
    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``.

        Args:
            t: Elapsed time in seconds.

        Returns:
            The raw position as ``(x, y)``, before boundary handling.
        """

    def position_at(self, t: float) -> Tuple[float, float]:
        """Return the constrained position at absolute time ``t``.

        Args:
            t: Elapsed time in seconds.

        Returns:
            The constrained position as ``(x, y)``.
        """
        return self.constrain(*self._evaluate(t))

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance by ``dt`` and return the new position.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new constrained position as ``(x, y)``.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        self._time += dt
        return self.position_at(self._time)


class LinearTrajectory(AnalyticTrajectory):
    """Straight-line motion at constant velocity (mandatory motion 1).

    ``x = x0 + vx*t``, ``y = y0 + vy*t``.
    """

    def __init__(self, x0: float, y0: float, velocity_x_px_s: float, velocity_y_px_s: float,
                 **kwargs) -> None:
        """Initialise straight-line motion.

        Args:
            x0: Initial x position in canvas coordinates.
            y0: Initial y position in canvas coordinates.
            velocity_x_px_s: Horizontal velocity in pixels per second.
            velocity_y_px_s: Vertical velocity in pixels per second.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self.x0 = x0
        self.y0 = y0
        self.vx = velocity_x_px_s
        self.vy = velocity_y_px_s

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``."""
        return self.x0 + self.vx * t, self.y0 + self.vy * t

    @property
    def speed_px_s(self) -> float:
        """Scalar speed in pixels per second.

        Worth comparing against ``CameraConfig.max_px_per_frame``: a target faster than the slew
        ceiling cannot be kept centred by any controller.
        """
        return math.hypot(self.vx, self.vy)


class CircularTrajectory(AnalyticTrajectory):
    """Circular motion about a fixed centre (mandatory motion 2).

    ``x = xc + R*cos(w*t)``, ``y = yc + R*sin(w*t)``.
    """

    def __init__(self, center_x: float, center_y: float, radius_px: float,
                 angular_velocity_rad_s: float, phase_offset_rad: float = 0.0, **kwargs) -> None:
        """Initialise circular motion.

        Args:
            center_x: Circle centre x in canvas coordinates.
            center_y: Circle centre y in canvas coordinates.
            radius_px: Circle radius in pixels.
            angular_velocity_rad_s: Angular rate in radians per second.
            phase_offset_rad: Starting phase in radians.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius_px
        self.omega = angular_velocity_rad_s
        self.phase = phase_offset_rad

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``."""
        angle = self.omega * t + self.phase
        return (self.center_x + self.radius * math.cos(angle),
                self.center_y + self.radius * math.sin(angle))

    @property
    def tangential_speed_px_s(self) -> float:
        """Tangential speed ``R*w`` in pixels per second."""
        return abs(self.radius * self.omega)


class Figure8Trajectory(AnalyticTrajectory):
    """Lissajous figure-of-8 motion (mandatory motion 3).

    ``x = xc + A*sin(w*t + d)``, ``y = yc + B*sin(2*w*t)``. The 1:2 frequency ratio is what makes
    the path a figure-of-8 rather than an ellipse.
    """

    def __init__(self, amplitude_x_px: float, amplitude_y_px: float,
                 angular_velocity_rad_s: float, phase_offset_rad: float = 0.0,
                 center_x: float = 0.0, center_y: float = 0.0, **kwargs) -> None:
        """Initialise figure-of-8 motion.

        Args:
            amplitude_x_px: Horizontal amplitude in pixels.
            amplitude_y_px: Vertical amplitude in pixels.
            angular_velocity_rad_s: Base angular rate in radians per second.
            phase_offset_rad: Horizontal phase offset in radians.
            center_x: Figure centre x in canvas coordinates.
            center_y: Figure centre y in canvas coordinates.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self.amplitude_x = amplitude_x_px
        self.amplitude_y = amplitude_y_px
        self.omega = angular_velocity_rad_s
        self.phase = phase_offset_rad
        self.center_x = center_x
        self.center_y = center_y

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``."""
        return (self.center_x + self.amplitude_x * math.sin(self.omega * t + self.phase),
                self.center_y + self.amplitude_y * math.sin(2.0 * self.omega * t))


class SpiralTrajectory(AnalyticTrajectory):
    """Archimedean spiral motion (optional).

    ``r = b*theta``, ``x = xc + r*cos(theta)``, ``y = yc + r*sin(theta)``. The same geometry
    doubles as the acquisition search pattern in ``src/control/search.py`` (Phase 4).
    """

    def __init__(self, center_x: float, center_y: float, growth_rate_b: float,
                 angular_velocity_rad_s: float, **kwargs) -> None:
        """Initialise spiral motion.

        Args:
            center_x: Spiral centre x in canvas coordinates.
            center_y: Spiral centre y in canvas coordinates.
            growth_rate_b: Radial growth per radian, in pixels. Arm spacing is ``2*pi*b``.
            angular_velocity_rad_s: Angular rate in radians per second.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self.center_x = center_x
        self.center_y = center_y
        self.b = growth_rate_b
        self.omega = angular_velocity_rad_s

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``."""
        theta = self.omega * t
        radius = self.b * theta
        return (self.center_x + radius * math.cos(theta),
                self.center_y + radius * math.sin(theta))

    @property
    def arm_spacing_px(self) -> float:
        """Radial distance between successive spiral arms, ``2*pi*b``."""
        return 2.0 * math.pi * self.b


class SinusoidalTrajectory(AnalyticTrajectory):
    """Horizontal drift with vertical oscillation (optional).

    ``x = x0 + vx*t``, ``y = y0 + A*sin(w*t)``.
    """

    def __init__(self, x0: float, y0: float, velocity_x_px_s: float, amplitude_px: float,
                 angular_velocity_rad_s: float, **kwargs) -> None:
        """Initialise sinusoidal motion.

        Args:
            x0: Initial x position in canvas coordinates.
            y0: Vertical centre of the oscillation, in canvas coordinates.
            velocity_x_px_s: Horizontal drift velocity in pixels per second.
            amplitude_px: Vertical amplitude in pixels.
            angular_velocity_rad_s: Oscillation rate in radians per second.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self.x0 = x0
        self.y0 = y0
        self.vx = velocity_x_px_s
        self.amplitude = amplitude_px
        self.omega = angular_velocity_rad_s

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Return the unconstrained position at time ``t``."""
        return self.x0 + self.vx * t, self.y0 + self.amplitude * math.sin(self.omega * t)


class RandomWalkTrajectory(Trajectory):
    """Gaussian random-walk motion (mandatory motion 4).

    ``x[k+1] = x[k] + xi``, ``xi ~ N(0, sigma_step^2)``, with the per-step displacement capped so
    apparent speed stays bounded. The cap matters: an uncapped Gaussian walk occasionally emits a
    step far beyond the camera's slew ceiling, which would appear in the logs as a tracker
    failure when it is really a physically impossible target.
    """

    def __init__(self, x0: float, y0: float, step_sigma_px: float,
                 max_speed_px_s: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None, **kwargs) -> None:
        """Initialise a random walk.

        Args:
            x0: Initial x position in canvas coordinates.
            y0: Initial y position in canvas coordinates.
            step_sigma_px: Standard deviation of the per-step displacement, in pixels.
            max_speed_px_s: Optional cap on apparent speed, in pixels per second.
            rng: Seeded generator. One is created if omitted, but then the path is not
                reproducible -- always pass a seeded generator for anything that will be logged.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self._x0 = x0
        self._y0 = y0
        self.step_sigma = step_sigma_px
        self.max_speed = max_speed_px_s
        self._rng = rng if rng is not None else np.random.default_rng()
        self._x, self._y = self.constrain(x0, y0)

    @property
    def is_stochastic(self) -> bool:
        """Random walks are path-dependent."""
        return True

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance the walk by one step of duration ``dt``.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new constrained position as ``(x, y)``.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        self._time += dt
        dx, dy = self._rng.normal(0.0, self.step_sigma, size=2)

        if self.max_speed is not None and dt > 0:
            limit = self.max_speed * dt
            magnitude = math.hypot(dx, dy)
            if magnitude > limit:
                scale = limit / magnitude
                dx, dy = dx * scale, dy * scale

        self._x, self._y = self.constrain(self._x + dx, self._y + dy)
        return self._x, self._y

    def reset(self) -> None:
        """Return to the initial position. Does not reseed the generator."""
        super().reset()
        self._x, self._y = self.constrain(self._x0, self._y0)


class OrnsteinUhlenbeckTrajectory(Trajectory):
    """Mean-reverting stochastic motion (optional, preferred stochastic model).

    ``x[k+1] = x[k] + theta*(mu - x[k])*dt + sigma*sqrt(dt)*eps``.

    Preferred over a plain random walk for realism because it is mean-reverting: an unbounded
    walk drifts off the canvas and then spends the run pinned against a boundary, which is not a
    useful test of a tracker. The same process also models beam wander in
    ``src/noise/turbulence.py`` (Phase 2).
    """

    def __init__(self, x0: float, y0: float, theta: float, mu_x: float, mu_y: float,
                 sigma: float, rng: Optional[np.random.Generator] = None, **kwargs) -> None:
        """Initialise an Ornstein-Uhlenbeck process.

        Args:
            x0: Initial x position in canvas coordinates.
            y0: Initial y position in canvas coordinates.
            theta: Mean-reversion rate. Larger values pull back to the mean faster.
            mu_x: Long-run mean x position.
            mu_y: Long-run mean y position.
            sigma: Diffusion coefficient, in pixels per sqrt(second).
            rng: Seeded generator. One is created if omitted.
            **kwargs: Boundary options forwarded to :class:`Trajectory`.
        """
        super().__init__(**kwargs)
        self._x0 = x0
        self._y0 = y0
        self.theta = theta
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma = sigma
        self._rng = rng if rng is not None else np.random.default_rng()
        self._x, self._y = self.constrain(x0, y0)

    @property
    def is_stochastic(self) -> bool:
        """Ornstein-Uhlenbeck motion is path-dependent."""
        return True

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance the process by ``dt`` seconds.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new constrained position as ``(x, y)``.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        self._time += dt
        noise = self._rng.normal(0.0, 1.0, size=2) * self.sigma * math.sqrt(dt)
        x = self._x + self.theta * (self.mu_x - self._x) * dt + noise[0]
        y = self._y + self.theta * (self.mu_y - self._y) * dt + noise[1]
        self._x, self._y = self.constrain(x, y)
        return self._x, self._y

    def reset(self) -> None:
        """Return to the initial position. Does not reseed the generator."""
        super().reset()
        self._x, self._y = self.constrain(self._x0, self._y0)


#: Registry of motion type name to implementing class.
_REGISTRY: Dict[str, Type[Trajectory]] = {
    "linear": LinearTrajectory,
    "circular": CircularTrajectory,
    "figure8": Figure8Trajectory,
    "random": RandomWalkTrajectory,
    "spiral": SpiralTrajectory,
    "sinusoidal": SinusoidalTrajectory,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckTrajectory,
}


def _initial_position(config: AppConfig, rng: np.random.Generator) -> Tuple[float, float]:
    """Determine the target's starting position from configuration.

    Args:
        config: Validated application configuration.
        rng: Seeded generator, used when the initial position is random.

    Returns:
        The starting position as ``(x, y)`` in canvas coordinates.

    Notes:
        A random start is drawn from an inset region so the beacon begins fully on-canvas. Note
        that a random start usually places the beacon *outside* the initial camera viewport,
        which puts acquisition into the search-limited population -- see ``docs/DESIGN.md``
        section 7.5. That is intended behaviour, not a defect, but it is why acquisition time is
        never reported as a single pooled number.
    """
    target = config.target
    scene = config.scene
    if target.initial_position == "custom":
        assert target.initial_x is not None and target.initial_y is not None
        return float(target.initial_x), float(target.initial_y)
    if target.initial_position == "center":
        return (scene.width - 1) / 2.0, (scene.height - 1) / 2.0

    margin = max(target.size_px, 1) * 2.0
    return (float(rng.uniform(margin, scene.width - 1 - margin)),
            float(rng.uniform(margin, scene.height - 1 - margin)))


def build_trajectory(config: AppConfig,
                     rng: Optional[np.random.Generator] = None) -> Trajectory:
    """Construct the configured trajectory.

    Args:
        config: Validated application configuration.
        rng: Seeded generator for stochastic motions. When omitted, one is created from
            ``config.run.random_seed`` so runs remain reproducible for the report.

    Returns:
        A ready-to-step :class:`Trajectory`.

    Raises:
        ConfigError: If the configured motion type is unknown, or its parameter block is missing
            a value the model requires.
    """
    if rng is None:
        rng = np.random.default_rng(config.run.random_seed)

    motion_type = config.target.motion_type
    if motion_type not in _REGISTRY:
        raise ConfigError(
            f"Unknown motion type {motion_type!r}. Known: {sorted(_REGISTRY)}"
        )

    params = dict(config.target.motion_params())
    x0, y0 = _initial_position(config, rng)

    # Keep the beacon centre on-canvas. Bounds use the pixel-centre convention.
    bounds = (0.0, 0.0, float(config.scene.width - 1), float(config.scene.height - 1))
    common = {"boundary": config.target.boundary_behaviour, "bounds": bounds}

    if motion_type == "linear":
        params.setdefault("x0", x0)
        params.setdefault("y0", y0)
    elif motion_type == "sinusoidal":
        params.setdefault("x0", x0)
        params.setdefault("y0", y0)
    elif motion_type in ("random", "ornstein_uhlenbeck"):
        params.setdefault("x0", x0)
        params.setdefault("y0", y0)
        params["rng"] = rng
    elif motion_type == "figure8":
        params.setdefault("center_x", (config.scene.width - 1) / 2.0)
        params.setdefault("center_y", (config.scene.height - 1) / 2.0)

    cls = _REGISTRY[motion_type]
    try:
        return cls(**params, **common)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ConfigError(
            f"Motion block 'target.motion.{motion_type}' does not match the parameters of "
            f"{cls.__name__}: {exc}"
        ) from exc
