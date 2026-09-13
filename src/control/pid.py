"""PID controller with anti-windup and slew-rate limiting.

Acts on *angular* error in degrees and produces an angular *rate* command in degrees per second,
so the proportional gain has units of inverse seconds and is directly comparable to a closed-loop
bandwidth.

Two design points that are not optional here:

* **The derivative term is supplied, not differenced.** Differencing successive measurements to
  get a rate amplifies exactly the high-frequency content the loop cannot reject. Camera jitter is
  specified at up to +-20 px/frame, zero-mean, changing every frame -- its energy sits at the
  Nyquist edge of a 30 Hz loop, above the closed-loop bandwidth, so it is unobservable and
  uncontrollable. A differentiator turns it into large spurious rate commands. The caller passes
  the Kalman velocity state instead (``docs/DESIGN.md`` section 7.2).

* **Anti-windup clamps the integral, not the output.** Clamping only the output lets the integral
  accumulate while the actuator is saturated, so the controller keeps commanding in the same
  direction long after the error has reversed. With a 5 deg/s slew ceiling and errors that can
  exceed 300 px, saturation is the normal case during acquisition rather than an edge case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["PidParams", "PidState", "Pid"]


@dataclass(frozen=True)
class PidParams:
    """PID gains and limits.

    Attributes:
        kp: Proportional gain, in inverse seconds. Sized so a ~100 px error saturates the slew
            limit: 100 px x 0.00625 deg/px = 0.625 deg, and 5.0 / 0.625 = 8.0.
        ki: Integral gain. Rejects the low-frequency bias that platform motion introduces.
        kd: Derivative gain, applied to a supplied rate rather than a differenced measurement.
        integral_limit: Anti-windup clamp on the accumulated term, in degree-seconds.
        output_limit: Slew ceiling in degrees per second.
    """

    kp: float = 8.0
    ki: float = 0.5
    kd: float = 1.5
    integral_limit: float = 5.0
    output_limit: float = 5.0


@dataclass(frozen=True)
class PidState:
    """One controller evaluation, decomposed for the trace.

    Attributes:
        output: Commanded rate after clamping, in degrees per second.
        proportional: Proportional contribution.
        integral: Integral contribution.
        derivative: Derivative contribution.
        feedforward: Feedforward contribution.
        saturated: Whether the command hit the slew ceiling. Logged because slew saturation is a
            physical limit rather than a tracking failure, and the two look identical in a
            pointing-error plot.
        integral_clamped: Whether anti-windup engaged this step.
    """

    output: float = 0.0
    proportional: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    feedforward: float = 0.0
    saturated: bool = False
    integral_clamped: bool = False


class Pid:
    """Single-axis PID on angular error.

    Attributes:
        params: Gains and limits.
    """

    def __init__(self, params: Optional[PidParams] = None) -> None:
        """Create a controller with a zeroed integrator.

        Args:
            params: Gains and limits.
        """
        self.params = params or PidParams()
        self._integral = 0.0

    @property
    def integral(self) -> float:
        """Current accumulated integral term, in degree-seconds."""
        return self._integral

    def step(self, error_deg: float, dt: float, rate_deg_s: float = 0.0,
             feedforward_deg_s: float = 0.0) -> PidState:
        """Evaluate the controller for one interval.

        Args:
            error_deg: Angular error in degrees, target minus boresight.
            dt: Interval in seconds. Must be non-negative.
            rate_deg_s: Measured rate of the error, supplied by the estimator. Used for the
                derivative term; never obtained by differencing raw measurements.
            feedforward_deg_s: Target angular velocity, added directly to the command. This is
                what cancels the steady-state phase lag a feedback-only loop always carries
                against a continuously moving reference.

        Returns:
            A :class:`PidState`.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")

        params = self.params
        proportional = params.kp * error_deg

        candidate = self._integral + error_deg * dt
        limit = params.integral_limit
        clamped = abs(candidate) > limit
        self._integral = max(-limit, min(limit, candidate))
        integral = params.ki * self._integral

        derivative = params.kd * rate_deg_s
        raw = proportional + integral + derivative + feedforward_deg_s

        ceiling = params.output_limit
        output = max(-ceiling, min(ceiling, raw))
        saturated = abs(raw) > ceiling

        if saturated:
            # Back off the integral while saturated. Continuing to accumulate against an actuator
            # that cannot respond is the classic windup, and during acquisition -- where errors of
            # hundreds of pixels are normal -- saturation is the usual state, not an edge case.
            self._integral = max(-limit, min(limit, self._integral - error_deg * dt))
            integral = params.ki * self._integral

        return PidState(output=output, proportional=proportional, integral=integral,
                        derivative=derivative, feedforward=feedforward_deg_s,
                        saturated=saturated, integral_clamped=clamped)

    def reset(self) -> None:
        """Zero the integrator."""
        self._integral = 0.0
