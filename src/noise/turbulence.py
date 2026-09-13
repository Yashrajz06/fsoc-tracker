"""Atmospheric turbulence: scintillation and beam wander.

Turbulence affects an optical link in two distinct ways, and we model both because they have
opposite implications for the tracker:

* **Scintillation** modulates received *intensity* over time. It threatens detection -- a deep
  fade can drop the beacon below the detection threshold entirely, which is what produces the
  dropouts the Kalman coast logic exists to survive. It does **not** move the spot.
* **Beam wander** displaces the apparent *position* of the beacon. It threatens pointing
  accuracy directly, and it is genuinely part of the target's motion rather than an image
  artefact, so it is applied to the true position before rendering -- meaning ground truth
  includes it.

Keeping these separate matters: conflating them would let an intensity model quietly move the
spot, producing tracking error attributable to nothing.

Intensity is modelled as a cheap multiplicative time series, not by phase-screen propagation.
Full split-step propagation is the physically complete approach but its compute cost is not
remotely justified for coarse-alignment simulation, where we need a plausible fade distribution
rather than a faithful wavefront.

Distributions follow the standard weak/strong turbulence split:

* **Log-normal** for weak turbulence (Rytov variance below ~1).
* **Gamma-Gamma** for moderate-to-strong turbulence, with ``alpha`` and ``beta`` derived from
  the Rytov variance. Indicative values: ``sigma_R^2 = 0.1`` gives alpha 20.8 / beta 19.8
  (near log-normal); ``= 1`` gives 2.95 / 2.46; ``= 10`` gives 2.48 / 0.98 (saturation regime).

See ``docs/DESIGN.md`` section 4.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "TurbulenceParams",
    "BeamWanderParams",
    "ScintillationModel",
    "BeamWander",
    "gamma_gamma_parameters",
]


def gamma_gamma_parameters(rytov_variance: float) -> Tuple[float, float]:
    """Derive Gamma-Gamma shape parameters from the Rytov variance.

    Uses the standard plane-wave relations::

        alpha = 1 / (exp(0.49 * s2 / (1 + 1.11 * s2^(6/5))^(7/6)) - 1)
        beta  = 1 / (exp(0.51 * s2 / (1 + 0.69 * s2^(6/5))^(5/6)) - 1)

    Args:
        rytov_variance: Rytov variance ``sigma_R^2``. Must be positive.

    Returns:
        ``(alpha, beta)``, the large- and small-scale scattering parameters.

    Raises:
        ValueError: If the Rytov variance is not positive.

    Examples:
        >>> a, b = gamma_gamma_parameters(1.0)
        >>> round(a, 2), round(b, 2)
        (2.95, 2.46)
    """
    if rytov_variance <= 0:
        raise ValueError(f"Rytov variance must be positive, got {rytov_variance}")
    s2 = float(rytov_variance)
    alpha = 1.0 / (math.exp(0.49 * s2 / (1.0 + 1.11 * s2 ** 1.2) ** (7.0 / 6.0)) - 1.0)
    beta = 1.0 / (math.exp(0.51 * s2 / (1.0 + 0.69 * s2 ** 1.2) ** (5.0 / 6.0)) - 1.0)
    return alpha, beta


@dataclass(frozen=True)
class BeamWanderParams:
    """Slow random displacement of the apparent beacon position.

    Modelled as an Ornstein-Uhlenbeck process, which is mean-reverting: unlike a random walk it
    does not drift away without bound, so the beacon wanders about its true position rather than
    escaping it.

    Attributes:
        enabled: Whether beam wander is active.
        theta: Mean-reversion rate, per second. Larger values pull back faster.
        sigma_px: Diffusion coefficient in pixels per sqrt(second).
    """

    enabled: bool = False
    theta: float = 0.2
    sigma_px: float = 5.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BeamWanderParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.turbulence.beam_wander`` block.

        Returns:
            The corresponding parameters.
        """
        return cls(enabled=bool(raw.get("enabled", False)),
                   theta=float(raw.get("theta", 0.2)),
                   sigma_px=float(raw.get("sigma_px", 5.0)))

    @property
    def stationary_sigma_px(self) -> float:
        """Long-run standard deviation of the displacement, ``sigma / sqrt(2*theta)``.

        This is the number to compare against the 10 px tracking budget: it says how far the
        beacon typically sits from its nominal position once the process has settled.
        """
        if self.theta <= 0:
            return float("inf")
        return self.sigma_px / math.sqrt(2.0 * self.theta)


@dataclass(frozen=True)
class TurbulenceParams:
    """Scintillation parameters.

    Attributes:
        enabled: Whether scintillation is active.
        model: ``"lognormal"`` or ``"gamma_gamma"``.
        rytov_variance: Rytov variance ``sigma_R^2``, setting the fade severity.
        beam_wander: Beam-wander sub-parameters.
    """

    enabled: bool = False
    model: str = "lognormal"
    rytov_variance: float = 0.1
    beam_wander: BeamWanderParams = BeamWanderParams()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TurbulenceParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.turbulence`` block.

        Returns:
            The corresponding parameters.
        """
        wander = raw.get("beam_wander", {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            model=str(raw.get("model", "lognormal")),
            rytov_variance=float(raw.get("rytov_variance", 0.1)),
            beam_wander=BeamWanderParams.from_mapping(
                wander if isinstance(wander, Mapping) else {}),
        )


class ScintillationModel:
    """Generates a multiplicative intensity time series.

    The series is normalised to unit mean, so turbulence redistributes intensity over time
    without changing the average received power. That keeps the Rytov variance the single knob
    controlling fade severity, rather than having it also dim the scene -- dimming is what the
    atmospheric model is for.
    """

    def __init__(self, params: TurbulenceParams,
                 rng: Optional[np.random.Generator] = None) -> None:
        """Initialise the scintillation model.

        Args:
            params: Turbulence parameters.
            rng: Seeded generator. One is created if omitted.

        Raises:
            ValueError: On an unknown model name.
        """
        if params.model not in ("lognormal", "gamma_gamma"):
            raise ValueError(f"Unknown scintillation model: {params.model!r}")
        self.params = params
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def scintillation_index(self) -> float:
        """Normalised intensity variance, ``var(I)/mean(I)^2``.

        For log-normal this is ``exp(sigma_R^2) - 1``; for Gamma-Gamma it is
        ``1/alpha + 1/beta + 1/(alpha*beta)``. This is the standard severity measure and is what
        a test should check against, rather than the raw distribution parameters.
        """
        if self.params.model == "lognormal":
            return math.exp(self.params.rytov_variance) - 1.0
        alpha, beta = gamma_gamma_parameters(self.params.rytov_variance)
        return 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)

    def sample(self) -> float:
        """Draw one unit-mean intensity scale factor.

        Returns:
            A positive multiplier. 1.0 exactly when turbulence is disabled.
        """
        if not self.params.enabled:
            return 1.0

        if self.params.model == "lognormal":
            # Choose mu so that E[exp(N(mu, s2))] = 1, i.e. mu = -s2/2.
            s2 = self.params.rytov_variance
            return float(np.exp(self._rng.normal(-s2 / 2.0, math.sqrt(s2))))

        alpha, beta = gamma_gamma_parameters(self.params.rytov_variance)
        # Gamma-Gamma is the product of two unit-mean Gamma variates.
        large = self._rng.gamma(alpha, 1.0 / alpha)
        small = self._rng.gamma(beta, 1.0 / beta)
        return float(large * small)

    def sample_many(self, count: int) -> np.ndarray:
        """Draw a series of intensity scale factors.

        Args:
            count: Number of samples.

        Returns:
            A ``float64`` array of positive multipliers.
        """
        return np.array([self.sample() for _ in range(count)], dtype=np.float64)


class BeamWander:
    """Ornstein-Uhlenbeck displacement of the apparent beacon position.

    Applied to the *true* position before rendering, so ground truth reflects where the beacon
    actually appeared. Beam wander is a real displacement of the received beam, not an imaging
    artefact -- treating it as one would mean measuring the tracker against a position the light
    never came from.
    """

    def __init__(self, params: BeamWanderParams,
                 rng: Optional[np.random.Generator] = None) -> None:
        """Initialise the beam-wander process at zero displacement.

        Args:
            params: Beam-wander parameters.
            rng: Seeded generator. One is created if omitted.
        """
        self.params = params
        self._rng = rng if rng is not None else np.random.default_rng()
        self._dx = 0.0
        self._dy = 0.0

    @property
    def offset(self) -> Tuple[float, float]:
        """Current displacement as ``(dx, dy)`` in pixels."""
        return self._dx, self._dy

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance the process by ``dt`` seconds.

        Args:
            dt: Time increment in seconds. Must be non-negative.

        Returns:
            The new displacement as ``(dx, dy)`` in pixels.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        if not self.params.enabled or dt == 0:
            return self.offset

        theta, sigma = self.params.theta, self.params.sigma_px
        noise = self._rng.normal(0.0, 1.0, size=2) * sigma * math.sqrt(dt)
        self._dx += -theta * self._dx * dt + noise[0]
        self._dy += -theta * self._dy * dt + noise[1]
        return self.offset

    def reset(self) -> None:
        """Return the displacement to zero."""
        self._dx = 0.0
        self._dy = 0.0
