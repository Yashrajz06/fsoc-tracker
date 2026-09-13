"""Configurable composition of the noise and degradation stages.

The pipeline applies enabled stages in a configured order and quantises to ``uint8`` exactly
once, at the end. Two decisions here are load-bearing:

**Order is configurable, and it matters physically.** Atmospheric scattering happens in the
world, before the light reaches the sensor; sensor noise happens in the detector. Running
salt-and-pepper *before* the atmosphere would have fog attenuate the impulse noise, which is
backwards -- impulses originate in the readout electronics. The default order in
``config/default.json`` follows the physical path: turbulence, atmosphere, then sensor effects.

**Quantisation happens once.** Each stage works in ``float32`` and only the final result is
rounded to 8 bits. Rounding after every stage would accumulate quantisation error across a
five-stage pipeline, which at low signal levels is a meaningful fraction of the noise being
modelled.

The pipeline also reports **saturation**: the fraction of pixels driven to the 8-bit ceiling.
Saturation flattens a bright spot's peak and biases the intensity-weighted centroid toward the
geometric centre of the saturated region -- a smooth, systematic, noise-free error with the same
shape as the Phase 1 edge-clipping finding. Surfacing it here means Phase 3 can set the
``saturated`` detection flag from a measured quantity rather than a guess.

See ``docs/DESIGN.md`` section 4 and specification parameters 21-25.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.config import AppConfig
from src.noise.atmospheric import AtmosphericParams, apply_atmosphere
from src.noise.sensor import (
    MAX_LEVEL,
    GaussianNoiseParams,
    PoissonNoiseParams,
    SaltPepperParams,
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_pepper,
    as_float,
    to_uint8,
)
from src.noise.turbulence import BeamWander, ScintillationModel, TurbulenceParams

__all__ = ["NoisePipeline", "NoiseResult", "KNOWN_STAGES"]

#: Stages the pipeline knows how to apply, in their default physical order.
KNOWN_STAGES: Tuple[str, ...] = ("turbulence", "atmospheric", "gaussian", "poisson",
                                 "salt_pepper")


@dataclass(frozen=True)
class NoiseResult:
    """A degraded frame plus the diagnostics needed to interpret it.

    Attributes:
        frame: The degraded ``uint8`` frame.
        saturated_fraction: Fraction of *frame* pixels at the 8-bit ceiling. Reported because
            saturation biases the centroid by a known systematic mechanism, not by noise -- see
            the module docstring and ``docs/DESIGN.md`` section 2.

            **Read this as a scene-level indicator, not as the Phase 3 detection flag.** Salt
            impulses also sit at the ceiling, so with salt-and-pepper enabled this figure has a
            floor of ``density * salt_ratio`` regardless of how bright the beacon is -- 0.025 at
            the default 0.05 density. The actionable ``saturated`` flag is computed per *blob*
            in Phase 3, where impulse pixels have already been removed by the median prefilter
            and the measurement is confined to the target.
        intensity_scale: Scintillation multiplier applied to the beacon this frame. A deep fade
            is the usual cause of a dropout, so logging it makes a lost lock attributable.
        stages_applied: Names of the stages that actually ran, in order.
    """

    frame: np.ndarray
    saturated_fraction: float = 0.0
    intensity_scale: float = 1.0
    stages_applied: Tuple[str, ...] = ()

    @property
    def is_saturated(self) -> bool:
        """Whether any pixel reached the 8-bit ceiling."""
        return self.saturated_fraction > 0.0


class NoisePipeline:
    """Applies the configured degradation stages to frames.

    The pipeline is stateful only where physics demands it: scintillation and beam wander evolve
    over time. The per-frame image operations themselves remain pure functions.

    Attributes:
        order: Stage names in the order they will be applied.
    """

    def __init__(self, order: List[str],
                 gaussian: GaussianNoiseParams,
                 poisson: PoissonNoiseParams,
                 salt_pepper: SaltPepperParams,
                 atmospheric: AtmosphericParams,
                 turbulence: TurbulenceParams,
                 enabled: bool = True,
                 rng: Optional[np.random.Generator] = None) -> None:
        """Initialise the pipeline.

        Args:
            order: Stage names to apply, in order.
            gaussian: Additive Gaussian noise parameters.
            poisson: Shot noise parameters.
            salt_pepper: Impulse noise parameters.
            atmospheric: Koschmieder-model parameters.
            turbulence: Scintillation and beam-wander parameters.
            enabled: Master switch. When false the pipeline only quantises.
            rng: Seeded generator shared by every stochastic stage, so a whole run is
                reproducible from one seed.

        Raises:
            ValueError: If ``order`` names a stage the pipeline does not implement.
        """
        unknown = [name for name in order if name not in KNOWN_STAGES]
        if unknown:
            raise ValueError(
                f"Unknown noise stage(s) {unknown}. Known: {list(KNOWN_STAGES)}")

        self.order = list(order)
        self.enabled = enabled
        self.gaussian = gaussian
        self.poisson = poisson
        self.salt_pepper = salt_pepper
        self.atmospheric = atmospheric
        self.turbulence = turbulence
        self._rng = rng if rng is not None else np.random.default_rng()
        self.scintillation = ScintillationModel(turbulence, self._rng)
        self.beam_wander = BeamWander(turbulence.beam_wander, self._rng)

    @classmethod
    def from_config(cls, config: AppConfig,
                    rng: Optional[np.random.Generator] = None) -> "NoisePipeline":
        """Build a pipeline from an application configuration.

        Args:
            config: Validated application configuration.
            rng: Seeded generator. Derived from ``config.run.random_seed`` when omitted.

        Returns:
            A configured :class:`NoisePipeline`.
        """
        if rng is None:
            rng = np.random.default_rng(config.run.random_seed)
        noise = config.noise
        order = list(noise.pipeline_order) or list(KNOWN_STAGES)
        return cls(
            order=order,
            gaussian=GaussianNoiseParams.from_mapping(noise.gaussian),
            poisson=PoissonNoiseParams.from_mapping(noise.poisson),
            salt_pepper=SaltPepperParams.from_mapping(noise.salt_pepper),
            atmospheric=AtmosphericParams.from_mapping(noise.atmospheric_preset),
            turbulence=TurbulenceParams.from_mapping(noise.turbulence),
            enabled=noise.enabled,
            rng=rng,
        )

    def next_intensity_scale(self) -> float:
        """Draw this frame's scintillation multiplier.

        Called by the scene *before* rendering, since scintillation modulates the beacon's
        emitted intensity rather than the captured image.

        Returns:
            A positive multiplier, 1.0 when turbulence is disabled.
        """
        if not self.enabled:
            return 1.0
        return self.scintillation.sample()

    def next_beam_wander(self, dt: float) -> Tuple[float, float]:
        """Advance beam wander and return the current displacement.

        Applied to the *true* beacon position before rendering, so ground truth reflects where
        the beacon actually appeared.

        Args:
            dt: Time increment in seconds.

        Returns:
            The displacement as ``(dx, dy)`` in pixels.
        """
        if not self.enabled:
            return 0.0, 0.0
        return self.beam_wander.step(dt)

    def apply(self, frame: np.ndarray, intensity_scale: float = 1.0) -> NoiseResult:
        """Apply the configured degradation stages to a frame.

        Args:
            frame: Input frame. Not modified.
            intensity_scale: The scintillation multiplier already applied when rendering,
                recorded in the result for logging.

        Returns:
            A :class:`NoiseResult` holding the degraded frame and its diagnostics.
        """
        working = as_float(frame).copy()
        applied: List[str] = []

        if self.enabled:
            for stage in self.order:
                if stage == "gaussian" and self.gaussian.enabled:
                    working = add_gaussian_noise(working, self.gaussian, self._rng)
                elif stage == "poisson" and self.poisson.enabled:
                    working = add_poisson_noise(working, self.poisson, self._rng)
                elif stage == "salt_pepper" and self.salt_pepper.enabled:
                    working = add_salt_pepper(working, self.salt_pepper, self._rng)
                elif stage == "atmospheric" and self.atmospheric.enabled:
                    working = apply_atmosphere(working, self.atmospheric, self._rng)
                elif stage == "turbulence" and self.turbulence.enabled:
                    # Scintillation acts on the beacon at render time, not on the captured
                    # frame, so there is no image operation here. It is still recorded as
                    # applied, because it did affect this frame -- via the intensity scale the
                    # scene used when rendering -- and a dropout traced to a deep fade needs
                    # that to be visible in the log.
                    pass
                else:
                    continue
                applied.append(stage)

        # Measure saturation before quantising, so pixels pushed above the ceiling are counted
        # even though uint8 will pin them at exactly 255.
        saturated = float(np.count_nonzero(working >= MAX_LEVEL) / working.size)
        return NoiseResult(frame=to_uint8(working),
                           saturated_fraction=saturated,
                           intensity_scale=intensity_scale,
                           stages_applied=tuple(applied))

    def reset(self) -> None:
        """Return the time-evolving stages to their initial state."""
        self.beam_wander.reset()
