"""Atmospheric degradation via the Koschmieder scattering model.

The standard atmospheric scattering model in computer vision::

    I(x) = J(x) * t(x) + A * (1 - t(x))

where ``J`` is the clear scene, ``A`` is the atmospheric light (airlight) and ``t`` is
transmission. Under Beer-Lambert attenuation ``t = exp(-beta * d)``. For our 2-D scene the depth
``d`` is uniform (optionally gently graded), so this reduces to a per-pixel affine blend --
trivially fast, and comfortably real-time at well over 30 FPS.

Physically this is a *contrast reduction*, not merely a brightness change: the scene is scaled
toward zero contrast and offset toward the airlight level. That distinction matters for us,
because it is exactly the regime where a fixed intensity threshold fails and an adaptive one
survives. Fog with high airlight can leave the beacon *dimmer in contrast* while the frame as a
whole gets *brighter* -- a combination that breaks any absolute threshold.

A second consequence worth naming: raising airlight pushes the whole frame up the 8-bit range,
so a bright beacon can be driven into saturation by fog that has actually reduced its contrast.
Saturation flattens the spot's peak and biases the centroid toward the geometric centre of the
saturated region -- the sibling of the Phase 1 edge-clipping finding (``docs/DESIGN.md``
section 2).

See ``docs/DESIGN.md`` section 4.2 and specification parameter 24.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.noise.sensor import MAX_LEVEL, as_float

__all__ = ["AtmosphericParams", "apply_atmosphere", "transmission"]


@dataclass(frozen=True)
class AtmosphericParams:
    """Koschmieder-model parameters for one atmospheric preset.

    Attributes:
        enabled: Whether the stage is active.
        beta: Extinction coefficient. ``t = exp(-beta * depth)``; larger values mean lower
            transmission and therefore lower contrast.
        airlight: Atmospheric light level ``A``, in grey levels. This is what the scene fades
            *toward*, so a high airlight brightens the frame while reducing contrast.
        blur_sigma: Optional Gaussian blur in pixels, modelling forward scattering. Note this
            widens the spot, which changes the runtime spot-scale estimate -- correctly, since
            the observed spot really is wider.
        brightness_scale: Overall multiplicative dimming applied before scattering, used for the
            low-light preset.
        depth: Uniform scene depth used in the Beer-Lambert term.
        streaks: Overlay sparse bright streaks, modelling rain.
        streak_density: Fraction of frame rows carrying a streak.
        streak_intensity: Peak grey level of a streak.
    """

    enabled: bool = True
    beta: float = 0.0
    airlight: float = 0.0
    blur_sigma: float = 0.0
    brightness_scale: float = 1.0
    depth: float = 1.0
    streaks: bool = False
    streak_density: float = 0.004
    streak_intensity: float = 200.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AtmosphericParams":
        """Build from a preset mapping, ignoring documentation keys.

        Args:
            raw: A single preset block from ``noise.atmospheric.presets``.

        Returns:
            The corresponding parameters.
        """
        return cls(
            enabled=bool(raw.get("enabled", True)),
            beta=float(raw.get("beta", 0.0)),
            airlight=float(raw.get("airlight", 0.0)),
            blur_sigma=float(raw.get("blur_sigma", 0.0)),
            brightness_scale=float(raw.get("brightness_scale", 1.0)),
            depth=float(raw.get("depth", 1.0)),
            streaks=bool(raw.get("streaks", False)),
            streak_density=float(raw.get("streak_density", 0.004)),
            streak_intensity=float(raw.get("streak_intensity", 200.0)),
        )

    @property
    def transmission(self) -> float:
        """Scene transmission ``t = exp(-beta * depth)``, in ``(0, 1]``."""
        return transmission(self.beta, self.depth)

    @property
    def contrast_retained(self) -> float:
        """Fraction of original scene contrast surviving this atmosphere.

        Equal to the transmission: the Koschmieder blend scales scene *differences* by ``t``,
        so a transmission of 0.5 halves every contrast in the frame regardless of airlight.
        """
        return self.transmission


def transmission(beta: float, depth: float = 1.0) -> float:
    """Compute Beer-Lambert transmission.

    Args:
        beta: Extinction coefficient. Must be non-negative.
        depth: Path depth through the medium.

    Returns:
        Transmission in ``(0, 1]``.

    Raises:
        ValueError: If ``beta`` or ``depth`` is negative.
    """
    if beta < 0:
        raise ValueError(f"Extinction coefficient must be non-negative, got {beta}")
    if depth < 0:
        raise ValueError(f"Depth must be non-negative, got {depth}")
    return float(np.exp(-beta * depth))


def _gaussian_blur(frame: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply a separable Gaussian blur with edge replication.

    Args:
        frame: 2-D array to blur.
        sigma_px: Blur standard deviation in pixels.

    Returns:
        The blurred array, same shape and dtype family as the input.
    """
    if sigma_px <= 0:
        return frame
    radius = max(1, int(np.ceil(3.0 * sigma_px)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)
    kernel /= kernel.sum()

    padded = np.pad(frame, ((0, 0), (radius, radius)), mode="edge")
    blurred = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0,
                               padded).astype(np.float32)


def _add_streaks(frame: np.ndarray, params: AtmosphericParams,
                 rng: np.random.Generator) -> np.ndarray:
    """Overlay sparse near-vertical bright streaks, modelling rain.

    Args:
        frame: Frame to draw onto. Modified in place.
        params: Atmospheric parameters supplying streak density and intensity.
        rng: Seeded generator.

    Returns:
        The frame, with streaks added.
    """
    height, width = frame.shape
    count = int(round(params.streak_density * height * width / max(height, 1)))
    count = max(count, 0)
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, max(1, height - 20)))
        length = int(rng.integers(8, 25))
        lean = int(rng.integers(-2, 3))
        for step in range(length):
            yy = y + step
            xx = x + (lean * step) // max(length, 1)
            if 0 <= yy < height and 0 <= xx < width:
                frame[yy, xx] = min(MAX_LEVEL, frame[yy, xx] + params.streak_intensity)
    return frame


def apply_atmosphere(frame: np.ndarray, params: AtmosphericParams,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply atmospheric degradation to a frame.

    Order of operations: dim, blur, scatter, then streaks. Blurring *before* scattering is
    deliberate -- forward scattering spreads light from the scene itself, whereas airlight is
    an additive veil that is not part of what gets blurred. Streaks are added last because rain
    is in the foreground, between the scene and the sensor.

    Args:
        frame: Input frame. Not modified.
        params: Atmospheric parameters.
        rng: Seeded generator, used only when ``streaks`` is enabled.

    Returns:
        A new ``float32`` frame, unclipped.

    Raises:
        ValueError: If ``airlight`` is outside the representable range or
            ``brightness_scale`` is negative.
    """
    if not 0.0 <= params.airlight <= MAX_LEVEL:
        raise ValueError(f"Airlight must be in [0, {MAX_LEVEL}], got {params.airlight}")
    if params.brightness_scale < 0:
        raise ValueError(
            f"Brightness scale must be non-negative, got {params.brightness_scale}")

    out = as_float(frame).copy()
    if not params.enabled:
        return out

    if params.brightness_scale != 1.0:
        out *= params.brightness_scale
    if params.blur_sigma > 0:
        out = _gaussian_blur(out, params.blur_sigma)

    t = params.transmission
    if t < 1.0:
        out = out * t + params.airlight * (1.0 - t)

    if params.streaks:
        rng = rng if rng is not None else np.random.default_rng()
        out = _add_streaks(out, params, rng)
    return out
