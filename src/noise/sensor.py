"""Sensor noise models: Gaussian, Poisson and salt-and-pepper.

All noise stages are pure functions of the form ``(frame, params, rng) -> frame``. They never
mutate their input and never hold state, which makes them trivially composable in any order
(:mod:`src.noise.pipeline`), independently testable, and safe to apply to a shared buffer.

The ``rng`` argument is explicit rather than global. Every run must be reproducible from
``config.run.random_seed`` for the technical report, and a module-level generator would silently
couple the noise sequence to call order elsewhere in the program.

Precision: stages operate in ``float32`` internally and only quantise back to ``uint8`` at the
end of the pipeline. Rounding to 8 bits after *every* stage would accumulate quantisation error
across a five-stage pipeline, which at low signal levels is a meaningful fraction of the noise
being modelled.

See ``docs/DESIGN.md`` section 4.1 and specification parameters 21-22.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "GaussianNoiseParams",
    "PoissonNoiseParams",
    "SaltPepperParams",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_salt_pepper",
    "as_float",
    "to_uint8",
]

#: Maximum representable grey level. Saturation at this value flattens a bright spot's peak and
#: biases the intensity-weighted centroid toward the geometric centre of the saturated region --
#: see ``docs/DESIGN.md`` section 2 and the Phase 2 saturation-bias test.
MAX_LEVEL: float = 255.0


def as_float(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to ``float32`` for intermediate processing.

    Args:
        frame: Input frame of any numeric dtype.

    Returns:
        A ``float32`` copy, or the input unchanged if it is already ``float32``.
    """
    if frame.dtype == np.float32:
        return frame
    return frame.astype(np.float32)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Quantise a float frame back to ``uint8``, clipping to the representable range.

    Call this once, at the end of a pipeline. Clipping here is what produces saturation, so a
    frame emerging from this function may contain pixels pinned at 255 whose true value was
    higher -- information that cannot be recovered downstream.

    Args:
        frame: Float frame.

    Returns:
        A ``uint8`` frame with values clipped to ``[0, 255]``.
    """
    return np.clip(frame, 0.0, MAX_LEVEL).astype(np.uint8)


@dataclass(frozen=True)
class GaussianNoiseParams:
    """Additive white Gaussian noise (specification parameter 21).

    Attributes:
        enabled: Whether the stage is active.
        sigma: Noise standard deviation in grey levels. Specification parameter 22 caps this at
            20; the cap is enforced in :mod:`src.config`, not here, so this function remains a
            pure numerical primitive usable in sweeps beyond the spec range.
    """

    enabled: bool = True
    sigma: float = 10.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GaussianNoiseParams":
        """Build from a configuration mapping, ignoring documentation keys.

        Args:
            raw: The ``noise.gaussian`` configuration block.

        Returns:
            The corresponding parameters.
        """
        return cls(enabled=bool(raw.get("enabled", True)),
                   sigma=float(raw.get("sigma", 10.0)))


@dataclass(frozen=True)
class PoissonNoiseParams:
    """Shot noise (specification parameter 21).

    Physically the correct model for photon counting: the variance equals the mean, so bright
    regions carry more absolute noise than dark ones. This is why centroiding precision scales
    as ``1/sqrt(N_photons)`` in the shot-noise limit (``docs/DESIGN.md`` section 5.3).

    Attributes:
        enabled: Whether the stage is active.
        scale: Photon-conversion factor. Values above 1 mean more photons per grey level, giving
            *less* relative noise; values below 1 mean fewer photons and more relative noise.
            This is the knob that sets the effective SNR of a low-light scene.
    """

    enabled: bool = False
    scale: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PoissonNoiseParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.poisson`` configuration block.

        Returns:
            The corresponding parameters.
        """
        return cls(enabled=bool(raw.get("enabled", False)),
                   scale=float(raw.get("scale", 1.0)))


@dataclass(frozen=True)
class SaltPepperParams:
    """Impulse noise (specification parameter 21, up to 10% of the image).

    This is the **primary false-positive threat** to centroiding: a single salt pixel is a
    maximum-intensity point that survives thresholding and, if it reaches the centroid stage,
    drags the estimate an arbitrary distance. It is the reason the median prefilter is mandatory
    and the reason the Kalman validation gate exists.

    Attributes:
        enabled: Whether the stage is active.
        density: Fraction of pixels corrupted, in ``[0, 1]``.
        salt_ratio: Fraction of corrupted pixels set to maximum rather than zero.
    """

    enabled: bool = False
    density: float = 0.05
    salt_ratio: float = 0.5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SaltPepperParams":
        """Build from a configuration mapping.

        Args:
            raw: The ``noise.salt_pepper`` configuration block.

        Returns:
            The corresponding parameters.
        """
        return cls(enabled=bool(raw.get("enabled", False)),
                   density=float(raw.get("density", 0.05)),
                   salt_ratio=float(raw.get("salt_ratio", 0.5)))


def add_gaussian_noise(frame: np.ndarray, params: GaussianNoiseParams,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add zero-mean Gaussian noise to a frame.

    The result is **not** clipped: clipping is deferred to :func:`to_uint8` at the end of the
    pipeline. Clipping here would bias the noise non-zero-mean wherever the signal is near 0 or
    255, which would quietly corrupt the statistical tests that validate this function.

    Args:
        frame: Input frame. Not modified.
        params: Noise parameters.
        rng: Seeded generator. One is created if omitted, in which case the result is not
            reproducible.

    Returns:
        A new ``float32`` frame with noise added, unclipped.

    Raises:
        ValueError: If ``sigma`` is negative.
    """
    if params.sigma < 0:
        raise ValueError(f"Gaussian sigma must be non-negative, got {params.sigma}")
    out = as_float(frame).copy()
    if not params.enabled or params.sigma == 0:
        return out
    rng = rng if rng is not None else np.random.default_rng()
    return out + rng.normal(0.0, params.sigma, size=out.shape).astype(np.float32)


def add_poisson_noise(frame: np.ndarray, params: PoissonNoiseParams,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply signal-dependent Poisson (shot) noise.

    The frame is interpreted as an expected photon count scaled by ``scale``, sampled, and
    scaled back. The result is therefore unbiased -- its expectation is the input -- while its
    variance grows with the signal.

    Args:
        frame: Input frame. Not modified. Negative values are treated as zero, since a negative
            photon count is meaningless.
        params: Noise parameters.
        rng: Seeded generator. One is created if omitted.

    Returns:
        A new ``float32`` frame.

    Raises:
        ValueError: If ``scale`` is not positive.
    """
    if params.scale <= 0:
        raise ValueError(f"Poisson scale must be positive, got {params.scale}")
    out = as_float(frame).copy()
    if not params.enabled:
        return out
    rng = rng if rng is not None else np.random.default_rng()
    expected = np.maximum(out, 0.0) * params.scale
    return (rng.poisson(expected).astype(np.float32) / params.scale)


def add_salt_pepper(frame: np.ndarray, params: SaltPepperParams,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Corrupt a fraction of pixels to the extremes of the range.

    Pixels are chosen without replacement, so the realised density matches the requested density
    exactly rather than approximately. Sampling each pixel independently would give a binomial
    spread around the target, which makes the density test below unnecessarily loose and, more
    importantly, makes a scenario's stated noise level only approximately what was asked for.

    Args:
        frame: Input frame. Not modified.
        params: Noise parameters.
        rng: Seeded generator. One is created if omitted.

    Returns:
        A new ``float32`` frame with impulses applied.

    Raises:
        ValueError: If ``density`` or ``salt_ratio`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= params.density <= 1.0:
        raise ValueError(f"Salt-and-pepper density must be in [0, 1], got {params.density}")
    if not 0.0 <= params.salt_ratio <= 1.0:
        raise ValueError(f"Salt ratio must be in [0, 1], got {params.salt_ratio}")

    out = as_float(frame).copy()
    if not params.enabled or params.density == 0:
        return out

    rng = rng if rng is not None else np.random.default_rng()
    total = out.size
    count = int(round(params.density * total))
    if count == 0:
        return out

    flat_indices = rng.choice(total, size=count, replace=False)
    salt_count = int(round(count * params.salt_ratio))
    flat = out.reshape(-1)
    flat[flat_indices[:salt_count]] = MAX_LEVEL
    flat[flat_indices[salt_count:]] = 0.0
    return out
