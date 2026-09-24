"""Typed configuration loading, validation and derived quantities.

Every tunable value in the system lives in ``config/default.json`` and reaches the rest of the
codebase through the dataclasses defined here. Nothing tunable may be a literal in any other
module (see ``CLAUDE.md`` -> Working conventions).

This module carries three responsibilities beyond plain deserialisation, and each exists to
defuse a specific, identified failure mode:

1. **Scale-relative vision geometry.** Kernel sizes, blob-area gates, the centroid window and the
   ROI size are expressed as *multiples of the beacon spot scale* (its FWHM in pixels), which is
   estimated at runtime. Absolute pixel values survive only as fallbacks for when that estimation
   fails. Benchmark Performance-2 runs on evaluator video whose resolution and spot size we
   cannot predict; absolute pixel geometry tuned to a 640x480 frame with a 10 px spot is the
   single most likely cause of failure there. See :meth:`VisionConfig.resolve_geometry`.

2. **Derived quantities are computed, never stored.** ``deg_per_pixel``, the per-frame slew
   ceiling and the search arm spacing are properties. A stored literal goes stale silently the
   moment somebody changes the FOV or resolution, and a stale angular scale corrupts every
   downstream metric without ever raising an error.

3. **Startup validation against the problem statement.** Spec limits (noise sigma <= 20, jitter
   <= 20 px/frame, slew 5-10 deg/s, target 5-20 px, salt-and-pepper density <= 0.10) are enforced
   on load, so an evaluator-supplied scenario file cannot silently put us out of spec.

The module additionally computes the **worst-case spiral search time** for the active
configuration and warns at startup when it exceeds the specified acquisition budget. For the
default 2000x2000 canvas at the 5 deg/s slew ceiling that bound is roughly 11 s against a 2 s
requirement. That is arithmetic set by canvas area, FOV and slew rate alone -- no algorithm
tuning changes it -- so we surface it from the first run rather than discover it in Phase 4.
See ``docs/DESIGN.md`` section 7.5.

Coordinate convention used throughout: pixel centres sit at integer indices, origin at the
top-left pixel, coordinate tuples ordered ``(x, y)``. See ``CLAUDE.md`` -> Coordinate conventions.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Type, TypeVar)

__all__ = [
    "ConfigError",
    "AppConfig",
    "RunConfig",
    "SceneConfig",
    "CameraConfig",
    "TargetConfig",
    "NoiseConfig",
    "VisionConfig",
    "ResolvedVisionGeometry",
    "FilteringConfig",
    "ControlConfig",
    "TelemetryConfig",
    "VideoInputConfig",
    "GuiConfig",
    "PerformanceConfig",
    "AiConfig",
    "load_config",
    "merge_overrides",
    "load_json_document",
]


# --------------------------------------------------------------------------------------------
# Spec limits. These are facts from docs/PROBLEM_STATEMENT.md, not tunables, so they are module
# constants rather than config entries -- a config file must not be able to relax the spec.
# --------------------------------------------------------------------------------------------

SPEC_MAX_NOISE_SIGMA: float = 20.0           # Parameter 22
SPEC_MAX_SALT_PEPPER_DENSITY: float = 0.10   # Parameter 21
SPEC_MAX_JITTER_PX_PER_FRAME: float = 20.0   # Parameter 23
SPEC_MAX_PLATFORM_PX_PER_FRAME: float = 20.0  # Parameter 25
SPEC_SLEW_RANGE_DEG_S: Tuple[float, float] = (5.0, 10.0)   # Parameters 13, 14
SPEC_TARGET_SIZE_RANGE_PX: Tuple[int, int] = (5, 20)       # Parameter 10
SPEC_MIN_SCENE_PX: int = 2000                # Parameter 1
SPEC_MIN_CAMERA_RATE_HZ: float = 30.0        # Parameter 5
SPEC_MIN_CONTROL_RATE_HZ: float = 20.0       # Parameter 15

#: Conversion from Gaussian sigma to full width at half maximum: ``2 * sqrt(2 * ln 2)``.
FWHM_PER_SIGMA: float = 2.0 * math.sqrt(2.0 * math.log(2.0))


class ConfigError(ValueError):
    """Raised when a configuration is structurally invalid or violates a specification limit.

    Distinct from a warning: a :class:`ConfigError` means the run cannot proceed meaningfully,
    whereas an out-of-budget derived quantity (such as worst-case search time) is reported as a
    warning because the run is still valid and the number is a finding worth logging.
    """


T = TypeVar("T")


# --------------------------------------------------------------------------------------------
# Deserialisation helpers
# --------------------------------------------------------------------------------------------


def _strip_comments(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop documentation keys from a config mapping.

    Keys beginning with an underscore carry commentary, spec references and derivations for human
    readers. They are never fields on a dataclass.

    Args:
        raw: A mapping parsed from JSON.

    Returns:
        A new dict containing only the keys that do not start with ``"_"``.
    """
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _build(cls: Type[T], raw: Mapping[str, Any], path: str) -> T:
    """Construct a dataclass from a mapping, rejecting unknown keys.

    Unknown keys are an error rather than being ignored. A silently ignored key is a typo that
    looks like it took effect -- the most expensive kind of configuration bug, because the run
    completes and produces plausible but wrong numbers.

    Args:
        cls: The dataclass type to construct.
        raw: Mapping of field name to value, comments already stripped.
        path: Dotted path of this block within the config, used in error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigError: If ``raw`` contains a key that is not a field of ``cls``, or omits a
            required field.
    """
    if not is_dataclass(cls):  # pragma: no cover - programming error, not user error
        raise TypeError(f"{cls!r} is not a dataclass")

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown configuration key(s) in '{path}': {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )
    try:
        return cls(**raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration block '{path}': {exc}") from exc


def _section(raw: Mapping[str, Any], name: str, path: str = "") -> Dict[str, Any]:
    """Extract and comment-strip a named sub-block.

    Args:
        raw: Parent mapping.
        name: Key of the sub-block.
        path: Dotted path of the parent, for error messages.

    Returns:
        The comment-stripped sub-block, or an empty dict when absent (so that every block can
        fall back entirely to its dataclass defaults).
    """
    full = f"{path}.{name}" if path else name
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration block '{full}' must be an object, got {type(value).__name__}")
    return _strip_comments(value)


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigError` when ``condition`` is false.

    Args:
        condition: Predicate that must hold.
        message: Explanation shown to the user, which should state both what is wrong and what
            the acceptable range is.

    Raises:
        ConfigError: When ``condition`` is false.
    """
    if not condition:
        raise ConfigError(message)


def _odd(value: float, minimum: int, maximum: int) -> int:
    """Round to the nearest odd integer within an inclusive bound.

    Morphological structuring elements and centroid windows must be odd so that they have a
    single defined centre pixel; an even window has its centre on a pixel *boundary*, which
    introduces exactly the half-pixel bias the coordinate convention exists to prevent.

    Args:
        value: Desired size in pixels.
        minimum: Smallest permitted size. Rounded up to odd if even.
        maximum: Largest permitted size. Rounded down to odd if even.

    Returns:
        An odd integer in ``[minimum, maximum]``.
    """
    lo = minimum if minimum % 2 == 1 else minimum + 1
    hi = maximum if maximum % 2 == 1 else maximum - 1
    n = int(round(value))
    if n % 2 == 0:
        n += 1
    return max(lo, min(hi, n))


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive interval ``[low, high]``.

    Args:
        value: Input value.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped value.
    """
    return max(low, min(high, value))


# --------------------------------------------------------------------------------------------
# Section dataclasses
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Top-level run settings.

    Attributes:
        mode: ``"simulation"`` (Mode A) or ``"video"`` (Mode B).
        duration_seconds: Wall-clock duration of a simulation run. Ignored in video mode, where
            the file length governs.
        random_seed: Seed for every RNG in the system, so runs are reproducible for the report.
        headless: Suppress the GUI and run compute-only.
    """

    mode: str = "simulation"
    duration_seconds: float = 60.0
    random_seed: int = 42
    headless: bool = False

    def validate(self) -> None:
        """Check run settings against permitted values.

        Raises:
            ConfigError: On an unknown mode or a non-positive duration.
        """
        _require(self.mode in ("simulation", "video"),
                 f"run.mode must be 'simulation' or 'video', got {self.mode!r}")
        _require(self.duration_seconds > 0,
                 f"run.duration_seconds must be positive, got {self.duration_seconds}")


@dataclass(frozen=True)
class SceneConfig:
    """Virtual world canvas.

    Attributes:
        width: Canvas width in pixels. Spec parameter 1 requires at least 2000.
        height: Canvas height in pixels.
        background_level: Uniform background grey level in ``[0, 255]``.
        background_gradient: Apply a gentle spatial gradient to the background, which exercises
            the top-hat background suppression.
    """

    width: int = 2000
    height: int = 2000
    background_level: int = 10
    background_gradient: bool = False

    @property
    def area_px(self) -> float:
        """Total canvas area in pixels, the uncertainty region a search must sweep."""
        return float(self.width) * float(self.height)

    def validate(self) -> None:
        """Check canvas dimensions against the specification minimum.

        Raises:
            ConfigError: If either dimension is below the 2000 px spec minimum, or the background
                level is outside the 8-bit range.
        """
        _require(self.width >= SPEC_MIN_SCENE_PX and self.height >= SPEC_MIN_SCENE_PX,
                 f"scene must be at least {SPEC_MIN_SCENE_PX}x{SPEC_MIN_SCENE_PX} px per spec "
                 f"parameter 1, got {self.width}x{self.height}")
        _require(0 <= self.background_level <= 255,
                 f"scene.background_level must be in [0, 255], got {self.background_level}")


@dataclass(frozen=True)
class CameraConfig:
    """Virtual pan/tilt camera and its mechanical limits.

    All angular scale factors are computed properties rather than stored values. Storing
    ``deg_per_pixel`` would let it drift out of agreement with ``fov`` and ``resolution``
    silently, corrupting every angular metric downstream without raising an error.

    Attributes:
        resolution_width: Viewport width in pixels.
        resolution_height: Viewport height in pixels.
        fov_horizontal_deg: Horizontal field of view in degrees.
        fov_vertical_deg: Vertical field of view in degrees.
        monochrome: Whether the sensor is single-channel.
        update_rate_hz: Frame generation clock. Spec parameter 5 requires >= 30 Hz.
        initial_position: ``"center"``, ``"random"`` or ``"custom"``.
        initial_pan_px: Custom initial boresight x in canvas coordinates.
        initial_tilt_px: Custom initial boresight y in canvas coordinates.
        max_pan_speed_deg_s: Slew ceiling in pan. Spec parameter 13 allows 5-10 deg/s.
        max_tilt_speed_deg_s: Slew ceiling in tilt. Spec parameter 14 allows 5-10 deg/s.
        max_acceleration_deg_s2: Angular acceleration limit of the simulated gimbal.
    """

    resolution_width: int = 640
    resolution_height: int = 480
    fov_horizontal_deg: float = 4.0
    fov_vertical_deg: float = 3.0
    monochrome: bool = True
    update_rate_hz: float = 30.0
    initial_position: str = "center"
    initial_pan_px: Optional[float] = None
    initial_tilt_px: Optional[float] = None
    max_pan_speed_deg_s: float = 5.0
    max_tilt_speed_deg_s: float = 5.0
    max_acceleration_deg_s2: float = 20.0

    @property
    def deg_per_pixel(self) -> Tuple[float, float]:
        """Angular resolution as ``(horizontal, vertical)`` degrees per pixel.

        For the default 4 deg x 3 deg FOV over 640x480 px this is 0.00625 deg/px on both axes.

        Returns:
            ``(deg_per_pixel_x, deg_per_pixel_y)``.
        """
        return (self.fov_horizontal_deg / self.resolution_width,
                self.fov_vertical_deg / self.resolution_height)

    @property
    def max_slew_px_s(self) -> Tuple[float, float]:
        """Slew ceilings expressed in pixels per second as ``(pan, tilt)``.

        Returns:
            ``(pan_px_s, tilt_px_s)``. At the 5 deg/s default and 0.00625 deg/px this is
            800 px/s on both axes.
        """
        dpp_x, dpp_y = self.deg_per_pixel
        return (self.max_pan_speed_deg_s / dpp_x, self.max_tilt_speed_deg_s / dpp_y)

    @property
    def max_px_per_frame(self) -> Tuple[float, float]:
        """Maximum boresight travel per frame as ``(pan, tilt)`` pixels.

        This is the binding feasibility constraint of the whole system: a target whose apparent
        motion between frames exceeds this value cannot be kept centred, regardless of how good
        the vision pipeline is. At 5 deg/s and 30 Hz it is 26.7 px/frame; at 10 deg/s, 53.3.

        Returns:
            ``(pan_px_per_frame, tilt_px_per_frame)``.
        """
        pan_px_s, tilt_px_s = self.max_slew_px_s
        return (pan_px_s / self.update_rate_hz, tilt_px_s / self.update_rate_hz)

    @property
    def fov_px(self) -> Tuple[int, int]:
        """Viewport size in pixels as ``(width, height)``."""
        return (self.resolution_width, self.resolution_height)

    @property
    def boresight_px(self) -> Tuple[float, float]:
        """Boresight position in frame-local coordinates as ``(x, y)``.

        Uses the pixel-centre convention: a frame of width ``W`` has its centre at ``(W-1)/2``.
        This must agree exactly with ``FrameData.center`` in :mod:`src.framesource`; pointing
        error is measured against this point.
        """
        return ((self.resolution_width - 1) / 2.0, (self.resolution_height - 1) / 2.0)

    def validate(self) -> None:
        """Check camera parameters against specification limits.

        Raises:
            ConfigError: On non-positive geometry, a slew rate outside the 5-10 deg/s spec range,
                an unknown initial-position mode, or a missing custom position.
        """
        _require(self.resolution_width > 0 and self.resolution_height > 0,
                 "camera resolution must be positive")
        _require(self.fov_horizontal_deg > 0 and self.fov_vertical_deg > 0,
                 "camera FOV must be positive")
        lo, hi = SPEC_SLEW_RANGE_DEG_S
        for name, value in (("max_pan_speed_deg_s", self.max_pan_speed_deg_s),
                            ("max_tilt_speed_deg_s", self.max_tilt_speed_deg_s)):
            _require(lo <= value <= hi,
                     f"camera.{name} must be within the spec range {lo}-{hi} deg/s "
                     f"(parameters 13-14), got {value}")
        _require(self.update_rate_hz >= SPEC_MIN_CAMERA_RATE_HZ,
                 f"camera.update_rate_hz must be >= {SPEC_MIN_CAMERA_RATE_HZ} Hz per spec "
                 f"parameter 5, got {self.update_rate_hz}")
        _require(self.initial_position in ("center", "random", "custom"),
                 f"camera.initial_position must be center/random/custom, got {self.initial_position!r}")
        if self.initial_position == "custom":
            _require(self.initial_pan_px is not None and self.initial_tilt_px is not None,
                     "camera.initial_position='custom' requires initial_pan_px and initial_tilt_px")
        _require(self.max_acceleration_deg_s2 > 0,
                 "camera.max_acceleration_deg_s2 must be positive")


@dataclass(frozen=True)
class TargetConfig:
    """Beacon appearance and motion.

    Attributes:
        count: Number of simultaneous targets. One is mandatory; more is optional bonus.
        shape: ``"gaussian"`` (default, most realistic), ``"square"`` (spec default) or
            ``"circle"``.
        size_px: Nominal target extent. Spec parameter 10 allows 5-20 px.
        gaussian_sigma_px: Standard deviation of the Gaussian profile.
        peak_intensity: Peak grey level of the beacon.
        supersample_factor: Rendering supersample factor, downsampled afterwards so the
            ground-truth centroid remains exact at sub-pixel positions.
        initial_position: ``"random"``, ``"center"`` or ``"custom"``.
        initial_x: Custom initial x in canvas coordinates.
        initial_y: Custom initial y in canvas coordinates.
        boundary_behaviour: ``"bounce"``, ``"wrap"`` or ``"clamp"`` at canvas edges.
        motion: Raw motion sub-block, kept as a mapping because each motion type has a different
            parameter set. Access via :meth:`motion_params`.
    """

    count: int = 1
    shape: str = "gaussian"
    size_px: int = 10
    gaussian_sigma_px: float = 2.5
    peak_intensity: int = 255
    supersample_factor: int = 4
    initial_position: str = "random"
    initial_x: Optional[float] = None
    initial_y: Optional[float] = None
    boundary_behaviour: str = "bounce"
    motion: Dict[str, Any] = field(default_factory=dict)

    #: Motion types the specification requires (parameter 12). ``ClassVar`` so that dataclass
    #: field discovery does not mistake these for configurable values.
    MANDATORY_MOTIONS: ClassVar[Tuple[str, ...]] = ("linear", "circular", "figure8", "random")
    #: Motion types we support as optional extras.
    OPTIONAL_MOTIONS: ClassVar[Tuple[str, ...]] = ("spiral", "sinusoidal", "ornstein_uhlenbeck", "custom")

    @property
    def nominal_fwhm_px(self) -> float:
        """Expected full width at half maximum of the beacon, in pixels.

        This is the fallback spot scale used when runtime estimation is unavailable or fails, and
        the reference length for every scale-relative vision parameter. For a Gaussian profile it
        is ``2.355 * sigma``; for hard-edged shapes the rendered extent is used directly.

        Returns:
            FWHM in pixels.
        """
        if self.shape == "gaussian":
            return FWHM_PER_SIGMA * self.gaussian_sigma_px
        return float(self.size_px)

    @property
    def motion_type(self) -> str:
        """Selected motion type, defaulting to ``"circular"`` when unspecified."""
        return str(self.motion.get("type", "circular"))

    def motion_params(self) -> Dict[str, Any]:
        """Return the parameter block for the selected motion type.

        Returns:
            The comment-stripped parameters for :attr:`motion_type`, or an empty dict if the
            motion type has no parameters.
        """
        block = self.motion.get(self.motion_type, {})
        return _strip_comments(block) if isinstance(block, Mapping) else {}

    def validate(self) -> None:
        """Check target parameters against specification limits.

        Raises:
            ConfigError: On an out-of-range size, unknown shape or motion type, a custom initial
                position with no coordinates, or a supersample factor below 1.
        """
        lo, hi = SPEC_TARGET_SIZE_RANGE_PX
        _require(lo <= self.size_px <= hi,
                 f"target.size_px must be within the spec range {lo}-{hi} px (parameter 10), "
                 f"got {self.size_px}")
        _require(self.count >= 1, f"target.count must be at least 1, got {self.count}")
        _require(self.shape in ("gaussian", "square", "circle"),
                 f"target.shape must be gaussian/square/circle, got {self.shape!r}")
        _require(self.gaussian_sigma_px > 0,
                 f"target.gaussian_sigma_px must be positive, got {self.gaussian_sigma_px}")
        _require(0 < self.peak_intensity <= 255,
                 f"target.peak_intensity must be in (0, 255], got {self.peak_intensity}")
        _require(self.supersample_factor >= 1,
                 f"target.supersample_factor must be >= 1, got {self.supersample_factor}")
        _require(self.boundary_behaviour in ("bounce", "wrap", "clamp"),
                 f"target.boundary_behaviour must be bounce/wrap/clamp, "
                 f"got {self.boundary_behaviour!r}")
        _require(self.initial_position in ("random", "center", "custom"),
                 f"target.initial_position must be random/center/custom, "
                 f"got {self.initial_position!r}")
        if self.initial_position == "custom":
            _require(self.initial_x is not None and self.initial_y is not None,
                     "target.initial_position='custom' requires initial_x and initial_y")
        allowed = self.MANDATORY_MOTIONS + self.OPTIONAL_MOTIONS
        _require(self.motion_type in allowed,
                 f"target.motion.type must be one of {list(allowed)}, got {self.motion_type!r}")


@dataclass(frozen=True)
class NoiseConfig:
    """Sensor noise, atmospheric degradation and mechanical disturbance settings.

    Sub-blocks are kept as mappings because they are consumed by the pure noise functions in
    ``src/noise/``, each of which owns its own parameter dataclass. This block's job is to hold
    them and to enforce the specification's magnitude limits.

    Attributes:
        enabled: Master switch for the whole noise pipeline.
        pipeline_order: Order in which noise stages are composed.
        gaussian: Additive white Gaussian noise parameters.
        poisson: Shot noise parameters.
        salt_pepper: Impulse noise parameters.
        atmospheric: Koschmieder-model parameters and presets.
        turbulence: Scintillation and beam-wander parameters.
        camera_jitter: High-frequency zero-mean viewport offset.
        platform_motion: Low-frequency boresight drift.
    """

    enabled: bool = True
    pipeline_order: List[str] = field(default_factory=list)
    gaussian: Dict[str, Any] = field(default_factory=dict)
    poisson: Dict[str, Any] = field(default_factory=dict)
    salt_pepper: Dict[str, Any] = field(default_factory=dict)
    atmospheric: Dict[str, Any] = field(default_factory=dict)
    turbulence: Dict[str, Any] = field(default_factory=dict)
    camera_jitter: Dict[str, Any] = field(default_factory=dict)
    platform_motion: Dict[str, Any] = field(default_factory=dict)

    @property
    def atmospheric_preset(self) -> Dict[str, Any]:
        """Return the parameters of the currently selected atmospheric preset.

        Returns:
            The preset's parameter dict, or an empty dict when atmospherics are unconfigured.

        Raises:
            ConfigError: If the named preset does not exist.
        """
        name = self.atmospheric.get("preset", "clear")
        presets = self.atmospheric.get("presets", {})
        if not presets:
            return {}
        if name not in presets:
            raise ConfigError(
                f"noise.atmospheric.preset {name!r} is not defined. "
                f"Available: {sorted(presets)}"
            )
        return _strip_comments(presets[name])

    def validate(self) -> None:
        """Check noise magnitudes against specification limits.

        Raises:
            ConfigError: If any noise magnitude exceeds its spec ceiling (parameters 21-25), or
                an unknown stage appears in ``pipeline_order``.
        """
        sigma = float(self.gaussian.get("sigma", 0.0))
        _require(0.0 <= sigma <= SPEC_MAX_NOISE_SIGMA,
                 f"noise.gaussian.sigma must be in [0, {SPEC_MAX_NOISE_SIGMA}] per spec "
                 f"parameter 22, got {sigma}")

        density = float(self.salt_pepper.get("density", 0.0))
        _require(0.0 <= density <= SPEC_MAX_SALT_PEPPER_DENSITY,
                 f"noise.salt_pepper.density must be in [0, {SPEC_MAX_SALT_PEPPER_DENSITY}] per "
                 f"spec parameter 21, got {density}")

        jitter = float(self.camera_jitter.get("max_px_per_frame", 0.0))
        _require(0.0 <= jitter <= SPEC_MAX_JITTER_PX_PER_FRAME,
                 f"noise.camera_jitter.max_px_per_frame must be in "
                 f"[0, {SPEC_MAX_JITTER_PX_PER_FRAME}] per spec parameter 23, got {jitter}")

        platform = float(self.platform_motion.get("max_px_per_frame", 0.0))
        _require(0.0 <= platform <= SPEC_MAX_PLATFORM_PX_PER_FRAME,
                 f"noise.platform_motion.max_px_per_frame must be in "
                 f"[0, {SPEC_MAX_PLATFORM_PX_PER_FRAME}] per spec parameter 25, got {platform}")

        known_stages = {"gaussian", "poisson", "salt_pepper", "atmospheric", "turbulence"}
        unknown = set(self.pipeline_order) - known_stages
        _require(not unknown,
                 f"noise.pipeline_order contains unknown stage(s) {sorted(unknown)}. "
                 f"Known: {sorted(known_stages)}")
        # Touch the preset so a bad name fails at load time, not mid-run.
        self.atmospheric_preset


# --------------------------------------------------------------------------------------------
# Vision: scale-relative geometry
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedVisionGeometry:
    """Concrete pixel geometry for the vision pipeline at one particular spot scale.

    Produced by :meth:`VisionConfig.resolve_geometry`. Every value here is derived from the spot
    FWHM rather than configured directly, which is what lets the identical pipeline run on a
    640x480 frame with a 10 px beacon and on an unseen evaluator video at a different resolution
    with a different spot size.

    Attributes:
        fwhm_px: The spot scale these values were derived from, in pixels.
        from_fallback: True when spot-scale estimation was unavailable or failed and the
            configured fallback scale was used instead. Log this: it tells us, after the fact,
            whether a poor result came from the estimator giving up.
        tophat_kernel_px: Side length of the top-hat structuring element. Odd.
        min_blob_area_px: Lower area gate for candidate blobs, in square pixels.
        max_blob_area_px: Upper area gate for candidate blobs, in square pixels.
        centroid_window_px: Side length of the centroiding window. Odd.
        roi_size_px: Side length of the tracking ROI around the Kalman prediction.
        max_roi_size_px: Ceiling on ROI growth after a loss.
    """

    fwhm_px: float
    from_fallback: bool
    tophat_kernel_px: int
    min_blob_area_px: float
    max_blob_area_px: float
    centroid_window_px: int
    roi_size_px: int
    max_roi_size_px: int

    @property
    def nominal_spot_area_px(self) -> float:
        """Area of a circular spot of diameter :attr:`fwhm_px`, i.e. ``(pi/4) * FWHM^2``.

        This is the reference area the blob gates are multiples of.
        """
        return math.pi / 4.0 * self.fwhm_px ** 2


@dataclass(frozen=True)
class SpotScaleConfig:
    """Runtime estimation of the beacon spot scale.

    Attributes:
        estimation_enabled: Estimate the scale from the frame. When False, the fallback is always
            used, which reproduces fixed-geometry behaviour.
        method: Estimator to use.
        fallback_fwhm_px: FWHM assumed when estimation is disabled or fails.
        min_fwhm_px: Lower clamp on the estimate. Below roughly 1.5 px a spot is undersampled.
        max_fwhm_px: Upper clamp on the estimate.
        smoothing_alpha: Exponential-moving-average coefficient across frames. The true spot
            scale changes slowly while per-frame estimates are noisy, so smoothing is essential
            to stop the whole vision geometry breathing frame to frame.
    """

    estimation_enabled: bool = True
    method: str = "residual_second_moment"
    fallback_fwhm_px: float = 5.89
    min_fwhm_px: float = 1.5
    max_fwhm_px: float = 60.0
    smoothing_alpha: float = 0.2

    def clamp(self, fwhm_px: float) -> float:
        """Clamp an estimated FWHM into the configured valid range.

        Args:
            fwhm_px: Raw estimate in pixels.

        Returns:
            The estimate constrained to ``[min_fwhm_px, max_fwhm_px]``.
        """
        return _clamp(fwhm_px, self.min_fwhm_px, self.max_fwhm_px)

    def validate(self) -> None:
        """Check estimator settings.

        Raises:
            ConfigError: On an unknown method, an inverted or non-positive clamp range, a
                fallback outside that range, or a smoothing coefficient outside ``(0, 1]``.
        """
        _require(self.method in ("residual_second_moment", "blob_equivalent_diameter", "fixed"),
                 f"vision.spot_scale.method is unknown: {self.method!r}")
        _require(0 < self.min_fwhm_px < self.max_fwhm_px,
                 f"vision.spot_scale requires 0 < min_fwhm_px < max_fwhm_px, "
                 f"got {self.min_fwhm_px} and {self.max_fwhm_px}")
        _require(self.min_fwhm_px <= self.fallback_fwhm_px <= self.max_fwhm_px,
                 f"vision.spot_scale.fallback_fwhm_px ({self.fallback_fwhm_px}) must lie within "
                 f"[{self.min_fwhm_px}, {self.max_fwhm_px}]")
        _require(0.0 < self.smoothing_alpha <= 1.0,
                 f"vision.spot_scale.smoothing_alpha must be in (0, 1], "
                 f"got {self.smoothing_alpha}")


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing stage: impulse rejection, background suppression, normalisation.

    Attributes:
        median_filter_enabled: Apply a median prefilter. Mandatory in practice: salt-and-pepper
            noise is the primary false-positive threat to centroiding.
        median_kernel_size: Median kernel side length. Deliberately scale-independent -- it
            targets single-pixel impulses, not the spot.
        tophat_enabled: Apply white top-hat morphology to suppress smooth background.
        tophat_kernel_fwhm_multiple: Structuring element size as a multiple of the spot FWHM.
            Must exceed 1 so the spot survives the opening and appears in the residual.
        tophat_kernel_fallback_px: Absolute size used only when the spot scale is unknown.
        tophat_kernel_min_px: Lower clamp on the resolved kernel size.
        tophat_kernel_max_px: Upper clamp on the resolved kernel size.
        normalise_per_frame: Rescale intensity per frame. Essential in Mode B, where we cannot
            assume anything about the brightness range of evaluator video.
    """

    median_filter_enabled: bool = True
    median_kernel_size: int = 3
    tophat_enabled: bool = True
    tophat_kernel_fwhm_multiple: float = 2.5
    tophat_kernel_fallback_px: int = 15
    tophat_kernel_min_px: int = 3
    tophat_kernel_max_px: int = 101
    normalise_per_frame: bool = True

    def validate(self) -> None:
        """Check preprocessing parameters.

        Raises:
            ConfigError: On an even or non-positive median kernel, a top-hat multiple that would
                erase the spot, or an inverted clamp range.
        """
        _require(self.median_kernel_size >= 1 and self.median_kernel_size % 2 == 1,
                 f"vision.preprocess.median_kernel_size must be a positive odd integer, "
                 f"got {self.median_kernel_size}")
        _require(self.tophat_kernel_fwhm_multiple > 1.0,
                 f"vision.preprocess.tophat_kernel_fwhm_multiple must exceed 1.0, otherwise the "
                 f"structuring element fits inside the spot and the opening removes the beacon "
                 f"itself; got {self.tophat_kernel_fwhm_multiple}")
        _require(0 < self.tophat_kernel_min_px <= self.tophat_kernel_max_px,
                 "vision.preprocess top-hat clamp range is inverted or non-positive")


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholding and candidate gating.

    Attributes:
        threshold_method: Adaptive operator. Default ``"mean_plus_k_sigma"`` applied to the
            top-hat residual. See ``docs/DESIGN.md`` 5.1.1 for why this beats Otsu on the raw
            frame at our fill factor, and why Otsu remains selectable.
        k_sigma: Multiplier for the ``mean + k*sigma`` operator.
        percentile: Percentile for the percentile operator.
        min_blob_area_spot_multiple: Lower area gate as a multiple of the nominal spot area.
        max_blob_area_spot_multiple: Upper area gate as a multiple of the nominal spot area.
        min_blob_area_fallback_px: Absolute lower gate, used only when the spot scale is unknown.
        max_blob_area_fallback_px: Absolute upper gate, used only when the spot scale is unknown.
        min_circularity: Shape gate. Dimensionless, therefore already scale-free.
    """

    threshold_method: str = "mean_plus_k_sigma"
    k_sigma: float = 3.0
    percentile: float = 99.5
    min_blob_area_spot_multiple: float = 0.15
    max_blob_area_spot_multiple: float = 33.0
    min_blob_area_fallback_px: float = 4.0
    max_blob_area_fallback_px: float = 900.0
    min_circularity: float = 0.3

    #: Threshold operators the pipeline implements. All are parameter-free or single-parameter;
    #: none may be a fixed intensity constant.
    METHODS: ClassVar[Tuple[str, ...]] = ("mean_plus_k_sigma", "otsu", "adaptive_gaussian",
                                          "percentile")

    def validate(self) -> None:
        """Check detection parameters.

        Raises:
            ConfigError: On an unknown threshold method, inverted area gates, or an out-of-range
                percentile or circularity.
        """
        _require(self.threshold_method in self.METHODS,
                 f"vision.detection.threshold_method must be one of {list(self.METHODS)}, "
                 f"got {self.threshold_method!r}")
        _require(self.k_sigma > 0,
                 f"vision.detection.k_sigma must be positive, got {self.k_sigma}")
        _require(0.0 < self.percentile < 100.0,
                 f"vision.detection.percentile must be in (0, 100), got {self.percentile}")
        _require(0.0 < self.min_blob_area_spot_multiple < self.max_blob_area_spot_multiple,
                 f"vision.detection blob-area multiples must satisfy 0 < min < max, got "
                 f"{self.min_blob_area_spot_multiple} and {self.max_blob_area_spot_multiple}")
        _require(0.0 < self.min_blob_area_fallback_px < self.max_blob_area_fallback_px,
                 "vision.detection fallback blob-area gates must satisfy 0 < min < max")
        _require(0.0 <= self.min_circularity <= 1.0,
                 f"vision.detection.min_circularity must be in [0, 1], got {self.min_circularity}")


@dataclass(frozen=True)
class CentroidConfig:
    """Centroid estimator settings.

    Attributes:
        method: ``"cog"``, ``"thresholded_cog"``, ``"iwcog"`` (default) or ``"gaussian_fit"``.
        iterations: Re-centring iterations for the iteratively-weighted estimator.
        window_fwhm_multiple: Centroiding window size as a multiple of the spot FWHM.
        window_fallback_px: Absolute window size when the spot scale is unknown.
        window_min_px: Lower clamp on the resolved window.
        window_max_px: Upper clamp on the resolved window.
        background_k_sigma: Background subtraction level before summing, as ``mean + k*sigma``.
            Unthresholded centre-of-gravity is badly biased by background and impulse noise, so
            this is not optional.
    """

    method: str = "iwcog"
    iterations: int = 3
    window_fwhm_multiple: float = 3.5
    window_fallback_px: int = 21
    window_min_px: int = 5
    window_max_px: int = 201
    background_k_sigma: float = 3.0

    #: Estimators the pipeline implements.
    METHODS: ClassVar[Tuple[str, ...]] = ("cog", "thresholded_cog", "iwcog", "gaussian_fit")

    def validate(self) -> None:
        """Check centroid estimator settings.

        Raises:
            ConfigError: On an unknown method, non-positive iteration count, a window multiple
                too small to contain the spot, or an inverted clamp range.
        """
        _require(self.method in self.METHODS,
                 f"vision.centroid.method must be one of {list(self.METHODS)}, "
                 f"got {self.method!r}")
        _require(self.iterations >= 1,
                 f"vision.centroid.iterations must be >= 1, got {self.iterations}")
        _require(self.window_fwhm_multiple >= 2.0,
                 f"vision.centroid.window_fwhm_multiple must be >= 2.0 so the window contains "
                 f"the spot wings that carry the sub-pixel information; "
                 f"got {self.window_fwhm_multiple}")
        _require(0 < self.window_min_px <= self.window_max_px,
                 "vision.centroid window clamp range is inverted or non-positive")
        _require(self.background_k_sigma > 0,
                 f"vision.centroid.background_k_sigma must be positive, "
                 f"got {self.background_k_sigma}")


@dataclass(frozen=True)
class RoiConfig:
    """Region-of-interest processing, the main throughput lever.

    Attributes:
        enabled: Restrict processing to a window around the Kalman prediction once locked.
        size_fwhm_multiple: ROI side length as a multiple of the spot FWHM. Must comfortably
            exceed per-frame target motion plus jitter, or the target walks out of the ROI
            between frames.
        size_fallback_px: Absolute ROI size when the spot scale is unknown.
        expand_on_loss_factor: Multiplier applied to the ROI on each missed detection.
        max_size_fwhm_multiple: Ceiling on ROI growth, as a multiple of the spot FWHM.
        max_size_fallback_px: Absolute ceiling when the spot scale is unknown.
    """

    enabled: bool = True
    size_fwhm_multiple: float = 11.0
    size_fallback_px: int = 64
    expand_on_loss_factor: float = 2.0
    max_size_fwhm_multiple: float = 44.0
    max_size_fallback_px: int = 256

    def validate(self) -> None:
        """Check ROI settings.

        Raises:
            ConfigError: If the ROI multiple is not smaller than its ceiling, or the expansion
                factor does not actually expand.
        """
        _require(0 < self.size_fwhm_multiple <= self.max_size_fwhm_multiple,
                 f"vision.roi requires 0 < size_fwhm_multiple <= max_size_fwhm_multiple, got "
                 f"{self.size_fwhm_multiple} and {self.max_size_fwhm_multiple}")
        _require(0 < self.size_fallback_px <= self.max_size_fallback_px,
                 "vision.roi fallback sizes must satisfy 0 < size <= max_size")
        _require(self.expand_on_loss_factor > 1.0,
                 f"vision.roi.expand_on_loss_factor must exceed 1.0 to widen the search after a "
                 f"loss, got {self.expand_on_loss_factor}")


@dataclass(frozen=True)
class VisionConfig:
    """Vision pipeline configuration, parameterised by spot scale rather than absolute pixels.

    Attributes:
        spot_scale: Runtime spot-scale estimation settings.
        preprocess: Preprocessing stage settings.
        detection: Thresholding and gating settings.
        centroid: Centroid estimator settings.
        roi: Region-of-interest settings.
    """

    spot_scale: SpotScaleConfig = field(default_factory=SpotScaleConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    centroid: CentroidConfig = field(default_factory=CentroidConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)

    def resolve_geometry(self, fwhm_px: Optional[float] = None) -> ResolvedVisionGeometry:
        """Convert scale-relative parameters into concrete pixel geometry.

        This is the function that makes the pipeline resolution-agnostic, and therefore the
        function Benchmark Performance-2 depends on. Call it whenever the smoothed spot-scale
        estimate changes materially -- not every frame, since the estimate is smoothed and the
        resolved integers only change at discrete steps.

        Args:
            fwhm_px: Estimated spot full width at half maximum in pixels. Pass ``None`` when the
                estimate is unavailable or the estimator failed, in which case the configured
                absolute fallbacks are used and the result is flagged via
                :attr:`ResolvedVisionGeometry.from_fallback`.

        Returns:
            A :class:`ResolvedVisionGeometry` with every value clamped to its configured range.
            Kernel and window sizes are forced odd so they have a defined centre pixel.

        Raises:
            ConfigError: If ``fwhm_px`` is supplied but is not a positive finite number.
        """
        use_fallback = fwhm_px is None or not self.spot_scale.estimation_enabled
        if fwhm_px is not None:
            if not math.isfinite(fwhm_px) or fwhm_px <= 0:
                raise ConfigError(
                    f"Estimated spot FWHM must be a positive finite number, got {fwhm_px!r}"
                )

        if use_fallback:
            scale = self.spot_scale.fallback_fwhm_px
            return ResolvedVisionGeometry(
                fwhm_px=scale,
                from_fallback=True,
                tophat_kernel_px=_odd(self.preprocess.tophat_kernel_fallback_px,
                                      self.preprocess.tophat_kernel_min_px,
                                      self.preprocess.tophat_kernel_max_px),
                min_blob_area_px=self.detection.min_blob_area_fallback_px,
                max_blob_area_px=self.detection.max_blob_area_fallback_px,
                centroid_window_px=_odd(self.centroid.window_fallback_px,
                                        self.centroid.window_min_px,
                                        self.centroid.window_max_px),
                roi_size_px=self.roi.size_fallback_px,
                max_roi_size_px=self.roi.max_size_fallback_px,
            )

        scale = self.spot_scale.clamp(float(fwhm_px))
        spot_area = math.pi / 4.0 * scale ** 2
        roi = int(round(self.roi.size_fwhm_multiple * scale))
        max_roi = int(round(self.roi.max_size_fwhm_multiple * scale))
        max_roi = max(max_roi, roi)

        # Upper bound uses 2x the normal multiple -- deliberate asymmetry.
        # The area gate argument (DESIGN §5.1) is one-directional: underestimating
        # scale is catastrophic (beacon gated out), overestimating admits larger blobs
        # but does not remove the beacon. The lower bound is unchanged.
        # This is a correctness fix on its own merits. It is NOT a fix for the
        # initiation-velocity bug (HANDOFF §2) -- those are separate defects.
        upper_multiple = self.detection.max_blob_area_spot_multiple * 2.0
        lower_multiple = self.detection.min_blob_area_spot_multiple

        return ResolvedVisionGeometry(
            fwhm_px=scale,
            from_fallback=False,
            tophat_kernel_px=_odd(self.preprocess.tophat_kernel_fwhm_multiple * scale,
                                  self.preprocess.tophat_kernel_min_px,
                                  self.preprocess.tophat_kernel_max_px),
            min_blob_area_px=lower_multiple * spot_area,
            max_blob_area_px=upper_multiple * spot_area,
            centroid_window_px=_odd(self.centroid.window_fwhm_multiple * scale,
                                    self.centroid.window_min_px,
                                    self.centroid.window_max_px),
            roi_size_px=max(1, roi),
            max_roi_size_px=max(1, max_roi),
        )

    def validate(self) -> None:
        """Validate every vision sub-block.

        Raises:
            ConfigError: Propagated from the sub-block that failed.
        """
        self.spot_scale.validate()
        self.preprocess.validate()
        self.detection.validate()
        self.centroid.validate()
        self.roi.validate()


@dataclass(frozen=True)
class FilteringConfig:
    """Kalman filter, validation gating and track management.

    Attributes:
        kalman: Filter model and noise parameters. Note that ``measurement_noise_sigma_px`` is
            only the *initial* value: Phase 4 derives ``R`` per frame from the measured detection
            SNR via ``sigma ~ FWHM / (2*SNR)``. See ``docs/DESIGN.md`` section 6.
        gating: Mahalanobis validation gate settings.
        track_management: M-of-N confirmation and deletion logic.
    """

    kalman: Dict[str, Any] = field(default_factory=dict)
    gating: Dict[str, Any] = field(default_factory=dict)
    track_management: Dict[str, Any] = field(default_factory=dict)

    def kalman_params(self, unobservable_sigma_px: float = 0.0) -> "KalmanParams":
        """Build :class:`~src.filtering.kalman.KalmanParams` from this configuration.

        Every consumer must go through here. Constructing ``KalmanParams()`` directly and
        relying on its dataclass defaults silently ignores the JSON, which is exactly what
        happened before this method existed: ``process_noise_psd`` was validated by
        :meth:`validate` and read by nobody, so it agreed with the filter only because the
        default happened to equal the shipped value. Editing it in a scenario file -- the way an
        evaluator would -- changed nothing at all, and no test caught it because every test also
        used the default.

        Args:
            unobservable_sigma_px: Sigma of disturbances the estimator cannot observe, chiefly
                camera jitter. Derived from the noise configuration by the caller, since it is
                not a filtering parameter.

        Returns:
            Filter parameters reflecting the loaded configuration.
        """
        from src.filtering.kalman import KalmanParams

        return KalmanParams(
            process_noise_psd=float(self.kalman.get("process_noise_psd", 50.0)),
            initial_position_sigma_px=float(
                self.kalman.get("initial_position_uncertainty_px", 50.0)),
            initial_velocity_sigma_px_s=float(
                self.kalman.get("initial_velocity_uncertainty_px_s", 200.0)),
            gate_threshold=float(self.gating.get("mahalanobis_threshold", 9.21)),
            unobservable_sigma_px=float(unobservable_sigma_px),
        )

    @property
    def delete_after_missed(self) -> int:
        """Consecutive missed detections that declare a track lost.

        This is the ``N`` in the re-acquisition metric definition: the re-acquisition clock
        starts when this many detections have been missed in a row.
        """
        return int(self.track_management.get("delete_after_missed", 5))

    def validate(self) -> None:
        """Check filtering parameters.

        Raises:
            ConfigError: On an unknown Kalman model, non-positive noise parameters, a
                non-positive gate threshold, or a malformed M-of-N specification.
        """
        model = self.kalman.get("model", "constant_velocity")
        _require(model in ("constant_velocity", "constant_acceleration", "imm"),
                 f"filtering.kalman.model is unknown: {model!r}")
        _require(float(self.kalman.get("process_noise_psd", 1.0)) > 0,
                 "filtering.kalman.process_noise_psd must be positive")
        _require(float(self.kalman.get("measurement_noise_sigma_px", 1.0)) > 0,
                 "filtering.kalman.measurement_noise_sigma_px must be positive")
        _require(float(self.gating.get("mahalanobis_threshold", 9.21)) > 0,
                 "filtering.gating.mahalanobis_threshold must be positive")

        m_of_n = self.track_management.get("confirm_m_of_n", [3, 5])
        _require(isinstance(m_of_n, (list, tuple)) and len(m_of_n) == 2,
                 f"filtering.track_management.confirm_m_of_n must be a two-element [M, N], "
                 f"got {m_of_n!r}")
        m, n = int(m_of_n[0]), int(m_of_n[1])
        _require(1 <= m <= n,
                 f"filtering.track_management.confirm_m_of_n requires 1 <= M <= N, got M={m}, N={n}")
        _require(self.delete_after_missed >= 1,
                 "filtering.track_management.delete_after_missed must be >= 1")


@dataclass(frozen=True)
class ControlConfig:
    """Control loop, state machine and acquisition search.

    The PID gains reaching this class are **provisional** until the Phase 4 step-response test
    passes; see :meth:`pid_gains_are_provisional` and the hard gate in ``docs/ROADMAP.md``.

    Attributes:
        update_rate_hz: Control clock. Spec parameter 15 requires >= 20 Hz. This is a separate
            clock from frame generation and from processing throughput; all three are logged
            independently and must never be conflated.
        pid: PID gains and anti-windup limit.
        feedforward: Kalman-velocity feedforward settings.
        state_machine: Lock/loss thresholds and hysteresis windows.
        search: Acquisition scan pattern settings.
    """

    update_rate_hz: float = 30.0
    pid: Dict[str, Any] = field(default_factory=dict)
    feedforward: Dict[str, Any] = field(default_factory=dict)
    state_machine: Dict[str, Any] = field(default_factory=dict)
    search: Dict[str, Any] = field(default_factory=dict)

    @property
    def pid_gains_are_provisional(self) -> bool:
        """Whether the PID gains are still analytically-sized rather than test-validated.

        True while the ``_PROVISIONAL`` marker is present in the config's PID block. While this
        is true, no performance number produced with these gains may be quoted in the technical
        report, the live demo, or any benchmark submission -- the gains were sized so that a
        ~100 px error saturates the slew limit, which is a sanity floor, not a tuning result.

        Returns:
            True if the marker is present.
        """
        return "_PROVISIONAL" in self.pid

    def search_arm_spacing_px(self, camera: CameraConfig) -> float:
        """Compute the spiral search arm spacing for a given camera.

        Derived from the camera FOV rather than stored, because a stored spacing either wastes
        scan time or silently opens coverage gaps the moment the FOV or resolution changes. The
        *limiting* (smaller) FOV dimension is used so the sweep is guaranteed gap-free on both
        axes, and the configured fraction supplies overlap margin.

        Args:
            camera: Active camera configuration.

        Returns:
            Arm spacing in pixels. For the default 640x480 viewport at a fraction of 0.9 this is
            432 px.
        """
        fraction = float(self.search.get("arm_spacing_fov_fraction", 0.9))
        width, height = camera.fov_px
        return fraction * float(min(width, height))

    def worst_case_search_time_s(self, camera: CameraConfig, scene: SceneConfig) -> float:
        """Estimate the worst-case time to sweep the whole uncertainty region.

        A spiral (or raster) covering an area ``A`` with arm spacing ``d`` traverses a path of
        length roughly ``A / d``. Dividing by the scan speed in pixels per second gives the time
        to visit every part of the canvas, which bounds acquisition when the beacon starts
        outside the initial viewport.

        For the default configuration this evaluates to roughly 11 s against a 2 s acquisition
        requirement. **That gap is set by canvas area, FOV and the mechanical slew limit alone**
        -- it is arithmetic, not a tuning failure, and no algorithm improves it. The honest
        response is to report in-FOV and search-limited acquisition as two separate populations
        (``docs/DESIGN.md`` section 7.5), which is why :meth:`AppConfig.startup_warnings` surfaces
        this number from the very first run.

        Args:
            camera: Active camera configuration, supplying FOV and slew limits.
            scene: Active scene configuration, supplying the uncertainty area.

        Returns:
            Worst-case sweep time in seconds.
        """
        spacing = self.search_arm_spacing_px(camera)
        if spacing <= 0:
            return math.inf
        path_px = scene.area_px / spacing

        scan_deg_s = float(self.search.get("scan_speed_deg_s", camera.max_pan_speed_deg_s))
        # Never claim a scan faster than the gimbal can physically slew.
        effective_deg_s = min(scan_deg_s, camera.max_pan_speed_deg_s, camera.max_tilt_speed_deg_s)
        dpp_x, dpp_y = camera.deg_per_pixel
        # The coarser axis (larger deg/px) yields fewer px/s and therefore bounds the sweep.
        scan_px_s = effective_deg_s / max(dpp_x, dpp_y)
        if scan_px_s <= 0:
            return math.inf
        return path_px / scan_px_s

    def validate(self) -> None:
        """Check control parameters.

        Raises:
            ConfigError: On a control rate below the spec minimum, non-positive gains, an unknown
                search pattern, a search fraction outside ``(0, 1]``, or hysteresis windows that
                are not strictly ordered.
        """
        _require(self.update_rate_hz >= SPEC_MIN_CONTROL_RATE_HZ,
                 f"control.update_rate_hz must be >= {SPEC_MIN_CONTROL_RATE_HZ} Hz per spec "
                 f"parameter 15, got {self.update_rate_hz}")
        for gain in ("kp", "ki", "kd"):
            value = float(self.pid.get(gain, 0.0))
            _require(value >= 0.0, f"control.pid.{gain} must be non-negative, got {value}")
        _require(float(self.pid.get("kp", 0.0)) > 0.0,
                 "control.pid.kp must be positive or the loop has no proportional action")
        _require(float(self.pid.get("integral_limit", 1.0)) > 0,
                 "control.pid.integral_limit must be positive (anti-windup clamp)")

        pattern = self.search.get("pattern", "archimedean_spiral")
        _require(pattern in ("archimedean_spiral", "raster"),
                 f"control.search.pattern is unknown: {pattern!r}")
        fraction = float(self.search.get("arm_spacing_fov_fraction", 0.9))
        _require(0.0 < fraction <= 1.0,
                 f"control.search.arm_spacing_fov_fraction must be in (0, 1]; values above 1 "
                 f"open coverage gaps between spiral arms. Got {fraction}")

        sm = self.state_machine
        lock_px = float(sm.get("lock_window_px", 40))
        unlock_px = float(sm.get("unlock_window_px", 80))
        _require(0 < lock_px < unlock_px,
                 f"control.state_machine requires 0 < lock_window_px < unlock_window_px for "
                 f"hysteresis; got {lock_px} and {unlock_px}")
        _require(int(sm.get("lock_confirm_frames", 3)) >= 1,
                 "control.state_machine.lock_confirm_frames must be >= 1")
        _require(int(sm.get("loss_declare_frames", 5)) >= 1,
                 "control.state_machine.loss_declare_frames must be >= 1")
        _require(float(sm.get("coast_timeout_seconds", 1.0)) > 0,
                 "control.state_machine.coast_timeout_seconds must be positive")


@dataclass(frozen=True)
class TelemetryConfig:
    """Logging, metrics and automatic report generation.

    Attributes:
        enabled: Master switch for telemetry.
        output_dir: Directory for logs and reports.
        per_frame_csv: Emit a per-frame CSV record.
        per_frame_json: Emit a per-frame JSON record.
        summary_report: Auto-generate a summary report at end of run.
        summary_format: ``"html"``, ``"pdf"`` or ``"both"``.
        include_config_snapshot: Embed the full configuration in the report for reproducibility.
        include_metric_definitions: Write the exact metric definitions into every log header.
            The specification is ambiguous about several of these; stating our definitions
            removes the ambiguity for the evaluator rather than leaving it to be guessed.
        split_acquisition_populations: Report in-FOV and search-limited acquisition separately.
        capacity_benchmark: Unthrottled pipeline-capacity benchmark settings.
        metrics: Target values from the specification, used for pass/fail annotation.
    """

    enabled: bool = True
    output_dir: str = "logs"
    per_frame_csv: bool = True
    per_frame_json: bool = False
    summary_report: bool = True
    summary_format: str = "html"
    include_config_snapshot: bool = True
    include_metric_definitions: bool = True
    split_acquisition_populations: bool = True
    capacity_benchmark: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def acquisition_target_s(self) -> float:
        """Acquisition time budget from the specification, in seconds (parameter 16)."""
        return float(self.metrics.get("acquisition_target_s", 2.0))

    @property
    def fps_target(self) -> float:
        """Processing throughput target from the specification, in FPS (parameter 20)."""
        return float(self.metrics.get("fps_target", 20.0))

    def validate(self) -> None:
        """Check telemetry settings.

        Raises:
            ConfigError: On an unknown summary format, non-positive targets, or a capacity
                benchmark configured with no measured frames.
        """
        _require(self.summary_format in ("html", "pdf", "both"),
                 f"telemetry.summary_format must be html/pdf/both, got {self.summary_format!r}")
        _require(self.output_dir != "", "telemetry.output_dir must not be empty")
        for key in ("acquisition_target_s", "reacquisition_target_s",
                    "tracking_error_target_px", "fps_target"):
            if key in self.metrics:
                _require(float(self.metrics[key]) > 0,
                         f"telemetry.metrics.{key} must be positive, got {self.metrics[key]}")
        if self.capacity_benchmark.get("enabled", False):
            frames = int(self.capacity_benchmark.get("frames", 0))
            _require(frames > 0,
                     "telemetry.capacity_benchmark.frames must be positive when the benchmark "
                     "is enabled")
            _require(int(self.capacity_benchmark.get("warmup_frames", 0)) >= 0,
                     "telemetry.capacity_benchmark.warmup_frames must be non-negative")


@dataclass(frozen=True)
class VideoInputConfig:
    """Mode B settings: pre-recorded video ingestion with the PTZ loop bypassed.

    Attributes:
        path: Path to the input ``.mp4``. Required when ``run.mode == "video"``.
        auto_detect_resolution: Never assume a frame size; read it from the file.
        force_grayscale: Convert colour input to single-channel at the source boundary, so no
            downstream module ever considers channel count.
        normalise_intensity: Rescale per frame; never assume a brightness range.
        ground_truth_path: Optional sidecar CSV of true centroids, if evaluators supply one.
        playback_realtime: Throttle to the file's native rate instead of decoding as fast as
            possible. Off by default so benchmark runs are not rate-limited by playback.
        colour_display: When ``True`` and the video source provides colour frames, the GUI
            displays them in colour. The vision pipeline always receives a single-channel grayscale
            ``frame``; this flag governs the optional ``display_frame`` field only.
    """

    path: Optional[str] = None
    auto_detect_resolution: bool = True
    force_grayscale: bool = True
    normalise_intensity: bool = True
    ground_truth_path: Optional[str] = None
    colour_display: bool = False
    playback_realtime: bool = False

    def validate(self, mode: str) -> None:
        """Check video-input settings for the active run mode.

        Args:
            mode: The active ``run.mode``.

        Raises:
            ConfigError: If video mode is selected without a path, or a supplied path does not
                exist.
        """
        if mode == "video":
            _require(self.path is not None,
                     "run.mode='video' requires video_input.path to be set")
            assert self.path is not None
            _require(Path(self.path).exists(),
                     f"video_input.path does not exist: {self.path}")
        if self.ground_truth_path is not None:
            _require(Path(self.ground_truth_path).exists(),
                     f"video_input.ground_truth_path does not exist: {self.ground_truth_path}")


@dataclass(frozen=True)
class GuiConfig:
    """GUI view-layer settings. Never imported by core modules.

    Attributes:
        enabled: Show the GUI.
        display_rate_hz: Repaint rate, independent of the frame and control clocks.
        show_ground_truth_overlay: Draw the true centroid when available.
        show_estimate_overlay: Draw the estimated centroid.
        show_roi_overlay: Draw the active ROI box.
        show_search_pattern: Draw the acquisition scan path.
        plot_history_seconds: Rolling window for the live strip charts.
    """

    enabled: bool = True
    display_rate_hz: float = 30.0
    show_ground_truth_overlay: bool = True
    show_estimate_overlay: bool = True
    show_roi_overlay: bool = True
    show_search_pattern: bool = True
    plot_history_seconds: float = 30.0

    def validate(self) -> None:
        """Check GUI settings.

        Raises:
            ConfigError: On a non-positive display rate or plot history window.
        """
        _require(self.display_rate_hz > 0, "gui.display_rate_hz must be positive")
        _require(self.plot_history_seconds > 0, "gui.plot_history_seconds must be positive")


@dataclass(frozen=True)
class PerformanceConfig:
    """Throughput and concurrency settings.

    Attributes:
        use_numba: JIT-compile the residual Python loops.
        threading_enabled: Run producer and consumer on separate threads.
        frame_queue_size: Bounded queue depth between them. Bounded on purpose: an unbounded
            queue converts a throughput deficit into unbounded memory growth and latency instead
            of a visible dropped-frame count.
        profile_enabled: Enable cProfile instrumentation.
    """

    use_numba: bool = True
    threading_enabled: bool = True
    frame_queue_size: int = 4
    profile_enabled: bool = False

    def validate(self) -> None:
        """Check performance settings.

        Raises:
            ConfigError: If the frame queue size is not at least 1.
        """
        _require(self.frame_queue_size >= 1,
                 f"performance.frame_queue_size must be >= 1, got {self.frame_queue_size}")


@dataclass(frozen=True)
class AiConfig:
    """Optional lightweight CNN fallback. Classical CV remains the primary path.

    Attributes:
        enabled: Enable the AI validator.
        model_path: Path to the ONNX model.
        backend: Inference backend.
        invoke_on: ``"never"``, ``"classical_failure"`` (default) or ``"always"``.
        confidence_threshold: Minimum confidence to accept a detection.
        max_inference_ms: Per-frame inference budget. Exceeding it must not be allowed to break
            the >= 20 FPS requirement, so the caller treats this as a hard timeout.
        flux_ratio_threshold: Minimum flux ratio between the top two candidates that must be
            exceeded before the discriminator is invoked. When ``detections[0].flux /
            detections[1].flux > flux_ratio_threshold``, classical flux ranking is already
            reliable and the AI is skipped. Only used when ``invoke_on="multi_candidate"``.
    """

    enabled: bool = False
    model_path: Optional[str] = None
    backend: str = "onnxruntime"
    invoke_on: str = "classical_failure"
    confidence_threshold: float = 0.5
    max_inference_ms: float = 20.0
    flux_ratio_threshold: float = 1.5

    def validate(self) -> None:
        """Check AI settings.

        Raises:
            ConfigError: If enabled without a model path, or given an unknown invocation mode or
                an out-of-range confidence threshold.
        """
        _require(self.invoke_on in ("never", "classical_failure", "always", "multi_candidate"),
                 f"ai.invoke_on must be never/classical_failure/always/multi_candidate, got {self.invoke_on!r}")
        _require(0.0 <= self.confidence_threshold <= 1.0,
                 f"ai.confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        _require(self.max_inference_ms > 0, "ai.max_inference_ms must be positive")
        _require(self.flux_ratio_threshold >= 1.0,
                 f"ai.flux_ratio_threshold must be >= 1.0, got {self.flux_ratio_threshold}")
        if self.enabled:
            _require(self.model_path is not None,
                     "ai.enabled is true but ai.model_path is not set")
            assert self.model_path is not None
            _require(Path(self.model_path).exists(),
                     f"ai.model_path does not exist: {self.model_path}")


# --------------------------------------------------------------------------------------------
# Top-level configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """The complete, validated configuration for one run.

    Construct via :meth:`from_dict` or :func:`load_config`; both validate on construction, so an
    :class:`AppConfig` instance is guaranteed internally consistent and within specification.

    Attributes:
        run: Top-level run settings.
        scene: Virtual world canvas.
        camera: Virtual pan/tilt camera and mechanical limits.
        target: Beacon appearance and motion.
        noise: Noise, atmosphere and disturbance settings.
        vision: Vision pipeline, parameterised by spot scale.
        filtering: Kalman filter and track management.
        control: Control loop, state machine and search.
        telemetry: Logging and reporting.
        video_input: Mode B settings.
        gui: View-layer settings.
        performance: Throughput and concurrency.
        ai: Optional CNN fallback.
        source_path: Path the configuration was loaded from, if any.
        override_paths: Paths of any partial override documents applied on top of the base
            configuration, in the order applied. Recorded so the log header states exactly which
            scenario files produced a given result -- without this, a benchmark number cannot be
            reproduced from the log alone.
    """

    run: RunConfig = field(default_factory=RunConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    video_input: VideoInputConfig = field(default_factory=VideoInputConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    source_path: Optional[str] = None
    override_paths: Tuple[str, ...] = ()

    # ---------------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source_path: Optional[str] = None,
                  override_paths: Sequence[str] = ()) -> "AppConfig":
        """Build and validate a configuration from a parsed JSON mapping.

        Args:
            raw: The parsed configuration document. Keys beginning with ``"_"`` are treated as
                documentation and ignored.
            source_path: Optional originating path, recorded for the log header.
            override_paths: Paths of any partial overrides already merged into ``raw``, recorded
                for the log header.

        Returns:
            A validated :class:`AppConfig`.

        Raises:
            ConfigError: If any block is malformed, contains an unknown key, or violates a
                specification limit.
        """
        vision_raw = _section(raw, "vision")
        vision = VisionConfig(
            spot_scale=_build(SpotScaleConfig, _section(vision_raw, "spot_scale", "vision"),
                              "vision.spot_scale"),
            preprocess=_build(PreprocessConfig, _section(vision_raw, "preprocess", "vision"),
                              "vision.preprocess"),
            detection=_build(DetectionConfig, _section(vision_raw, "detection", "vision"),
                             "vision.detection"),
            centroid=_build(CentroidConfig, _section(vision_raw, "centroid", "vision"),
                            "vision.centroid"),
            roi=_build(RoiConfig, _section(vision_raw, "roi", "vision"), "vision.roi"),
        )

        config = cls(
            run=_build(RunConfig, _section(raw, "run"), "run"),
            scene=_build(SceneConfig, _section(raw, "scene"), "scene"),
            camera=_build(CameraConfig, _section(raw, "camera"), "camera"),
            target=_build(TargetConfig, _section(raw, "target"), "target"),
            noise=_build(NoiseConfig, _section(raw, "noise"), "noise"),
            vision=vision,
            filtering=_build(FilteringConfig, _section(raw, "filtering"), "filtering"),
            control=_build(ControlConfig, _section(raw, "control"), "control"),
            telemetry=_build(TelemetryConfig, _section(raw, "telemetry"), "telemetry"),
            video_input=_build(VideoInputConfig, _section(raw, "video_input"), "video_input"),
            gui=_build(GuiConfig, _section(raw, "gui"), "gui"),
            performance=_build(PerformanceConfig, _section(raw, "performance"), "performance"),
            ai=_build(AiConfig, _section(raw, "ai"), "ai"),
            source_path=source_path,
            override_paths=tuple(override_paths),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate every block and the cross-block relationships between them.

        Raises:
            ConfigError: On the first violation found.
        """
        self.run.validate()
        self.scene.validate()
        self.camera.validate()
        self.target.validate()
        self.noise.validate()
        self.vision.validate()
        self.filtering.validate()
        self.control.validate()
        self.telemetry.validate()
        self.video_input.validate(self.run.mode)
        self.gui.validate()
        self.performance.validate()
        self.ai.validate()

        # --- Cross-block consistency ------------------------------------------------------
        cam_w, cam_h = self.camera.fov_px
        _require(cam_w <= self.scene.width and cam_h <= self.scene.height,
                 f"camera viewport {cam_w}x{cam_h} does not fit inside the "
                 f"{self.scene.width}x{self.scene.height} scene")

        if self.camera.initial_position == "custom":
            assert self.camera.initial_pan_px is not None
            assert self.camera.initial_tilt_px is not None
            _require(0 <= self.camera.initial_pan_px < self.scene.width
                     and 0 <= self.camera.initial_tilt_px < self.scene.height,
                     "camera custom initial position lies outside the scene")

        if self.target.initial_position == "custom":
            assert self.target.initial_x is not None
            assert self.target.initial_y is not None
            _require(0 <= self.target.initial_x < self.scene.width
                     and 0 <= self.target.initial_y < self.scene.height,
                     "target custom initial position lies outside the scene")

        _require(self.control.update_rate_hz <= self.camera.update_rate_hz,
                 f"control.update_rate_hz ({self.control.update_rate_hz}) exceeds "
                 f"camera.update_rate_hz ({self.camera.update_rate_hz}); the controller cannot "
                 f"run faster than frames arrive")

        # The ROI must be able to contain the target's per-frame motion, or the target leaves
        # the ROI between frames and the tracker drops lock for a reason no threshold can fix.
        geometry = self.vision.resolve_geometry(None)
        resolved, clamped = self.roi_size_px(geometry.fwhm_px)
        if clamped:
            warnings.warn(
                f"The slew-aware ROI needs {self.roi_size_unclamped_px(geometry.fwhm_px):.0f} px "
                f"but vision.roi.max_size_* caps it at {resolved} px. The target may leave the "
                f"ROI between frames. Raise vision.roi.max_size_fwhm_multiple, reduce "
                f"camera.max_pan_speed_deg_s, or reduce camera jitter.",
                RuntimeWarning,
                stacklevel=2,
            )

    def roi_size_unclamped_px(self, fwhm_px: Optional[float] = None,
                              steerable: bool = True) -> float:
        """The ROI side length the physics demands, before clamping.

        Sized from the largest displacement the target can have relative to the ROI centre
        between two frames, which is what determines whether it is still inside the window when
        the next frame arrives:

        * **Boresight slew** -- the camera may move up to ``camera.max_px_per_frame`` while
          chasing the target, and the ROI is placed on the *predicted* position rather than the
          realised one.
        * **Camera jitter** -- unobservable by construction (it is why ``KalmanParams`` carries
          ``unobservable_sigma_px``), so it cannot be predicted out and must be budgeted for.
        * **Spot extent** -- the window has to contain the whole spot, not merely its centre.

        Half-width is therefore ``boresight + jitter``, and the full side length twice that plus
        the spot. Deriving it from the slew ceiling rather than a constant is what lets the same
        configuration work at 640x480 and 2000x2000: ``deg_per_pixel`` shrinks with resolution,
        so the same 5 deg/s ceiling is 27 px/frame at 640x480 and 111 px/frame at 2000x2000, and
        a fixed 64 px window is comfortable at one and hopeless at the other.

        Args:
            fwhm_px: Spot scale. ``None`` uses the configured fallback.
            steerable: Whether the frame source actually has a pan/tilt camera. Take it from
                ``FrameSource.supports_pan_tilt``, never from a mode string. With pre-recorded
                video there is no boresight to slew and no camera jitter to absorb, so budgeting
                for either sizes the window from hardware the run does not have.

        Returns:
            The required side length in pixels, unclamped.
        """
        geometry = self.vision.resolve_geometry(fwhm_px)
        if not steerable:
            return 2.0 * geometry.fwhm_px
        jitter_px = float(self.noise.camera_jitter.get("max_px_per_frame", 0.0)) \
            if self.noise.camera_jitter.get("enabled") else 0.0
        pan_px, tilt_px = self.camera.max_px_per_frame
        half = max(pan_px, tilt_px) + jitter_px
        return 2.0 * half + 2.0 * geometry.fwhm_px

    def roi_size_px(self, fwhm_px: Optional[float] = None,
                    steerable: bool = True) -> Tuple[int, bool]:
        """The ROI side length to actually use, clamped to the configured ceiling.

        Args:
            fwhm_px: Spot scale. ``None`` uses the configured fallback.
            steerable: Whether the source has a pan/tilt camera. See
                :meth:`roi_size_unclamped_px`.

        Returns:
            ``(size_px, clamped)``. ``clamped`` is true when the physics wanted a larger window
            than ``vision.roi.max_size_*`` permits, which is a real risk of losing the target
            rather than a tuning preference -- hence the startup warning.
        """
        geometry = self.vision.resolve_geometry(fwhm_px)
        required = self.roi_size_unclamped_px(fwhm_px, steerable=steerable)
        ceiling = float(geometry.max_roi_size_px)
        # Never smaller than the scale-derived window: at low slew rates the spot itself, not the
        # motion, sets the useful size.
        size = max(required, float(geometry.roi_size_px))
        return int(round(min(size, ceiling))), size > ceiling

    # ---------------------------------------------------------------------------------------
    # Derived quantities
    # ---------------------------------------------------------------------------------------

    @property
    def search_arm_spacing_px(self) -> float:
        """Spiral search arm spacing in pixels, derived from the camera FOV."""
        return self.control.search_arm_spacing_px(self.camera)

    @property
    def worst_case_search_time_s(self) -> float:
        """Worst-case time to sweep the full uncertainty region, in seconds.

        See :meth:`ControlConfig.worst_case_search_time_s` for the derivation and for why this
        number is surfaced at startup rather than discovered during Phase 4.
        """
        return self.control.worst_case_search_time_s(self.camera, self.scene)

    @property
    def search_is_within_acquisition_budget(self) -> bool:
        """Whether worst-case search fits inside the specified acquisition budget.

        Returns:
            True if :attr:`worst_case_search_time_s` does not exceed
            ``telemetry.metrics.acquisition_target_s``. False for the default configuration, by
            arithmetic rather than by any deficiency of the algorithm.
        """
        return self.worst_case_search_time_s <= self.telemetry.acquisition_target_s

    def startup_warnings(self) -> List[str]:
        """Return human-readable warnings about the active configuration.

        These are findings rather than errors: the run is valid and should proceed, but the
        numbers here materially affect how its results must be interpreted, so they belong in
        the log header and on the console at startup rather than in a post-hoc analysis.

        Returns:
            A list of warning strings, empty when the configuration raises no concerns.
        """
        messages: List[str] = []

        if not self.search_is_within_acquisition_budget:
            messages.append(
                f"Worst-case spiral search takes {self.worst_case_search_time_s:.1f} s to sweep "
                f"the {self.scene.width}x{self.scene.height} canvas at "
                f"{self.search_arm_spacing_px:.0f} px arm spacing, which EXCEEDS the "
                f"{self.telemetry.acquisition_target_s:.1f} s acquisition budget. This is set by "
                f"canvas area, FOV and the slew ceiling alone -- no algorithm change removes it. "
                f"Acquisition must therefore be reported as two populations (in-FOV vs "
                f"search-limited); see docs/DESIGN.md section 7.5."
            )

        if self.control.pid_gains_are_provisional:
            messages.append(
                "PID gains are marked PROVISIONAL: analytically sized, not validated against a "
                "step-response test. No performance number produced with these gains may be "
                "quoted in the report, the demo, or a benchmark submission until the Phase 4 "
                "step-response test passes. See docs/ROADMAP.md Phase 4 HARD GATE."
            )

        if not self.vision.spot_scale.estimation_enabled:
            messages.append(
                "vision.spot_scale.estimation_enabled is false, so all vision geometry uses "
                "fixed absolute pixel fallbacks. This reproduces the failure mode we are "
                "specifically defending against in Benchmark-2 and should only be used for "
                "controlled comparisons."
            )

        if self.run.mode == "video" and self.camera.deg_per_pixel and self.video_input.path:
            messages.append(
                "Mode B: the virtual PTZ camera is bypassed and angular scale is unknown for "
                "evaluator video. All errors will be reported in pixels only; angular metrics "
                "are omitted from the logs."
            )

        return messages

    def metric_definitions(self, steerable_camera: bool = True) -> Dict[str, str]:
        """Return the exact metric definitions used by this run.

        The specification is genuinely ambiguous about several of these. Writing our definitions
        into every log header removes the ambiguity for the evaluator instead of leaving it to be
        inferred, which is worth marks in both benchmark stages.

        Args:
            steerable_camera: Whether the frame source has a steerable camera, i.e.
                ``FrameSource.supports_pan_tilt``. This changes the lock criterion, so **both
                forms are stated in the header and the active one is named** -- a Benchmark-2 log
                should declare on its face why its lock criterion differs from a Mode A log,
                rather than leaving an evaluator to infer it.

        Returns:
            Mapping of metric name to its precise definition.
        """
        sm = self.control.state_machine
        k = int(sm.get("lock_confirm_frames", 3))
        n = int(sm.get("loss_declare_frames", 5))
        return {
            "coordinate_convention": (
                "Pixel centres at integer indices; origin at the top-left pixel; tuples ordered "
                f"(x, y). A {self.camera.resolution_width}x{self.camera.resolution_height} frame "
                f"has its boresight at {self.camera.boresight_px}."
            ),
            "acquisition_time_s": (
                f"Clock starts at the first frame of the run; stops on the first frame the lock "
                f"criterion has held for K={k} consecutive frames. Reported as two separate "
                f"populations: in-FOV (beacon inside the initial viewport) and search-limited "
                f"(beacon outside it). Never pooled."
            ),
            "lock_criterion": (
                "ACTIVE (" + ("steerable camera" if steerable_camera else "no steerable camera")
                + "): " + (
                    "a valid detection whose SNR exceeds the adaptive threshold, whose centroid "
                    "passes the Kalman Mahalanobis validation gate, AND whose pointing error is "
                    f"within {self.control.state_machine.get('lock_window_px', 40)} px of "
                    "boresight."
                    if steerable_camera else
                    "a valid detection whose SNR exceeds the adaptive threshold AND whose "
                    "centroid passes the Kalman Mahalanobis validation gate. The pointing-error "
                    "window is deliberately EXCLUDED."
                )
            ),
            "lock_criterion_rationale": (
                "The criterion is selected from FrameSource.supports_pan_tilt, never from a mode "
                "string. With a steerable camera, holding the target near boresight is the "
                "objective, so failing to do so is a real loss of lock. With pre-recorded video "
                "there is no camera to steer: the target's distance from frame centre is a "
                "property of the file rather than of our tracking, so including it would report "
                "a perfectly tracked beacon as never locked and suppress the centroiding "
                "statistics that Benchmark-2 scores."
            ),
            "reacquisition_time_s": (
                f"Clock starts when an established lock is lost (N={n} consecutive missed "
                f"detections); stops when the lock criterion is re-satisfied."
            ),
            "centroid_error_px": (
                "Euclidean distance between the estimated centroid and the ground-truth "
                "centroid, in pixels, per frame. This is the quantity Benchmark-2 scores."
            ),
            "pointing_error_px": (
                "Euclidean distance between the target position and the camera boresight "
                "(viewport centre), in pixels, per frame. Logged separately from centroid error "
                "because the spec's 'tracking error <= 10 px' is ambiguous between the two."
            ),
            "rmse_px": "sqrt(mean(error^2)) over frames where lock was held.",
            "loss_rate": "Frames without valid lock divided by frames where the target was present.",
            "frame_rate_hz": (
                f"Frame generation clock, nominally {self.camera.update_rate_hz} Hz. Distinct "
                f"from the control and processing clocks."
            ),
            "control_rate_hz": (
                f"Control update clock, nominally {self.control.update_rate_hz} Hz. Distinct "
                f"from the frame and processing clocks."
            ),
            "processing_fps": (
                "End-to-end frames processed per wall-clock second during a real-time run. "
                "Capped by the frame generation clock, so it understates capability; see "
                "pipeline_capacity_fps."
            ),
            "pipeline_capacity_fps": (
                "Maximum sustainable throughput measured unthrottled, with rendering, noise "
                "synthesis, GUI and disk I/O excluded from the timed region. This is the figure "
                "that substantiates the >= 20 FPS requirement with headroom."
            ),
        }

    def summary_lines(self) -> List[str]:
        """Return a human-readable summary of the derived quantities for this configuration.

        Suitable for printing at startup and for embedding in a log header.

        Returns:
            A list of formatted lines.
        """
        dpp_x, dpp_y = self.camera.deg_per_pixel
        pan_px, tilt_px = self.camera.max_px_per_frame
        geometry = self.vision.resolve_geometry(None)
        override_line = (", ".join(self.override_paths) if self.override_paths else "(none)")
        return [
            f"mode                    : {self.run.mode}",
            f"scenario overrides      : {override_line}",
            f"scene                   : {self.scene.width} x {self.scene.height} px",
            f"viewport                : {self.camera.resolution_width} x "
            f"{self.camera.resolution_height} px",
            f"boresight (x, y)        : {self.camera.boresight_px[0]:.1f}, "
            f"{self.camera.boresight_px[1]:.1f}  (pixel-centre convention)",
            f"deg per pixel           : {dpp_x:.6f} h, {dpp_y:.6f} v",
            f"slew ceiling            : {self.camera.max_pan_speed_deg_s:.1f} deg/s pan, "
            f"{self.camera.max_tilt_speed_deg_s:.1f} deg/s tilt",
            f"max travel per frame    : {pan_px:.1f} px pan, {tilt_px:.1f} px tilt "
            f"@ {self.camera.update_rate_hz:.0f} Hz",
            f"search arm spacing      : {self.search_arm_spacing_px:.0f} px (FOV-derived)",
            f"worst-case search time  : {self.worst_case_search_time_s:.1f} s "
            f"(budget {self.telemetry.acquisition_target_s:.1f} s)",
            f"fallback spot FWHM      : {geometry.fwhm_px:.2f} px",
            f"fallback ROI            : {geometry.roi_size_px} px",
            f"threshold operator      : {self.vision.detection.threshold_method}",
            f"centroid estimator      : {self.vision.centroid.method}",
        ]


def merge_overrides(base: Mapping[str, Any], override: Mapping[str, Any],
                    _path: str = "") -> Dict[str, Any]:
    """Recursively merge a partial override document on top of a base configuration.

    This is how evaluator scenarios are applied: Benchmark Performance-1 supplies scenarios we
    have never seen, and they must be expressible as a small file naming only the values that
    differ, rather than as a full copy of ``config/default.json`` that would silently freeze
    every other default at whatever it happened to be when the scenario was written.

    Merge rules, chosen so a scenario cannot surprise us:

    * **Nested objects merge recursively.** Setting ``noise.gaussian.sigma`` leaves
      ``noise.gaussian.enabled`` and every sibling block untouched.
    * **Scalars are replaced.**
    * **Lists are replaced wholesale, never concatenated.** A scenario overriding
      ``noise.pipeline_order`` means "use exactly this order"; appending would produce a
      pipeline that runs stages twice, which is both wrong and very hard to spot in a log.
    * **Underscore-prefixed keys in the override are ignored**, exactly as in the base document,
      so a scenario may carry its own commentary.
    * A type conflict between a block and a scalar is an error rather than a silent replacement,
      since it always means the override targets the wrong key.

    Args:
        base: The base configuration document.
        override: A partial document naming only the values to change.
        _path: Internal dotted-path accumulator used for error messages.

    Returns:
        A new merged dict. Neither input is modified.

    Raises:
        ConfigError: If the override replaces an object with a scalar, or vice versa.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        full = f"{_path}.{key}" if _path else key
        if key in merged and isinstance(merged[key], Mapping) != isinstance(value, Mapping):
            raise ConfigError(
                f"Override for '{full}' changes the shape of the configuration: base holds "
                f"{'an object' if isinstance(merged[key], Mapping) else 'a scalar'} but the "
                f"override supplies {'an object' if isinstance(value, Mapping) else 'a scalar'}. "
                f"This almost always means the override targets the wrong key."
            )
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = merge_overrides(merged[key], value, full)
        else:
            merged[key] = value
    return merged


def load_json_document(path: str | Path) -> Dict[str, Any]:
    """Read and parse a JSON configuration document.

    Args:
        path: Path to the document.

    Returns:
        The parsed top-level object.

    Raises:
        ConfigError: If the file is missing, is not valid JSON, or does not contain a JSON
            object at the top level.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Configuration file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Configuration file {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file {p} must contain a JSON object at the top level")
    return raw


def load_config(path: str | Path,
                overrides: Optional[Sequence[str | Path]] = None) -> AppConfig:
    """Load a configuration, apply any partial overrides, and validate the result.

    Validation runs **after** merging, never before, so an evaluator scenario cannot push the
    system out of specification: a scenario setting ``noise.gaussian.sigma`` to 25 fails on load
    with a message citing spec parameter 22, rather than producing a completed run whose numbers
    are quietly invalid.

    Args:
        path: Path to the base JSON configuration document.
        overrides: Optional paths to partial override documents, applied in order. Later
            overrides win over earlier ones.

    Returns:
        A validated :class:`AppConfig` recording both the base path and the override paths.

    Raises:
        ConfigError: If any file is missing or malformed, an override changes the shape of the
            configuration, or the merged result fails validation.
    """
    raw = load_json_document(path)
    applied: List[str] = []
    for override_path in overrides or ():
        raw = merge_overrides(raw, load_json_document(override_path))
        applied.append(str(override_path))
    return AppConfig.from_dict(raw, source_path=str(Path(path)), override_paths=applied)
