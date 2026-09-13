"""Sub-pixel beacon rendering with exact ground truth.

The beacon is rendered on a supersampled grid and block-averaged down, so its commanded centre
``(xc, yc)`` may be an arbitrary non-integer position and the ground-truth centroid is *exactly*
that position rather than an approximation of it. Ground truth being exact is the premise every
accuracy claim in this project rests on (``CLAUDE.md`` -> constraint 4): the centroid-error-vs-SNR
curve measures the estimator against these numbers, so any bias here is silently inherited by
every result we report.

Coordinate convention
---------------------
Pixel centres sit at integer indices, origin at the top-left pixel, tuples ordered ``(x, y)``.
See ``CLAUDE.md`` -> Coordinate conventions.

The whole correctness of this module reduces to one mapping. For a supersample factor ``S``,
supersample index ``C`` corresponds to output coordinate::

    x_out = (C + 0.5) / S - 0.5

The natural-looking ``x_out = C / S`` is **wrong** and introduces a systematic ``(S-1)/(2S)``
pixel offset -- 0.375 px at S=4. That is large enough to invalidate the SNR curve while being
small enough to survive an inattentive test, which is why :func:`supersample_to_output` is a
public function tested in isolation to <0.01 px, separately from the end-to-end render test.

A sanity check on the mapping: at ``S=4``, output pixel 0 spans ``[-0.5, +0.5]`` and its four
supersamples land at ``-0.375, -0.125, +0.125, +0.375``. Their mean is exactly ``0.0``, the
centre of pixel 0 -- so block-averaging consecutive groups of ``S`` supersamples starting at
``C=0`` reconstructs the output pixel grid with no offset. Under ``C/S`` the same four samples
would land at ``0, 0.25, 0.5, 0.75`` with mean ``0.375``.

See ``docs/DESIGN.md`` section 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from src.config import FWHM_PER_SIGMA, AppConfig, ConfigError

__all__ = [
    "BeaconParams",
    "BeaconPatch",
    "supersample_to_output",
    "output_to_supersample",
    "render_beacon",
    "centroid_of",
]

#: Numeric type accepted by the coordinate mapping helpers.
Coordinate = Union[float, np.ndarray]


def supersample_to_output(index: Coordinate, factor: int) -> Coordinate:
    """Map a supersample-grid index to its centre in output-pixel coordinates.

    This is *the* coordinate mapping of the renderer. Supersample index ``C`` covers the output
    interval ``[C/S - 0.5, (C+1)/S - 0.5]``, whose centre is ``(C + 0.5)/S - 0.5``.

    Args:
        index: Supersample index, or an array of indices. Need not be integral.
        factor: Supersample factor ``S``. Must be at least 1.

    Returns:
        The corresponding output-pixel coordinate(s), same shape as ``index``.

    Raises:
        ValueError: If ``factor`` is less than 1.

    Examples:
        At ``S=1`` the mapping is the identity, since supersampling is disabled::

            >>> supersample_to_output(3.0, 1)
            3.0

        At ``S=4`` the four supersamples of output pixel 0 straddle zero symmetrically::

            >>> [round(supersample_to_output(c, 4), 3) for c in range(4)]
            [-0.375, -0.125, 0.125, 0.375]
    """
    if factor < 1:
        raise ValueError(f"Supersample factor must be at least 1, got {factor}")
    return (index + 0.5) / factor - 0.5


def output_to_supersample(coord: Coordinate, factor: int) -> Coordinate:
    """Map an output-pixel coordinate to its position on the supersample grid.

    Exact inverse of :func:`supersample_to_output`.

    Args:
        coord: Output-pixel coordinate, or an array of them.
        factor: Supersample factor ``S``. Must be at least 1.

    Returns:
        The corresponding supersample-grid index/indices.

    Raises:
        ValueError: If ``factor`` is less than 1.
    """
    if factor < 1:
        raise ValueError(f"Supersample factor must be at least 1, got {factor}")
    return (coord + 0.5) * factor - 0.5


@dataclass(frozen=True)
class BeaconParams:
    """Rendering parameters for a single beacon.

    Attributes:
        shape: ``"gaussian"``, ``"square"`` or ``"circle"``.
        size_px: Nominal extent for hard-edged shapes, in pixels.
        sigma_px: Gaussian standard deviation in pixels. Also the width of the PSF convolved
            onto hard-edged shapes when ``psf_sigma_px`` is not given.
        peak_intensity: Peak grey level of the rendered profile.
        supersample_factor: Supersample factor ``S``.
        support_sigma_multiple: Patch half-width in units of sigma, for Gaussian shapes. The
            default of 5 leaves a truncated tail below 1e-5 of peak, so truncation cannot bias
            the centroid at the precision we care about.
        psf_sigma_px: Optional Gaussian blur applied to hard-edged shapes. ``None`` leaves the
            supersampled hard edge, anti-aliased by the downsample alone.
    """

    shape: str = "gaussian"
    size_px: float = 10.0
    sigma_px: float = 2.5
    peak_intensity: float = 255.0
    supersample_factor: int = 4
    support_sigma_multiple: float = 5.0
    psf_sigma_px: Optional[float] = None

    @property
    def fwhm_px(self) -> float:
        """Full width at half maximum of the rendered profile, in pixels."""
        if self.shape == "gaussian":
            return FWHM_PER_SIGMA * self.sigma_px
        return float(self.size_px)

    @property
    def support_radius_px(self) -> float:
        """Half-width of the render patch in output pixels.

        Chosen so the profile is negligible at the patch edge. Truncating a symmetric profile
        inside an integer-aligned patch is very slightly asymmetric about a sub-pixel centre, so
        the support must be generous enough that the asymmetry is far below our error budget.

        Returns:
            Patch half-width in pixels.
        """
        if self.shape == "gaussian":
            return self.support_sigma_multiple * self.sigma_px
        # Hard shapes need only their own extent plus room for anti-aliasing / PSF.
        blur = self.psf_sigma_px if self.psf_sigma_px is not None else 0.0
        return self.size_px / 2.0 + 3.0 * blur + 2.0

    @classmethod
    def from_config(cls, config: AppConfig) -> "BeaconParams":
        """Build rendering parameters from an application configuration.

        Args:
            config: Validated application configuration.

        Returns:
            The corresponding :class:`BeaconParams`.
        """
        target = config.target
        return cls(
            shape=target.shape,
            size_px=float(target.size_px),
            sigma_px=float(target.gaussian_sigma_px),
            peak_intensity=float(target.peak_intensity),
            supersample_factor=int(target.supersample_factor),
        )

    def validate(self) -> None:
        """Check rendering parameters.

        Raises:
            ConfigError: On an unknown shape or a non-positive size, sigma or supersample factor.
        """
        if self.shape not in ("gaussian", "square", "circle"):
            raise ConfigError(f"Unknown beacon shape: {self.shape!r}")
        if self.sigma_px <= 0:
            raise ConfigError(f"Beacon sigma must be positive, got {self.sigma_px}")
        if self.size_px <= 0:
            raise ConfigError(f"Beacon size must be positive, got {self.size_px}")
        if self.supersample_factor < 1:
            raise ConfigError(
                f"Supersample factor must be at least 1, got {self.supersample_factor}"
            )


@dataclass(frozen=True)
class BeaconPatch:
    """A rendered beacon and its placement on the canvas.

    Rendering a small patch rather than the full 2000x2000 canvas is what keeps frame generation
    cheap: a default beacon occupies a 27x27 patch, roughly 5500 times smaller than the canvas.

    Attributes:
        data: 2-D ``float32`` array of intensities, not yet quantised to 8 bits.
        x0: Canvas x coordinate of the patch's top-left pixel centre.
        y0: Canvas y coordinate of the patch's top-left pixel centre.
        true_x: Commanded beacon centre x in canvas coordinates. Exact ground truth.
        true_y: Commanded beacon centre y in canvas coordinates. Exact ground truth.
    """

    data: np.ndarray
    x0: int
    y0: int
    true_x: float
    true_y: float

    @property
    def shape(self) -> Tuple[int, int]:
        """Patch size as ``(height, width)``."""
        return self.data.shape[0], self.data.shape[1]

    @property
    def local_true_xy(self) -> Tuple[float, float]:
        """Ground-truth centre expressed in patch-local coordinates as ``(x, y)``."""
        return self.true_x - self.x0, self.true_y - self.y0


def centroid_of(patch: np.ndarray, x0: float = 0.0, y0: float = 0.0) -> Tuple[float, float]:
    """Compute the intensity-weighted centroid of an array.

    This is a plain, unthresholded centre of gravity, used here to *verify* the renderer against
    its own ground truth on clean noiseless data. It is deliberately not the tracking estimator:
    unthresholded CoG is badly biased by background and impulse noise, so the pipeline in
    ``src/vision/centroid.py`` uses thresholded and iteratively-weighted variants instead.

    Args:
        patch: 2-D array of non-negative intensities.
        x0: Canvas x coordinate of the array's top-left pixel centre.
        y0: Canvas y coordinate of the array's top-left pixel centre.

    Returns:
        The centroid as ``(x, y)``, offset by ``(x0, y0)``.

    Raises:
        ValueError: If the array is not 2-D, or its total intensity is not positive.
    """
    if patch.ndim != 2:
        raise ValueError(f"Centroid requires a 2-D array, got shape {patch.shape}")
    weights = patch.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Cannot compute a centroid of an array with non-positive total intensity")
    rows = np.arange(weights.shape[0], dtype=np.float64)
    cols = np.arange(weights.shape[1], dtype=np.float64)
    cx = float((weights.sum(axis=0) * cols).sum() / total)
    cy = float((weights.sum(axis=1) * rows).sum() / total)
    return cx + x0, cy + y0


def _coverage_1d(offsets: np.ndarray, half: float, cell: float) -> np.ndarray:
    """Fraction of each sample cell that lies inside a centred interval of half-width ``half``.

    Each sample at offset ``c`` represents a cell spanning ``[c - cell/2, c + cell/2]``. This
    returns the fraction of that cell inside ``[-half, +half]``, which is exact area sampling in
    one dimension.

    Args:
        offsets: Sample-centre offsets from the shape centre, in output-pixel units.
        half: Half-width of the interval, in output-pixel units.
        cell: Width of one sample cell, in output-pixel units.

    Returns:
        Coverage fractions in ``[0, 1]``, same shape as ``offsets``.
    """
    lower = np.maximum(offsets - cell / 2.0, -half)
    upper = np.minimum(offsets + cell / 2.0, half)
    return np.clip((upper - lower) / cell, 0.0, 1.0)


def _profile(dx: np.ndarray, dy: np.ndarray, params: BeaconParams,
             cell: float = 1.0) -> np.ndarray:
    """Evaluate the beacon's intensity profile on a supersampled offset grid.

    Hard-edged shapes are **area-sampled**, not point-sampled. A binary inside/outside test at
    each supersample centre quantises the shape's edge to the supersample grid, which caps
    centroid accuracy at ``1/(2S)`` pixels -- 0.125 px at S=4, well outside our 0.05 px budget
    and impossible to fix downstream. Blurring afterwards does not recover it, because the
    position information is already lost by the time the blur is applied. Area sampling keeps
    the edge continuous, so the centroid stays exact at any sub-pixel phase.

    The Gaussian shape needs no such treatment: it is smooth, so every supersample carries
    partial information about the true centre.

    Args:
        dx: Horizontal offsets from the beacon centre, shape ``(1, W)``.
        dy: Vertical offsets from the beacon centre, shape ``(H, 1)``.
        params: Rendering parameters.
        cell: Width of one supersample cell in output-pixel units, i.e. ``1/S``.

    Returns:
        A ``(H, W)`` array of intensities normalised to a peak of 1.0.

    Raises:
        ConfigError: On an unknown shape.
    """
    if params.shape == "gaussian":
        return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * params.sigma_px ** 2))

    half = params.size_px / 2.0
    if params.shape == "square":
        # Separable, so one-dimensional coverage multiplies out to exact area coverage.
        return _coverage_1d(dx, half, cell) * _coverage_1d(dy, half, cell)
    if params.shape == "circle":
        # Not separable; ramp coverage linearly across one cell at the boundary, which is a
        # close approximation to true area coverage for a curve of this radius.
        radius = np.sqrt(dx ** 2 + dy ** 2)
        return np.clip(0.5 - (radius - half) / cell, 0.0, 1.0)
    raise ConfigError(f"Unknown beacon shape: {params.shape!r}")


def _block_average(grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a supersampled grid by averaging non-overlapping ``factor x factor`` blocks.

    Block averaging is the correct downsample here because it approximates each output pixel's
    integral of the continuous profile over its own area, which is what a real detector pixel
    measures. Its grouping -- consecutive supersamples starting at index 0 -- is what makes
    :func:`supersample_to_output` the correct coordinate mapping; the two must stay consistent.

    Args:
        grid: 2-D array whose dimensions are both exact multiples of ``factor``.
        factor: Supersample factor ``S``.

    Returns:
        The downsampled array, with both dimensions divided by ``factor``.

    Raises:
        ValueError: If either dimension is not a multiple of ``factor``.
    """
    height, width = grid.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Supersampled grid {grid.shape} is not an exact multiple of factor {factor}"
        )
    return grid.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _gaussian_blur(grid: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply a separable Gaussian blur, used as an optical PSF for hard-edged shapes.

    Implemented directly rather than via OpenCV so the renderer has no dependency on a specific
    border-handling convention: the kernel is normalised over its truncated support and applied
    with edge replication, which keeps a symmetric input symmetric and therefore preserves the
    centroid.

    Args:
        grid: 2-D array to blur.
        sigma_px: Blur standard deviation in units of the grid's own samples.

    Returns:
        The blurred array, same shape as the input.
    """
    if sigma_px <= 0:
        return grid
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma_px ** 2))
    kernel /= kernel.sum()

    padded = np.pad(grid, ((0, 0), (radius, radius)), mode="edge")
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)


def render_beacon(x: float, y: float, params: BeaconParams,
                  intensity_scale: float = 1.0) -> BeaconPatch:
    """Render a beacon centred at an arbitrary sub-pixel canvas position.

    The patch is aligned to the integer output-pixel grid and sized from
    :attr:`BeaconParams.support_radius_px`, so the commanded centre sits at a sub-pixel offset
    *within* the patch. That offset is the whole point: it is what makes the rendered spot carry
    genuine sub-pixel information rather than snapping to the nearest pixel.

    Args:
        x: Beacon centre x in canvas coordinates. May be non-integer.
        y: Beacon centre y in canvas coordinates. May be non-integer.
        params: Rendering parameters.
        intensity_scale: Multiplicative factor on the peak intensity, used for scintillation and
            range-dependent fading. 1.0 leaves the configured peak unchanged.

    Returns:
        A :class:`BeaconPatch` whose ``true_x``/``true_y`` are exactly the requested ``x``/``y``.

    Raises:
        ConfigError: If ``params`` is invalid.
        ValueError: If ``x`` or ``y`` is not finite.
    """
    params.validate()
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"Beacon centre must be finite, got ({x}, {y})")

    factor = params.supersample_factor
    half = int(math.ceil(params.support_radius_px))
    # Anchor the patch on the pixel nearest the true centre, so the sub-pixel offset within the
    # patch stays in [-0.5, +0.5] and the profile is never truncated asymmetrically.
    cx_pixel = int(math.floor(x + 0.5))
    cy_pixel = int(math.floor(y + 0.5))
    x0 = cx_pixel - half
    y0 = cy_pixel - half
    width = height = 2 * half + 1

    # Supersample indices -> output coordinates within the patch -> canvas coordinates.
    ss_cols = supersample_to_output(np.arange(width * factor, dtype=np.float64), factor)
    ss_rows = supersample_to_output(np.arange(height * factor, dtype=np.float64), factor)
    dx = (x0 + ss_cols - x).reshape(1, -1)
    dy = (y0 + ss_rows - y).reshape(-1, 1)

    grid = _profile(dx, dy, params, cell=1.0 / factor)

    if params.shape != "gaussian" and params.psf_sigma_px:
        # Blur on the supersampled grid, so the PSF sigma is expressed in output pixels.
        grid = _gaussian_blur(grid, params.psf_sigma_px * factor)

    patch = _block_average(grid, factor) * params.peak_intensity * intensity_scale
    return BeaconPatch(data=patch.astype(np.float32), x0=x0, y0=y0, true_x=float(x),
                       true_y=float(y))
