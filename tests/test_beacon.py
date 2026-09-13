"""Tests for sub-pixel beacon rendering.

Two tiers of test, deliberately kept separate (see ``docs/ROADMAP.md`` Phase 1):

* **Tier 1 -- the coordinate mapping in isolation, to <0.01 px.** Exercises only
  :func:`supersample_to_output` and block averaging, with the supersampled grid built directly
  in the test. Nothing from the renderer's patch anchoring, parameter handling or ``float32``
  storage participates. The mapping is exact arithmetic, so it earns a tolerance five times
  tighter than the end-to-end test can support.
* **Tier 2 -- the full render path, to <0.05 px.** Commanded position in, rendered patch out,
  centroid recovered.

Both tiers run at three position classes -- exact integer, exact half-integer, and an
irrational-ish offset. The first two are symmetric and can each pass under a wrong convention;
only the asymmetric third reliably fails one. A suite containing only integers and half-integers
looks thorough and is exactly what a ``C/S`` mapping would survive.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import ConfigError, load_config
from src.sim.beacon import (
    BeaconParams,
    _block_average,
    centroid_of,
    output_to_supersample,
    render_beacon,
    supersample_to_output,
)

#: Exact integer, exact half-integer, and an irrational-ish asymmetric offset.
POSITION_CLASSES = [
    pytest.param(100.0, id="exact_integer"),
    pytest.param(100.5, id="exact_half_integer"),
    pytest.param(100.37, id="irrational_ish"),
]

#: Supersample factors to exercise. S=1 must degenerate to the identity mapping.
FACTORS = [1, 2, 3, 4, 8]


# ------------------------------------------------------------------------------------------
# Tier 1: the coordinate mapping, in isolation
# ------------------------------------------------------------------------------------------


def test_mapping_known_values_at_factor_four() -> None:
    """The four supersamples of output pixel 0 must straddle zero symmetrically.

    Their mean is exactly 0.0 -- the centre of pixel 0 -- which is what makes block-averaging
    consecutive groups of S samples reconstruct the output grid with no offset.
    """
    values = [supersample_to_output(c, 4) for c in range(4)]
    assert values == pytest.approx([-0.375, -0.125, 0.125, 0.375])
    assert np.mean(values) == pytest.approx(0.0, abs=1e-15)


def test_mapping_is_identity_at_factor_one() -> None:
    """With supersampling disabled the mapping must not move anything."""
    for c in (0.0, 1.0, 17.0, 123.456):
        assert supersample_to_output(c, 1) == pytest.approx(c)


def test_mapping_round_trips() -> None:
    """output_to_supersample must invert supersample_to_output exactly."""
    for factor in FACTORS:
        coords = np.linspace(-5.0, 50.0, 97)
        back = supersample_to_output(output_to_supersample(coords, factor), factor)
        assert back == pytest.approx(coords, abs=1e-12)


def test_mapping_rejects_invalid_factor() -> None:
    """A factor below 1 is a programming error and must not be silently accepted."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            supersample_to_output(0.0, bad)
        with pytest.raises(ValueError, match="at least 1"):
            output_to_supersample(0.0, bad)


def _supersampled_gaussian(center_ss: float, factor: int, width_out: int,
                           sigma_ss: float) -> np.ndarray:
    """Build a supersampled 1-D-separable Gaussian directly, without the renderer.

    Args:
        center_ss: Centre of the profile, in supersample index units.
        factor: Supersample factor.
        width_out: Output width in pixels.
        sigma_ss: Profile sigma in supersample units.

    Returns:
        A ``(width_out*factor, width_out*factor)`` array.
    """
    idx = np.arange(width_out * factor, dtype=np.float64)
    line = np.exp(-((idx - center_ss) ** 2) / (2.0 * sigma_ss ** 2))
    return np.outer(line, line)


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("offset", [0.0, 0.5, 0.37])
def test_mapping_preserves_centroid_to_one_hundredth_pixel(factor: int, offset: float) -> None:
    """The supersample->output mapping must preserve centroid position to <0.01 px.

    This is the isolated test. It measures the centroid on the supersampled grid, maps that
    single number through :func:`supersample_to_output`, block-averages the grid, measures the
    centroid again in output space, and requires the two to agree. The renderer is not involved,
    so a failure here localises squarely to the mapping.
    """
    width_out = 41
    # Place the centre near the middle of the patch, displaced by the requested sub-pixel offset.
    center_out = width_out // 2 + offset
    center_ss = output_to_supersample(center_out, factor)
    sigma_ss = 2.5 * factor

    grid = _supersampled_gaussian(center_ss, factor, width_out, sigma_ss)

    measured_ss, _ = centroid_of(grid)
    expected_out = supersample_to_output(measured_ss, factor)
    measured_out, _ = centroid_of(_block_average(grid, factor))

    assert abs(measured_out - expected_out) < 0.01, (
        f"factor={factor} offset={offset}: mapping shifted the centroid by "
        f"{measured_out - expected_out:+.6f} px"
    )


def test_wrong_mapping_is_caught_by_the_isolated_test() -> None:
    """The natural-looking ``C/S`` mapping must fail, at the predicted (S-1)/(2S) magnitude.

    Without this, the isolated test above could pass vacuously. Here we deliberately build the
    grid under the wrong mapping and confirm the bias appears at exactly 0.375 px for S=4 --
    38 times the 0.01 px tolerance, so the test has ample margin to catch it.
    """
    factor, width_out = 4, 41
    center_out = width_out // 2 + 0.37
    sigma_ss = 2.5 * factor

    # Build the grid as if supersample index C mapped to output coordinate C/S.
    idx = np.arange(width_out * factor, dtype=np.float64)
    wrong_coords = idx / factor
    line = np.exp(-((wrong_coords - center_out) ** 2) / (2.0 * (sigma_ss / factor) ** 2))
    grid = np.outer(line, line)

    measured_out, _ = centroid_of(_block_average(grid, factor))
    bias = measured_out - center_out
    assert bias == pytest.approx(-(factor - 1) / (2 * factor), abs=1e-3)
    assert abs(bias) > 0.01  # comfortably outside the isolated test's tolerance


# ------------------------------------------------------------------------------------------
# Tier 2: the full render path
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("position", POSITION_CLASSES)
def test_rendered_centroid_matches_ground_truth(position: float) -> None:
    """A clean render must return its commanded centre to better than 0.05 px."""
    params = BeaconParams()
    patch = render_beacon(position, position, params)
    cx, cy = centroid_of(patch.data, patch.x0, patch.y0)

    assert abs(cx - position) < 0.05
    assert abs(cy - position) < 0.05
    assert (patch.true_x, patch.true_y) == (position, position)


@pytest.mark.parametrize("position", POSITION_CLASSES)
@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_rendered_centroid_is_accurate_at_every_supersample_factor(position: float,
                                                                   factor: int) -> None:
    """Accuracy must not depend on the supersample factor.

    A mapping wrong by ``(S-1)/(2S)`` would pass at S=1 (where the error is zero) and fail
    progressively at higher factors, so sweeping S is what exposes it.
    """
    patch = render_beacon(position, position, BeaconParams(supersample_factor=factor))
    cx, cy = centroid_of(patch.data, patch.x0, patch.y0)
    assert abs(cx - position) < 0.05, f"S={factor}"
    assert abs(cy - position) < 0.05, f"S={factor}"


def test_asymmetric_position_distinguishes_x_from_y() -> None:
    """Distinct x and y offsets must be recovered independently.

    Rendering at equal x and y would let a transposed row/column mapping pass unnoticed.
    """
    patch = render_beacon(100.37, 250.62, BeaconParams())
    cx, cy = centroid_of(patch.data, patch.x0, patch.y0)
    assert abs(cx - 100.37) < 0.05
    assert abs(cy - 250.62) < 0.05


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.25, 0.37, 0.5, 0.63, 0.75, 0.9])
def test_no_systematic_bias_across_the_sub_pixel_phase(offset: float) -> None:
    """Error must not grow with sub-pixel phase.

    A phase-dependent error signature -- worst near 0.5, zero at integers -- is the fingerprint
    of a rendering or downsample bias, and would show up in the SNR curve as a noise floor that
    no amount of estimator work could remove.
    """
    position = 300.0 + offset
    patch = render_beacon(position, position, BeaconParams())
    cx, _ = centroid_of(patch.data, patch.x0, patch.y0)
    assert abs(cx - position) < 0.01


def test_bias_has_no_phase_dependent_structure() -> None:
    """Across a dense phase sweep the mean and spread of the error must both be negligible."""
    errors = []
    for offset in np.linspace(0.0, 1.0, 51):
        position = 300.0 + offset
        patch = render_beacon(position, position, BeaconParams())
        cx, _ = centroid_of(patch.data, patch.x0, patch.y0)
        errors.append(cx - position)
    errors = np.asarray(errors)
    assert abs(errors.mean()) < 0.001
    assert errors.std() < 0.001
    assert np.abs(errors).max() < 0.005


# ------------------------------------------------------------------------------------------
# Shapes, parameters and patch geometry
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["gaussian", "square", "circle"])
@pytest.mark.parametrize("position", POSITION_CLASSES)
def test_all_shapes_recover_their_centroid(shape: str, position: float) -> None:
    """Every supported shape must be symmetric about its commanded centre.

    Hard-edged shapes rely entirely on the downsample for anti-aliasing, so they are the most
    sensitive to a mapping error -- there is no smooth profile to hide it.
    """
    params = BeaconParams(shape=shape, size_px=10.0)
    patch = render_beacon(position, position, params)
    cx, cy = centroid_of(patch.data, patch.x0, patch.y0)
    assert abs(cx - position) < 0.05, shape
    assert abs(cy - position) < 0.05, shape


@pytest.mark.parametrize("size", [5, 10, 15, 20])
def test_spec_target_size_range_is_supported(size: int) -> None:
    """Specification parameter 10 allows 5-20 px; all of it must render correctly."""
    params = BeaconParams(shape="square", size_px=float(size))
    patch = render_beacon(100.37, 100.37, params)
    cx, _ = centroid_of(patch.data, patch.x0, patch.y0)
    assert abs(cx - 100.37) < 0.05


def test_square_shape_has_the_requested_extent() -> None:
    """A square of side L must illuminate about L*L pixels' worth of intensity."""
    params = BeaconParams(shape="square", size_px=10.0, peak_intensity=1.0)
    patch = render_beacon(100.0, 100.0, params)
    assert patch.data.sum() == pytest.approx(100.0, rel=0.05)


def test_circle_shape_has_the_requested_area() -> None:
    """A circle of diameter L must illuminate about pi/4 * L*L pixels' worth of intensity."""
    params = BeaconParams(shape="circle", size_px=10.0, peak_intensity=1.0)
    patch = render_beacon(100.0, 100.0, params)
    assert patch.data.sum() == pytest.approx(math.pi / 4.0 * 100.0, rel=0.05)


def test_gaussian_fwhm_matches_the_analytic_relation() -> None:
    """FWHM must be 2.355 * sigma, the relation the spot-scale config depends on."""
    params = BeaconParams(shape="gaussian", sigma_px=2.5)
    assert params.fwhm_px == pytest.approx(5.887, abs=0.01)


def test_peak_intensity_is_respected() -> None:
    """The rendered peak must approach the configured peak intensity."""
    patch = render_beacon(100.0, 100.0, BeaconParams(peak_intensity=200.0))
    assert patch.data.max() == pytest.approx(200.0, rel=0.02)


def test_intensity_scale_preserves_the_centroid() -> None:
    """Scintillation scales brightness, and must not move the spot.

    Phase 2 modulates intensity frame to frame; if that shifted the centroid it would appear as
    tracking error attributable to nothing.
    """
    bright = render_beacon(100.37, 100.37, BeaconParams(), intensity_scale=1.0)
    faint = render_beacon(100.37, 100.37, BeaconParams(), intensity_scale=0.05)
    assert centroid_of(faint.data, faint.x0, faint.y0) == pytest.approx(
        centroid_of(bright.data, bright.x0, bright.y0), abs=1e-4)
    assert faint.data.max() < bright.data.max()


def test_patch_is_odd_sized_and_brackets_the_target() -> None:
    """The patch must be centred on the nearest pixel, keeping truncation symmetric."""
    patch = render_beacon(100.37, 250.62, BeaconParams())
    height, width = patch.shape
    assert height == width and width % 2 == 1
    local_x, local_y = patch.local_true_xy
    assert abs(local_x - (width - 1) / 2.0) <= 0.5
    assert abs(local_y - (height - 1) / 2.0) <= 0.5


def test_patch_support_is_wide_enough_that_truncation_is_negligible() -> None:
    """Edge intensity must be a vanishing fraction of the peak.

    Asymmetric truncation of the profile tail is a bias source; 5 sigma of support keeps it far
    below the error budget.
    """
    patch = render_beacon(100.0, 100.0, BeaconParams())
    edge = max(patch.data[0].max(), patch.data[-1].max(),
               patch.data[:, 0].max(), patch.data[:, -1].max())
    assert edge / patch.data.max() < 1e-4


def test_invalid_parameters_are_rejected() -> None:
    """Bad rendering parameters must fail loudly rather than render nonsense."""
    with pytest.raises(ConfigError, match="Unknown beacon shape"):
        render_beacon(0.0, 0.0, BeaconParams(shape="triangle"))
    with pytest.raises(ConfigError, match="sigma must be positive"):
        render_beacon(0.0, 0.0, BeaconParams(sigma_px=0.0))
    with pytest.raises(ConfigError, match="Supersample factor"):
        render_beacon(0.0, 0.0, BeaconParams(supersample_factor=0))


def test_non_finite_position_is_rejected() -> None:
    """NaN or infinite centres indicate an upstream bug and must not render."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            render_beacon(bad, 0.0, BeaconParams())


def test_params_from_config_match_the_configuration() -> None:
    """Rendering parameters must be derived from config, not duplicated as literals."""
    config = load_config("config/default.json")
    params = BeaconParams.from_config(config)
    assert params.shape == config.target.shape
    assert params.sigma_px == config.target.gaussian_sigma_px
    assert params.supersample_factor == config.target.supersample_factor
    assert params.fwhm_px == pytest.approx(config.target.nominal_fwhm_px)


def test_centroid_of_rejects_degenerate_input() -> None:
    """An empty or 1-D array is a caller error, not something to guess at."""
    with pytest.raises(ValueError, match="2-D"):
        centroid_of(np.zeros(5))
    with pytest.raises(ValueError, match="non-positive total intensity"):
        centroid_of(np.zeros((5, 5)))


# ------------------------------------------------------------------------------------------
# Hard-edge area sampling
#
# A binary inside/outside test at each supersample centre quantises a hard edge to the
# supersample grid, capping centroid accuracy at 1/(2S) px -- 0.125 px at S=4, outside our
# 0.05 px budget and unrecoverable downstream. The spec's *default* target shape is a square
# (parameter 9), so this is not an edge case.
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["square", "circle"])
@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_hard_edges_beat_the_point_sampling_quantisation_limit(shape: str, factor: int) -> None:
    """Hard-edged shapes must be far more accurate than ``1/(2S)`` at every factor.

    Point sampling would sit right at that bound. Area sampling must beat it by a wide margin,
    including at S=1 where point sampling would be catastrophic (0.5 px).
    """
    quantisation_bound = 1.0 / (2.0 * factor)
    worst = 0.0
    for offset in np.linspace(0.0, 1.0, 21):
        position = 100.0 + offset
        params = BeaconParams(shape=shape, size_px=10.0, supersample_factor=factor)
        patch = render_beacon(position, position, params)
        cx, _ = centroid_of(patch.data, patch.x0, patch.y0)
        worst = max(worst, abs(cx - position))

    assert worst < 0.05, f"{shape} at S={factor} misses the render budget: {worst:.4f} px"
    assert worst < quantisation_bound / 4.0, (
        f"{shape} at S={factor} shows {worst:.4f} px error, close to the {quantisation_bound:.4f} "
        f"px point-sampling bound -- area sampling has probably regressed to a binary test"
    )


def test_square_conserves_total_intensity_across_sub_pixel_phase() -> None:
    """A square's total intensity must be exactly its area, at every sub-pixel position.

    This is the direct signature of area sampling. A 9x9 square encloses 81 px^2 of light no
    matter where its edges fall, so exact area sampling holds the sum perfectly constant. A
    point-sampled renderer instead gains and loses whole supersample cells as the edge sweeps
    past their centres: measured at S=4 that swings the total by 2.25 px^2, roughly 2.8%. So
    conservation here is not a weak sanity check -- it fails loudly the moment area sampling
    regresses to a binary inside/outside test.
    """
    params = BeaconParams(shape="square", size_px=9.0, supersample_factor=4,
                          peak_intensity=1.0)
    sums = np.array([render_beacon(100.0 + off, 100.0, params).data.sum()
                     for off in np.linspace(0.0, 1.0, 41)])

    assert sums == pytest.approx(81.0, abs=1e-3)
    assert sums.max() - sums.min() < 1e-3, (
        f"total intensity varies by {sums.max() - sums.min():.4f} px^2 with sub-pixel phase; "
        f"point sampling at S=4 would give about 2.25"
    )


def test_coverage_of_a_cell_fully_inside_is_one() -> None:
    """Sanity check on the coverage primitive itself."""
    from src.sim.beacon import _coverage_1d

    offsets = np.array([0.0, 1.0, -1.0])
    assert _coverage_1d(offsets, half=5.0, cell=0.25) == pytest.approx([1.0, 1.0, 1.0])
    # A cell straddling the edge is half covered.
    assert _coverage_1d(np.array([5.0]), half=5.0, cell=0.25) == pytest.approx([0.5])
    # A cell entirely outside is uncovered.
    assert _coverage_1d(np.array([6.0]), half=5.0, cell=0.25) == pytest.approx([0.0])
