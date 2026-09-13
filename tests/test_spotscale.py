"""Tier-0 spot-scale bootstrap: the Benchmark-2 keystone.

``resolve_geometry`` needs the spot scale, the scale needs a detection, and detection geometry
needs the scale. Tier 0 breaks that circle with a strictly scale-free pass. These tests protect
three things:

1. **It is genuinely scale-free** -- the same code recovers a 3.5 px and a 14 px spot, at
   640x480 and at other resolutions, with no parameter changes.
2. **The area gate holds against salt-and-pepper** -- the arithmetic in ``docs/DESIGN.md``
   section 5.1.2 says that without it the bootstrap locks onto salt rather than the beacon.
3. **It fails closed** -- when there is no confident answer it returns ``None`` so the caller
   sets ``from_fallback``, rather than returning a confident wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.noise.sensor import (
    GaussianNoiseParams,
    SaltPepperParams,
    add_gaussian_noise,
    add_salt_pepper,
    to_uint8,
)
from src.sim.beacon import BeaconParams, render_beacon
from src.vision.spotscale import (
    BootstrapParams,
    bootstrap_scale,
    robust_background,
)

#: Tier 0 only has to land inside the Tier-0/Tier-1 agreement band; Tier 1 supplies precision.
AGREEMENT_BAND = 0.30


def make_frame(sigma_px: float = 2.5, peak: float = 200.0, shape=(480, 640),
               position=None, background: int = 20,
               gaussian_sigma: float = 0.0, sp_density: float = 0.0,
               seed: int = 0) -> np.ndarray:
    """Render a beacon on a noisy background.

    Args:
        sigma_px: Beacon Gaussian sigma.
        peak: Beacon peak intensity.
        shape: Frame shape as ``(height, width)``.
        position: Beacon centre as ``(x, y)``; defaults to a non-integer frame centre.
        background: Uniform background level.
        gaussian_sigma: Additive Gaussian noise sigma.
        sp_density: Salt-and-pepper density.
        seed: RNG seed.

    Returns:
        A ``uint8`` frame.
    """
    height, width = shape
    if position is None:
        position = (width / 2.0 + 0.4, height / 2.0 + 0.6)
    rng = np.random.default_rng(seed)

    frame = np.full(shape, float(background), dtype=np.float32)
    patch = render_beacon(position[0], position[1],
                          BeaconParams(sigma_px=sigma_px, peak_intensity=peak))
    # Clip the patch against the frame, so a beacon placed near an edge renders its visible
    # part rather than raising. Edge cases are exactly what the clipped-blob test needs.
    ph, pw = patch.data.shape
    fx0, fy0 = max(0, patch.x0), max(0, patch.y0)
    fx1, fy1 = min(width, patch.x0 + pw), min(height, patch.y0 + ph)
    if fx0 < fx1 and fy0 < fy1:
        frame[fy0:fy1, fx0:fx1] += patch.data[fy0 - patch.y0:fy1 - patch.y0,
                                              fx0 - patch.x0:fx1 - patch.x0]
    out = to_uint8(frame)
    if gaussian_sigma:
        out = to_uint8(add_gaussian_noise(out, GaussianNoiseParams(sigma=gaussian_sigma), rng))
    if sp_density:
        out = to_uint8(add_salt_pepper(
            out, SaltPepperParams(enabled=True, density=sp_density), rng))
    return out


def true_fwhm(sigma_px: float) -> float:
    """Analytic FWHM for a Gaussian of the given sigma."""
    return 2.3548200450309493 * sigma_px


# ------------------------------------------------------------------------------------------
# Scale-freedom: the whole point of Tier 0
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_px", [1.5, 2.5, 4.0, 6.0, 9.0])
def test_recovers_scale_across_a_wide_spot_size_range(sigma_px: float) -> None:
    """One code path, no parameter changes, across a 6x range of spot sizes.

    This is what Benchmark-2 needs: an evaluator video's spot size is unknown, and absolute
    pixel geometry tuned to our 10 px default would fail on it.
    """
    frame = make_frame(sigma_px=sigma_px, gaussian_sigma=8.0, sp_density=0.05)
    result = bootstrap_scale(frame)
    assert result.succeeded, f"sigma={sigma_px}: no scale recovered"
    expected = true_fwhm(sigma_px)
    assert abs(result.fwhm_px - expected) / expected < AGREEMENT_BAND


@pytest.mark.parametrize("shape", [(240, 320), (480, 640), (720, 1280), (1080, 1920)])
def test_recovers_scale_across_resolutions(shape) -> None:
    """Gates are expressed as frame fractions, so resolution must not matter.

    The evaluator video may be any resolution -- the specification says only that it covers "a
    complete screen".
    """
    frame = make_frame(sigma_px=3.0, shape=shape, gaussian_sigma=8.0, sp_density=0.05)
    result = bootstrap_scale(frame)
    assert result.succeeded, f"{shape}: no scale recovered"
    expected = true_fwhm(3.0)
    assert abs(result.fwhm_px - expected) / expected < AGREEMENT_BAND


def test_locates_the_beacon_not_a_noise_blob() -> None:
    """The candidate must actually be at the beacon.

    Regression guard for a real bug: an earlier version capped the candidate list by *label
    index*. Connected-component labels run in raster order, so on a noisy frame with hundreds of
    spurious blobs the beacon -- near frame centre -- got a high label and was never examined.
    Any cap must be applied after ranking by strength, never before.
    """
    frame = make_frame(gaussian_sigma=10.0, sp_density=0.10)
    result = bootstrap_scale(frame)
    assert result.candidate is not None
    assert result.candidate.x == pytest.approx(320.4, abs=3.0)
    assert result.candidate.y == pytest.approx(240.6, abs=3.0)


# ------------------------------------------------------------------------------------------
# The area gate: load-bearing, not hygiene
# ------------------------------------------------------------------------------------------


def test_survives_ten_percent_salt_and_pepper() -> None:
    """The specification's worst-case impulse noise must not defeat the bootstrap.

    A 3x3 median fails wherever 5 of 9 pixels are corrupted; at 10% density that leaves roughly
    273 residual impulse pixels on a 640x480 frame, against a 99.9th-percentile selection of
    about 307. The area gate is what separates them.
    """
    frame = make_frame(sp_density=0.10, gaussian_sigma=10.0)
    result = bootstrap_scale(frame)
    assert result.succeeded
    expected = true_fwhm(2.5)
    assert abs(result.fwhm_px - expected) / expected < AGREEMENT_BAND


def test_area_gate_actually_rejects_impulse_sized_blobs() -> None:
    """The gate must be doing measurable work, not passing everything through.

    If this count ever falls to zero under heavy impulse noise, the gate has stopped filtering
    and the next salt cluster to exceed the beacon's flux will capture the bootstrap.
    """
    result = bootstrap_scale(make_frame(sp_density=0.10, gaussian_sigma=10.0))
    assert result.rejected_impulse_like > 0


def test_removing_the_area_gate_breaks_the_bootstrap() -> None:
    """Demonstrate the gate is necessary, not merely present.

    With the lower gate disabled, impulse-sized blobs enter the candidate list. This pins the
    claim in DESIGN section 5.1.2 rather than asserting it on faith -- and it is the test that
    will show the H.264 re-validation failing in Phase 6, if compression artifacts grow into the
    beacon's size class.
    """
    frame = make_frame(sp_density=0.10, gaussian_sigma=10.0)
    gated = bootstrap_scale(frame)
    ungated = bootstrap_scale(frame, BootstrapParams(min_area_px=0.0))
    assert len(ungated.candidates) > len(gated.candidates)
    assert ungated.rejected_impulse_like == 0


def test_ranking_uses_flux_not_peak() -> None:
    """Salt sits at the 8-bit ceiling and out-ranks any beacon dimmer than 255 on peak.

    Regression guard: ranking by peak made the bootstrap fail outright under 10% salt-and-pepper,
    because a two-pixel salt cluster at 255 beat a 200-peak beacon. Integrated flux separates
    them decisively -- the beacon spreads comparable intensity over many more pixels.
    """
    frame = make_frame(peak=200.0, sp_density=0.10)
    result = bootstrap_scale(frame)
    assert result.candidate is not None
    assert result.candidate.x == pytest.approx(320.4, abs=3.0)
    # There is genuinely something brighter than the beacon in this frame.
    assert frame.max() > 200


def test_dim_beacon_is_still_found_under_impulse_noise() -> None:
    """A beacon well below the salt level must still win on flux."""
    result = bootstrap_scale(make_frame(peak=60.0, gaussian_sigma=8.0, sp_density=0.05))
    assert result.succeeded
    assert result.candidate is not None
    assert result.candidate.x == pytest.approx(320.4, abs=3.0)


# ------------------------------------------------------------------------------------------
# Failing closed
# ------------------------------------------------------------------------------------------


def test_empty_frame_yields_no_scale() -> None:
    """A frame with no target must return no answer, driving from_fallback."""
    blank = np.full((480, 640), 20, dtype=np.uint8)
    result = bootstrap_scale(blank)
    assert not result.succeeded
    assert result.fwhm_px is None


def test_pure_noise_yields_no_confident_scale() -> None:
    """Noise alone must not produce a confident scale.

    Returning a confident wrong number is worse than returning nothing: the caller would resolve
    geometry to it and never discover the error.
    """
    rng = np.random.default_rng(3)
    noise = to_uint8(add_gaussian_noise(np.full((480, 640), 20, dtype=np.uint8),
                                        GaussianNoiseParams(sigma=15.0), rng))
    result = bootstrap_scale(noise)
    assert not result.succeeded, (
        "pure noise produced a confident scale; the peak-significance floor has regressed")
    assert result.fwhm_px is None


@pytest.mark.parametrize("seed", [3, 4, 5, 6, 7])
def test_peak_significance_floor_separates_signal_from_noise(seed: int) -> None:
    """The 5-sigma floor must reject noise while admitting a genuinely dim beacon.

    Measured separation on 640x640 frames: pure Gaussian noise peaks at 3.4-3.9 sigma, while a
    beacon of peak 45 against sigma 15 scores 6.1. The floor sits between, and this test pins
    both sides of it across several noise realisations rather than trusting one.
    """
    rng = np.random.default_rng(seed)
    noise = to_uint8(add_gaussian_noise(np.full((480, 640), 20, dtype=np.uint8),
                                        GaussianNoiseParams(sigma=15.0), rng))
    assert not bootstrap_scale(noise).succeeded

    dim = make_frame(peak=45.0, gaussian_sigma=15.0, sp_density=0.05, seed=seed)
    assert bootstrap_scale(dim).succeeded, "a real dim beacon was rejected by the floor"


def test_oversized_blob_is_rejected() -> None:
    """A large bright region is background structure, not a beacon."""
    frame = np.full((480, 640), 20, dtype=np.uint8)
    frame[100:300, 100:400] = 220
    result = bootstrap_scale(frame)
    assert result.rejected_too_large > 0


def test_non_2d_frame_is_rejected() -> None:
    """A colour frame must be converted at the source boundary, not here."""
    with pytest.raises(ValueError, match="2-D"):
        bootstrap_scale(np.zeros((10, 10, 3), dtype=np.uint8))


# ------------------------------------------------------------------------------------------
# Supporting machinery
# ------------------------------------------------------------------------------------------


def test_robust_background_is_unaffected_by_impulses() -> None:
    """Median and MAD must not move when 10% of pixels are driven to the extremes.

    Plain mean and standard deviation would both be badly inflated -- at 10% density the sigma
    estimate roughly doubles, which would silently halve every SNR we report.
    """
    rng = np.random.default_rng(0)
    clean = to_uint8(add_gaussian_noise(np.full((400, 400), 100, dtype=np.uint8),
                                        GaussianNoiseParams(sigma=10.0), rng))
    corrupted = to_uint8(add_salt_pepper(
        clean, SaltPepperParams(enabled=True, density=0.10), rng))

    clean_median, clean_sigma = robust_background(clean)
    dirty_median, dirty_sigma = robust_background(corrupted)

    assert dirty_median == pytest.approx(clean_median, abs=2.0)
    assert dirty_sigma == pytest.approx(clean_sigma, rel=0.15)

    # And demonstrate that the naive estimator really would have failed here.
    assert corrupted.std() > 1.8 * clean.std()


def test_edge_touching_blob_is_flagged() -> None:
    """A clipped blob has both a biased centroid and an underestimated scale.

    Recorded at the moment it is knowable, so Phase 3/4 need not re-derive it -- the same
    reasoning as ``from_fallback``.
    """
    frame = make_frame(position=(2.0, 240.0))
    result = bootstrap_scale(frame)
    assert result.candidate is not None
    assert result.candidate.touches_edge


def test_centre_blob_is_not_flagged_as_clipped() -> None:
    """A fully visible beacon must not be flagged."""
    result = bootstrap_scale(make_frame())
    assert result.candidate is not None
    assert not result.candidate.touches_edge


def test_gates_scale_with_frame_size() -> None:
    """Both area and FWHM ceilings are frame fractions, never absolute pixel counts."""
    params = BootstrapParams()
    small = params.max_area_px((240, 320))
    large = params.max_area_px((1080, 1920))
    assert large / small == pytest.approx((1080 * 1920) / (240 * 320))
    assert params.max_fwhm_px((480, 640)) == pytest.approx(0.05 * 480)


def test_bootstrap_does_not_mutate_its_input() -> None:
    """The caller may reuse the frame buffer."""
    frame = make_frame(gaussian_sigma=8.0, sp_density=0.05)
    original = frame.copy()
    bootstrap_scale(frame)
    assert np.array_equal(frame, original)


# ------------------------------------------------------------------------------------------
# Shape dependence
#
# Both conversions in Tier 0 are Gaussian-derived: half-max area -> FWHM assumes a circular
# half-maximum disc, and the Tier-1 LoG extremum location is derived for a Gaussian profile.
# The specification's *default* target shape is a square (parameter 9), and in Mode B we do not
# know the shape at all -- so any shape bias has to be absorbed, not looked up.
# ------------------------------------------------------------------------------------------

#: Tier 0 must report the same scale for the same nominal target size, whatever the profile.
SHAPE_SPREAD_LIMIT = AGREEMENT_BAND


def _reported_scale_ratio(shape: str, size_px: float, blur: float = 0.0,
                          beta: float = 0.0, seed: int = 0) -> float:
    """Report Tier-0 FWHM divided by nominal target size, for one shape.

    Args:
        shape: ``"gaussian"``, ``"square"`` or ``"circle"``.
        size_px: Nominal target size. For the Gaussian, sigma is chosen so its natural FWHM
            equals this, making the three shapes directly comparable.
        blur: Atmospheric blur sigma in pixels.
        beta: Atmospheric extinction coefficient.
        seed: RNG seed.

    Returns:
        Reported FWHM divided by ``size_px``, or NaN when no scale was recovered.
    """
    from src.noise.atmospheric import AtmosphericParams, apply_atmosphere

    rng = np.random.default_rng(seed)
    sigma = size_px / 2.3548200450309493 if shape == "gaussian" else 2.5
    frame = np.full((480, 640), 20.0, dtype=np.float32)
    patch = render_beacon(320.4, 240.6, BeaconParams(shape=shape, size_px=size_px,
                                                     sigma_px=sigma, peak_intensity=220.0))
    frame[patch.y0:patch.y0 + patch.data.shape[0],
          patch.x0:patch.x0 + patch.data.shape[1]] += patch.data
    if blur or beta:
        frame = apply_atmosphere(frame, AtmosphericParams(beta=beta, airlight=180.0 if beta else 0.0,
                                                          blur_sigma=blur), rng)
    noisy = to_uint8(add_gaussian_noise(to_uint8(frame), GaussianNoiseParams(sigma=8.0), rng))
    result = bootstrap_scale(noisy)
    return result.fwhm_px / size_px if result.fwhm_px else float("nan")


@pytest.mark.parametrize("size_px", [6.0, 10.0, 14.0, 20.0])
def test_shape_bias_fits_inside_the_agreement_band(size_px: float) -> None:
    """Square and circle must report a scale close enough to the Gaussian to be absorbed.

    Measured spread across the three shapes is 10.2-13.2%, comfortably inside the 30% band. The
    bias is real and systematic, but it does not need correcting -- which matters because in
    Mode B we do not know the shape and so could not apply a per-shape constant anyway.
    """
    ratios = [_reported_scale_ratio(shape, size_px)
              for shape in ("gaussian", "square", "circle")]
    assert not any(math.isnan(r) for r in ratios)
    spread = max(ratios) / min(ratios) - 1.0
    assert spread < SHAPE_SPREAD_LIMIT, f"size={size_px}: shape spread {spread:.1%}"


def test_square_bias_matches_its_analytic_value() -> None:
    """A square's half-max contour is the square itself, giving a predictable 2/sqrt(pi) bias.

    For a square of side L the half-maximum region has area L^2, and the disc-equivalent
    conversion returns ``2*sqrt(L^2/pi) = 1.128*L``. Pinning the analytic value means a future
    change to the estimator that alters this bias shows up as an explained number rather than a
    mystery, and it confirms the measurement is tracking geometry rather than noise.
    """
    analytic = 2.0 / math.sqrt(math.pi)
    ratio = (_reported_scale_ratio("square", 20.0)
             / _reported_scale_ratio("gaussian", 20.0))
    assert ratio == pytest.approx(analytic, rel=0.05)


def test_circle_is_essentially_unbiased() -> None:
    """A circle *is* a disc, so the disc-equivalent conversion is exact for it by construction."""
    assert _reported_scale_ratio("circle", 14.0) == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("beta", [0.0, 0.35, 0.75])
def test_scale_estimate_is_contrast_invariant(beta: float) -> None:
    """Atmospheric contrast reduction must not move the reported scale at all.

    This is the property Mode B depends on most: an evaluator video of unknown haze or fog must
    not shift our geometry. Measured spread is identical to three decimal places from clear
    through heavy fog, because both the percentile threshold and the half-maximum contour are
    defined relative to the spot's own peak rather than to an absolute level.
    """
    ratios = [_reported_scale_ratio(shape, 14.0, beta=beta)
              for shape in ("gaussian", "square", "circle")]
    spread = max(ratios) / min(ratios) - 1.0
    assert spread < SHAPE_SPREAD_LIMIT
    assert ratios[0] == pytest.approx(_reported_scale_ratio("gaussian", 14.0), rel=0.05)


@pytest.mark.parametrize("blur", [0.0, 0.8, 1.5])
def test_shape_spread_survives_every_configured_atmospheric_blur(blur: float) -> None:
    """Blur widens the spot genuinely, but the shapes must stay inside the band.

    The heaviest blur in any configured preset is fog at 1.5 px. Measured spread reaches 13.6%
    there -- still inside the band, though it does consume a meaningful share of it.
    """
    ratios = [_reported_scale_ratio(shape, 14.0, blur=blur)
              for shape in ("gaussian", "square", "circle")]
    spread = max(ratios) / min(ratios) - 1.0
    assert spread < SHAPE_SPREAD_LIMIT, f"blur={blur}: spread {spread:.1%}"


def test_blur_widening_tracks_gaussian_quadrature() -> None:
    """A blurred Gaussian is genuinely wider, and the estimator must report that, not resist it.

    ``sigma_total = sqrt(sigma_spot^2 + sigma_blur^2)``. Confirming the reported growth matches
    quadrature is what distinguishes "the estimator is correctly tracking a real change in the
    spot" from "blur is breaking the estimator" -- the two look identical in a single number.
    """
    size_px = 14.0
    sigma_spot = size_px / 2.3548200450309493
    for blur in (2.5, 4.0):
        expected = math.sqrt(sigma_spot ** 2 + blur ** 2) / sigma_spot
        measured = _reported_scale_ratio("gaussian", size_px, blur=blur) / \
            _reported_scale_ratio("gaussian", size_px, blur=0.0)
        assert measured == pytest.approx(expected, rel=0.08), f"blur={blur}"


# ------------------------------------------------------------------------------------------
# Tier 1: scale-normalised LoG confirmation
# ------------------------------------------------------------------------------------------

from src.vision.spotscale import (  # noqa: E402
    MAX_SATURATION_FOR_SCALE,
    LadderParams,
    ProfileClass,
    classify_rho,
    confirm_scale,
    estimate_scale,
)


def _seeded(frame):
    """Return ``(frame, seed_xy)`` using Tier 0 to locate the candidate."""
    result = bootstrap_scale(frame)
    assert result.candidate is not None
    return frame, (int(round(result.candidate.x)), int(round(result.candidate.y)))


@pytest.mark.parametrize("sigma_px", [2.0, 3.0, 5.0])
def test_tier1_recovers_gaussian_scale(sigma_px: float) -> None:
    """The LoG extremum must land on the true characteristic scale."""
    frame, seed = _seeded(make_frame(sigma_px=sigma_px, gaussian_sigma=8.0))
    result = confirm_scale(frame, seed)
    assert result.succeeded
    assert result.fwhm_px == pytest.approx(true_fwhm(sigma_px), rel=0.20)


def test_tier1_curvature_separates_signal_from_noise() -> None:
    """Curvature is the confidence measure; peak-to-runner-up is not.

    Measured, the ratio is 1.01-1.06 for correct estimates on real spots and 1.12-1.14 for pure
    noise -- inverted, so it would reject good answers and accept bad ones. Curvature reads
    ~1.9-2.4 for real spots and 0.0 for noise.
    """
    frame, seed = _seeded(make_frame(gaussian_sigma=8.0))
    assert confirm_scale(frame, seed).curvature > 1.0

    rng = np.random.default_rng(11)
    noise = to_uint8(add_gaussian_noise(np.full((480, 640), 20, dtype=np.uint8),
                                        GaussianNoiseParams(sigma=15.0), rng))
    assert confirm_scale(noise, (320, 240)).curvature < 1.0


def test_tier1_flags_a_ladder_edge_argmax() -> None:
    """An argmax at a ladder end means the true scale is outside the searched range.

    The argmax then carries no information, so it must be reported as a failure rather than
    returned as if it were a measurement.
    """
    frame, seed = _seeded(make_frame(sigma_px=2.5, gaussian_sigma=8.0))
    # A ladder far too coarse for this spot forces the peak to the low end.
    narrow = LadderParams(sigma_min=8.0, ratio=1.3, steps=5)
    result = confirm_scale(frame, seed, narrow)
    assert result.at_ladder_edge
    assert not result.succeeded


def test_tier1_is_frame_size_independent() -> None:
    """The ladder must run on a padded crop, not the whole frame.

    Blurring the full frame and sampling only the ROI gives identical answers but costs 121 ms
    on a 2000x2000 frame against 18 ms on 640x480 -- it is not ROI-limited at all. Cropping
    first makes it 0.9-1.3 ms at every size. This test pins the *equivalence*; the speed is what
    motivated it.
    """
    small, seed_small = _seeded(make_frame(sigma_px=3.0, shape=(480, 640)))
    large, seed_large = _seeded(make_frame(sigma_px=3.0, shape=(1440, 1920)))
    a = confirm_scale(small, seed_small)
    b = confirm_scale(large, seed_large)
    assert a.succeeded and b.succeeded
    assert a.fwhm_px == pytest.approx(b.fwhm_px, rel=0.10)


def test_tier1_rejects_non_2d_input() -> None:
    """Colour conversion belongs at the source boundary."""
    with pytest.raises(ValueError, match="2-D"):
        confirm_scale(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4))


# ------------------------------------------------------------------------------------------
# The rho agreement rule
# ------------------------------------------------------------------------------------------


def test_rho_clusters_overlap_and_so_cannot_gate_behaviour() -> None:
    """Pin the refutation, so nobody re-introduces cluster gating.

    An initial narrow sweep (4 sizes, all >= 8 px, one brightness) suggested the smooth and
    hard-edged clusters separated with a +0.024 gap. A wider sweep over 10 sizes from 5 to 24 px
    showed a raw gap of **-0.204** with heavy overlap, and smoothing over 5, 10 or 20 frames does
    not rescue it because the spread is systematic in spot size rather than frame-to-frame noise.

    So rho is advisory only, and the agreement band must absorb the shape divergence rather than
    explain it away.
    """
    rng = np.random.default_rng(0)
    gaussian_rhos, hard_rhos = [], []
    for size_px in (5.0, 7.0, 10.0, 14.0, 20.0):
        for shape in ("gaussian", "square", "circle"):
            frame = np.full((480, 640), 20.0, dtype=np.float32)
            sigma = size_px / 2.3548200450309493 if shape == "gaussian" else 2.5
            patch = render_beacon(320.4, 240.6, BeaconParams(shape=shape, size_px=size_px,
                                                             sigma_px=sigma,
                                                             peak_intensity=220.0))
            frame[patch.y0:patch.y0 + patch.data.shape[0],
                  patch.x0:patch.x0 + patch.data.shape[1]] += patch.data
            noisy = to_uint8(add_gaussian_noise(to_uint8(frame),
                                                GaussianNoiseParams(sigma=8.0), rng))
            estimate = estimate_scale(noisy)
            if estimate.rho is None:
                continue
            (gaussian_rhos if shape == "gaussian" else hard_rhos).append(estimate.rho)

    assert gaussian_rhos and hard_rhos
    # The distributions overlap: the clusters are not separable.
    assert min(gaussian_rhos) < max(hard_rhos), (
        "rho clusters appear separable again -- re-check the sweep breadth before trusting it")


def test_rho_band_absorbs_every_supported_profile() -> None:
    """Whatever the profile, honest agreement must not trip from_fallback.

    The band is deliberately wide (+/-30%): shape divergence up to ~15% plus size/noise spread of
    ~11% at 3 sigma. That costs sensitivity, which is why from_fallback leans on the saturation
    gate, curvature floor, ladder-edge check and peak-significance floor instead -- all of which
    discriminate cleanly.
    """
    rng = np.random.default_rng(1)
    for shape in ("gaussian", "square", "circle"):
        for size_px in (6.0, 10.0, 16.0):
            frame = np.full((480, 640), 20.0, dtype=np.float32)
            sigma = size_px / 2.3548200450309493 if shape == "gaussian" else 2.5
            patch = render_beacon(320.4, 240.6, BeaconParams(shape=shape, size_px=size_px,
                                                             sigma_px=sigma,
                                                             peak_intensity=220.0))
            frame[patch.y0:patch.y0 + patch.data.shape[0],
                  patch.x0:patch.x0 + patch.data.shape[1]] += patch.data
            noisy = to_uint8(add_gaussian_noise(to_uint8(frame),
                                                GaussianNoiseParams(sigma=8.0), rng))
            estimate = estimate_scale(noisy)
            assert not estimate.from_fallback, f"{shape} at {size_px}px: {estimate.reason}"


def test_rho_band_still_rejects_gross_disagreement() -> None:
    """The band is wide but not vacuous."""
    from src.vision.spotscale import rho_in_agreement

    assert rho_in_agreement(1.00)
    assert rho_in_agreement(0.86)
    assert not rho_in_agreement(0.4)
    assert not rho_in_agreement(2.5)
    assert not rho_in_agreement(None)


def test_profile_class_is_advisory_only() -> None:
    """A label of UNEXPLAINED must not by itself force fallback."""
    estimate = estimate_scale(make_frame(sigma_px=3.0, gaussian_sigma=8.0))
    assert not estimate.from_fallback
    assert estimate.profile_class in tuple(ProfileClass)


# ------------------------------------------------------------------------------------------
# Saturation gating
#
# Saturated flat-top is not hypothetical: scintillation plus a bright atmospheric preset
# produces it in normal operation.
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("peak,size_px", [(600, 14.0), (2000, 14.0), (20000, 6.0)])
def test_saturation_blocks_scale_resolution(peak: int, size_px: float) -> None:
    """A saturated frame must not be used to resolve scale, at any beacon size.

    Clipping flattens the top, widening the half-maximum contour: measured scale error is +53%
    at 24-29% saturation, +102% at 37-39%, and +173% at 45%. The ``max_fwhm`` clamp is *not* a
    safety net -- a 6 px beacon at heavy saturation reports 16.4 px and stays under the clamp,
    claiming success. Only an explicit saturation gate catches it.
    """
    estimate = estimate_scale(make_frame(sigma_px=size_px / 2.3548200450309493, peak=float(peak),
                                         gaussian_sigma=8.0))
    assert estimate.saturated_fraction > MAX_SATURATION_FOR_SCALE
    assert estimate.from_fallback
    assert estimate.reason == "saturated"
    assert estimate.fwhm_px is None


def test_mild_saturation_is_still_usable() -> None:
    """The gate must not be so tight that ordinary brightness variation blocks estimation."""
    estimate = estimate_scale(make_frame(sigma_px=3.0, peak=255.0, gaussian_sigma=8.0))
    assert estimate.saturated_fraction <= MAX_SATURATION_FOR_SCALE
    assert not estimate.from_fallback


def test_saturation_policy_is_opposite_to_the_centroid_policy() -> None:
    """Saturation rejects a *scale* estimate but only inflates R for a *centroid*.

    Same physical effect, opposite conclusion: the Phase 2 measurement put centroid bias at
    ~0.19 px against a 10 px budget (bounded, keep the measurement), whereas scale error reaches
    +173% and poisons every downstream kernel and ROI size for subsequent frames. This test
    documents the asymmetry so nobody later "harmonises" the two policies.
    """
    saturated = make_frame(sigma_px=3.0, peak=5000.0, gaussian_sigma=8.0)
    estimate = estimate_scale(saturated)
    assert estimate.from_fallback and estimate.reason == "saturated"
    # But the frame is still perfectly usable for centroiding -- Tier 0 still finds the target.
    assert bootstrap_scale(saturated).candidate is not None


# ------------------------------------------------------------------------------------------
# The diagnostic trace
# ------------------------------------------------------------------------------------------


def test_trace_is_populated_even_when_estimation_fails() -> None:
    """A flag is worth less than the flag plus the trace that explains it.

    Every failure path must still carry enough context to attribute the failure after the fact,
    rather than leaving a bare from_fallback with no history.
    """
    estimate = estimate_scale(make_frame(peak=5000.0, gaussian_sigma=8.0))
    assert estimate.from_fallback
    assert estimate.reason
    assert estimate.saturated_fraction > 0.0
    assert estimate.tier0_fwhm_px is not None or estimate.reason in {"saturated", "no_candidate"}


def test_rho_migrates_toward_hard_edged_as_saturation_rises() -> None:
    """A saturated Gaussian is hard-edged in exactly the way rho keys on.

    This is why rho must be logged *per frame* rather than only at resolve time: a genuinely
    smooth beacon can migrate between clusters mid-run as scintillation brightens it, and a rho
    trajectory that walks across the gap is diagnosable where a bare from_fallback is not.
    Measured: rho falls 1.009 -> 0.971 -> 0.935 -> 0.926 -> 0.906 as saturation goes 0 -> 0.28.
    """
    rhos = []
    for peak in (200, 255, 300):
        estimate = estimate_scale(make_frame(sigma_px=14.0 / 2.3548200450309493,
                                             peak=float(peak), gaussian_sigma=8.0),
                                  max_saturation=1.0)  # disable the gate to observe migration
        if estimate.rho is not None:
            rhos.append(estimate.rho)
    assert len(rhos) >= 2
    assert rhos[-1] < rhos[0], "rho should fall as the top flattens"


def test_reason_codes_are_distinct_and_machine_readable() -> None:
    """Reason codes go in the per-frame log, so they must be stable identifiers."""
    blank = np.full((480, 640), 20, dtype=np.uint8)
    assert estimate_scale(blank).reason == "no_candidate"
    assert estimate_scale(make_frame(peak=5000.0)).reason == "saturated"
    assert estimate_scale(make_frame(sigma_px=3.0, gaussian_sigma=8.0)).reason == "ok"


# ------------------------------------------------------------------------------------------
# Tier 2: tracking, re-check scheduling, and the ratchet
# ------------------------------------------------------------------------------------------

from src.vision.spotscale import (  # noqa: E402
    ScaleEstimate,
    ScaleTracker,
    TrackerParams,
    bootstrap_scale_roi,
    recommended_recheck_interval_s,
)


def _good(fwhm: float) -> ScaleEstimate:
    """A valid scale estimate at the given FWHM."""
    return ScaleEstimate(fwhm_px=fwhm, from_fallback=False, reason="ok", rho=1.0,
                         tier0_fwhm_px=fwhm)


def _blind() -> ScaleEstimate:
    """An estimate where no candidate was found at all."""
    return ScaleEstimate(reason="no_candidate")


def _blocked(reason: str = "saturated") -> ScaleEstimate:
    """An estimate where a candidate exists but its scale cannot be trusted."""
    return ScaleEstimate(from_fallback=True, reason=reason, saturated_fraction=0.4,
                         tier0_fwhm_px=99.0)


def _adopt(tracker: ScaleTracker, fwhm: float, t: float = 0.0) -> float:
    """Drive a tracker through M-of-N adoption at a constant scale."""
    for index in range(5):
        tracker.update(_good(fwhm), t + index * 0.1)
    assert tracker.state.adopted
    return t + 0.5


# --- the ratchet ---------------------------------------------------------------------------


def test_intermittent_blindness_does_not_ratchet_the_resolved_scale() -> None:
    """A long run with intermittent detection failures at CONSTANT true scale must not drift.

    Growing the assumed scale when blind is necessary -- too small a scale puts the top-hat
    element inside the spot and deletes the beacon, and that failure is self-reinforcing. But
    growth plus the integer deadband can interact into a one-way ratchet: each failure nudges the
    scale up, the deadband lets accumulated growth take effect, and nothing symmetric pushes it
    back down.

    The defence is that growth lives *only* in the transient search scale and never touches the
    persistent estimate.
    """
    true_fwhm_px = 6.0
    tracker = ScaleTracker(fallback_fwhm_px=5.0)
    now = _adopt(tracker, true_fwhm_px)
    resolved_at_start = tracker.state.resolved_fwhm_px

    for cycle in range(200):
        now += 0.1
        # Three blind frames, then two good ones, repeatedly.
        estimate = _blind() if cycle % 5 < 3 else _good(true_fwhm_px)
        tracker.update(estimate, now)

    state = tracker.state
    assert state.resolved_fwhm_px == pytest.approx(resolved_at_start, abs=0.5)
    assert state.estimated_fwhm_px == pytest.approx(true_fwhm_px, abs=0.5)
    assert state.search_inflation == pytest.approx(1.0), "search inflation failed to reset"


def test_saturation_gating_does_not_ratchet_either() -> None:
    """A gated stretch looks like sustained failure to a naive grow policy. It is not.

    Saturation gating leaves us with a perfectly good detection whose scale we decline to trust.
    We can see the target; we simply cannot measure it. So blocked frames must *hold* the scale
    and never inflate the search -- otherwise saturation becomes a second path into the same
    ratchet.
    """
    true_fwhm_px = 6.0
    tracker = ScaleTracker(fallback_fwhm_px=5.0)
    now = _adopt(tracker, true_fwhm_px)
    resolved_at_start = tracker.state.resolved_fwhm_px

    # A long saturated stretch, as scintillation on a bright preset would produce.
    for index in range(120):
        now += 0.1
        tracker.update(_blocked("saturated"), now)

    held = tracker.state
    assert held.resolved_fwhm_px == pytest.approx(resolved_at_start)
    assert held.search_inflation == pytest.approx(1.0), "blocked frames must not inflate search"
    assert held.consecutive_blocked == 120
    assert held.reason.startswith("blocked:")

    # And it recovers cleanly once saturation clears.
    for index in range(10):
        now += 0.1
        tracker.update(_good(true_fwhm_px), now)
    assert tracker.state.resolved_fwhm_px == pytest.approx(resolved_at_start, abs=0.5)
    assert tracker.state.consecutive_blocked == 0


def test_blind_and_blocked_are_counted_separately() -> None:
    """The two failure modes are distinguishable in the trace, because they mean different things."""
    tracker = ScaleTracker()
    _adopt(tracker, 6.0)
    tracker.update(_blind(), 1.0)
    assert tracker.state.consecutive_blind == 1 and tracker.state.consecutive_blocked == 0
    tracker.update(_blocked(), 1.1)
    assert tracker.state.consecutive_blind == 0 and tracker.state.consecutive_blocked == 1


def test_search_inflates_while_blind_and_is_bounded() -> None:
    """Growth is real, transient and capped -- it must not run away."""
    tracker = ScaleTracker(TrackerParams(max_search_inflation=3.0), fallback_fwhm_px=5.0)
    _adopt(tracker, 6.0)
    for index in range(200):
        tracker.update(_blind(), 1.0 + index * 0.1)
    state = tracker.state
    assert state.search_inflation == pytest.approx(3.0)
    assert state.search_fwhm_px == pytest.approx(6.0 * 3.0, rel=0.1)
    # The persistent estimate was never touched by any of that.
    assert state.estimated_fwhm_px == pytest.approx(6.0, abs=0.5)


def test_search_inflation_resets_on_the_first_candidate() -> None:
    """Seeing the target again ends the widening immediately, not gradually."""
    tracker = ScaleTracker(fallback_fwhm_px=5.0)
    _adopt(tracker, 6.0)
    for index in range(10):
        tracker.update(_blind(), 1.0 + index * 0.1)
    assert tracker.state.search_inflation > 1.5
    tracker.update(_blocked(), 3.0)  # a candidate exists, even though its scale is untrusted
    assert tracker.state.search_inflation == pytest.approx(1.0)


def test_tracker_follows_a_genuine_scale_change_downward() -> None:
    """A receding target must be tracked down, against the grow bias and the deadband.

    Shrinking is the direction that fights both policies, so it is the case most likely to expose
    a ratchet.
    """
    tracker = ScaleTracker(fallback_fwhm_px=5.0)
    now = _adopt(tracker, 16.0)
    assert tracker.state.resolved_fwhm_px == pytest.approx(16.0, abs=0.5)
    for index in range(80):
        now += 0.1
        tracker.update(_good(8.0), now)
    assert tracker.state.estimated_fwhm_px == pytest.approx(8.0, abs=0.5)
    assert tracker.state.resolved_fwhm_px == pytest.approx(8.0, abs=1.0)


def test_tracker_follows_a_genuine_scale_change_upward() -> None:
    """An approaching target is the easy direction, but must still converge."""
    tracker = ScaleTracker(fallback_fwhm_px=5.0)
    now = _adopt(tracker, 8.0)
    for index in range(80):
        now += 0.1
        tracker.update(_good(16.0), now)
    assert tracker.state.estimated_fwhm_px == pytest.approx(16.0, abs=0.5)


# --- adoption, deadband, scheduling ---------------------------------------------------------


def test_one_frame_cannot_set_the_scale() -> None:
    """M-of-N confirmation, so a bad first estimate cannot poison the run."""
    tracker = ScaleTracker(TrackerParams(confirm_m_of_n=(3, 5)))
    tracker.update(_good(6.0), 0.0)
    assert not tracker.state.adopted
    assert tracker.state.from_fallback
    tracker.update(_good(6.0), 0.1)
    assert not tracker.state.adopted
    tracker.update(_good(6.0), 0.2)
    assert tracker.state.adopted


def test_disagreeing_measurements_delay_adoption() -> None:
    """Wildly inconsistent measurements must not satisfy M-of-N."""
    tracker = ScaleTracker(TrackerParams(confirm_m_of_n=(3, 5), agreement_tolerance=0.10))
    for index, value in enumerate((4.0, 12.0, 7.0)):
        tracker.update(_good(value), index * 0.1)
    assert not tracker.state.adopted


def test_deadband_suppresses_kernel_chatter() -> None:
    """Small oscillations must not re-resolve geometry.

    Resolved geometry yields integer kernel sizes; an estimate oscillating near a boundary would
    chatter the kernel frame to frame and shift the detection statistics underneath the tracker.
    """
    tracker = ScaleTracker(TrackerParams(deadband_px=2.0))
    now = _adopt(tracker, 10.0)
    resolved = tracker.state.resolved_fwhm_px
    for index in range(40):
        now += 0.1
        tracker.update(_good(10.0 + (0.6 if index % 2 else -0.6)), now)
    assert tracker.state.resolved_fwhm_px == pytest.approx(resolved)
    assert tracker.state.reason == "tracking"


def test_recheck_interval_is_frame_size_aware_and_clamped() -> None:
    """One interval cannot be right for a 640x480 and a 2000x2000 frame.

    Steady-state ROI re-checks are near frame-size independent (~2.5 ms at every size), so the
    interval barely varies once locked -- but it is derived from measured cost and a duty budget
    rather than assumed.
    """
    small = recommended_recheck_interval_s((480, 640), measured_cost_ms=2.5)
    large = recommended_recheck_interval_s((2000, 2000), measured_cost_ms=2.5)
    assert small == pytest.approx(large)
    assert 0.5 <= small <= 5.0

    # A costly full-frame re-check must stretch the interval, not be run regardless.
    costly = recommended_recheck_interval_s((2000, 2000), measured_cost_ms=234.0)
    assert costly > small
    assert costly <= 5.0

    with pytest.raises(ValueError, match="Duty budget"):
        recommended_recheck_interval_s((480, 640), duty_budget=0.0)


def test_full_frame_is_only_needed_when_position_is_unknown() -> None:
    """Cold bootstrap and post-loss need full-frame; steady state does not.

    Full-frame Tier 0 costs up to 234 ms against ~2.5 ms for an ROI re-check, so getting this
    distinction right is the difference between 11% and 0.1% of the acquisition budget.
    """
    tracker = ScaleTracker()
    assert tracker.needs_full_frame()  # cold
    _adopt(tracker, 6.0)
    assert not tracker.needs_full_frame()  # locked
    tracker.update(_blind(), 1.0)
    assert tracker.needs_full_frame()  # lost
    tracker.update(_good(6.0), 1.1)
    assert not tracker.needs_full_frame()  # re-acquired


def test_recheck_scheduling_respects_the_interval() -> None:
    """Re-checks fire on schedule, not every frame."""
    tracker = ScaleTracker(TrackerParams(recheck_interval_s=2.0))
    tracker.update(_good(6.0), 10.0)
    assert not tracker.due_for_recheck(11.0)
    assert tracker.due_for_recheck(12.0)


def test_roi_bootstrap_matches_full_frame() -> None:
    """An ROI re-check must give the same answer as a full-frame one.

    Gates are frame-fraction based, so the ROI pass must be told the *full* frame shape --
    otherwise ``max_fwhm`` collapses to 6.4 px on a 128 px ROI and rejects a legitimate 14 px
    beacon. That was a real bug caught by this comparison.
    """
    frame = make_frame(sigma_px=14.0 / 2.3548200450309493, gaussian_sigma=8.0, sp_density=0.05)
    full = bootstrap_scale(frame)
    roi = bootstrap_scale_roi(frame, (320.4, 240.6), 128)
    assert full.fwhm_px is not None and roi.fwhm_px is not None
    assert roi.fwhm_px == pytest.approx(full.fwhm_px, rel=0.05)
    assert roi.candidate.x == pytest.approx(full.candidate.x, abs=2.0)


def test_roi_bootstrap_rejects_bad_arguments() -> None:
    """Caller errors must fail loudly."""
    frame = make_frame()
    with pytest.raises(ValueError, match="ROI size"):
        bootstrap_scale_roi(frame, (100.0, 100.0), 0)
    with pytest.raises(ValueError, match="2-D"):
        bootstrap_scale_roi(np.zeros((4, 4, 3), np.uint8), (2.0, 2.0), 4)


def test_tracker_reset_clears_everything() -> None:
    """Reset must restore the unadopted state so a scenario can be re-run."""
    tracker = ScaleTracker()
    _adopt(tracker, 6.0)
    tracker.reset()
    state = tracker.state
    assert not state.adopted and state.resolved_fwhm_px is None
    assert state.search_inflation == 1.0
    assert tracker.needs_full_frame()
