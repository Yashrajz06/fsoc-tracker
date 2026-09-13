"""Centroid bias versus saturation fraction.

Saturation is the sibling of the Phase 1 edge-clipping finding: once brightening or
scintillation drives the beacon's peak to the 8-bit ceiling, the spot's top is flattened and
sub-pixel information is destroyed inside the plateau. Like clipping it is smooth, systematic
and noise-free, so it can masquerade as tracker lag.

**What the measurement actually shows**, which is more nuanced than the concern that motivated it:

Clipping a *symmetric* spot *symmetrically* preserves its centroid exactly. So saturation on its
own does not produce a large bias -- the surviving unsaturated wings still carry the sub-pixel
information, and they dominate the intensity-weighted sum. Bias only appears as the estimator
discards those wings, and it is worst in the degenerate case where nothing but the fully
saturated plateau survives thresholding, because then the centroid is the geometric centre of an
*integer pixel set* and quantises to the pixel grid.

Measured worst-case bias across sub-pixel phase, default 10 px Gaussian beacon:

| regime | what survives thresholding | worst bias |
|---|---|---|
| unthresholded centre of gravity | everything | < 0.005 px |
| threshold at 0.7 x peak | bright core, partial edge pixels | < 0.06 px |
| threshold at the ceiling | fully saturated pixels only | ~0.19 px |

**Conclusion for the report and for Phase 3/4:** saturation costs *precision*, not *lock*. Even
fully saturated the bias is ~0.19 px against a 10 px tracking requirement -- roughly 50x margin.
It matters for the sub-pixel accuracy claims and the SNR curve, not for lock retention. The
practical guidance is therefore that the ``saturated`` flag should raise Kalman ``R`` modestly
rather than reject the detection: throwing away a 0.19 px-biased measurement would cost far more
than keeping it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.noise.sensor import MAX_LEVEL, to_uint8
from src.sim.beacon import BeaconParams, centroid_of, render_beacon

#: Sub-pixel phases to sweep. Bias from a quantisation mechanism is phase-dependent, so a single
#: phase would understate it -- and integer/half-integer phases can pass by symmetry.
PHASES = np.linspace(0.0, 1.0, 33)

#: Beacon peak intensities, spanning unsaturated through heavily saturated.
PEAKS = [200, 255, 300, 400, 600, 1000, 2000, 5000]


def _render(peak: float, offset: float):
    """Render the default beacon at a given peak intensity and sub-pixel phase.

    Args:
        peak: Peak intensity before 8-bit clipping.
        offset: Sub-pixel offset added to an integer base position.

    Returns:
        ``(quantised_patch, patch, true_position)``.
    """
    params = BeaconParams(peak_intensity=float(peak))
    patch = render_beacon(100.0 + offset, 100.0 + offset, params)
    return to_uint8(patch.data).astype(np.float64), patch, 100.0 + offset


def _saturated_fraction(quantised: np.ndarray) -> float:
    """Fraction of the spot's core pixels sitting at the 8-bit ceiling.

    Args:
        quantised: The uint8-quantised patch as floats.

    Returns:
        Fraction in ``[0, 1]``, measured over pixels above half the maximum.
    """
    core = quantised >= 0.5 * quantised.max()
    if not core.any():
        return 0.0
    return float(np.count_nonzero(quantised[core] >= MAX_LEVEL) / np.count_nonzero(core))


def _worst_bias(peak: float, threshold_fraction: float | None) -> tuple[float, float]:
    """Measure worst centroid bias across sub-pixel phase at one peak intensity.

    Args:
        peak: Peak intensity before clipping.
        threshold_fraction: Fraction of the frame maximum used as a background threshold before
            centroiding, or ``None`` for an unthresholded centre of gravity.

    Returns:
        ``(worst_bias_px, worst_saturated_fraction)``.
    """
    worst_bias = 0.0
    worst_saturation = 0.0
    for offset in PHASES:
        quantised, patch, truth = _render(peak, float(offset))
        worst_saturation = max(worst_saturation, _saturated_fraction(quantised))

        if threshold_fraction is None:
            weights = quantised
        else:
            weights = np.maximum(quantised - quantised.max() * threshold_fraction, 0.0)
        if weights.sum() <= 0:
            continue
        cx, _ = centroid_of(weights, patch.x0, patch.y0)
        worst_bias = max(worst_bias, abs(cx - truth))
    return worst_bias, worst_saturation


# ------------------------------------------------------------------------------------------
# The bias curve
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("peak", PEAKS)
def test_unthresholded_centroid_is_insensitive_to_saturation(peak: int) -> None:
    """With the wings retained, saturation barely moves the centroid.

    Symmetric clipping of a symmetric spot preserves the centre of mass, and the unsaturated
    wings carry the sub-pixel information. This is the reassuring half of the finding.
    """
    bias, _ = _worst_bias(peak, threshold_fraction=None)
    assert bias < 0.01, f"peak={peak}: {bias:.4f} px"


@pytest.mark.parametrize("peak", PEAKS)
def test_thresholded_centroid_stays_inside_the_render_budget(peak: int) -> None:
    """A realistic bright-core threshold keeps bias near the 0.05 px render budget."""
    bias, _ = _worst_bias(peak, threshold_fraction=0.7)
    assert bias < 0.06, f"peak={peak}: {bias:.4f} px"


@pytest.mark.parametrize("peak", [260, 400, 1000, 5000])
def test_degenerate_plateau_bias_is_bounded_and_quantisation_shaped(peak: int) -> None:
    """When only saturated pixels survive, the centroid quantises to the pixel grid.

    This is the worst case: the estimate becomes the geometric centre of an integer pixel set,
    so all sub-pixel information within the plateau is gone. Bias stays around 0.19 px -- large
    against the 0.05 px render budget, negligible against the 10 px tracking requirement.
    """
    worst = 0.0
    for offset in PHASES:
        quantised, patch, truth = _render(peak, float(offset))
        plateau = (quantised >= MAX_LEVEL).astype(np.float64)
        if plateau.sum() < 1:
            continue
        cx, _ = centroid_of(plateau, patch.x0, patch.y0)
        worst = max(worst, abs(cx - truth))

    assert worst > 0.05, (
        f"peak={peak}: bias {worst:.4f} px is suspiciously small for a fully quantised "
        f"plateau -- the degenerate case may not be being exercised")
    assert worst < 0.5, f"peak={peak}: {worst:.4f} px"


def test_saturation_costs_precision_not_lock() -> None:
    """The headline conclusion, asserted so a future change cannot quietly invalidate it.

    Even in the degenerate fully-saturated case the bias is roughly 50x inside the 10 px
    tracking requirement. Therefore the ``saturated`` flag (Phase 3) should *inflate* Kalman R
    rather than reject the detection: discarding a 0.19 px-biased measurement costs far more
    than keeping it.
    """
    worst = 0.0
    for peak in (260, 400, 1000, 5000, 20000):
        for offset in PHASES:
            quantised, patch, truth = _render(peak, float(offset))
            plateau = (quantised >= MAX_LEVEL).astype(np.float64)
            if plateau.sum() < 1:
                continue
            cx, _ = centroid_of(plateau, patch.x0, patch.y0)
            worst = max(worst, abs(cx - truth))
    tracking_budget_px = 10.0
    assert worst < tracking_budget_px / 20.0, (
        f"worst saturated-plateau bias {worst:.4f} px has grown to within 20x of the "
        f"{tracking_budget_px} px tracking budget; the Phase 3 saturated-flag policy of "
        f"inflating R rather than rejecting the detection needs revisiting")


def test_saturation_fraction_rises_monotonically_with_peak_intensity() -> None:
    """The measured saturation fraction must track brightness, since Phase 3 keys a flag on it."""
    fractions = [_worst_bias(peak, threshold_fraction=0.7)[1] for peak in PEAKS]
    assert fractions[0] == 0.0  # peak 200: no saturation at all
    assert all(b >= a - 1e-9 for a, b in zip(fractions, fractions[1:]))

    # The fraction approaches but does not reach 1.0, and that is not a measurement artefact:
    # "core" is defined relative to the *quantised* maximum, which is pinned at 255 once the
    # spot saturates. The half-maximum contour therefore sits at a fixed 127.5 and the core
    # keeps growing outward into unsaturated wing pixels as the beacon brightens. Measured
    # 0.83 at peak 5000. Phase 3 should key its flag on this same definition so the threshold
    # it inherits means the same thing.
    assert 0.8 < fractions[-1] < 1.0


def test_saturation_threshold_for_the_phase_3_flag() -> None:
    """Locate where saturation first becomes measurable, which is where the flag should fire.

    Phase 3 sets the ``saturated`` flag from a measured quantity rather than a guess; this test
    records the crossing so the threshold has a documented basis.
    """
    first_saturating = None
    for peak in PEAKS:
        _, fraction = _worst_bias(peak, threshold_fraction=0.7)
        if fraction > 0.0:
            first_saturating = peak
            break
    # A Gaussian whose analytic peak just exceeds 255 saturates only its centre pixel; because
    # each pixel reports the profile integrated over its area, that needs a peak somewhat above
    # 255 rather than exactly 255.
    assert first_saturating == 300


def test_pipeline_reports_saturation_fraction() -> None:
    """The noise pipeline must surface saturation so telemetry can log it per frame."""
    from src.config import load_config
    from src.noise.pipeline import NoisePipeline

    pipeline = NoisePipeline.from_config(load_config("config/default.json"))
    bright = np.full((64, 64), 250, dtype=np.uint8)
    result = pipeline.apply(bright)
    assert result.saturated_fraction > 0.0
    assert result.is_saturated

    dark = np.full((64, 64), 10, dtype=np.uint8)
    assert pipeline.apply(dark).saturated_fraction == 0.0
