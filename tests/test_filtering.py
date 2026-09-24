"""Tests for the Kalman filter and track lifecycle.

The headline test is :func:`test_track_survives_where_a_bare_filter_locks_itself_out`, which pins
a real defect: a filter initiated from a single detection assumes zero velocity, falls behind a
moving target, and then rejects every subsequent measurement because the innovation dwarfs its
own tight gate. The measurements are fine throughout; only the output is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.filtering.kalman import (
    CHI2_99_2DOF,
    LAW_CALIBRATION,
    KalmanFilterCV,
    KalmanParams,
    measurement_sigma,
)
from src.filtering.track import Track, TrackParams, TrackStatus

DT = 1.0 / 30.0


# ------------------------------------------------------------------------------------------
# Adaptive measurement noise
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr", [200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0])
def test_sigma_follows_the_accuracy_law(snr: float) -> None:
    """``sigma ~ calibration * FWHM / (2 * SNR)``, clamped at the quantisation floor."""
    fwhm = 5.887
    sigma = measurement_sigma(snr, fwhm)
    expected = max(0.02, LAW_CALIBRATION * fwhm / (2.0 * snr))
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_sigma_is_calibrated_against_measurement_not_theory() -> None:
    """The bare law understates our measured error, so an uncorrected R would be optimistic.

    Comparing the law against measured centroid error across three shapes and four SNR bands gave
    a median ratio of 1.32. Using the raw law would tighten the validation gate by a third and
    reject genuine detections -- the failure mode adaptive R exists to prevent.
    """
    assert LAW_CALIBRATION > 1.0
    assert measurement_sigma(50.0, 5.887) > 5.887 / (2.0 * 50.0)


def test_sigma_is_floored_and_capped() -> None:
    """The floor reflects 8-bit quantisation; the cap stops a near-zero SNR exploding R."""
    assert measurement_sigma(1e6, 5.887) == pytest.approx(0.02)
    assert measurement_sigma(1e-6, 5.887) == pytest.approx(20.0)


def test_unmeasurable_snr_is_not_treated_as_worst_case_snr() -> None:
    """An SNR that could not be *measured* is not an SNR that is known to be terrible.

    Returning the ceiling for ``None`` was a real defect. On a compressed 2000x2000 clip the
    background MAD collapsed to zero, SNR became unmeasurable, sigma jumped straight to the 20 px
    ceiling, and the Kalman gain collapsed -- so the filter accepted every (good) measurement
    while weighting it to nothing and drifting open-loop past 2000 px.

    With no history the fallback is mid-range; with history it reuses the last confidently
    measured sigma, so a transient measurement failure cannot spike ``R``.
    """
    assert measurement_sigma(None, 5.887) < 5.0
    assert measurement_sigma(None, 5.887, last_known_sigma_px=1.67) == pytest.approx(1.67)
    # It must still be worse than a strong measured SNR, just not catastrophically so.
    assert measurement_sigma(None, 5.887) > measurement_sigma(100.0, 5.887)


def test_effective_sigma_keeps_the_gate_discriminating() -> None:
    """A gate only discriminates while R stays commensurate with P.

    Once R dominates, S is set almost entirely by R, every normalised innovation looks small, and
    the gate accepts anything. Capping the ratio preserves discrimination however sigma arose.
    """
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0)
    # Drive P down with confident updates, so a large sigma would otherwise dominate.
    for _ in range(40):
        kalman.predict(DT)
        kalman.update((100.0, 100.0), 0.1)

    position_sigma = kalman.position_sigma_px
    assert kalman.effective_sigma(1000.0) <= \
        kalman.params.max_sigma_to_p_ratio * position_sigma

    # With the cap in place a wild outlier is still rejected.
    kalman.predict(DT)
    assert not kalman.gate((900.0, 900.0), 1000.0).accepted


def test_bias_flags_widen_sigma_by_their_measured_budgets() -> None:
    """Clipping and saturation are bias, not variance, and are folded in as a budget.

    A Kalman filter has no representation for bias, so the honest treatment is to widen R by the
    measured displacement -- 2 px for clipping (Phase 1), 0.19 px for saturation (Phase 2) -- and
    let the filter lean on its prediction accordingly.
    """
    clean = measurement_sigma(100.0, 5.887)
    saturated = measurement_sigma(100.0, 5.887, saturated=True)
    clipped = measurement_sigma(100.0, 5.887, clipped=True)

    assert saturated > clean
    assert clipped > saturated
    assert clipped == pytest.approx(math.hypot(clean, 2.0), rel=0.01)
    assert saturated == pytest.approx(math.hypot(clean, 0.19), rel=0.01)


def test_sigma_rejects_bad_scale() -> None:
    """A non-positive FWHM is a caller error."""
    with pytest.raises(ValueError, match="FWHM must be positive"):
        measurement_sigma(10.0, 0.0)


# ------------------------------------------------------------------------------------------
# Filter mechanics
# ------------------------------------------------------------------------------------------


def test_transition_and_process_noise_have_the_documented_form() -> None:
    """F and Q must match DESIGN section 6 exactly."""
    F = KalmanFilterCV.transition(0.5)
    assert F[0, 2] == 0.5 and F[1, 3] == 0.5
    assert F[2, 2] == 1.0 and F[3, 3] == 1.0

    Q = KalmanFilterCV.process_noise(0.5, 2.0)
    assert Q[0, 0] == pytest.approx(2.0 * 0.125 / 3.0)
    assert Q[2, 2] == pytest.approx(2.0 * 0.5)
    assert Q[0, 2] == pytest.approx(2.0 * 0.25 / 2.0)
    assert np.allclose(Q, Q.T)


def test_filter_recovers_velocity_of_a_constant_velocity_target() -> None:
    """The velocity state is what feeds controller feedforward, so it must be right."""
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0)
    rng = np.random.default_rng(0)
    for step in range(300):
        kalman.predict(DT)
        t = DT * (step + 1)
        z = (100.0 + 120.0 * t + rng.normal(0, 0.5), 100.0 + 60.0 * t + rng.normal(0, 0.5))
        kalman.update(z, 0.5)
    vx, vy = kalman.velocity
    assert vx == pytest.approx(120.0, rel=0.1)
    assert vy == pytest.approx(60.0, rel=0.1)


def test_filter_is_statistically_consistent() -> None:
    """Mean NIS should sit near the 2 degrees of freedom of a 2-D measurement.

    This is the standard filter-consistency check and the evidence that adaptive R is scaled
    correctly rather than merely plausible.
    """
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0, 120.0, 60.0)
    rng = np.random.default_rng(1)
    scores = []
    for step in range(500):
        kalman.predict(DT)
        t = DT * (step + 1)
        z = (100.0 + 120.0 * t + rng.normal(0, 0.5), 100.0 + 60.0 * t + rng.normal(0, 0.5))
        gate = kalman.gate(z, 0.5)
        if gate.accepted:
            kalman.update(z, 0.5)
        scores.append(gate.nis)
    assert 1.0 < float(np.mean(scores[100:])) < 4.0


def test_covariance_stays_symmetric_and_positive_definite() -> None:
    """The Joseph form must hold up over a long run where the short form can drift negative."""
    kalman = KalmanFilterCV()
    kalman.initialise(0.0, 0.0)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        kalman.predict(DT)
        kalman.update((rng.normal(0, 1), rng.normal(0, 1)), 0.5)
    P = kalman.covariance
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_gate_rejects_a_distant_outlier() -> None:
    """The gate is what removes impulse false positives the detector cannot."""
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0)
    kalman.predict(DT)
    assert kalman.gate((100.2, 99.9), 0.5).accepted
    assert not kalman.gate((400.0, 50.0), 0.5).accepted


def test_gate_threshold_is_the_documented_chi_squared_value() -> None:
    """9.21 is chi-squared 99% for 2 DOF."""
    assert KalmanParams().gate_threshold == pytest.approx(CHI2_99_2DOF)


def test_operations_require_initialisation() -> None:
    """Using an empty filter is a caller error, not something to paper over."""
    kalman = KalmanFilterCV()
    with pytest.raises(ValueError, match="initialised"):
        kalman.predict(DT)
    with pytest.raises(ValueError, match="initialised"):
        kalman.gate((0.0, 0.0), 1.0)
    kalman.initialise(0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        kalman.predict(-DT)


# ------------------------------------------------------------------------------------------
# Track lifecycle -- the gate-lockout defect
# ------------------------------------------------------------------------------------------


def _moving_measurements(n: int, speed_px_s: float, noise_px: float,
                         seed: int = 0):
    """Yield ``(t, (x, y), truth)`` for a constant-velocity target with measurement noise."""
    rng = np.random.default_rng(seed)
    for step in range(n):
        t = step * DT
        truth = (40.0 + speed_px_s * t, 60.0 + 0.5 * speed_px_s * t)
        yield t, (truth[0] + rng.normal(0, noise_px), truth[1] + rng.normal(0, noise_px)), truth


def test_zero_velocity_initiation_rejects_correct_measurements() -> None:
    """The lockout mechanism itself, constructed deterministically rather than hoped for.

    A filter initiated from one detection starts at zero velocity. Against a target moving several
    pixels per frame the prediction is behind by exactly that much, and once the position
    covariance has collapsed the gate is far tighter than the lag. The measurement below is
    *perfect* -- it is the true position -- and it is rejected anyway. A rejected measurement
    produces no update, so the velocity is never learned and the lag grows without bound.
    """
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0)          # zero velocity: the defect
    # The state after a few accepted updates: confident in position (0.2 px) and moderately
    # confident in velocity (10 px/s) -- but the velocity it is confident about is *zero*, while
    # the target is moving. Collapsing only the position covariance does not reproduce the
    # lockout, because the initial velocity variance alone keeps the predicted covariance large.
    kalman._P = np.diag([0.04, 0.04, 100.0, 100.0])

    speed_px_s = 100.0
    kalman.predict(DT)
    truth = (100.0 + speed_px_s * DT, 100.0)

    gate = kalman.gate(truth, 0.2)
    assert not gate.accepted, "expected a correct measurement to be gated out"
    assert gate.mahalanobis_sq > CHI2_99_2DOF


def test_bare_filter_diverges_under_sustained_lockout() -> None:
    """With every measurement rejected, error grows without bound while detection is healthy."""
    kalman = KalmanFilterCV()
    kalman.initialise(100.0, 100.0)
    kalman._P = np.diag([0.04, 0.04, 100.0, 100.0])

    errors = []
    for step in range(60):
        kalman.predict(DT)
        t = DT * (step + 1)
        truth = (100.0 + 100.0 * t, 100.0)
        gate = kalman.gate(truth, 0.2)
        if gate.accepted:
            kalman.update(truth, 0.2)
        errors.append(math.dist(kalman.position, truth))
    assert errors[-1] > 10.0
    assert errors[-1] > errors[0]


def test_track_survives_where_a_bare_filter_locks_itself_out() -> None:
    """Two-point initiation plus ungated confirmation removes the lockout entirely.

    Measured end to end through the real vision pipeline on a dim target, this took track loss
    from 3 runs in 12 to 0 in 12.
    """
    track = Track()
    errors = []
    for t, z, truth in _moving_measurements(120, 100.0, 0.3, seed=3):
        update = track.update(z, DT, t, snr_aperture=40.0)
        if update.has_estimate:
            errors.append(math.dist(update.position, truth))
    assert track.is_established
    assert errors[-1] < 2.0
    assert float(np.median(errors[20:])) < 1.0


def test_two_point_initiation_recovers_velocity_immediately() -> None:
    """Velocity comes from the first detections, never assumed zero."""
    track = Track(TrackParams(confirm_m_of_n=(3, 5)))
    for t, z, _ in _moving_measurements(4, 120.0, 0.0, seed=0):
        track.update(z, DT, t, snr_aperture=40.0)
    assert track.status is TrackStatus.CONFIRMED
    vx, vy = track.filter.velocity
    assert vx == pytest.approx(120.0, rel=0.05)
    assert vy == pytest.approx(60.0, rel=0.05)


# --- the two cases that look identical for a few frames and must resolve oppositely ---------
#
# Both present as "the incumbent is rejecting every detection". They are opposites:
#
#   A) LOCKOUT  -- the incumbent's *state* is wrong (zero-velocity initiation against a moving
#      target). The detections are correct and keep coming. The incumbent must be replaced.
#   B) DETECTOR EXCURSION -- the incumbent is *correct* and the detector has latched onto a
#      spurious object for a few frames. The incumbent must be kept; replacing it discards the
#      only accurate state in the system, which is exactly what the old lockout rule did on
#      canvas_2000_fog.
#
# Over a short window the two are indistinguishable: a spurious object that persists for a few
# frames is as self-consistent as a real one, so a challenger fitted to it scores just as well.
# **Persistence is the discriminator.** An excursion ends and the true detections return, at
# which point the incumbent starts explaining them again; a genuinely wrong incumbent never
# recovers. The contest window is therefore longer than a typical excursion, and these two tests
# sit together so that relationship cannot be broken by tuning one without the other.


def test_contest_replaces_an_incumbent_whose_state_is_genuinely_wrong() -> None:
    """Case A: the Phase 4 lockout. Detections are correct and persistent, so the challenger wins."""
    track = Track(TrackParams(max_consecutive_rejections=3, contest_window_frames=12))
    for step in range(5):
        track.update((100.0, 100.0), DT, step * DT, snr_aperture=100.0)
    assert track.status is TrackStatus.CONFIRMED

    # The target was never really stationary: it has been moving fast all along, so the
    # incumbent's zero-velocity state cannot explain any detection, and never will.
    # Time is continuous with the initiation above -- a jump in `t` while still passing dt=DT
    # would leave the filter permanently behind and fake this result.
    reasons = []
    for step in range(30):
        target = (100.0 + 200.0 * (step + 1) * DT, 100.0)
        update = track.update(target, DT, (5 + step) * DT, snr_aperture=100.0)
        reasons.append(update.reason)

    assert "contest_opened" in reasons
    assert "contest_challenger_won" in reasons, reasons
    assert track.filter.position[0] > 120.0


def test_contest_keeps_a_correct_incumbent_through_a_detector_excursion() -> None:
    """Case B: canvas_2000_fog. The detector latches onto a spurious object, then recovers.

    The incumbent is right throughout and must survive. The old lockout rule discarded it here and
    adopted the spurious object, tracking it at NIS 0.0 while the true detections were rejected.
    """
    track = Track(TrackParams(max_consecutive_rejections=3, contest_window_frames=12))

    def truth_at(step: int):
        t = step * DT
        return 40.0 + 60.0 * t, 60.0 + 30.0 * t

    for step in range(10):
        track.update(truth_at(step), DT, step * DT, snr_aperture=100.0)
    assert track.status is TrackStatus.CONFIRMED

    # Six frames of spurious detections far away, then the true detections return. Time is
    # continuous throughout -- a discontinuity here would strand the incumbent behind the truth
    # and make it look wrong when it is not.
    reasons = []
    for step in range(10, 34):
        spurious = step < 16
        true_position = truth_at(step)
        z = ((true_position[0] + 2100.0, true_position[1]) if spurious else true_position)
        update = track.update(z, DT, step * DT, snr_aperture=100.0)
        reasons.append(update.reason)

    assert "contest_opened" in reasons
    assert "contest_challenger_won" not in reasons, \
        "the incumbent was correct; replacing it is the canvas_2000_fog failure"
    assert "contest_incumbent_held" in reasons, reasons
    assert math.dist(track.filter.position, truth_at(33)) < 20.0


def test_incumbent_keeps_producing_output_during_the_contest() -> None:
    """The incumbent is never suspended while a challenger is evaluated.

    During the failure that motivated this, the incumbent was the only correct state in the
    system for the whole contest, and the controller consumed its output every frame.
    """
    track = Track(TrackParams(max_consecutive_rejections=3, contest_window_frames=12))
    for t, z, _ in _moving_measurements(6, 60.0, 0.0, seed=0):
        track.update(z, DT, t, snr_aperture=100.0)

    for step in range(10):
        update = track.update((3000.0, 3000.0), DT, (6 + step) * DT, snr_aperture=100.0)
        assert update.has_estimate, "incumbent stopped producing output during the contest"
        assert update.position is not None


def test_contest_is_visible_in_the_trace() -> None:
    """A track replacement must never be invisible in the log."""
    track = Track(TrackParams(max_consecutive_rejections=3, contest_window_frames=8))
    for step in range(5):
        track.update((100.0, 100.0), DT, step * DT, snr_aperture=100.0)

    seen_open = False
    for step in range(20):
        update = track.update((100.0 + 200.0 * (step + 1) * DT, 100.0), DT, (5 + step) * DT,
                              snr_aperture=100.0)
        if update.reason == "contest_opened":
            seen_open = True
        if update.contest_open:
            assert update.contest_frames >= 0
            assert update.contest_incumbent_nis is not None
        if update.reason.startswith("contest_") and update.reason != "contest_opened":
            assert update.contest_frames > 0
            assert update.contest_challenger_nis is not None
    assert seen_open


def test_m_of_n_confirmation_requires_several_detections() -> None:
    """One detection must not establish a track."""
    track = Track(TrackParams(confirm_m_of_n=(3, 5)))
    assert track.update((10.0, 10.0), DT, 0.0).status is TrackStatus.INITIATING
    assert not track.is_established
    track.update((11.0, 11.0), DT, DT)
    assert track.update((12.0, 12.0), DT, 2 * DT).status is TrackStatus.CONFIRMED


def test_track_coasts_through_dropouts_then_recovers() -> None:
    """Coasting is what makes the <=1 s re-acquisition budget reachable."""
    track = Track(TrackParams(delete_after_missed=5))
    for t, z, _ in _moving_measurements(5, 90.0, 0.0, seed=0):
        track.update(z, DT, t)
    assert track.status is TrackStatus.CONFIRMED

    update = track.update(None, DT, 1.0)
    assert update.status is TrackStatus.COASTING
    assert update.coasting and update.has_estimate

    resumed = track.update(track.filter.position, DT, 1.0 + DT, snr_aperture=40.0)
    assert resumed.status is TrackStatus.CONFIRMED


def test_track_is_deleted_after_sustained_loss() -> None:
    """Sustained absence must delete the track, starting the re-acquisition clock."""
    track = Track(TrackParams(delete_after_missed=3))
    for t, z, _ in _moving_measurements(5, 90.0, 0.0, seed=0):
        track.update(z, DT, t)
    for step in range(5):
        update = track.update(None, DT, 1.0 + step * DT)
    assert update.status is TrackStatus.EMPTY
    assert not track.is_established


def test_track_propagates_flags_into_sigma() -> None:
    """A clipped detection must widen R, not be discarded."""
    track = Track()
    for t, z, _ in _moving_measurements(4, 60.0, 0.0, seed=0):
        track.update(z, DT, t, snr_aperture=50.0)
    clean = track.update((46.0, 63.0), DT, 1.0, snr_aperture=50.0)
    flagged = track.update((46.0, 63.0), DT, 1.0 + DT, snr_aperture=50.0, clipped=True)
    assert flagged.sigma_meas_px > clean.sigma_meas_px


def test_track_reset_clears_everything() -> None:
    """Reset must return the track to empty so a scenario can be re-run."""
    track = Track()
    for t, z, _ in _moving_measurements(5, 60.0, 0.0, seed=0):
        track.update(z, DT, t)
    track.reset()
    assert track.status is TrackStatus.EMPTY
    assert not track.filter.initialised


# ------------------------------------------------------------------------------------------
# Fix 1 — initiation velocity bound  (Phase A §P3)
# ------------------------------------------------------------------------------------------


def test_spurious_first_detection_does_not_produce_a_phantom_track() -> None:
    """A single bad first detection must not lock the track onto a phantom velocity.

    Phase A §P3: on a large-spot clip the first detection was 499.9 px from the true position.
    Two-point initiation derived 7550 px/s from that pair, far above the 279 px/s trackable
    ceiling. The track then flew at 251.7 px/frame and gated out every subsequent correct
    detection (39.8 % / 13.6 % association failure on 15 / 20 px spots).

    The fix: if the implied speed exceeds `max_initiation_velocity_px_s` the oldest pending
    detection is discarded and initiation waits for a fresh pair. The velocity should come from
    two *correct* detections, not from one bad and one good.

    The default bound is 4000 px/s (5× the slew ceiling). 7550 px/s >> 4000, so the phantom is
    still caught; and 900 px/s (the highest legitimate speed in the test suite) << 4000.
    """
    # Use the default bound (4000 px/s), which is well above any real target speed but far
    # below the 7550 px/s phantom that triggered this bug.
    params = TrackParams(confirm_m_of_n=(3, 5), max_initiation_velocity_px_s=4000.0)
    track = Track(params)

    true_speed_px_s = 71.1  # the actual target speed from Phase A
    true_start = (320.0, 240.0)

    # Frame 0: spurious detection ~500 px away from truth. With DT=1/30 s, the implied speed
    # between frame 0 and frame 1 (which lands ~2.4 px away) is roughly 499*30 = 14970 px/s --
    # far above 4000, so the oldest detection is discarded.
    t0 = 0.0
    track.update((true_start[0] + 499.0, true_start[1]), DT, t0)
    assert track.status is TrackStatus.INITIATING

    # Frames 1+: correct detections. Without the fix the track would confirm on frames 0-1
    # with a 7550+ px/s phantom and gate out everything from frame 3 onward.
    for step in range(1, 20):
        t = step * DT
        truth = (true_start[0] + true_speed_px_s * t, true_start[1] + 0.5 * true_speed_px_s * t)
        update = track.update(truth, DT, t, snr_aperture=100.0)

    # The track must have confirmed on the correct detections and hold a plausible state.
    assert track.is_established, "track never confirmed after correct detections"
    vx, vy = track.filter.velocity
    speed = math.hypot(vx, vy)
    assert speed < 4000.0, f"phantom velocity leaked through: {speed:.1f} px/s"
    assert speed == pytest.approx(true_speed_px_s, rel=0.20), \
        f"recovered velocity {speed:.1f} px/s is far from truth {true_speed_px_s:.1f} px/s"


def test_fast_but_realistic_target_still_initiates() -> None:
    """A target moving quickly but within the bound must still confirm normally.

    The velocity bound must not block legitimate fast targets -- only absurdly spurious ones.
    The default bound is 4000 px/s; targets at 900 px/s (the highest in the test suite) must
    pass through without triggering the discard.
    """
    # 900 px/s: above the slew ceiling (800) but well below the bound (4000).
    speed = 900.0
    params = TrackParams(confirm_m_of_n=(3, 5), max_initiation_velocity_px_s=4000.0)
    track = Track(params)
    for t, z, _ in _moving_measurements(8, speed, 0.0, seed=7):
        track.update(z, DT, t, snr_aperture=100.0)
    assert track.is_established


# ------------------------------------------------------------------------------------------
# Fix 2 — contest NIS gate  (Phase A §P5 fix 2)
# ------------------------------------------------------------------------------------------


def test_contest_rejects_an_incumbent_with_catastrophic_nis() -> None:
    """An incumbent with mean NIS in the hundreds must never win a contest.

    Phase A §P5 fix 2: at frame 18, a correct challenger lost the contest to an incumbent
    carrying NIS = 1147, because the classic NIS-margin test asks \"is the challenger better?\"
    rather than \"is the incumbent even coherent?\". A hypothesis explaining nothing should not
    be kept regardless of what the challenger scores over the window so far.
    """
    # Set a generous NIS ceiling (well below 1147) and a long enough window to accumulate NIS.
    params = TrackParams(
        confirm_m_of_n=(3, 5),
        max_consecutive_rejections=3,
        contest_window_frames=8,
        incumbent_max_mean_nis=100.0,
    )
    track = Track(params)

    # Confirm on a slow target.
    for step in range(6):
        track.update((100.0, 100.0), DT, step * DT, snr_aperture=100.0)
    assert track.status is TrackStatus.CONFIRMED

    # Now a fast target starts at a completely different location. The incumbent's state (near
    # 100, 100 with near-zero velocity) will accumulate enormous NIS against these detections.
    reasons = []
    for step in range(6, 6 + 20):
        fast_pos = (100.0 + 400.0 * (step - 5) * DT, 100.0)
        update = track.update(fast_pos, DT, step * DT, snr_aperture=100.0)
        reasons.append(update.reason)

    # The challenger must have won -- the incumbent's NIS ceiling forces it out.
    assert "contest_challenger_won" in reasons, \
        "incumbent with NIS >> ceiling should have been ejected"


def test_well_calibrated_incumbent_is_unaffected_by_nis_ceiling() -> None:
    """A healthy incumbent (mean NIS ~2) must not be evicted by the NIS ceiling.

    The ceiling is intentionally generous (100) and should only fire against states that are
    clearly broken, not against any incumbent that is merely slightly off.
    """
    params = TrackParams(
        confirm_m_of_n=(3, 5),
        max_consecutive_rejections=3,
        contest_window_frames=12,
        incumbent_max_mean_nis=100.0,
    )
    track = Track(params)

    # A well-estimated track following a real target.
    for t, z, _ in _moving_measurements(10, 60.0, 0.3, seed=5):
        track.update(z, DT, t, snr_aperture=80.0)
    assert track.is_established

    # A short burst of off-target detections opens a contest but should not eject the incumbent
    # (the incumbent has mean NIS well below 100 for a brief perturbation).
    reasons = []
    for step in range(10, 16):
        track.update((900.0, 900.0), DT, step * DT, snr_aperture=80.0)
        reasons.append(track.status)
    # The incumbent must still be producing output; a healthy state should survive.
    assert track.is_established
