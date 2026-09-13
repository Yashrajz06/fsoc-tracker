"""Tests for PID, controller, state machine and search.

Includes the step-response test that lifts the PROVISIONAL marker from the PID gains. That test
runs **with the real Kalman filter in the loop**, because the controller consumes smoothed state
and the filter's lag is therefore inside the loop -- gains validated against a perfect measurement
would not be validated against the system we actually ship.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.camera.model import CameraModel
from src.config import load_config
from src.control.controller import ControllerParams, PointingController
from src.control.pid import Pid, PidParams
from src.control.search import SearchParams, SpiralSearch
from src.control.statemachine import (
    AcquisitionClass,
    StateMachineParams,
    TrackState,
    TrackingStateMachine,
)
from src.filtering.track import Track, TrackStatus

DT = 1.0 / 30.0

#: Aperture SNR consistent with the 0.3 px measurement noise these synthetic tests inject.
#:
#: The value matters. Claiming SNR 40 while injecting 0.3 px of noise implies sigma = 0.097 px,
#: so a correct measurement scores NIS ~9.6 against a 9.21 gate and roughly two thirds of good
#: measurements are rejected. That inconsistency was invisible while the old lockout rule
#: re-initiated the track every five rejections; replacing it with a competing-hypothesis contest
#: exposed it. Deriving the SNR from the injected noise keeps the filter's own calibration honest
#: rather than relying on recovery machinery to paper over it.
CONSISTENT_SNR = 13.0


@pytest.fixture(scope="module")
def config():
    """The validated default configuration."""
    return load_config("config/default.json")


def _gains(config) -> PidParams:
    """PID parameters from the configuration."""
    pid = config.control.pid
    return PidParams(kp=float(pid["kp"]), ki=float(pid["ki"]), kd=float(pid["kd"]),
                     integral_limit=float(pid["integral_limit"]))


# ------------------------------------------------------------------------------------------
# PID
# ------------------------------------------------------------------------------------------


def test_proportional_gain_saturates_at_the_design_error(config) -> None:
    """The sizing rule: a ~100 px error should command the full slew rate."""
    pid = Pid(PidParams(kp=float(config.control.pid["kp"]), ki=0.0, kd=0.0, output_limit=5.0))
    error_deg = 100.0 * config.camera.deg_per_pixel[0]
    assert pid.step(error_deg, DT).output == pytest.approx(5.0)


def test_output_is_clamped_to_the_slew_ceiling() -> None:
    """A controller bug must degrade tracking, never produce an impossible command."""
    pid = Pid(PidParams(kp=100.0, output_limit=5.0))
    assert pid.step(10.0, DT).output == pytest.approx(5.0)
    assert pid.step(-10.0, DT).output == pytest.approx(-5.0)


def test_anti_windup_clamps_the_integral_not_just_the_output() -> None:
    """Clamping only the output lets the integral accumulate while the actuator is saturated.

    During acquisition, errors of hundreds of pixels make saturation the normal case, so an
    unclamped integrator would keep commanding in one direction long after the error reversed.
    """
    pid = Pid(PidParams(kp=8.0, ki=1.0, integral_limit=2.0, output_limit=5.0))
    for _ in range(200):
        pid.step(10.0, DT)
    assert abs(pid.integral) <= 2.0 + 1e-9


def test_saturation_is_reported() -> None:
    """Slew saturation is a physical limit, not a tracking failure; the two must be separable."""
    pid = Pid(PidParams(kp=100.0, output_limit=5.0))
    assert pid.step(10.0, DT).saturated
    assert not pid.step(0.001, DT).saturated


def test_pid_rejects_negative_dt() -> None:
    """Time must not run backwards."""
    with pytest.raises(ValueError, match="non-negative"):
        Pid().step(1.0, -DT)


# ------------------------------------------------------------------------------------------
# Step response -- the PROVISIONAL gate
# ------------------------------------------------------------------------------------------


def _step_response(config, step_px: float = 100.0, use_filter: bool = True,
                   seed: int = 0, noise_px: float = 0.3, frames: int = 200) -> np.ndarray:
    """Drive a step in pointing error and return the error history.

    Args:
        config: Application configuration.
        step_px: Step size in pixels.
        use_filter: Route measurements through the real track/Kalman pipeline.
        seed: RNG seed.
        noise_px: Measurement noise standard deviation.
        frames: Number of control steps.

    Returns:
        Pointing error per frame, in pixels.
    """
    camera = CameraModel.from_config(config)
    camera.set_boresight(1000.0, 1000.0)
    controller = PointingController(camera, ControllerParams(pid=_gains(config)))
    target = (1000.0 + step_px, 1000.0)
    track = Track()
    rng = np.random.default_rng(seed)

    errors = []
    for step in range(frames):
        t = step * DT
        z = (target[0] + rng.normal(0, noise_px), target[1] + rng.normal(0, noise_px))
        if use_filter:
            update = track.update(z, DT, t, snr_aperture=CONSISTENT_SNR)
            if not update.has_estimate:
                errors.append(math.dist(camera.boresight, target))
                continue
            controller.step(update.position, DT, update.velocity)
        else:
            controller.step(z, DT)
        errors.append(math.dist(camera.boresight, target))
    return np.asarray(errors)


def test_step_response_settles_within_budget_with_the_filter_in_the_loop(config) -> None:
    """**This is the test that lifts the PROVISIONAL marker.**

    Gains were originally sized analytically, assuming pointing error maps straight to a rate
    command. The controller consumes Kalman-smoothed state, so the filter's lag is inside the
    loop and the analytic value is optimistic: measured with the filter in the loop, Kp=8 settles
    a 100 px step in 1.07 s against 0.30 s for a perfect measurement. Kp=12 restores 0.40 s.

    A gain validated against an idealised measurement is not validated against the system we
    ship -- the same reason the R calibration had to go through the real pipeline rather than the
    theoretical law.
    """
    errors = _step_response(config, use_filter=True)
    settled = np.where(errors <= 5.0)[0]
    assert len(settled), "step never settled"
    settle_time = settled[0] * DT
    assert settle_time < 1.0, f"settled in {settle_time:.3f} s"
    assert float(np.median(errors[-60:])) < 3.0


def test_filter_lag_is_inside_the_loop_and_measurable(config) -> None:
    """Pin the effect that invalidated the analytic gain derivation."""
    filtered = _step_response(config, use_filter=True)
    ideal = _step_response(config, use_filter=False)

    def settle(errors):
        idx = np.where(errors <= 5.0)[0]
        return idx[0] * DT if len(idx) else float("inf")

    assert settle(filtered) >= settle(ideal), \
        "the filter should cost settling time; if not, it may be bypassed"


def test_gains_are_no_longer_provisional(config) -> None:
    """The marker may only be absent once the step-response test above exists and passes."""
    assert not config.control.pid_gains_are_provisional
    assert not any("PROVISIONAL" in message for message in config.startup_warnings())


def test_integral_rejects_platform_drift(config) -> None:
    """The integral term exists for low-frequency bias, and must be sized for it.

    Measured at 150 px/s platform drift: ki=0 leaves 19.5 px steady-state error, outside the
    10 px budget entirely; the configured ki holds it near 4 px.
    """
    def drift_error(ki: float) -> float:
        camera = CameraModel.from_config(config)
        camera.set_boresight(1000.0, 1000.0)
        gains = _gains(config)
        controller = PointingController(
            camera, ControllerParams(pid=PidParams(kp=gains.kp, ki=ki, kd=gains.kd,
                                                   integral_limit=gains.integral_limit)))
        track = Track()
        rng = np.random.default_rng(0)
        errors = []
        for step in range(450):
            t = step * DT
            target = (1000.0, 1000.0)
            z = (target[0] + rng.normal(0, 0.3), target[1] + rng.normal(0, 0.3))
            update = track.update(z, DT, t, snr_aperture=CONSISTENT_SNR)
            if update.has_estimate:
                controller.step(update.position, DT, update.velocity)
            camera.set_boresight(camera.boresight[0] + 150.0 * DT, camera.boresight[1])
            errors.append(math.dist(camera.boresight, target))
        return float(np.median(errors[300:]))

    assert drift_error(0.0) > 10.0
    assert drift_error(float(config.control.pid["ki"])) < 10.0


# ------------------------------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------------------------------


def test_controller_drives_boresight_toward_the_target(config) -> None:
    """The basic closed-loop behaviour."""
    camera = CameraModel.from_config(config)
    camera.set_boresight(900.0, 900.0)
    controller = PointingController(camera, ControllerParams(pid=_gains(config)))
    before = math.dist(camera.boresight, (1000.0, 1000.0))
    for _ in range(60):
        controller.step((1000.0, 1000.0), DT)
    assert math.dist(camera.boresight, (1000.0, 1000.0)) < before / 10.0


def test_feedforward_reduces_lag_on_a_moving_target(config) -> None:
    """Feedback alone lags a moving reference; feedforward cancels it.

    The integral term is disabled here to isolate the effect. That is not a convenience: for a
    *constant-velocity* target the integral reaches the same place by a different route -- it
    integrates a constant error into a constant rate -- so with both active they are largely
    redundant and the comparison measures nothing. Measured with ki active, feedforward looks
    mildly harmful (6.7 px against 4.5 px); with ki=0 its contribution is unambiguous
    (6.7 px against 10.0 px).
    """
    gains = _gains(config)

    def trail(enabled: bool) -> float:
        camera = CameraModel.from_config(config)
        camera.set_boresight(500.0, 1000.0)
        controller = PointingController(
            camera,
            ControllerParams(pid=PidParams(kp=gains.kp, ki=0.0, kd=gains.kd,
                                           integral_limit=gains.integral_limit),
                             feedforward_enabled=enabled))
        errors = []
        for step in range(300):
            t = step * DT
            target = (500.0 + 200.0 * t, 1000.0)
            if target[0] > 1800.0:
                break
            controller.step(target, DT, (200.0, 0.0))
            errors.append(math.dist(camera.boresight, target))
        return float(np.median(errors[len(errors) // 2:]))

    assert trail(True) < trail(False)


def test_feedforward_beats_the_integral_on_a_manoeuvring_target(config) -> None:
    """Where feedforward genuinely earns its place rather than merely duplicating the integral.

    Feedforward acts on the *current* velocity estimate, so it responds immediately to a change.
    The integral has to wind up again, which takes time proportional to its gain. On a target that
    keeps changing velocity the difference is real, whereas on a constant-velocity ramp the two
    are interchangeable.
    """
    gains = _gains(config)

    def trail(enabled: bool) -> float:
        camera = CameraModel.from_config(config)
        camera.set_boresight(1000.0, 1000.0)
        controller = PointingController(
            camera, ControllerParams(pid=gains, feedforward_enabled=enabled))
        errors = []
        omega = 2.0 * math.pi * 0.25          # quarter-hertz weave
        for step in range(400):
            t = step * DT
            target_x = 1000.0 + 250.0 * math.sin(omega * t)
            velocity_x = 250.0 * omega * math.cos(omega * t)
            controller.step((target_x, 1000.0), DT, (velocity_x, 0.0))
            errors.append(abs(camera.boresight[0] - target_x))
        return float(np.median(errors[150:]))

    assert trail(True) < trail(False)


def test_controller_never_exceeds_the_slew_limit(config) -> None:
    """Limits live in the camera model, so a controller bug cannot break physics."""
    camera = CameraModel.from_config(config)
    camera.set_boresight(100.0, 100.0)
    controller = PointingController(camera, ControllerParams(pid=_gains(config)))
    previous = camera.boresight
    for _ in range(120):
        controller.step((1900.0, 1900.0), DT)
        travelled = math.dist(camera.boresight, previous)
        assert travelled <= math.hypot(*camera.max_px_per_frame) + 1e-6
        previous = camera.boresight


def test_controller_rejects_negative_dt(config) -> None:
    """Caller errors fail loudly."""
    camera = CameraModel.from_config(config)
    controller = PointingController(camera)
    with pytest.raises(ValueError, match="non-negative"):
        controller.step((1000.0, 1000.0), -DT)


# ------------------------------------------------------------------------------------------
# State machine
# ------------------------------------------------------------------------------------------


def test_lock_requires_k_consecutive_frames() -> None:
    """K=3 is what stops a single noise blob establishing lock."""
    machine = TrackingStateMachine(StateMachineParams(lock_confirm_frames=3),
                                   target_initially_in_fov=True)
    assert machine.update(True, 10.0, 0.0, DT).state is TrackState.SEARCH
    assert machine.update(True, 10.0, DT, DT).state is TrackState.SEARCH
    assert machine.update(True, 10.0, 2 * DT, DT).state is TrackState.TRACK


def test_hysteresis_prevents_mode_chatter() -> None:
    """Different enter and exit windows, so a target near the boundary cannot chatter the mode.

    Mode chatter is worse than either state: it resets timers and restarts the integral.
    """
    machine = TrackingStateMachine(StateMachineParams(lock_window_px=40.0,
                                                      unlock_window_px=80.0),
                                   target_initially_in_fov=True)
    for step in range(3):
        machine.update(True, 20.0, step * DT, DT)
    assert machine.state is TrackState.TRACK

    # Between the two windows: would fail the enter test, but must not trigger the exit test.
    for step in range(20):
        machine.update(True, 60.0, (3 + step) * DT, DT)
    assert machine.state is TrackState.TRACK

    machine.update(True, 90.0, 1.0, DT)
    assert machine.state is TrackState.COAST


def test_loss_after_n_misses_then_coast_timeout_returns_to_search() -> None:
    """The full COAST path, including the timeout back to SEARCH."""
    params = StateMachineParams(loss_declare_frames=5, coast_timeout_seconds=0.2)
    machine = TrackingStateMachine(params, target_initially_in_fov=True)
    for step in range(3):
        machine.update(True, 10.0, step * DT, DT)
    assert machine.state is TrackState.TRACK

    for step in range(5):
        machine.update(False, math.inf, (3 + step) * DT, DT)
    assert machine.state is TrackState.COAST

    for step in range(10):
        machine.update(False, math.inf, 1.0 + step * DT, DT)
    assert machine.state is TrackState.SEARCH


def test_acquisition_is_split_into_two_populations() -> None:
    """In-FOV and search-limited must never be pooled.

    Pooling hides a physical limit behind an initial-condition lottery: the search-limited case is
    bounded below by the slew ceiling at 11.6 s for the default canvas, against a 2 s budget.
    """
    in_fov = TrackingStateMachine(target_initially_in_fov=True)
    for step in range(3):
        in_fov.update(True, 10.0, step * DT, DT)
    assert len(in_fov.acquisitions(AcquisitionClass.IN_FOV)) == 1
    assert len(in_fov.acquisitions(AcquisitionClass.SEARCH_LIMITED)) == 0

    searched = TrackingStateMachine(target_initially_in_fov=False)
    for step in range(3):
        searched.update(True, 10.0, step * DT, DT)
    assert len(searched.acquisitions(AcquisitionClass.SEARCH_LIMITED)) == 1
    assert len(searched.acquisitions(AcquisitionClass.IN_FOV)) == 0


def test_reacquisition_is_never_charged_to_the_search_population() -> None:
    """After a loss the last position is known, so re-acquisition is local by definition."""
    machine = TrackingStateMachine(StateMachineParams(loss_declare_frames=2),
                                   target_initially_in_fov=False)
    for step in range(3):
        machine.update(True, 10.0, step * DT, DT)
    for step in range(2):
        machine.update(False, math.inf, (3 + step) * DT, DT)
    for step in range(3):
        machine.update(True, 10.0, 1.0 + step * DT, DT)

    reacquisitions = [e for e in machine.events if e.reacquisition]
    assert reacquisitions
    assert all(e.population is AcquisitionClass.IN_FOV for e in reacquisitions)


def test_inverted_hysteresis_windows_are_rejected() -> None:
    """Equal windows remove the hysteresis entirely."""
    with pytest.raises(ValueError, match="Hysteresis"):
        TrackingStateMachine(StateMachineParams(lock_window_px=40.0, unlock_window_px=40.0))


# ------------------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------------------


def test_arm_spacing_is_fov_derived_not_a_stored_constant(config) -> None:
    """0.9 x the limiting FOV dimension, so coverage has no gaps at any resolution."""
    search = SpiralSearch((1000.0, 1000.0), config.camera.fov_px, 800.0)
    assert search.arm_spacing_px == pytest.approx(0.9 * 480.0)
    assert search.arm_spacing_px != 300.0  # the stale configured constant


def test_coverage_time_reproduces_the_documented_worst_case(config) -> None:
    """11.6 s to sweep the 2000x2000 canvas -- the search-limited acquisition bound."""
    search = SpiralSearch((1000.0, 1000.0), config.camera.fov_px, 800.0)
    assert search.coverage_time_s(config.scene.area_px) == pytest.approx(11.6, abs=0.2)
    assert search.coverage_time_s(config.scene.area_px) > \
        config.telemetry.acquisition_target_s


def test_spiral_expands_and_covers_the_region(config) -> None:
    """Radius must grow, and reach the uncertainty region within the coverage time."""
    search = SpiralSearch((1000.0, 1000.0), config.camera.fov_px, 800.0)
    radii = []
    for _ in range(int(11.6 / DT)):
        search.step(DT)
        radii.append(search.radius_px)
    assert radii[-1] > radii[0]
    assert max(radii) > 900.0


def test_spiral_restarts_past_its_maximum_radius(config) -> None:
    """A bounded search must re-sweep rather than run away."""
    search = SpiralSearch((1000.0, 1000.0), config.camera.fov_px, 800.0,
                          SearchParams(max_radius_px=300.0))
    seen_reset = False
    previous = 0.0
    for _ in range(2000):
        search.step(DT)
        if search.radius_px < previous:
            seen_reset = True
            break
        previous = search.radius_px
    assert seen_reset


def test_search_rejects_bad_configuration(config) -> None:
    """Caller errors fail loudly."""
    with pytest.raises(ValueError, match="FOV must be positive"):
        SpiralSearch((0.0, 0.0), (0, 480), 800.0)
    with pytest.raises(ValueError, match="Scan speed"):
        SpiralSearch((0.0, 0.0), (640, 480), 0.0)


# ------------------------------------------------------------------------------------------
# Re-acquisition must not re-enter the gate lockout
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("speed_px_s", [150.0, 400.0, 700.0, 900.0])
def test_reacquisition_does_not_inherit_a_stale_velocity(speed_px_s: float) -> None:
    """Re-initiation after a loss must do fresh two-point initiation, not resume stale state.

    The three lockout fixes cover *initial* acquisition. After a loss the track re-initiates, and
    if that path inherited a stale velocity it could walk straight back into the
    confident-but-wrong-velocity state that caused the original lockout. The fast-target case is
    the one that matters, because that is where a stale velocity is most wrong.
    """
    track = Track()
    rng = np.random.default_rng(0)
    dropout = range(60, 72)
    post_errors = []
    reacquired_at = None

    for step in range(200):
        t = step * DT
        truth = (40.0 + speed_px_s * t, 60.0 + 0.4 * speed_px_s * t)
        z = None if step in dropout else (truth[0] + rng.normal(0, 0.3),
                                          truth[1] + rng.normal(0, 0.3))
        update = track.update(z, DT, t, snr_aperture=CONSISTENT_SNR)
        if step > dropout.stop and reacquired_at is None and \
                update.status is TrackStatus.CONFIRMED:
            reacquired_at = step
        if reacquired_at is not None and step > reacquired_at + 5 and update.has_estimate:
            post_errors.append(math.dist(update.position, truth))

    assert reacquired_at is not None, "never re-acquired"
    assert post_errors
    assert float(np.median(post_errors)) < 2.0, "post-re-acquisition lockout"

    vx, vy = track.filter.velocity
    assert vx == pytest.approx(speed_px_s, rel=0.10), "velocity not re-derived after re-acquisition"
    assert vy == pytest.approx(0.4 * speed_px_s, rel=0.15)
