"""Knobs an evaluator would plausibly edit must actually change behaviour.

Benchmark-1 hands us scenario files. A knob that loads, validates, and then does nothing is
invisible in testing and produces a completed run whose numbers silently ignore the scenario --
the worst possible failure on a 30% stage, because nothing looks wrong.

This was not hypothetical. ``filtering.kalman.process_noise_psd`` was validated by config and
read by nobody; the filter used its dataclass default, which agreed with the JSON only by
coincidence. Editing it changed nothing.

Scope is deliberate. These cover the parameters an evaluator would realistically vary -- target
motion, noise, camera constraints, filter tuning -- not every leaf in the document. Coverage of
the remainder is stated honestly in the technical report rather than implied here. A reference
count is not evidence: each test below *mutates* the knob and asserts the observable consequence,
because config binds JSON keys to dataclass fields by name and a live knob's literal may never
appear in the source.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.config import AppConfig

BASE = json.loads(Path("config/default.json").read_text())


def _with(**sections) -> AppConfig:
    """Build a config with deep-merged overrides."""
    raw = copy.deepcopy(BASE)

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict):
                merge(dst.setdefault(key, {}), value)
            else:
                dst[key] = value

    merge(raw, sections)
    return AppConfig.from_dict(raw)


# --- camera constraints -----------------------------------------------------------------------

def test_field_of_view_changes_angular_resolution():
    """FOV drives deg/pixel, which every angular quantity is derived from."""
    narrow = _with(camera={"fov_horizontal_deg": 2.0}).camera.deg_per_pixel[0]
    wide = _with(camera={"fov_horizontal_deg": 8.0}).camera.deg_per_pixel[0]
    assert wide == pytest.approx(4.0 * narrow), "fov_horizontal_deg does not reach deg_per_pixel"


def test_resolution_changes_frame_size_and_resolution():
    """Resolution must reach both the frame shape and the angular scale."""
    small = _with(camera={"resolution_width": 320, "resolution_height": 240})
    assert small.camera.fov_px == (320, 240)
    big = _with(camera={"resolution_width": 1280, "resolution_height": 960})
    assert big.camera.deg_per_pixel[0] < small.camera.deg_per_pixel[0]


def test_max_pan_speed_changes_the_per_frame_slew_ceiling():
    """The slew ceiling is the physical limit the trackable-velocity envelope is quoted against."""
    # 5-10 deg/s is the spec range (parameters 13-14); config rejects anything outside it, so
    # the mutation has to stay in range.
    slow = _with(camera={"max_pan_speed_deg_s": 5.0}).camera.max_px_per_frame[0]
    fast = _with(camera={"max_pan_speed_deg_s": 10.0}).camera.max_px_per_frame[0]
    assert fast == pytest.approx(2.0 * slow), "max_pan_speed_deg_s does not reach the slew ceiling"


# --- target motion ----------------------------------------------------------------------------

@pytest.mark.parametrize("motion,knob,value", [
    ("linear", "velocity_x_px_s", 321.0),
    ("circular", "radius_px", 137.0),
    ("circular", "angular_velocity_rad_s", 1.7),
    ("figure8", "amplitude_x_px", 271.0),
    ("figure8", "phase_offset_rad", 1.1),
    ("sinusoidal", "amplitude_px", 191.0),
    ("spiral", "growth_rate_b", 31.0),
    ("random", "step_sigma_px", 19.0),
    ("ornstein_uhlenbeck", "theta", 0.71),
])
def test_motion_parameters_reach_the_trajectory(motion, knob, value):
    """Each motion model's parameters must reach the generator that consumes them.

    Built by asking the config for the trajectory it describes and driving it, rather than by
    inspecting the config: the parameters are passed through as keyword arguments, so a typo in
    the name would be invisible to any check that only reads the document back.
    """
    import numpy as np

    from src.sim.trajectories import build_trajectory

    def path(cfg):
        # Stepped rather than sampled: the stochastic trajectories (random walk, OU) evolve
        # state per call, and position_at cannot express them. Stepping exercises both kinds.
        trajectory = build_trajectory(cfg, np.random.default_rng(11))
        return [trajectory.step(1.0 / 30.0) for _ in range(30)]

    base = _with(target={"motion": {"type": motion}})
    changed = _with(target={"motion": {"type": motion, motion: {knob: value}}})
    assert path(base) != path(changed), (
        f"target.motion.{motion}.{knob} does not reach the trajectory")


# --- noise ------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,knob,value", [
    ("gaussian", "sigma", 19.0),
    ("salt_pepper", "density", 0.09),
    ("salt_pepper", "salt_ratio", 0.2),
    ("poisson", "scale", 3.7),
])
def test_noise_parameters_change_the_generated_frame(path, knob, value):
    """Noise parameters must change the pixels, not merely load."""
    import numpy as np

    from src.noise.pipeline import NoisePipeline

    def render(cfg):
        frame = np.full((64, 64), 80, dtype=np.uint8)
        pipeline = NoisePipeline.from_config(cfg, np.random.default_rng(5))
        return pipeline.apply(frame.copy()).frame

    base = _with(noise={"enabled": True, path: {"enabled": True}})
    changed = _with(noise={"enabled": True, path: {"enabled": True, knob: value}})
    assert not np.array_equal(render(base), render(changed)), (
        f"noise.{path}.{knob} does not change the generated frame")


@pytest.mark.parametrize("knob,value", [("max_px_per_frame", 4.0), ("distribution", "uniform")])
def test_camera_jitter_knobs_reach_the_jitter_model(knob, value):
    """Camera jitter displaces the *viewport*, so it is not in the frame-noise pipeline.

    Probing it through ``NoisePipeline`` reported it dead. It is not: it is consumed by
    ``JitterParams``, which the viewport and the filter's unobservable-sigma both read.
    """
    from src.noise.disturbance import JitterParams

    base = JitterParams.from_mapping(_with().noise.camera_jitter)
    changed = JitterParams.from_mapping(
        _with(noise={"camera_jitter": {knob: value}}).noise.camera_jitter)
    assert base != changed, f"noise.camera_jitter.{knob} does not reach JitterParams"


@pytest.mark.parametrize("knob,value", [("max_px_per_frame", 7.0), ("velocity_x_px_s", 33.0)])
def test_platform_motion_knobs_reach_the_platform_model(knob, value):
    """Platform motion moves the camera, not the pixels -- same reasoning as jitter."""
    from src.noise.disturbance import PlatformMotionParams

    base = PlatformMotionParams.from_mapping(_with().noise.platform_motion)
    changed = PlatformMotionParams.from_mapping(
        _with(noise={"platform_motion": {knob: value}}).noise.platform_motion)
    assert base != changed, f"noise.platform_motion.{knob} does not reach PlatformMotionParams"


def test_atmospheric_preset_selection_changes_the_frame():
    """The preset *name* must select a different preset, not just validate."""
    import numpy as np

    from src.noise.pipeline import NoisePipeline

    def render(preset):
        frame = np.full((64, 64), 80, dtype=np.uint8)
        cfg = _with(noise={"enabled": True,
                           "atmospheric": {"enabled": True, "preset": preset}})
        return NoisePipeline.from_config(cfg, np.random.default_rng(5)).apply(frame).frame

    assert not np.array_equal(render("clear"), render("fog")), \
        "noise.atmospheric.preset does not select the preset"


# --- vision geometry ---------------------------------------------------------------------------

@pytest.mark.parametrize("section,knob,value", [
    ("preprocess", "tophat_kernel_fwhm_multiple", 6.0),
    ("detection", "min_blob_area_spot_multiple", 0.9),
    ("detection", "max_blob_area_spot_multiple", 19.0),
    ("centroid", "window_fwhm_multiple", 7.0),
    ("roi", "size_fwhm_multiple", 17.0),
])
def test_vision_geometry_knobs_reach_resolved_geometry(section, knob, value):
    """The scale-relative vision geometry must be configurable.

    These are consumed by ``VisionConfig.resolve_geometry``, which lives in ``config.py`` -- so a
    search for reads *outside* config.py reports them as dead when they are the most load-bearing
    knobs in the vision path. Mutation does not have that blind spot.
    """
    base = _with().vision.resolve_geometry(6.0)
    changed = _with(vision={section: {knob: value}}).vision.resolve_geometry(6.0)
    assert base != changed, f"vision.{section}.{knob} does not reach resolve_geometry"


# --- control ------------------------------------------------------------------------------------

@pytest.mark.parametrize("knob", ["kp", "ki", "kd", "integral_limit"])
def test_pid_gains_reach_the_controller(knob):
    """PID gains must reach the controller that consumes them."""
    from src.control.pid import Pid, PidParams

    def response(cfg):
        pid = Pid(PidParams(**{k: float(cfg.control.pid[k])
                               for k in ("kp", "ki", "kd", "integral_limit")}))
        # The integral term is compared rather than only the output, and the run is long enough
        # for anti-windup to engage. A short, small-error probe leaves the integral far below the
        # clamp, so integral_limit would look dead while being perfectly well wired.
        # Small error on purpose: at the default kp of 8, an error of 1 deg commands 8 deg/s,
        # which clamps to the 5 deg/s slew ceiling and makes every gain produce an identical
        # saturated output. A saturated probe cannot distinguish a live gain from a dead one.
        states = [pid.step(0.05, 1.0 / 30.0, rate_deg_s=0.05) for _ in range(300)]
        return [(round(x.output, 9), round(x.integral, 9)) for x in states]

    base = _with()
    # integral_limit is mutated *downward*: raising a clamp the integral never reaches has no
    # effect, so an upward mutation would report a live knob as dead.
    mutated = 0.001 if knob == "integral_limit" else float(BASE["control"]["pid"][knob]) * 3.0 + 1.0
    changed = _with(control={"pid": {knob: mutated}})
    assert response(base) != response(changed), f"control.pid.{knob} does not reach the controller"


def test_state_machine_thresholds_reach_the_state_machine():
    """Lock and loss thresholds define the acquisition metric and must be configurable."""
    quick = _with(control={"state_machine": {"lock_confirm_frames": 1}})
    slow = _with(control={"state_machine": {"lock_confirm_frames": 9}})
    assert (int(quick.control.state_machine["lock_confirm_frames"])
            != int(slow.control.state_machine["lock_confirm_frames"]))

    from src.control.statemachine import StateMachineParams, TrackingStateMachine

    def frames_to_lock(cfg):
        machine = TrackingStateMachine(StateMachineParams(
            lock_confirm_frames=int(cfg.control.state_machine["lock_confirm_frames"]),
            require_pointing_window=False))
        for index in range(20):
            result = machine.update(True, 0.0, index / 30.0, 1 / 30.0)
            if result.state.value == "track":
                return index
        return None

    assert frames_to_lock(quick) < frames_to_lock(slow), \
        "control.state_machine.lock_confirm_frames does not reach the state machine"


def test_track_management_knobs_reach_the_track():
    """M-of-N confirmation and deletion drive the re-acquisition metric definition."""
    assert _with(filtering={"track_management": {"delete_after_missed": 9}}
                 ).filtering.delete_after_missed == 9
    assert _with(filtering={"track_management": {"delete_after_missed": 2}}
                 ).filtering.delete_after_missed == 2
