"""Tests for configuration loading, validation and derived quantities.

Three things are being protected here, each corresponding to an identified failure mode:

1. **Derived quantities are correct and computed, not stored.** A wrong ``deg_per_pixel``
   silently corrupts every angular metric downstream without raising an error.
2. **Scale-relative vision geometry actually scales.** This is the defence against Benchmark
   Performance-2 running on evaluator video at an unknown resolution with an unknown spot size.
3. **Specification limits are enforced on load.** An out-of-spec scenario file must fail loudly
   at startup rather than produce plausible but invalid results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest

from src.config import (
    AppConfig,
    CameraConfig,
    ConfigError,
    ControlConfig,
    SceneConfig,
    VisionConfig,
    load_config,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.json"


@pytest.fixture()
def raw_config() -> Dict[str, Any]:
    """Return the default configuration document as a mutable dict."""
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def config() -> AppConfig:
    """Return the validated default configuration."""
    return load_config(DEFAULT_CONFIG_PATH)


# ------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------


def test_default_config_loads_and_validates(config: AppConfig) -> None:
    """The shipped default configuration must load and pass every validation rule."""
    assert config.run.mode == "simulation"
    assert config.source_path is not None


def test_missing_file_raises_config_error() -> None:
    """A missing configuration file produces a ConfigError, not an OSError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config("config/does_not_exist.json")


def test_malformed_json_raises_config_error(tmp_path: Path) -> None:
    """Invalid JSON produces a ConfigError naming the file."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(bad)


def test_unknown_key_is_rejected(raw_config: Dict[str, Any]) -> None:
    """An unrecognised key is an error, never silently ignored.

    A silently ignored key is a typo that looks like it took effect, which produces a completed
    run with plausible but wrong numbers -- the most expensive kind of configuration bug.
    """
    raw_config["camera"]["max_pan_speed_deg_sec"] = 5.0  # plausible typo
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        AppConfig.from_dict(raw_config)


def test_comment_keys_are_ignored(raw_config: Dict[str, Any]) -> None:
    """Underscore-prefixed documentation keys must not be treated as fields."""
    raw_config["camera"]["_some_new_note"] = "documentation only"
    AppConfig.from_dict(raw_config)  # must not raise


# ------------------------------------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------------------------------------


def test_deg_per_pixel_matches_hand_calculation(config: AppConfig) -> None:
    """4 deg / 640 px and 3 deg / 480 px both give 0.00625 deg/px."""
    dpp_x, dpp_y = config.camera.deg_per_pixel
    assert dpp_x == pytest.approx(0.00625)
    assert dpp_y == pytest.approx(0.00625)


def test_max_px_per_frame_matches_hand_calculation(config: AppConfig) -> None:
    """5 deg/s at 0.00625 deg/px is 800 px/s, which is 26.67 px/frame at 30 Hz.

    This is the binding feasibility constraint of the system: a target moving faster than this
    between frames cannot be kept centred by any controller.
    """
    pan, tilt = config.camera.max_px_per_frame
    assert pan == pytest.approx(800.0 / 30.0, rel=1e-9)
    assert pan == pytest.approx(26.667, abs=1e-3)
    assert tilt == pytest.approx(26.667, abs=1e-3)


def test_derived_values_track_configuration_changes() -> None:
    """Derived quantities are computed, so changing the FOV changes them.

    This is precisely what a stored literal would fail to do, which is why the previously stored
    ``_derived_deg_per_pixel`` entries were removed from the config file.
    """
    camera = CameraConfig(fov_horizontal_deg=8.0, fov_vertical_deg=6.0)
    dpp_x, dpp_y = camera.deg_per_pixel
    assert dpp_x == pytest.approx(0.0125)
    assert dpp_y == pytest.approx(0.0125)
    # Doubling the FOV coarsens the angular resolution: each pixel now covers twice the angle,
    # so the same 5 deg/s slew rate sweeps HALF as many pixels per second. Widening the FOV
    # therefore makes fast targets harder to keep centred in pixel terms, not easier.
    assert camera.max_px_per_frame[0] == pytest.approx(0.5 * 800.0 / 30.0, rel=1e-9)


def test_boresight_uses_pixel_centre_convention(config: AppConfig) -> None:
    """Boresight must be at ((W-1)/2, (H-1)/2), agreeing with FrameData.center.

    Any disagreement here is a systematic half-pixel bias that would survive every test we would
    otherwise write, and would silently invalidate the centroid-error-vs-SNR curve.
    """
    from src.framesource import FrameData
    import numpy as np

    x, y = config.camera.boresight_px
    assert (x, y) == (319.5, 239.5)

    frame = np.zeros((config.camera.resolution_height, config.camera.resolution_width),
                     dtype=np.uint8)
    data = FrameData(frame=frame, timestamp=0.0, frame_index=0)
    assert data.center == (x, y)


def test_search_arm_spacing_is_fov_derived(config: AppConfig) -> None:
    """Arm spacing uses the limiting (smaller) FOV dimension so coverage has no gaps."""
    assert config.search_arm_spacing_px == pytest.approx(0.9 * 480)


def test_search_arm_spacing_follows_the_limiting_axis() -> None:
    """A tall narrow viewport must spread its arms by the narrower dimension."""
    control = ControlConfig(search={"arm_spacing_fov_fraction": 1.0})
    camera = CameraConfig(resolution_width=200, resolution_height=900)
    assert control.search_arm_spacing_px(camera) == pytest.approx(200.0)


def test_worst_case_search_time_matches_hand_calculation(config: AppConfig) -> None:
    """Path length is area/spacing; time is path divided by scan speed in px/s.

    2000*2000 / 432 = 9259 px of path, at 800 px/s, is 11.6 s -- against a 2 s budget.
    """
    expected_path = (2000.0 * 2000.0) / (0.9 * 480.0)
    expected_time = expected_path / 800.0
    assert config.worst_case_search_time_s == pytest.approx(expected_time, rel=1e-9)
    assert config.worst_case_search_time_s == pytest.approx(11.6, abs=0.1)


def test_search_time_exceeds_acquisition_budget_by_default(config: AppConfig) -> None:
    """The default configuration cannot meet the 2 s budget when the beacon starts off-screen.

    This is arithmetic set by canvas area, FOV and slew ceiling -- not a tuning failure. The test
    asserts it deliberately so that if someone later changes the canvas or FOV such that the
    budget *is* met, they are forced to notice and update the reporting story.
    """
    assert not config.search_is_within_acquisition_budget
    assert config.worst_case_search_time_s > config.telemetry.acquisition_target_s


def test_search_time_never_exceeds_the_physical_slew_limit() -> None:
    """A scan speed above the gimbal's slew ceiling must not shorten the predicted time."""
    camera = CameraConfig(max_pan_speed_deg_s=5.0, max_tilt_speed_deg_s=5.0)
    scene = SceneConfig()
    honest = ControlConfig(search={"arm_spacing_fov_fraction": 0.9, "scan_speed_deg_s": 5.0})
    optimistic = ControlConfig(search={"arm_spacing_fov_fraction": 0.9, "scan_speed_deg_s": 500.0})
    assert optimistic.worst_case_search_time_s(camera, scene) == pytest.approx(
        honest.worst_case_search_time_s(camera, scene)
    )


def test_startup_warning_reports_the_search_budget_overrun(config: AppConfig) -> None:
    """The overrun must be surfaced at startup, not discovered during Phase 4."""
    messages = " ".join(config.startup_warnings())
    assert "EXCEEDS" in messages
    assert "acquisition budget" in messages


def test_provisional_marker_is_lifted(config: AppConfig) -> None:
    """The gains were validated in Phase 4 with the real Kalman filter in the loop.

    The marker may only be absent once ``tests/test_control.py`` contains a passing step-response
    test that includes the filter -- gains validated against an idealised measurement are not
    validated against the system we ship. The mechanism that raises the warning is still tested
    in :func:`test_provisional_marker_warns_when_present`.
    """
    assert not config.control.pid_gains_are_provisional
    assert not any("PROVISIONAL" in m for m in config.startup_warnings())


def test_provisional_marker_warns_when_present(raw_config: Dict[str, Any]) -> None:
    """The warning mechanism itself must keep working, for the next set of untrusted gains."""
    raw_config["control"]["pid"]["_PROVISIONAL"] = "unvalidated"
    config = AppConfig.from_dict(raw_config)
    assert config.control.pid_gains_are_provisional
    assert any("PROVISIONAL" in m for m in config.startup_warnings())


def test_pid_kp_saturates_the_slew_limit_at_the_design_error(config: AppConfig) -> None:
    """Kp must be sized so a ~100 px error commands the full slew rate.

    100 px * 0.00625 deg/px = 0.625 deg of angular error; Kp * 0.625 must reach the 5 deg/s
    ceiling, so Kp = 8. The previous value of 0.8 gave a closed-loop time constant over a second
    and never approached the 26.7 px/frame slew ceiling.
    """
    dpp_x, _ = config.camera.deg_per_pixel
    angular_error_deg = 100.0 * dpp_x
    kp = float(config.control.pid["kp"])
    # Kp was raised from the analytic 8.0 to 12.0 in Phase 4, because measuring the step response
    # *with the Kalman filter in the loop* showed the analytic value settling in 1.07 s against
    # 0.30 s for a perfect measurement. So Kp now saturates the slew limit below 100 px, which is
    # the intended direction: the analytic figure is a floor, not a target.
    assert kp * angular_error_deg >= config.camera.max_pan_speed_deg_s
    saturating_error_px = config.camera.max_pan_speed_deg_s / (kp * dpp_x)
    assert 50.0 <= saturating_error_px <= 100.0


# ------------------------------------------------------------------------------------------
# Scale-relative vision geometry
#
# This is the defence against Benchmark Performance-2, which runs on evaluator video at a
# resolution and spot size we cannot predict. Absolute pixel geometry tuned to a 640x480 frame
# with a 10 px spot is the single most likely cause of failure there.
# ------------------------------------------------------------------------------------------


def test_resolved_geometry_reproduces_previous_absolutes_at_default_scale(config: AppConfig) -> None:
    """At the default spot scale the multiples must reproduce the old hardcoded values.

    The scale-relative parameterisation is a generalisation, not a retune: at the scale it was
    calibrated for it must land on the same numbers, so any behaviour change observed later is
    attributable to the estimator rather than to a silent shift in defaults.
    """
    # The calibration scale is stated explicitly rather than read from the current default
    # shape. The multiples were calibrated against a gaussian spot of sigma 2.5 (FWHM 5.89), and
    # that is what this anchor pins. Reading config.target.nominal_fwhm_px instead silently
    # coupled the anchor to the default shape, so changing the default from gaussian to square
    # (spec parameter 9 says "Default: Square") broke a test that has nothing to do with which
    # shape ships as the default.
    calibration_fwhm = 5.89

    geometry = config.vision.resolve_geometry(calibration_fwhm)
    assert geometry.tophat_kernel_px == 15
    assert geometry.centroid_window_px == 21
    assert geometry.roi_size_px == 65  # 11 * 5.89, versus the previous fixed 64
    assert geometry.min_blob_area_px == pytest.approx(4.0, abs=0.5)
    assert geometry.max_blob_area_px == pytest.approx(900.0, abs=20.0)


def test_default_target_shape_matches_the_specification(config: AppConfig) -> None:
    """Spec parameter 9: "Target Shape: User-defined, Default: Square".

    Square is the shipped default because the specification says so. A 10 px square has a
    half-max width of its full 10 px, against 5.89 px for the gaussian of sigma 2.5 -- so the
    default spot scale differs between them, and the scale-relative geometry follows it. All
    three shapes are implemented and characterised.
    """
    assert config.target.shape == "square"
    assert config.target.nominal_fwhm_px == pytest.approx(10.0, abs=0.01)


def test_resolved_geometry_scales_with_spot_size(config: AppConfig) -> None:
    """A four-times-larger beacon must produce proportionally larger vision geometry.

    This is the property that lets the identical pipeline handle an unseen evaluator video whose
    beacon is 40 px across rather than 10.
    """
    small = config.vision.resolve_geometry(6.0)
    large = config.vision.resolve_geometry(24.0)

    assert large.tophat_kernel_px > small.tophat_kernel_px
    assert large.centroid_window_px > small.centroid_window_px
    assert large.roi_size_px > small.roi_size_px
    # Area gates scale with the square of the linear scale.
    assert large.max_blob_area_px / small.max_blob_area_px == pytest.approx(16.0, rel=1e-6)
    assert large.roi_size_px / small.roi_size_px == pytest.approx(4.0, rel=0.05)


def test_kernel_and_window_sizes_are_odd(config: AppConfig) -> None:
    """Structuring elements and centroid windows must have a defined centre pixel.

    An even window centres on a pixel boundary, reintroducing exactly the half-pixel bias the
    coordinate convention exists to prevent.
    """
    for fwhm in (2.0, 5.89, 9.3, 17.0, 40.0, 55.0):
        geometry = config.vision.resolve_geometry(fwhm)
        assert geometry.tophat_kernel_px % 2 == 1, fwhm
        assert geometry.centroid_window_px % 2 == 1, fwhm


def test_resolved_geometry_is_clamped_at_both_extremes(config: AppConfig) -> None:
    """Absurd spot-scale estimates must be clamped, not propagated into kernel sizes."""
    tiny = config.vision.resolve_geometry(0.001)
    huge = config.vision.resolve_geometry(10_000.0)

    assert tiny.fwhm_px == pytest.approx(config.vision.spot_scale.min_fwhm_px)
    assert huge.fwhm_px == pytest.approx(config.vision.spot_scale.max_fwhm_px)
    assert huge.tophat_kernel_px <= config.vision.preprocess.tophat_kernel_max_px
    assert huge.centroid_window_px <= config.vision.centroid.window_max_px


def test_fallback_geometry_is_used_and_flagged_when_scale_unknown(config: AppConfig) -> None:
    """Passing None must fall back to absolutes and mark the result as such.

    The flag matters: it lets us tell, after the fact, whether a poor benchmark result came from
    the estimator giving up rather than from the tracker itself.
    """
    geometry = config.vision.resolve_geometry(None)
    assert geometry.from_fallback is True
    assert geometry.tophat_kernel_px == config.vision.preprocess.tophat_kernel_fallback_px
    assert geometry.roi_size_px == config.vision.roi.size_fallback_px
    assert geometry.min_blob_area_px == config.vision.detection.min_blob_area_fallback_px


def test_disabling_estimation_forces_fallback_even_with_an_estimate(config: AppConfig) -> None:
    """With estimation disabled the pipeline must use fixed geometry regardless of input."""
    from dataclasses import replace

    vision = replace(config.vision,
                     spot_scale=replace(config.vision.spot_scale, estimation_enabled=False))
    geometry = vision.resolve_geometry(30.0)
    assert geometry.from_fallback is True
    assert geometry.tophat_kernel_px == config.vision.preprocess.tophat_kernel_fallback_px


def test_invalid_spot_scale_estimate_is_rejected(config: AppConfig) -> None:
    """A non-finite or non-positive estimate is a bug upstream and must not be silently clamped."""
    for bad in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="positive finite"):
            config.vision.resolve_geometry(bad)


def test_nominal_spot_area_is_consistent(config: AppConfig) -> None:
    """The reference area the blob gates multiply is (pi/4) * FWHM^2."""
    geometry = config.vision.resolve_geometry(10.0)
    assert geometry.nominal_spot_area_px == pytest.approx(math.pi / 4.0 * 100.0)
    ratio = geometry.max_blob_area_px / geometry.nominal_spot_area_px
    assert ratio == pytest.approx(config.vision.detection.max_blob_area_spot_multiple)


# ------------------------------------------------------------------------------------------
# Specification limit enforcement
#
# An evaluator-supplied scenario file must not be able to put us silently out of spec.
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("noise", "gaussian", "sigma"), 25.0, "parameter 22"),
        (("noise", "salt_pepper", "density"), 0.25, "parameter 21"),
        (("noise", "camera_jitter", "max_px_per_frame"), 30.0, "parameter 23"),
        (("noise", "platform_motion", "max_px_per_frame"), 30.0, "parameter 25"),
        (("camera", "max_pan_speed_deg_s"), 15.0, "parameters 13-14"),
        (("camera", "max_tilt_speed_deg_s"), 1.0, "parameters 13-14"),
        (("camera", "update_rate_hz"), 24.0, "parameter 5"),
        (("target", "size_px"), 40, "parameter 10"),
        (("target", "size_px"), 2, "parameter 10"),
        (("scene", "width"), 800, "parameter 1"),
        (("control", "update_rate_hz"), 10.0, "parameter 15"),
    ],
)
def test_specification_limits_are_enforced(
    raw_config: Dict[str, Any], path: tuple, value: Any, match: str
) -> None:
    """Each specification limit must be rejected with a message citing its spec parameter."""
    node = raw_config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError, match=match):
        AppConfig.from_dict(raw_config)


def test_control_rate_cannot_exceed_frame_rate(raw_config: Dict[str, Any]) -> None:
    """The controller cannot run faster than frames arrive."""
    raw_config["control"]["update_rate_hz"] = 60.0
    raw_config["camera"]["update_rate_hz"] = 30.0
    with pytest.raises(ConfigError, match="cannot run faster than frames arrive"):
        AppConfig.from_dict(raw_config)


def test_hysteresis_windows_must_be_ordered(raw_config: Dict[str, Any]) -> None:
    """Equal or inverted lock/unlock windows remove the hysteresis and cause mode chatter."""
    raw_config["control"]["state_machine"]["unlock_window_px"] = 40
    raw_config["control"]["state_machine"]["lock_window_px"] = 40
    with pytest.raises(ConfigError, match="hysteresis"):
        AppConfig.from_dict(raw_config)


def test_tophat_kernel_smaller_than_the_spot_is_rejected(raw_config: Dict[str, Any]) -> None:
    """A structuring element inside the spot means the opening removes the beacon itself."""
    raw_config["vision"]["preprocess"]["tophat_kernel_fwhm_multiple"] = 0.5
    with pytest.raises(ConfigError, match="removes the beacon"):
        AppConfig.from_dict(raw_config)


def test_search_arm_spacing_above_one_fov_is_rejected(raw_config: Dict[str, Any]) -> None:
    """Arms spaced wider than the FOV leave uncovered strips between them."""
    raw_config["control"]["search"]["arm_spacing_fov_fraction"] = 1.5
    with pytest.raises(ConfigError, match="coverage gaps"):
        AppConfig.from_dict(raw_config)


def test_viewport_must_fit_inside_the_scene(raw_config: Dict[str, Any]) -> None:
    """A viewport larger than the canvas is nonsensical."""
    raw_config["camera"]["resolution_width"] = 4000
    with pytest.raises(ConfigError, match="does not fit inside"):
        AppConfig.from_dict(raw_config)


def test_video_mode_requires_a_path(raw_config: Dict[str, Any]) -> None:
    """Mode B without an input file must fail at load, not at first frame."""
    raw_config["run"]["mode"] = "video"
    with pytest.raises(ConfigError, match="video_input.path"):
        AppConfig.from_dict(raw_config)


def test_unknown_threshold_method_is_rejected(raw_config: Dict[str, Any]) -> None:
    """Only the implemented adaptive operators are selectable."""
    raw_config["vision"]["detection"]["threshold_method"] = "fixed_200"
    with pytest.raises(ConfigError, match="threshold_method"):
        AppConfig.from_dict(raw_config)


def test_otsu_remains_selectable(raw_config: Dict[str, Any]) -> None:
    """Otsu must stay available as a report comparison row, not be removed.

    The default moved to mean+k*sigma because of fill factor in the raw histogram, but Otsu
    applied to the top-hat residual is a reasonable operator and we want the comparison.
    """
    raw_config["vision"]["detection"]["threshold_method"] = "otsu"
    assert AppConfig.from_dict(raw_config).vision.detection.threshold_method == "otsu"


def test_roi_is_sized_from_the_slew_ceiling_not_from_the_configured_floor(
        raw_config: Dict[str, Any]) -> None:
    """A small configured ROI is raised to what the physics needs, rather than warned about.

    This used to warn. The warning was correct but useless: it told the operator the fallback was
    too small for the per-frame boresight motion and asked them to fix it by hand, when every
    quantity needed to compute the right size was already available. The window is now derived
    from ``2 * (max per-frame boresight motion + jitter) + 2 * FWHM``.
    """
    raw_config["vision"]["roi"]["size_fallback_px"] = 8
    raw_config["vision"]["roi"]["size_fwhm_multiple"] = 1.5
    config = AppConfig.from_dict(raw_config)
    size, clamped = config.roi_size_px()
    pan_px, tilt_px = config.camera.max_px_per_frame
    assert not clamped
    assert size > 2 * max(pan_px, tilt_px), (
        f"ROI {size} px cannot contain a single frame of boresight motion "
        f"({max(pan_px, tilt_px):.1f} px)")


def test_roi_warns_only_when_the_configured_ceiling_binds(raw_config: Dict[str, Any]) -> None:
    """The remaining warning is for a real, unfixable-at-runtime risk.

    When ``max_size_*`` caps the window below what the slew ceiling demands, the target genuinely
    can leave the ROI between frames and no derivation can prevent it -- the operator has to
    raise the cap, slow the camera, or reduce jitter. That is worth a warning; the old condition
    was not.
    """
    # The ceiling has to sit below what the slew ceiling demands (~75 px at these defaults)
    # while still respecting size_fwhm_multiple <= max_size_fwhm_multiple.
    raw_config["vision"]["roi"]["size_fwhm_multiple"] = 4.0
    raw_config["vision"]["roi"]["size_fallback_px"] = 24
    raw_config["vision"]["roi"]["max_size_fwhm_multiple"] = 6.0
    raw_config["vision"]["roi"]["max_size_fallback_px"] = 36
    with pytest.warns(RuntimeWarning, match="may leave the ROI"):
        AppConfig.from_dict(raw_config)


# ------------------------------------------------------------------------------------------
# Log header content
# ------------------------------------------------------------------------------------------


def test_metric_definitions_cover_every_ambiguous_metric(config: AppConfig) -> None:
    """Every metric the specification leaves ambiguous must be defined in the log header."""
    definitions = config.metric_definitions()
    for key in ("coordinate_convention", "acquisition_time_s", "reacquisition_time_s",
                "centroid_error_px", "pointing_error_px", "lock_criterion", "rmse_px",
                "loss_rate", "frame_rate_hz", "control_rate_hz", "processing_fps",
                "pipeline_capacity_fps"):
        assert key in definitions
        assert definitions[key].strip()


def test_acquisition_definition_states_the_population_split(config: AppConfig) -> None:
    """The header must say acquisition is reported split, so an evaluator cannot mistake it."""
    text = config.metric_definitions()["acquisition_time_s"]
    assert "in-FOV" in text and "search-limited" in text and "Never pooled" in text


def test_centroid_and_pointing_error_are_defined_separately(config: AppConfig) -> None:
    """The spec's 'tracking error' is ambiguous between these two; both must be defined."""
    definitions = config.metric_definitions()
    assert "ground-truth" in definitions["centroid_error_px"]
    assert "boresight" in definitions["pointing_error_px"]


def test_summary_lines_report_the_derived_quantities(config: AppConfig) -> None:
    """The startup summary must show the numbers a reader would otherwise have to derive."""
    text = "\n".join(config.summary_lines())
    for token in ("deg per pixel", "max travel per frame", "search arm spacing",
                  "worst-case search time", "pixel-centre convention"):
        assert token in text


# ------------------------------------------------------------------------------------------
# Partial override merging (evaluator scenarios)
#
# Benchmark Performance-1 supplies scenarios we have never seen. They must be expressible as a
# small file naming only what differs -- and must never be able to push us out of specification.
# ------------------------------------------------------------------------------------------


@pytest.fixture()
def write_json(tmp_path: Path):
    """Return a helper that writes a dict to a temporary JSON file and returns its path."""

    def _write(name: str, payload: Dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


def test_override_replaces_a_scalar(write_json) -> None:
    """A scenario naming one scalar changes only that scalar."""
    scenario = write_json("s.json", {"run": {"duration_seconds": 12.5}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.run.duration_seconds == 12.5
    assert config.run.random_seed == 42  # untouched


def test_override_merges_nested_blocks_without_dropping_siblings(write_json) -> None:
    """Setting one nested key must not erase its sibling keys.

    A shallow merge would replace the whole ``noise.gaussian`` block, silently discarding
    ``enabled`` and defaulting it -- exactly the kind of bug that produces a plausible-looking
    but wrong benchmark run.
    """
    scenario = write_json("s.json", {"noise": {"gaussian": {"sigma": 18.0}}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.noise.gaussian["sigma"] == 18.0
    assert config.noise.gaussian["enabled"] is True
    # Sibling blocks at the same level survive too.
    assert "salt_pepper" in config.noise.salt_pepper or config.noise.salt_pepper != {}
    assert config.noise.camera_jitter["max_px_per_frame"] == 5.0


def test_override_preserves_unrelated_motion_blocks(write_json) -> None:
    """Switching motion type must not discard the parameter blocks of other motions."""
    scenario = write_json("s.json", {"target": {"motion": {"type": "figure8"}}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.target.motion_type == "figure8"
    assert config.target.motion["circular"]["radius_px"] == 400.0
    assert config.target.motion_params()["amplitude_x_px"] == 500.0


def test_override_replaces_lists_wholesale(write_json) -> None:
    """Lists are replaced, never concatenated.

    Appending to ``noise.pipeline_order`` would run stages twice -- wrong, and near-invisible
    in a log.
    """
    scenario = write_json("s.json", {"noise": {"pipeline_order": ["gaussian"]}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.noise.pipeline_order == ["gaussian"]


def test_overrides_apply_in_order_with_later_winning(write_json) -> None:
    """Multiple overrides stack, and the last one wins on a contested key."""
    first = write_json("a.json", {"run": {"duration_seconds": 10.0, "random_seed": 7}})
    second = write_json("b.json", {"run": {"duration_seconds": 20.0}})
    config = load_config(DEFAULT_CONFIG_PATH, [first, second])
    assert config.run.duration_seconds == 20.0
    assert config.run.random_seed == 7  # from the first, not reverted to the default


def test_override_underscore_keys_are_ignored(write_json) -> None:
    """A scenario may carry its own commentary without it being treated as configuration."""
    scenario = write_json("s.json", {"_author": "evaluator", "run": {"_note": "x",
                                                                     "duration_seconds": 5.0}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.run.duration_seconds == 5.0


def test_override_cannot_bypass_specification_limits(write_json) -> None:
    """Validation runs after merging, so a scenario cannot push us out of spec.

    This is the property that matters most for Benchmark-1: an out-of-spec scenario must fail
    loudly at load rather than produce a completed run whose numbers are quietly invalid.
    """
    scenario = write_json("s.json", {"noise": {"gaussian": {"sigma": 25.0}}})
    with pytest.raises(ConfigError, match="parameter 22"):
        load_config(DEFAULT_CONFIG_PATH, [scenario])


def test_override_with_unknown_key_is_rejected(write_json) -> None:
    """A typo in a scenario file must fail rather than be silently ignored."""
    scenario = write_json("s.json", {"camera": {"max_pan_speed": 6.0}})
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        load_config(DEFAULT_CONFIG_PATH, [scenario])


def test_override_changing_shape_is_rejected(write_json) -> None:
    """Replacing an object with a scalar means the override targets the wrong key."""
    scenario = write_json("s.json", {"noise": {"gaussian": 5.0}})
    with pytest.raises(ConfigError, match="changes the shape"):
        load_config(DEFAULT_CONFIG_PATH, [scenario])


def test_missing_override_file_is_reported(write_json) -> None:
    """A missing scenario file is an error naming the file, not a silent no-op."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(DEFAULT_CONFIG_PATH, ["no_such_scenario.json"])


def test_merge_does_not_mutate_its_inputs() -> None:
    """merge_overrides is pure; callers may reuse the base document."""
    from src.config import merge_overrides

    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"x": 99}}
    merged = merge_overrides(base, override)
    assert merged == {"a": {"x": 99, "y": 2}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}
    assert override == {"a": {"x": 99}}


def test_override_provenance_is_recorded(write_json) -> None:
    """The log header must state which scenario files produced a result.

    Without this a benchmark number cannot be reproduced from its log alone.
    """
    scenario = write_json("s.json", {"run": {"duration_seconds": 5.0}})
    config = load_config(DEFAULT_CONFIG_PATH, [scenario])
    assert config.override_paths == (str(scenario),)
    assert str(scenario) in "\n".join(config.summary_lines())


def test_no_overrides_reports_none(config: AppConfig) -> None:
    """A run with no scenario says so explicitly rather than leaving the field blank."""
    assert config.override_paths == ()
    assert "scenario overrides      : (none)" in "\n".join(config.summary_lines())
