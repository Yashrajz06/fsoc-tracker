"""Tests for the virtual camera model and viewport extraction.

The properties under protection are the angular scale (a wrong deg/px silently corrupts every
angular metric), enforcement of the mechanical limits inside the model rather than in the
controller, and the coordinate round-trip between canvas and frame.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.camera.model import CameraModel
from src.camera.viewport import Viewport, ViewportFrame
from src.config import load_config
from src.framesource import FrameData
from src.sim.beacon import BeaconParams, centroid_of, render_beacon
from src.sim.canvas import Canvas

DT = 1.0 / 30.0


@pytest.fixture()
def config():
    """The validated default configuration."""
    return load_config("config/default.json")


@pytest.fixture()
def camera(config) -> CameraModel:
    """A camera at the default configuration, starting at canvas centre."""
    return CameraModel.from_config(config)


@pytest.fixture()
def canvas(config) -> Canvas:
    """A cleared canvas at the default configuration."""
    c = Canvas.from_config(config)
    c.clear()
    return c


# ------------------------------------------------------------------------------------------
# Angular scale
# ------------------------------------------------------------------------------------------


def test_camera_starts_at_the_scene_centre(camera: CameraModel) -> None:
    """Specification parameter 6: the camera begins at the centre of the screen."""
    assert camera.boresight == pytest.approx((999.5, 999.5))
    assert camera.pan_deg == pytest.approx(0.0)
    assert camera.tilt_deg == pytest.approx(0.0)


def test_angular_scale_matches_hand_calculation(camera: CameraModel) -> None:
    """4 deg / 640 px and 3 deg / 480 px both give 0.00625 deg/px."""
    assert camera.deg_per_pixel == pytest.approx((0.00625, 0.00625))


def test_pixel_and_degree_conversions_are_inverse(camera: CameraModel) -> None:
    """Converting pixels to degrees and back must be lossless."""
    for dx, dy in ((100.0, -50.0), (0.0, 0.0), (-333.3, 12.7)):
        pan, tilt = camera.pixels_to_degrees(dx, dy)
        assert camera.degrees_to_pixels(pan, tilt) == pytest.approx((dx, dy))


def test_one_hundred_pixels_is_the_design_angular_error(camera: CameraModel) -> None:
    """100 px must be 0.625 deg -- the error the PID gains were sized against."""
    pan, _ = camera.pixels_to_degrees(100.0, 0.0)
    assert pan == pytest.approx(0.625)


def test_max_travel_per_frame_matches_hand_calculation(camera: CameraModel) -> None:
    """5 deg/s at 0.00625 deg/px is 800 px/s, i.e. 26.7 px/frame at 30 Hz.

    This is the binding feasibility constraint: a target moving faster than this between frames
    cannot be kept centred by any controller.
    """
    pan, tilt = camera.max_px_per_frame
    assert pan == pytest.approx(800.0 / 30.0)
    assert pan == pytest.approx(26.667, abs=1e-3)
    assert tilt == pytest.approx(26.667, abs=1e-3)


# ------------------------------------------------------------------------------------------
# Mechanical limits
# ------------------------------------------------------------------------------------------


def test_rates_are_clamped_to_the_slew_ceiling(camera: CameraModel) -> None:
    """A wild command must be clamped, not executed."""
    assert camera.clamp_rates(1000.0, -1000.0) == pytest.approx((5.0, -5.0))
    assert camera.clamp_rates(1.0, -2.0) == pytest.approx((1.0, -2.0))


def test_limits_are_enforced_inside_the_model_not_the_controller(camera: CameraModel) -> None:
    """A controller bug must degrade tracking, never produce an impossible camera.

    Travel is checked against the acceleration-limited bound, which is tighter than the slew
    ceiling on the first step.
    """
    start = camera.boresight
    camera.apply_rates(1e6, 1e6, DT)
    travelled = math.hypot(camera.boresight[0] - start[0], camera.boresight[1] - start[1])
    max_pan, max_tilt = camera.max_px_per_frame
    assert travelled <= math.hypot(max_pan, max_tilt) + 1e-9


def test_acceleration_limit_produces_a_ramp_not_a_jump(camera: CameraModel) -> None:
    """A step command must ramp up over several frames.

    The first frame is limited by ``max_acceleration_deg_s2 * dt``, not by the slew ceiling, so
    an instantaneous jump to full rate would mean the acceleration limit is not applied.
    """
    first = camera.apply_rates(5.0, 0.0, DT)
    first_rate = camera.rates_deg_s[0]
    assert first_rate == pytest.approx(20.0 * DT)  # acceleration-limited
    assert first_rate < 5.0

    for _ in range(20):
        camera.apply_rates(5.0, 0.0, DT)
    assert camera.rates_deg_s[0] == pytest.approx(5.0)  # eventually reaches the ceiling
    assert camera.boresight[0] > first[0]


def test_executed_rates_are_reported_not_requested_rates(camera: CameraModel) -> None:
    """Logging the executed rate is what makes slew saturation visible in telemetry."""
    camera.apply_rates(1000.0, 0.0, 1.0)
    executed, _ = camera.rates_deg_s
    assert executed <= 5.0


def test_boresight_is_confined_to_the_canvas(camera: CameraModel, config) -> None:
    """The camera must not slew off the world."""
    for _ in range(2000):
        camera.apply_rates(-5.0, -5.0, DT)
    x, y = camera.boresight
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_negative_time_step_is_rejected(camera: CameraModel) -> None:
    """Time must not run backwards."""
    with pytest.raises(ValueError, match="non-negative"):
        camera.apply_rates(1.0, 1.0, -DT)


def test_reset_restores_the_initial_state(camera: CameraModel) -> None:
    """Reset must restore boresight and zero the rates so scenarios are reproducible."""
    for _ in range(30):
        camera.apply_rates(5.0, 5.0, DT)
    camera.reset()
    assert camera.boresight == pytest.approx((999.5, 999.5))
    assert camera.rates_deg_s == pytest.approx((0.0, 0.0))


# ------------------------------------------------------------------------------------------
# Viewport geometry
# ------------------------------------------------------------------------------------------


def test_frame_centre_matches_the_framedata_convention(camera: CameraModel,
                                                       canvas: Canvas) -> None:
    """The viewport boresight must agree exactly with ``FrameData.center``.

    Pointing error is measured against this point; a disagreement here would be a systematic
    half-pixel offset in every pointing-error number we report.
    """
    view = Viewport(camera, jitter_px=0.0).extract(canvas)
    assert view.center == (319.5, 239.5)

    data = FrameData(frame=view.frame, timestamp=0.0, frame_index=0)
    assert data.center == view.center
    assert view.center == camera.config.boresight_px


def test_viewport_is_the_configured_size(camera: CameraModel, canvas: Canvas) -> None:
    """Extraction must produce a 640x480 single-channel uint8 frame."""
    view = Viewport(camera, jitter_px=0.0).extract(canvas)
    assert view.shape == (480, 640)
    assert view.frame.dtype == np.uint8


@pytest.mark.parametrize("position", [(1000.0, 1000.0), (1003.5, 998.5), (1003.37, 998.63)])
def test_canvas_to_frame_round_trip(camera: CameraModel, canvas: Canvas, position) -> None:
    """Canvas and frame coordinates must convert both ways without loss."""
    view = Viewport(camera, jitter_px=0.0).extract(canvas)
    fx, fy = view.canvas_to_frame(*position)
    assert view.frame_to_canvas(fx, fy) == pytest.approx(position)
    assert camera.canvas_to_frame(*position) == pytest.approx((fx, fy))


@pytest.mark.parametrize("position", [1000.0, 1000.5, 1003.37])
def test_target_sub_pixel_position_survives_extraction(camera: CameraModel, canvas: Canvas,
                                                       config, position: float) -> None:
    """A beacon's sub-pixel position must be recoverable from the extracted frame.

    The viewport origin snaps to whole pixels because a detector reads out whole pixels, but the
    target's sub-pixel offset must live on inside the frame rather than being lost to the snap.
    """
    canvas.clear()
    canvas.composite(render_beacon(position, position + 1.25, BeaconParams()))
    view = Viewport(camera, jitter_px=0.0).extract(canvas)

    expected = view.canvas_to_frame(position, position + 1.25)
    measured = centroid_of(view.frame.astype(np.float64) - config.scene.background_level)
    assert measured == pytest.approx(expected, abs=0.05)

    # And back in canvas coordinates, it is the position we rendered.
    assert view.frame_to_canvas(*measured) == pytest.approx((position, position + 1.25),
                                                            abs=0.05)


def test_is_visible_classifies_in_fov_and_outside(camera: CameraModel) -> None:
    """Used to split acquisition into in-FOV and search-limited populations."""
    assert camera.is_visible(*camera.boresight)
    assert camera.is_visible(1000.0, 1000.0)
    assert not camera.is_visible(50.0, 50.0)
    # A point right at the frame edge is visible without margin but not with one.
    edge_x = camera.viewport_origin[0] + 639
    assert camera.is_visible(edge_x, 1000.0)
    assert not camera.is_visible(edge_x, 1000.0, margin_px=5.0)


def test_frame_contains_matches_is_visible(camera: CameraModel, canvas: Canvas) -> None:
    """The frame and the camera must agree on what is in view."""
    view = Viewport(camera, jitter_px=0.0).extract(canvas)
    for point in ((1000.0, 1000.0), (50.0, 50.0), (1300.0, 1200.0)):
        assert view.contains(*point) == camera.is_visible(*point)


def test_viewport_off_canvas_edge_pads_with_background(camera: CameraModel,
                                                       canvas: Canvas, config) -> None:
    """Pointing at a corner must still produce a full frame, padded with background."""
    camera.set_boresight(0.0, 0.0)
    view = Viewport(camera, jitter_px=0.0).extract(canvas)
    assert view.shape == (480, 640)
    assert view.frame.min() == config.scene.background_level


# ------------------------------------------------------------------------------------------
# Camera jitter
# ------------------------------------------------------------------------------------------


def test_jitter_never_exceeds_the_specified_maximum(camera: CameraModel,
                                                    canvas: Canvas) -> None:
    """Specification parameter 23 states a maximum, so the Gaussian tail must be clipped."""
    viewport = Viewport(camera, jitter_px=20.0, rng=np.random.default_rng(0))
    for _ in range(500):
        view = viewport.extract(canvas)
        assert abs(view.jitter_x) <= 20.0
        assert abs(view.jitter_y) <= 20.0


def test_jitter_is_zero_mean(camera: CameraModel, canvas: Canvas) -> None:
    """Jitter must be a zero-mean disturbance, not a drift.

    This is the property that distinguishes it from platform motion: a zero-mean disturbance at
    frame rate is above the loop bandwidth and cannot be rejected by the controller, whereas a
    biased one must be.
    """
    viewport = Viewport(camera, jitter_px=15.0, rng=np.random.default_rng(3))
    offsets = np.array([[v.jitter_x, v.jitter_y]
                        for v in (viewport.extract(canvas) for _ in range(2000))])
    assert abs(offsets[:, 0].mean()) < 1.0
    assert abs(offsets[:, 1].mean()) < 1.0
    assert offsets[:, 0].std() > 1.0  # it is actually moving


def test_jitter_can_be_disabled_for_deterministic_extraction(camera: CameraModel,
                                                             canvas: Canvas) -> None:
    """Ground-truth reference frames must be reproducible."""
    viewport = Viewport(camera, jitter_px=20.0, rng=np.random.default_rng(0))
    first = viewport.extract(canvas, apply_jitter=False)
    second = viewport.extract(canvas, apply_jitter=False)
    assert (first.x0, first.y0) == (second.x0, second.y0)
    assert first.jitter_x == 0.0 and first.jitter_y == 0.0


def test_jitter_shifts_the_extraction_window_not_the_scene(camera: CameraModel,
                                                           canvas: Canvas) -> None:
    """Jitter must move where we look, leaving the world untouched.

    That is what it physically is, and it means the target's canvas position -- the ground truth
    -- is unaffected by jitter, while its frame-local position is not.
    """
    canvas.clear()
    canvas.composite(render_beacon(1000.0, 1000.0, BeaconParams()))
    before = canvas.data.copy()

    viewport = Viewport(camera, jitter_px=15.0, rng=np.random.default_rng(1))
    origins = {(v.x0, v.y0) for v in (viewport.extract(canvas) for _ in range(50))}

    assert np.array_equal(canvas.data, before)  # scene untouched
    assert len(origins) > 1  # window did move


def test_invalid_jitter_settings_are_rejected(camera: CameraModel) -> None:
    """Bad jitter configuration must fail loudly."""
    with pytest.raises(ValueError, match="non-negative"):
        Viewport(camera, jitter_px=-1.0)
    with pytest.raises(ValueError, match="jitter distribution"):
        Viewport(camera, jitter_px=1.0, jitter_distribution="cauchy")


def test_uniform_jitter_respects_its_bound(camera: CameraModel, canvas: Canvas) -> None:
    """The uniform distribution must also stay within the stated maximum."""
    viewport = Viewport(camera, jitter_px=10.0, rng=np.random.default_rng(0),
                        jitter_distribution="uniform")
    offsets = [viewport.extract(canvas).jitter_x for _ in range(500)]
    assert max(abs(o) for o in offsets) <= 10.0
    assert max(abs(o) for o in offsets) > 8.0  # actually spanning the range
