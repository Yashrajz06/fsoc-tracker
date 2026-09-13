"""Phase 1 exit criterion: a clean beacon in every mandatory motion, with exact ground truth.

The headline test here sweeps all four mandatory motions and requires the rendered beacon's
centroid to match its commanded position to well inside the 0.05 px budget -- which is what
makes ground truth usable as the reference for every accuracy claim later.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.camera.model import CameraModel
from src.camera.viewport import Viewport
from src.config import AppConfig
from src.sim.beacon import centroid_of
from src.sim.canvas import Canvas
from src.sim.scene import Scene, SceneState
from src.sim.beacon import BeaconParams
from src.sim.trajectories import MANDATORY_MOTIONS, LinearTrajectory

DT = 1.0 / 30.0


def _config(motion: str, **target_overrides) -> AppConfig:
    """Build a validated configuration with the given motion selected."""
    raw = json.loads(open("config/default.json", encoding="utf-8").read())
    raw["target"]["motion"]["type"] = motion
    raw["target"]["initial_position"] = "center"
    raw["target"].update(target_overrides)
    return AppConfig.from_dict(raw)


def _measure(view, background: int):
    """Recover the beacon centroid from an extracted frame."""
    return centroid_of(view.frame.astype(np.float64) - background)


@pytest.mark.parametrize("motion", MANDATORY_MOTIONS)
def test_ground_truth_is_recoverable_for_every_mandatory_motion(motion: str) -> None:
    """The Phase 1 exit criterion, for each of the four mandatory motions.

    Only frames where the beacon is *fully* inside the viewport are scored. A spot clipped by
    the frame boundary is genuinely truncated, so its measured centroid is legitimately pulled
    inward -- see :func:`test_edge_clipping_biases_the_centroid_inward`, which pins that
    behaviour separately rather than letting it quietly relax this budget.
    """
    config = _config(motion)
    scene = Scene.from_config(config)
    camera = CameraModel.from_config(config)
    viewport = Viewport(camera, jitter_px=0.0)
    margin = 5.0 * config.target.gaussian_sigma_px

    worst = 0.0
    scored = 0
    for index in range(120):
        state = scene.step(0.0 if index == 0 else DT)
        view = viewport.extract(scene.canvas)
        if not camera.is_visible(state.x, state.y, margin_px=margin):
            continue
        scored += 1
        expected = view.canvas_to_frame(state.x, state.y)
        measured = _measure(view, config.scene.background_level)
        worst = max(worst, math.dist(measured, expected))

    if motion == "circular":
        # Radius 400 px exceeds the viewport half-extent of (320, 240), so with a static camera
        # the beacon is never in view. That is the search-limited acquisition case of
        # docs/DESIGN.md section 7.5, not a rendering failure -- assert it explicitly so the
        # zero-frame case can never silently masquerade as a pass.
        assert scored == 0
        return

    assert scored > 30, f"{motion}: too few scorable frames ({scored})"
    assert worst < 0.05, f"{motion}: worst centroid error {worst:.5f} px"


@pytest.mark.parametrize("motion", MANDATORY_MOTIONS)
def test_ground_truth_matches_the_trajectory_exactly(motion: str) -> None:
    """Scene state must report the trajectory position verbatim, with no quantisation.

    Ground truth is sacred: the scene must never round the position it reports to the pixel grid
    it rendered onto.
    """
    config = _config(motion)
    scene = Scene.from_config(config)
    seen_non_integer = False
    for index in range(60):
        state = scene.step(0.0 if index == 0 else DT)
        assert isinstance(state, SceneState)
        if abs(state.x - round(state.x)) > 1e-6:
            seen_non_integer = True
    assert seen_non_integer, "ground truth appears to be pixel-quantised"


def test_circular_motion_is_trackable_once_the_camera_follows() -> None:
    """With the camera pointed at it, the circular beacon is recovered to budget.

    Complements the exit-criterion test above: circular motion is invisible to a *static* camera
    at the default radius, but the rendering itself is exact. Pointing the camera at the target
    isolates the renderer from the acquisition geometry.
    """
    config = _config("circular")
    scene = Scene.from_config(config)
    camera = CameraModel.from_config(config)
    viewport = Viewport(camera, jitter_px=0.0)

    worst = 0.0
    for index in range(90):
        state = scene.step(0.0 if index == 0 else DT)
        camera.set_boresight(state.x, state.y)  # perfect tracking, for test isolation only
        view = viewport.extract(scene.canvas)
        expected = view.canvas_to_frame(state.x, state.y)
        measured = _measure(view, config.scene.background_level)
        worst = max(worst, math.dist(measured, expected))
    assert worst < 0.05


def test_edge_clipping_biases_the_centroid_inward() -> None:
    """A spot truncated by the frame edge measures as pulled inward, by a quantified amount.

    This is real, not a defect: half a spot has its centre of mass inside the surviving half.
    It matters for Phase 3/4 -- a detection near the frame boundary should not be trusted at the
    same precision as one in the middle -- so it is pinned here rather than discovered later.
    """
    config = _config("linear")
    camera = CameraModel.from_config(config)
    viewport = Viewport(camera, jitter_px=0.0)
    canvas = Canvas.from_config(config)
    stationary = LinearTrajectory(x0=camera.viewport_origin[0] + 639.0, y0=1000.0,
                                  velocity_x_px_s=0.0, velocity_y_px_s=0.0)
    scene = Scene(canvas, stationary, BeaconParams())

    state = scene.step(0.0)
    view = viewport.extract(scene.canvas)
    expected = view.canvas_to_frame(state.x, state.y)
    measured = _measure(view, config.scene.background_level)

    # Half the spot is outside the frame, so the measured centroid sits inward of the truth.
    assert measured[0] < expected[0]
    assert 0.5 < expected[0] - measured[0] < 5.0


def test_canvas_buffer_is_reused_across_scene_steps() -> None:
    """Stepping the scene must not reallocate the canvas."""
    config = _config("circular")
    scene = Scene.from_config(config)
    address = scene.canvas.data.__array_interface__["data"][0]
    for index in range(30):
        scene.step(0.0 if index == 0 else DT)
        assert scene.canvas.data.__array_interface__["data"][0] == address


def test_scene_is_reproducible_from_the_configured_seed() -> None:
    """Two scenes from one config must produce identical ground-truth sequences."""
    config = _config("random")
    first = Scene.from_config(config)
    second = Scene.from_config(config)
    for index in range(60):
        dt = 0.0 if index == 0 else DT
        assert first.step(dt).position == second.step(dt).position


def test_reset_replays_the_same_scene() -> None:
    """After reset, an analytic scene must retrace its original path."""
    config = _config("figure8")
    scene = Scene.from_config(config)
    original = [scene.step(0.0 if i == 0 else DT).position for i in range(30)]
    scene.reset()
    replayed = [scene.step(0.0 if i == 0 else DT).position for i in range(30)]
    assert original == pytest.approx(replayed)


def test_intensity_scale_dims_the_beacon_without_moving_it() -> None:
    """Scintillation must change brightness only.

    Phase 2 modulates intensity every frame; if that moved the spot it would appear in the logs
    as tracking error with no cause.
    """
    config = _config("linear")
    scene = Scene.from_config(config)
    bright = scene.step(0.0, intensity_scale=1.0)
    bright_peak = scene.canvas.data.max()
    bright_centroid = centroid_of(
        scene.canvas.extract(int(bright.x) - 20, int(bright.y) - 20, 41, 41).astype(np.float64)
        - config.scene.background_level, int(bright.x) - 20, int(bright.y) - 20)

    scene.reset()
    faint = scene.step(0.0, intensity_scale=0.2)
    faint_peak = scene.canvas.data.max()
    faint_centroid = centroid_of(
        scene.canvas.extract(int(faint.x) - 20, int(faint.y) - 20, 41, 41).astype(np.float64)
        - config.scene.background_level, int(faint.x) - 20, int(faint.y) - 20)

    assert faint_peak < bright_peak
    assert faint_centroid == pytest.approx(bright_centroid, abs=0.05)


def test_scene_reports_visibility_off_canvas() -> None:
    """A beacon driven off the canvas must be reported as not visible."""
    config = _config("linear")
    scene = Scene.from_config(config)
    scene.step(0.0)
    assert scene.state.visible
