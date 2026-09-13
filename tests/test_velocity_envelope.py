"""Trackable-velocity envelope: where the closed loop holds, and where it does not.

Produces ``docs/figures/trackable_velocity.png``. The sweep deliberately continues past the point
of failure -- a curve that only shows the regime where tracking works says nothing about where
the system's limits are, and characterising our own limits is worth more in technical Q&A than a
headline that quietly avoids them.

Two limits are in play and they are not the same:

* **The slew ceiling** is mechanical: 5 deg/s at 0.00625 deg/px is 800 px/s, i.e. 26.7 px/frame
  at 30 Hz. Beyond it the camera physically cannot keep up, whatever the controller does.
* **The loop bandwidth** is the binding limit in practice, and it bites far earlier. It includes
  the Kalman filter's lag, which sits inside the loop.

The FOV tension
---------------
Widening the FOV helps acquisition -- larger arm spacing means fewer spiral turns to cover the
uncertainty region -- but hurts tracking of fast targets *in pixel terms*, because each pixel then
subtends more angle and the same angular slew rate sweeps fewer pixels per second. The two
requirements pull in opposite directions on the same parameter, which is why the sweep reports
both.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.camera.model import CameraModel
from src.config import load_config
from src.control.controller import ControllerParams, PointingController
from src.control.pid import PidParams
from src.control.search import SpiralSearch
from src.filtering.track import Track

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
POINTING_BUDGET_PX = 10.0
FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "trackable_velocity.png"


def _gains(config) -> PidParams:
    """Build PID parameters from the validated configuration gains."""
    pid = config.control.pid
    return PidParams(kp=float(pid["kp"]), ki=float(pid["ki"]), kd=float(pid["kd"]),
                     integral_limit=float(pid["integral_limit"]))


def steady_state_error(config, velocity_px_s: float, fov_h_deg: float = None,
                       seed: int = 0, noise_px: float = 0.3) -> float:
    """Measure steady-state pointing error while a target crosses the canvas.

    Args:
        config: Application configuration.
        velocity_px_s: Target speed in pixels per second.
        fov_h_deg: Override the horizontal FOV, for the FOV-tension sweep.
        seed: RNG seed.
        noise_px: Measurement noise standard deviation.

    Returns:
        Median pointing error over the second half of the traverse, in pixels.
    """
    from dataclasses import replace

    camera_config = config.camera
    if fov_h_deg is not None:
        scale = fov_h_deg / camera_config.fov_horizontal_deg
        camera_config = replace(camera_config, fov_horizontal_deg=fov_h_deg,
                                fov_vertical_deg=camera_config.fov_vertical_deg * scale)

    camera = CameraModel(camera_config, config.scene)
    start_x = 200.0
    camera.set_boresight(start_x, 1000.0)
    controller = PointingController(camera, ControllerParams(pid=_gains(config)))
    track = Track()
    rng = np.random.default_rng(seed)

    errors: List[float] = []
    limit = config.scene.width - 150.0
    for step in range(2000):
        t = step * DT
        target_x = start_x + velocity_px_s * t
        if target_x > limit:
            break
        z = (target_x + rng.normal(0, noise_px), 1000.0 + rng.normal(0, noise_px))
        update = track.update(z, DT, t, snr_aperture=CONSISTENT_SNR)
        if update.has_estimate:
            controller.step(update.position, DT, update.velocity)
        errors.append(math.hypot(camera.boresight[0] - target_x,
                                 camera.boresight[1] - 1000.0))

    if not errors:
        return float("nan")
    tail = np.asarray(errors[len(errors) // 2:])
    return float(np.median(tail))


def max_trackable_velocity(config, budget_px: float = POINTING_BUDGET_PX) -> float:
    """Find the highest target speed holding steady-state error inside the budget.

    Args:
        config: Application configuration.
        budget_px: Pointing-error budget.

    Returns:
        Maximum trackable speed in pixels per second.
    """
    low, high = 10.0, 1200.0
    for _ in range(18):
        mid = 0.5 * (low + high)
        if steady_state_error(config, mid) <= budget_px:
            low = mid
        else:
            high = mid
    return low


@pytest.fixture(scope="module")
def config():
    """The validated default configuration."""
    return load_config("config/default.json")


def test_slew_ceiling_matches_hand_calculation(config) -> None:
    """5 deg/s at 0.00625 deg/px is 800 px/s, i.e. 26.7 px/frame at 30 Hz."""
    pan, _ = config.camera.max_px_per_frame
    assert pan == pytest.approx(800.0 / 30.0, rel=1e-9)
    assert pan * config.camera.update_rate_hz == pytest.approx(800.0)


def test_loop_bandwidth_binds_before_the_slew_ceiling(config) -> None:
    """The practical limit is the loop, not the mechanism -- and by a wide margin.

    This is the honest headline: the mechanical ceiling is 800 px/s, but steady-state pointing
    error crosses the 10 px budget far below it, because the loop bandwidth and the filter lag
    inside it are what actually bind.
    """
    slew_ceiling_px_s = config.camera.max_px_per_frame[0] * config.camera.update_rate_hz
    trackable = max_trackable_velocity(config)
    assert trackable < slew_ceiling_px_s
    assert trackable > 100.0, "envelope collapsed; the loop may be misconfigured"


def test_error_grows_monotonically_with_target_speed(config) -> None:
    """Faster targets must cost more pointing error, with no surprises in between."""
    speeds = [100.0, 200.0, 400.0, 600.0]
    errors = [steady_state_error(config, v) for v in speeds]
    assert all(b > a for a, b in zip(errors, errors[1:])), errors


def test_beyond_the_slew_ceiling_tracking_fails(config) -> None:
    """Past 800 px/s the camera cannot keep up at all, whatever the controller does."""
    assert steady_state_error(config, 1000.0) > 100.0


def test_fov_tension_runs_the_opposite_way_in_a_pixel_defined_canvas(config) -> None:
    """The FOV trade-off, measured -- and it does not go the way the usual argument suggests.

    The standard reasoning is that a wider FOV helps acquisition (more sky per frame) while
    hurting fast tracking in pixel terms (each pixel subtends more angle). The second half holds
    here. **The first half does not**, and the reason is structural rather than incidental:

    Our uncertainty region is a **pixel** canvas and the viewport is a fixed **pixel** count, so
    widening the FOV does not let the camera see more of the canvas -- the viewport is still
    640x480 px. Spiral arm spacing is ``0.9 * min(fov_px)`` = 432 px whatever the FOV, so the
    path length to cover the canvas is unchanged. What does change is the angular scale: a wider
    FOV means more degrees per pixel, so a fixed angular slew rate sweeps *fewer* pixels per
    second. Measured coverage time for the 2000x2000 canvas:

    ===========  =============  ==================
    FOV          deg/px         coverage time
    ===========  =============  ==================
    2 deg        0.00313        5.8 s
    4 deg        0.00625        11.6 s
    8 deg        0.01250        23.1 s
    ===========  =============  ==================

    So in this formulation a wider FOV is worse on *both* counts. The usual argument applies to a
    system whose uncertainty region is angular and whose detector subtends the FOV; ours is
    specified in pixels (spec parameters 1 and 3 fix both canvas and resolution independently of
    parameter 4), which decouples them. Worth stating explicitly in the report, because the
    intuition is otherwise very natural and wrong here.
    """
    def coverage_time(fov_deg: float) -> float:
        width = config.camera.resolution_width
        height = config.camera.resolution_height
        deg_per_px = fov_deg / width
        search = SpiralSearch((1000.0, 1000.0), (width, height),
                              config.camera.max_pan_speed_deg_s / deg_per_px)
        return search.coverage_time_s(config.scene.area_px)

    # Wider FOV searches SLOWER here, not faster.
    assert coverage_time(8.0) > coverage_time(4.0) > coverage_time(2.0)

    # And a wider FOV lowers the pixel-space slew ceiling, so fast targets fail sooner. At
    # FOV 8 deg that ceiling is 5.0 / 0.0125 = 400 px/s. The onset is abrupt rather than gradual,
    # which is itself the evidence that the ceiling is the mechanism: below it the wide FOV is
    # marginally *better* (10.07 px against 10.74 px at 300 px/s), and at it the error jumps by
    # nearly an order of magnitude (110 px against 13.9 px).
    assert steady_state_error(config, 300.0, fov_h_deg=8.0) < 15.0
    # The ratio itself is tuning-dependent -- it measured 7.9x before the track-contest change
    # and 4.2x after -- so the assertion pins the *effect*, not a particular magnitude.
    assert steady_state_error(config, 400.0, fov_h_deg=8.0) > \
        3.0 * steady_state_error(config, 400.0, fov_h_deg=4.0)


def test_pixel_slew_ceiling_scales_inversely_with_fov(config) -> None:
    """The mechanism behind the tracking half of the trade-off."""
    for fov, expected in ((2.0, 1600.0), (4.0, 800.0), (8.0, 400.0)):
        deg_per_px = fov / config.camera.resolution_width
        assert config.camera.max_pan_speed_deg_s / deg_per_px == pytest.approx(expected)


@pytest.mark.slow
def test_trackable_velocity_figure(config) -> None:
    """Produce the envelope figure, including the regime where tracking fails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = np.geomspace(30.0, 1400.0, 26)
    errors = np.array([steady_state_error(config, float(v)) for v in speeds])
    trackable = max_trackable_velocity(config)
    slew_ceiling = config.camera.max_px_per_frame[0] * config.camera.update_rate_hz

    fov_options = (2.0, 4.0, 8.0)
    fov_curves: Dict[float, np.ndarray] = {}
    fov_speeds = np.geomspace(50.0, 900.0, 14)
    for fov in fov_options:
        fov_curves[fov] = np.array(
            [steady_state_error(config, float(v), fov_h_deg=fov) for v in fov_speeds])

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax_fov) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax.plot(speeds, errors, marker="o", ms=4, color="#1f77b4", label="steady-state pointing error")
    ax.axhline(POINTING_BUDGET_PX, color="#d62728", ls="--", lw=1.2,
               label=f"{POINTING_BUDGET_PX:.0f} px spec budget")
    ax.axvline(slew_ceiling, color="0.35", ls=":", lw=1.4,
               label=f"slew ceiling {slew_ceiling:.0f} px/s (26.7 px/frame)")
    ax.axvline(trackable, color="#2ca02c", lw=1.4,
               label=f"max trackable {trackable:.0f} px/s")
    ax.axvspan(trackable, speeds[-1], color="#d62728", alpha=0.07, zorder=0)
    ax.text(trackable * 1.12, 0.6, "CANNOT TRACK\nto budget", fontsize=8.5, color="#8b1a1a")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target speed (px/s)   [26.7 px/frame = 800 px/s at default FOV]")
    ax.set_ylabel("steady-state pointing error (px)")
    ax.set_title("Trackable-velocity envelope\nthe loop bandwidth binds well before the slew ceiling",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")

    ceilings = {2.0: 1600.0, 4.0: 800.0, 8.0: 400.0}
    colours = {2.0: "#1f77b4", 4.0: "#ff7f0e", 8.0: "#2ca02c"}
    coverage = {2.0: 5.8, 4.0: 11.6, 8.0: 23.1}
    for fov, curve in fov_curves.items():
        ax_fov.plot(fov_speeds, curve, marker="s", ms=3.5, color=colours[fov],
                    label=f"FOV {fov:.0f} deg  |  {fov / 640:.5f} deg/px  |  "
                          f"search {coverage[fov]:.1f} s  |  ceiling {ceilings[fov]:.0f} px/s")
        ax_fov.axvline(ceilings[fov], color=colours[fov], ls=":", lw=1.0, alpha=0.7)
    ax_fov.axhline(POINTING_BUDGET_PX, color="#d62728", ls="--", lw=1.2)
    ax_fov.set_xscale("log")
    ax_fov.set_yscale("log")
    ax_fov.set_xlabel("target speed (px/s)")
    ax_fov.set_ylabel("steady-state pointing error (px)")
    ax_fov.set_title(
        "FOV trade-off: in a pixel-defined canvas a wider FOV is worse on BOTH counts\n"
        "slower to search (5.8 / 11.6 / 23.1 s) and a lower pixel slew ceiling",
        fontsize=10)
    ax_fov.grid(True, which="both", alpha=0.25)
    ax_fov.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert FIGURE_PATH.exists()
