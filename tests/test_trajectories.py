"""Tests for target motion models.

Covers the four mandatory motions (spec parameter 12), the optional extras, boundary handling,
and the reproducibility property that makes benchmark runs repeatable for the report.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.config import AppConfig, ConfigError
from src.sim.trajectories import (
    MANDATORY_MOTIONS,
    CircularTrajectory,
    Figure8Trajectory,
    LinearTrajectory,
    OrnsteinUhlenbeckTrajectory,
    RandomWalkTrajectory,
    SpiralTrajectory,
    apply_boundary,
    build_trajectory,
)

DT = 1.0 / 30.0


def _config(motion_type: str, **scene_overrides) -> AppConfig:
    """Build a validated configuration with the given motion type selected."""
    raw = json.loads(open("config/default.json", encoding="utf-8").read())
    raw["target"]["motion"]["type"] = motion_type
    raw["target"]["initial_position"] = "center"
    raw.update(scene_overrides)
    return AppConfig.from_dict(raw)


# ------------------------------------------------------------------------------------------
# Boundary handling
# ------------------------------------------------------------------------------------------


def test_clamp_pins_at_the_boundary() -> None:
    """Clamping saturates at the bound."""
    assert apply_boundary(150.0, 0.0, 100.0, "clamp") == 100.0
    assert apply_boundary(-50.0, 0.0, 100.0, "clamp") == 0.0
    assert apply_boundary(50.0, 0.0, 100.0, "clamp") == 50.0


def test_wrap_is_modular() -> None:
    """Wrapping is modulo the span."""
    assert apply_boundary(150.0, 0.0, 100.0, "wrap") == pytest.approx(50.0)
    assert apply_boundary(-10.0, 0.0, 100.0, "wrap") == pytest.approx(90.0)


def test_bounce_reflects_like_an_elastic_collision() -> None:
    """Folding through a triangle wave reproduces elastic reflection exactly."""
    assert apply_boundary(110.0, 0.0, 100.0, "bounce") == pytest.approx(90.0)
    assert apply_boundary(-10.0, 0.0, 100.0, "bounce") == pytest.approx(10.0)
    # A full double-span excursion returns to the start.
    assert apply_boundary(200.0, 0.0, 100.0, "bounce") == pytest.approx(0.0)
    assert apply_boundary(250.0, 0.0, 100.0, "bounce") == pytest.approx(50.0)


def test_bounce_stays_in_range_for_large_excursions() -> None:
    """No excursion, however large, may escape the bounds."""
    for value in np.linspace(-1000.0, 1000.0, 501):
        result = apply_boundary(float(value), 0.0, 100.0, "bounce")
        assert 0.0 <= result <= 100.0


def test_unknown_boundary_behaviour_is_rejected() -> None:
    """An unknown policy must fail rather than pass the coordinate through untouched."""
    with pytest.raises(ValueError, match="Unknown boundary behaviour"):
        apply_boundary(1.0, 0.0, 10.0, "teleport")


# ------------------------------------------------------------------------------------------
# Analytic motions
# ------------------------------------------------------------------------------------------


def test_linear_motion_matches_its_closed_form() -> None:
    """``x = x0 + vx*t`` exactly."""
    traj = LinearTrajectory(x0=100.0, y0=200.0, velocity_x_px_s=120.0, velocity_y_px_s=-60.0)
    assert traj.position_at(0.0) == pytest.approx((100.0, 200.0))
    assert traj.position_at(2.0) == pytest.approx((340.0, 80.0))
    assert traj.speed_px_s == pytest.approx(math.hypot(120.0, 60.0))


def test_circular_motion_holds_its_radius() -> None:
    """Every sampled point must lie on the circle."""
    traj = CircularTrajectory(center_x=1000.0, center_y=1000.0, radius_px=400.0,
                              angular_velocity_rad_s=0.3)
    for t in np.linspace(0.0, 40.0, 200):
        x, y = traj.position_at(float(t))
        assert math.hypot(x - 1000.0, y - 1000.0) == pytest.approx(400.0)
    assert traj.tangential_speed_px_s == pytest.approx(120.0)


def test_circular_motion_is_periodic() -> None:
    """One full period must return to the starting point."""
    omega = 0.3
    traj = CircularTrajectory(1000.0, 1000.0, 400.0, omega)
    period = 2.0 * math.pi / omega
    assert traj.position_at(period) == pytest.approx(traj.position_at(0.0), abs=1e-6)


def test_figure8_crosses_itself_at_the_centre() -> None:
    """A Lissajous 1:2 path passes through its own centre twice per period."""
    traj = Figure8Trajectory(amplitude_x_px=500.0, amplitude_y_px=300.0,
                             angular_velocity_rad_s=0.25, center_x=1000.0, center_y=1000.0)
    # At w*t = 0 and w*t = pi the path is at the centre.
    assert traj.position_at(0.0) == pytest.approx((1000.0, 1000.0), abs=1e-9)
    assert traj.position_at(math.pi / 0.25) == pytest.approx((1000.0, 1000.0), abs=1e-6)


def test_figure8_has_the_one_to_two_frequency_ratio() -> None:
    """y must complete two cycles for every one of x -- that is what makes it an 8, not an ellipse.

    Asserted as a closed-form identity over a half period rather than by counting zero
    crossings: a crossing counter on a half-open window silently drops the crossing that falls
    at the wrap point, which makes the expected count depend on sampling phase rather than on
    the property being tested.

    With ``x = A*sin(w*t)`` and ``y = B*sin(2*w*t)``, advancing by half a period ``pi/w`` gives
    ``y`` a full extra cycle (unchanged) while ``x`` advances half a cycle (negated).
    """
    omega = 0.25
    traj = Figure8Trajectory(500.0, 300.0, omega, center_x=0.0, center_y=0.0)
    half_period = math.pi / omega

    for t in np.linspace(0.0, half_period, 97):
        x_now, y_now = traj.position_at(float(t))
        x_later, y_later = traj.position_at(float(t) + half_period)
        assert y_later == pytest.approx(y_now, abs=1e-6)
        assert x_later == pytest.approx(-x_now, abs=1e-6)


def test_figure8_is_not_an_ellipse() -> None:
    """The path must self-intersect, which an ellipse never does.

    A 1:1 frequency ratio would trace an ellipse and pass a naive amplitude check, so this pins
    the distinguishing feature: the centre is visited twice per period from opposite directions.
    """
    omega = 0.25
    traj = Figure8Trajectory(500.0, 300.0, omega, center_x=0.0, center_y=0.0)
    period = 2.0 * math.pi / omega

    at_start = traj.position_at(0.0)
    at_half = traj.position_at(period / 2.0)
    assert at_start == pytest.approx((0.0, 0.0), abs=1e-6)
    assert at_half == pytest.approx((0.0, 0.0), abs=1e-6)

    # Same point, opposite horizontal direction: that crossing is the waist of the 8.
    delta = 1e-4
    vx_start = traj.position_at(delta)[0] - at_start[0]
    vx_half = traj.position_at(period / 2.0 + delta)[0] - at_half[0]
    assert vx_start * vx_half < 0


def test_spiral_arm_spacing_matches_two_pi_b() -> None:
    """Arm spacing is ``2*pi*b``, the relation the acquisition search depends on."""
    traj = SpiralTrajectory(1000.0, 1000.0, growth_rate_b=12.0, angular_velocity_rad_s=0.4)
    assert traj.arm_spacing_px == pytest.approx(2.0 * math.pi * 12.0)
    # Radius grows linearly with angle.
    r1 = math.hypot(*np.subtract(traj.position_at(10.0), (1000.0, 1000.0)))
    r2 = math.hypot(*np.subtract(traj.position_at(20.0), (1000.0, 1000.0)))
    assert r2 == pytest.approx(2.0 * r1)


def test_analytic_motion_is_independent_of_step_size() -> None:
    """Stepping in small or large increments must reach the same place.

    This is the property that makes analytic trajectories reproducible regardless of how the
    simulation happens to be clocked.
    """
    fine = CircularTrajectory(1000.0, 1000.0, 400.0, 0.3)
    coarse = CircularTrajectory(1000.0, 1000.0, 400.0, 0.3)
    for _ in range(60):
        fine.step(1.0 / 60.0)
    for _ in range(30):
        coarse.step(1.0 / 30.0)
    assert fine.position_at(fine.elapsed) == pytest.approx(coarse.position_at(coarse.elapsed))


def test_analytic_trajectories_expose_a_closed_form() -> None:
    """Analytic motions must not be marked stochastic."""
    traj = LinearTrajectory(0.0, 0.0, 1.0, 1.0)
    assert traj.is_stochastic is False
    assert traj.position_at(1.0) == pytest.approx((1.0, 1.0))


def test_negative_time_step_is_rejected() -> None:
    """Time must not run backwards."""
    with pytest.raises(ValueError, match="non-negative"):
        LinearTrajectory(0.0, 0.0, 1.0, 1.0).step(-0.1)


# ------------------------------------------------------------------------------------------
# Stochastic motions
# ------------------------------------------------------------------------------------------


def test_random_walk_is_reproducible_from_a_seed() -> None:
    """Identical seeds must produce identical paths, or benchmark runs are not repeatable."""
    def walk(seed: int):
        traj = RandomWalkTrajectory(1000.0, 1000.0, step_sigma_px=8.0,
                                    rng=np.random.default_rng(seed))
        return [traj.step(DT) for _ in range(50)]

    assert walk(42) == walk(42)
    assert walk(42) != walk(43)


def test_random_walk_respects_its_speed_cap() -> None:
    """Per-step displacement must never exceed the configured maximum speed.

    An uncapped Gaussian walk occasionally emits a step far beyond the camera's slew ceiling,
    which would appear in the logs as a tracker failure when it is really an impossible target.
    """
    traj = RandomWalkTrajectory(1000.0, 1000.0, step_sigma_px=50.0, max_speed_px_s=200.0,
                                rng=np.random.default_rng(0))
    previous = (1000.0, 1000.0)
    for _ in range(500):
        current = traj.step(DT)
        step = math.hypot(current[0] - previous[0], current[1] - previous[1])
        assert step <= 200.0 * DT + 1e-9
        previous = current


def test_stochastic_motion_has_no_closed_form() -> None:
    """Asking a path-dependent motion for a closed form must fail, not return something wrong."""
    traj = RandomWalkTrajectory(0.0, 0.0, 1.0, rng=np.random.default_rng(0))
    assert traj.is_stochastic is True
    with pytest.raises(NotImplementedError, match="stochastic"):
        traj.position_at(1.0)


def test_ornstein_uhlenbeck_reverts_to_its_mean() -> None:
    """Started far from the mean, the process must be pulled back towards it.

    Mean reversion is why this is preferred over a plain random walk: an unbounded walk drifts
    off-canvas and spends the run pinned against a boundary, which tests nothing useful.
    """
    traj = OrnsteinUhlenbeckTrajectory(x0=1800.0, y0=1800.0, theta=1.5, mu_x=1000.0,
                                       mu_y=1000.0, sigma=5.0,
                                       rng=np.random.default_rng(1))
    positions = [traj.step(DT) for _ in range(600)]
    late = np.array(positions[-100:])
    assert abs(late[:, 0].mean() - 1000.0) < 100.0
    assert abs(late[:, 1].mean() - 1000.0) < 100.0


def test_reset_returns_a_stochastic_trajectory_to_its_start() -> None:
    """Reset must restore the initial position so a scenario can be re-run."""
    traj = RandomWalkTrajectory(1000.0, 1000.0, 8.0, rng=np.random.default_rng(0))
    for _ in range(20):
        traj.step(DT)
    traj.reset()
    assert traj.elapsed == 0.0
    assert traj.step(0.0) == pytest.approx((1000.0, 1000.0), abs=20.0)


# ------------------------------------------------------------------------------------------
# Construction from configuration
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("motion", MANDATORY_MOTIONS)
def test_every_mandatory_motion_builds_and_stays_on_canvas(motion: str) -> None:
    """Specification parameter 12 requires all four; each must run bounded for 30 seconds."""
    config = _config(motion)
    traj = build_trajectory(config)
    for _ in range(900):
        x, y = traj.step(DT)
        assert 0.0 <= x <= config.scene.width - 1
        assert 0.0 <= y <= config.scene.height - 1


@pytest.mark.parametrize("motion", ["spiral", "sinusoidal", "ornstein_uhlenbeck"])
def test_optional_motions_build_and_stay_on_canvas(motion: str) -> None:
    """Optional motions must be equally well behaved."""
    config = _config(motion)
    traj = build_trajectory(config)
    for _ in range(900):
        x, y = traj.step(DT)
        assert 0.0 <= x <= config.scene.width - 1
        assert 0.0 <= y <= config.scene.height - 1


def test_build_is_reproducible_from_the_configured_seed() -> None:
    """Two builds from the same config must produce identical paths."""
    config = _config("random")
    a = [build_trajectory(config).step(DT) for _ in range(1)]
    first = build_trajectory(config)
    second = build_trajectory(config)
    assert [first.step(DT) for _ in range(50)] == [second.step(DT) for _ in range(50)]
    assert a  # the single-step build also succeeded


def test_unknown_motion_type_is_rejected() -> None:
    """A motion type the registry does not implement must fail clearly."""
    config = _config("circular")
    object.__setattr__(config.target, "motion", {"type": "brownian_bridge"})
    with pytest.raises(ConfigError, match="Unknown motion type"):
        build_trajectory(config)


def test_mismatched_motion_parameters_are_reported_clearly() -> None:
    """A motion block missing a required parameter must name the block and the class."""
    config = _config("circular")
    object.__setattr__(config.target, "motion",
                       {"type": "circular", "circular": {"radius_px": 10.0}})
    with pytest.raises(ConfigError, match="target.motion.circular"):
        build_trajectory(config)
