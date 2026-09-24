"""Tests for CustomTrajectory — the user-defined drift-plus-oscillation motion type."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from src.sim.trajectories import CustomTrajectory, build_trajectory
from src.config import load_config, merge_overrides


def _make_trajectory(**kwargs) -> CustomTrajectory:
    """Helper: build a CustomTrajectory with sensible defaults, overridable via kwargs."""
    defaults = dict(
        x0=500.0, y0=300.0,
        velocity_x_px_s=80.0, velocity_y_px_s=40.0,
        amplitude_x_px=120.0, amplitude_y_px=200.0,
        frequency_x_hz=0.15, frequency_y_hz=0.08,
    )
    defaults.update(kwargs)
    return CustomTrajectory(**defaults)


def test_initial_position() -> None:
    """At t=0, drift and sinusoidal terms are both zero, so position_at(0) == (x0, y0)."""
    traj = _make_trajectory(x0=500.0, y0=300.0)
    x, y = traj.position_at(0.0)
    assert abs(x - 500.0) < 1e-12
    assert abs(y - 300.0) < 1e-12


def test_deterministic() -> None:
    """Calling position_at(t) twice on the same instance returns bit-identical results."""
    traj = _make_trajectory()
    first = traj.position_at(3.7)
    second = traj.position_at(3.7)
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_is_not_stochastic() -> None:
    """CustomTrajectory is analytic, so is_stochastic must be False."""
    traj = _make_trajectory()
    assert traj.is_stochastic is False


def test_boundary_clamp() -> None:
    """With large amplitudes and tight bounds, every position must stay within the bounds."""
    traj = CustomTrajectory(
        x0=150.0, y0=150.0,
        velocity_x_px_s=80.0, velocity_y_px_s=40.0,
        amplitude_x_px=5000.0, amplitude_y_px=5000.0,
        frequency_x_hz=0.15, frequency_y_hz=0.08,
        boundary="clamp",
        bounds=(100.0, 100.0, 200.0, 200.0),
    )
    for t in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
        x, y = traj.position_at(t)
        assert 100.0 <= x <= 200.0, f"x={x} out of [100, 200] at t={t}"
        assert 100.0 <= y <= 200.0, f"y={y} out of [100, 200] at t={t}"


def test_build_trajectory_custom() -> None:
    """build_trajectory() with motion type 'custom' must return a CustomTrajectory instance."""
    raw = json.load(open("config/default.json"))
    raw2 = merge_overrides(raw, {"target": {"motion": {"type": "custom"}}})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(raw2, f)
        tmppath = f.name
    try:
        config = load_config(tmppath)
        traj = build_trajectory(config)
    finally:
        os.unlink(tmppath)

    assert isinstance(traj, CustomTrajectory)


def test_known_position() -> None:
    """At t = quarter period of frequency_x, sin(2π*fx*t) = 1, giving a known exact position."""
    x0, y0 = 1000.0, 1000.0
    vx, vy = 80.0, 40.0
    amplitude_x, amplitude_y = 120.0, 200.0
    fx, fy = 0.15, 0.08

    t = 1.0 / (4.0 * fx)  # ≈ 1.6667 s, quarter period of the x oscillation

    expected_x = x0 + vx * t + amplitude_x * 1.0  # sin(π/2) = 1
    expected_y = y0 + vy * t + amplitude_y * math.sin(2.0 * math.pi * fy * t)

    traj = CustomTrajectory(
        x0=x0, y0=y0,
        velocity_x_px_s=vx, velocity_y_px_s=vy,
        amplitude_x_px=amplitude_x, amplitude_y_px=amplitude_y,
        frequency_x_hz=fx, frequency_y_hz=fy,
        # No bounds — unconstrained so boundary handling doesn't affect the result
    )

    got_x, got_y = traj.position_at(t)
    assert abs(got_x - expected_x) < 1e-9, f"x mismatch: {got_x} vs {expected_x}"
    assert abs(got_y - expected_y) < 1e-9, f"y mismatch: {got_y} vs {expected_y}"
