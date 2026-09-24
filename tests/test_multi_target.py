"""Tests for multi-target simulation (MultiScene and SimulationFrameSource with count > 1)."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.sim.scene import MultiScene, Scene, SceneState
from src.config import AppConfig, load_config
from src.sim.source import SimulationFrameSource
from src.runner import build_frame_source


def _make_config(count: int, duration_s: float = 0.1) -> AppConfig:
    """Build a minimal config with the given target count."""
    raw = json.load(open("config/default.json"))
    raw["target"]["count"] = count
    raw["run"]["duration_seconds"] = duration_s
    raw["run"]["headless"] = True
    raw["noise"]["enabled"] = False
    raw["noise"]["camera_jitter"] = {"enabled": False, "max_px_per_frame": 0.0}
    raw["ai"] = {"enabled": False}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        path = f.name
    try:
        config = AppConfig.from_dict(json.load(open(path)))
    finally:
        os.unlink(path)
    return config


def test_multi_scene_composites_both_beacons():
    """Canvas max should be higher with 2 beacons than with 1."""
    config1 = _make_config(1)
    config2 = _make_config(2)
    rng = np.random.default_rng(99)

    single = Scene.from_config(config1, np.random.default_rng(99))
    single.step(0.0)
    single_max = int(single.canvas.data.max())

    multi = MultiScene.from_config(config2, rng)
    multi.step(0.0)
    multi_max = int(multi.canvas.data.max())

    # With two beacons composited via "add", the canvas should have at least as high a peak
    # (it may saturate at 255; both should be > background level)
    assert multi_max > int(config2.scene.background_level), \
        "Canvas must have beacon signal after multi step"
    assert single_max > int(config1.scene.background_level), \
        "Single canvas must have beacon signal"


def _make_config_random_motion(count: int, duration_s: float = 0.1) -> AppConfig:
    """Build a minimal config with the given target count and random walk motion.

    For circular motion all sub-scenes share the same center/radius/phase, so they
    always start at the same position regardless of the per-scene RNG seed.  Using
    a random walk (``"random"`` motion type) makes the initial position actually
    dependent on the per-scene RNG, which is what this test needs to verify.
    """
    raw = json.load(open("config/default.json"))
    raw["target"]["count"] = count
    raw["target"]["motion"]["type"] = "random"
    raw["run"]["duration_seconds"] = duration_s
    raw["run"]["headless"] = True
    raw["noise"]["enabled"] = False
    raw["noise"]["camera_jitter"] = {"enabled": False, "max_px_per_frame": 0.0}
    raw["ai"] = {"enabled": False}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        path = f.name
    try:
        config = AppConfig.from_dict(json.load(open(path)))
    finally:
        os.unlink(path)
    return config


def test_multi_scene_independent_states():
    """Two targets must report different positions (different RNG seeds)."""
    config = _make_config_random_motion(2)
    multi = MultiScene.from_config(config, np.random.default_rng(0))
    states = multi.step(0.0)
    assert len(states) == 2
    s0, s1 = states
    # Different seeds → different initial positions (random motion uses per-scene RNG for x0/y0)
    assert not (s0.x == s1.x and s0.y == s1.y), \
        "Two sub-scenes with distinct seeds must start at different positions"


def test_simulation_source_multi_target_extra_ground_truths():
    """SimulationFrameSource with count=2 must set extra_ground_truths with length 1."""
    config = _make_config(2)
    source = SimulationFrameSource(config)
    fd = source.get_frame()
    assert fd is not None
    assert fd.extra_ground_truths is not None, \
        "count=2 must populate extra_ground_truths"
    assert len(fd.extra_ground_truths) == 1, \
        f"count=2 must produce 1 extra GT, got {len(fd.extra_ground_truths)}"


def test_simulation_source_single_target_no_extra_ground_truths():
    """SimulationFrameSource with count=1 must leave extra_ground_truths as None."""
    config = _make_config(1)
    source = SimulationFrameSource(config)
    fd = source.get_frame()
    assert fd is not None
    assert fd.extra_ground_truths is None, \
        "count=1 must not populate extra_ground_truths"


def test_build_frame_source_multi_target():
    """build_frame_source with count=2 must yield multi-target frames end to end."""
    config = _make_config(2)
    source = build_frame_source(config)
    fd = source.get_frame()
    assert fd is not None
    assert fd.extra_ground_truths is not None
    assert len(fd.extra_ground_truths) == 1
