"""Tests for the headless entry point.

These cover the Phase 0 exit criterion: a full run must complete without error while the
simulation, vision, filtering and control modules are still stubs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ConfigError, load_config
from src.framesource import FrameData, FrameSource
from src.main import PlaceholderFrameSource, apply_overrides, build_frame_source, main, parse_args

CONFIG = "config/default.json"


def test_headless_run_completes(capsys: pytest.CaptureFixture[str]) -> None:
    """The Phase 0 exit criterion: a headless run exits zero."""
    assert main(["--config", CONFIG, "--headless", "--duration", "1"]) == 0
    assert "frames processed" in capsys.readouterr().out


def test_check_config_prints_header_without_running(capsys: pytest.CaptureFixture[str]) -> None:
    """``--check-config`` validates and reports without executing a run."""
    assert main(["--config", CONFIG, "--check-config"]) == 0
    out = capsys.readouterr().out
    assert "metric definitions in force" in out
    # Match the run-summary line specifically. A bare "frames processed" substring collides with
    # the processing_fps metric definition, which legitimately contains that phrase.
    assert "frames processed        :" not in out
    assert "loop throughput" not in out


def test_startup_output_surfaces_the_search_budget_overrun(
        capsys: pytest.CaptureFixture[str]) -> None:
    """The search-budget overrun must stay visible at startup.

    The PROVISIONAL gain warning was also checked here until Phase 4 validated the gains against
    a step response with the real filter in the loop. That warning's mechanism is still covered
    in ``tests/test_config.py``.
    """
    main(["--config", CONFIG, "--check-config"])
    out = capsys.readouterr().out
    assert "EXCEEDS" in out
    assert "acquisition budget" in out


def test_missing_config_exits_with_code_two(capsys: pytest.CaptureFixture[str]) -> None:
    """A configuration error is reported on stderr with a distinct exit code."""
    assert main(["--config", "config/nope.json"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_simulation_source_satisfies_the_protocol() -> None:
    """Simulation mode now builds the real source; it must still match the seam structurally.

    The placeholder remains for video mode until Phase 6 supplies ``VideoFrameSource``.
    """
    from src.sim.source import SimulationFrameSource

    source = build_frame_source(load_config(CONFIG))
    assert isinstance(source, FrameSource)
    assert isinstance(source, SimulationFrameSource)


def test_placeholder_source_satisfies_the_protocol() -> None:
    """The stub must structurally match FrameSource, or the seam is not being exercised."""
    source = PlaceholderFrameSource(load_config(CONFIG))
    assert isinstance(source, FrameSource)


def test_placeholder_source_yields_the_configured_frame_count() -> None:
    """Duration times rate frames are produced, then the source is exhausted."""
    config = load_config(CONFIG)
    source = PlaceholderFrameSource(config)
    expected = int(config.run.duration_seconds * config.camera.update_rate_hz)
    frames = list(source)
    assert len(frames) == expected
    assert source.get_frame() is None


def test_placeholder_frames_match_the_configured_geometry() -> None:
    """Frames must be single-channel uint8 at the configured viewport size."""
    config = load_config(CONFIG)
    frame_data = PlaceholderFrameSource(config).get_frame()
    assert frame_data is not None
    assert frame_data.frame.dtype == np.uint8
    assert frame_data.shape == (config.camera.resolution_height,
                                config.camera.resolution_width)
    assert frame_data.center == config.camera.boresight_px


def test_pan_tilt_commands_are_clamped_to_the_slew_limit() -> None:
    """A source must enforce the mechanical limits itself, so a controller cannot exceed them."""
    config = load_config(CONFIG)
    source = PlaceholderFrameSource(config)
    source.apply_pan_tilt(1000.0, -1000.0, dt=1.0)
    assert source.pan_deg == pytest.approx(config.camera.max_pan_speed_deg_s)
    assert source.tilt_deg == pytest.approx(-config.camera.max_tilt_speed_deg_s)


def test_reset_restores_initial_state() -> None:
    """Reset must rewind frames and zero the boresight so scenarios are reproducible."""
    config = load_config(CONFIG)
    source = PlaceholderFrameSource(config)
    source.get_frame()
    source.apply_pan_tilt(1.0, 1.0, dt=1.0)
    source.reset()
    assert source.pan_deg == 0.0
    first = source.get_frame()
    assert first is not None and first.frame_index == 0


def test_headless_override_disables_the_gui() -> None:
    """``--headless`` must actually switch the GUI off in the resulting configuration."""
    config = apply_overrides(load_config(CONFIG), parse_args(["--headless"]))
    assert config.run.headless is True
    assert config.gui.enabled is False


def test_video_override_switches_mode_and_is_revalidated() -> None:
    """``--video`` implies video mode, and the override is re-validated, not trusted."""
    with pytest.raises(ConfigError, match="does not exist"):
        apply_overrides(load_config(CONFIG), parse_args(["--video", "no_such_file.mp4"]))


def test_overrides_do_not_mutate_the_loaded_config() -> None:
    """Overrides rebuild frozen dataclasses rather than mutating shared state."""
    original = load_config(CONFIG)
    apply_overrides(original, parse_args(["--headless", "--duration", "5"]))
    assert original.run.headless is False
    assert original.run.duration_seconds == 60.0


def test_scenario_flag_applies_an_override(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--scenario`` merges a partial override and records it in the header."""
    import json

    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({"target": {"motion": {"type": "figure8"}}}), encoding="utf-8")
    assert main(["--config", CONFIG, "--scenario", str(scenario), "--check-config"]) == 0
    assert str(scenario) in capsys.readouterr().out


def test_scenario_flag_is_repeatable(tmp_path) -> None:
    """Multiple ``--scenario`` files stack in order."""
    import json

    from src.config import load_config as _load

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"run": {"duration_seconds": 10.0}}), encoding="utf-8")
    b.write_text(json.dumps({"run": {"duration_seconds": 20.0}}), encoding="utf-8")
    args = parse_args(["--scenario", str(a), "--scenario", str(b)])
    assert args.scenario == [str(a), str(b)]
    assert _load(CONFIG, args.scenario).run.duration_seconds == 20.0


def test_out_of_spec_scenario_exits_with_code_two(tmp_path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """An out-of-spec evaluator scenario must fail loudly at startup, not run and mislead."""
    import json

    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({"noise": {"gaussian": {"sigma": 25.0}}}), encoding="utf-8")
    assert main(["--config", CONFIG, "--scenario", str(scenario), "--headless"]) == 2
    assert "parameter 22" in capsys.readouterr().err
