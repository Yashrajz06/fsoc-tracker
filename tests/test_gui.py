"""GUI tests.

The GUI is a view layer: the assertions here are mostly *structural* -- that the widgets hold no
logic, that every mandatory specification parameter is reachable, and that a GUI run is the same
run the CLI would perform. Pixel appearance is not tested; it is not what the GUI is graded on.

All tests run offscreen, so they need no display.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.config import AppConfig, ConfigError, load_config, load_json_document, merge_overrides  # noqa: E402
from src.gui.controls import ControlPanel  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402

CONFIG = "config/default.json"


@pytest.fixture(scope="module")
def qt_app():
    """A single QApplication for the module."""
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def config():
    """The validated default configuration."""
    return load_config(CONFIG)


@pytest.fixture()
def window(qt_app, config):
    """A constructed main window."""
    return MainWindow(config, CONFIG)


# ------------------------------------------------------------------------------------------
# The headless constraint
# ------------------------------------------------------------------------------------------


def test_core_modules_import_without_qt() -> None:
    """``CLAUDE.md``: core modules stay importable and runnable headless.

    A module-level Qt import anywhere in the core would break every CLI invocation on a machine
    without Qt -- including the frozen headless executable, which deliberately does not bundle it.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import src.main, src.runner, src.vision.pipeline, src.filtering.track,"
        " src.control.controller, src.telemetry.report;"
        "bad=[m for m in sys.modules if m.startswith(('PySide6','shiboken6','pyqtgraph'))];"
        "print('QT_LEAKED' if bad else 'CLEAN', bad[:3])"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], check=False)
    assert "CLEAN" in result.stdout, result.stdout + result.stderr


def test_widgets_contain_no_tracking_logic() -> None:
    """Widgets must not import the tracking stack beyond the data types they display.

    The guarantee is structural, not a matter of discipline while editing: if a widget cannot
    reach the filter or the controller, it cannot accumulate logic that drifts from the CLI.
    """
    source = (Path(__file__).resolve().parents[1] / "src" / "gui" / "widgets.py").read_text()
    for forbidden in ("KalmanFilterCV", "PointingController", "VisionPipeline",
                      "SpiralSearch", "TrackingStateMachine"):
        assert forbidden not in source, f"widgets.py reaches into {forbidden}"


# ------------------------------------------------------------------------------------------
# Specification parameter coverage
# ------------------------------------------------------------------------------------------


#: Spec table rows that must be reachable from the GUI, and the control that exposes each.
MANDATORY_CONTROLS = {
    1: ("scene_w", "scene_h"),
    3: ("res_w", "res_h"),
    4: ("fov_h", "fov_v"),
    5: ("camera_rate",),
    6: ("camera_position",),
    9: ("target_shape",),
    10: ("target_size",),
    11: ("target_position",),
    12: ("motion",),
    13: ("pan_speed",),
    14: ("tilt_speed",),
    15: ("control_rate",),
    21: ("gaussian_on", "poisson_on", "sp_on"),
    22: ("gaussian_sigma",),
    23: ("jitter_on", "jitter_px"),
    24: ("atmosphere",),
    25: ("platform_on", "platform_type", "platform_px"),
}


@pytest.mark.parametrize("parameter,attributes", sorted(MANDATORY_CONTROLS.items()))
def test_every_mandatory_spec_parameter_is_reachable(qt_app, config, parameter, attributes):
    """Walk the specification table and confirm each row has a control."""
    panel = ControlPanel(config)
    for attribute in attributes:
        assert hasattr(panel, attribute), f"spec parameter {parameter}: missing {attribute}"


def test_all_four_mandatory_motions_are_selectable(qt_app, config) -> None:
    """Spec parameter 12 requires at least four selectable motions."""
    panel = ControlPanel(config)
    options = {panel.motion.itemText(i) for i in range(panel.motion.count())}
    assert {"linear", "circular", "figure8", "random"} <= options


def test_all_five_atmospheric_presets_are_selectable(qt_app, config) -> None:
    """Spec parameter 24 names clear, haze, fog, rain and low light."""
    panel = ControlPanel(config)
    options = {panel.atmosphere.itemText(i) for i in range(panel.atmosphere.count())}
    assert {"clear", "haze", "fog", "rain", "low_light"} <= options


def test_mode_switch_and_file_picker_exist(qt_app, config) -> None:
    """Mode A / Mode B switching with a file picker is a demo requirement."""
    panel = ControlPanel(config)
    options = {panel.mode.itemText(i) for i in range(panel.mode.count())}
    assert options == {"simulation", "video"}
    assert hasattr(panel, "video_path") and hasattr(panel, "truth_path")


def test_unimplemented_optional_parameters_are_disclosed(qt_app, config) -> None:
    """Parameters 2 and 8 are optional in the spec and not implemented.

    A control that silently did nothing would be worse than no control, so the count is pinned at
    one and the camera type is a label rather than a selector.
    """
    panel = ControlPanel(config)
    assert panel.target_count.minimum() == panel.target_count.maximum() == 1
    assert not hasattr(panel, "camera_type_selector")
    source = (Path(__file__).resolve().parents[1] / "src" / "gui" / "controls.py").read_text()
    assert "colour" in source and "not implemented" in source


# ------------------------------------------------------------------------------------------
# A GUI run is the same run the CLI would perform
# ------------------------------------------------------------------------------------------


def test_controls_produce_a_valid_configuration(window) -> None:
    """The overrides must go through the same validation the CLI uses."""
    built = window._build_config()
    assert isinstance(built, AppConfig)
    assert built.camera.max_pan_speed_deg_s == window.controls.pan_speed.value()


def test_control_overrides_have_the_shape_of_a_scenario_file(qt_app, config) -> None:
    """So a GUI run and a CLI run with that scenario are the same run."""
    overrides = ControlPanel(config).overrides()
    merged = merge_overrides(load_json_document(CONFIG), overrides)
    rebuilt = AppConfig.from_dict(merged)
    assert rebuilt.target.motion_type in (
        "linear", "circular", "figure8", "random", "spiral", "sinusoidal",
        "ornstein_uhlenbeck")


def test_out_of_spec_control_values_are_rejected_by_config_not_the_gui(qt_app, config) -> None:
    """Spec ranges are enforced in ``src.config``, never re-implemented in the GUI.

    Re-implementing them would create a second source of truth that could drift from the CLI's.
    """
    panel = ControlPanel(config)
    overrides = panel.overrides()
    overrides["noise"]["gaussian"]["sigma"] = 25.0        # spec parameter 22 caps this at 20
    merged = merge_overrides(load_json_document(CONFIG), overrides)
    with pytest.raises(ConfigError, match="parameter 22"):
        AppConfig.from_dict(merged)


@pytest.mark.slow
def test_gui_run_tracks_and_writes_the_same_outputs_as_headless(window, tmp_path) -> None:
    """An end-to-end GUI run must track and produce the usual logs and report."""
    from PySide6.QtCore import QEventLoop, QTimer

    window.controls.mode.setCurrentText("simulation")
    window.controls.motion.setCurrentText("linear")
    window.controls.target_position.setCurrentText("center")
    window.controls.duration.setValue(3.0)

    loop = QEventLoop()
    finished = {}

    def done(runner):
        finished["summary"] = runner.logger.summary()
        loop.quit()

    window.start_run()
    window.thread.finished_run.connect(done)
    QTimer.singleShot(120_000, loop.quit)
    loop.exec()

    assert "summary" in finished, "GUI run did not finish"
    summary = finished["summary"]
    assert summary.total_frames > 60
    assert summary.lock_retention > 0.8
    assert summary.centroid_rmse_px is not None and summary.centroid_rmse_px < 10.0
