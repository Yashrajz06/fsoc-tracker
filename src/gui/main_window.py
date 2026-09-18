"""Main window: assembles the view widgets and drives runs on a worker thread.

Holds no tracking logic. It builds a configuration from the control panel, hands it to
:class:`~src.gui.runner_thread.RunnerThread`, and routes the resulting signals to widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QVBoxLayout, QWidget,
)

from src.config import AppConfig, ConfigError, load_json_document, merge_overrides
from src.gui.controls import ControlPanel
from src.gui.runner_thread import RunnerThread
from src.gui.widgets import MetricsPanel, StripCharts, ViewportWidget
from src.runner import FrameOutcome, TrackingRunner
from src.telemetry.report import write_report

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, config: AppConfig, config_path: str,
                 parent: Optional[QWidget] = None) -> None:
        """Create the window.

        Args:
            config: Starting configuration.
            config_path: Path the configuration was loaded from, used to rebuild it when
                controls change so validation runs exactly as it does for the CLI.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle("FSOC coarse-alignment tracker")
        self.base_config = config
        self.config_path = config_path
        self.thread: Optional[RunnerThread] = None
        self._last_state = ""

        self.viewport = ViewportWidget()
        self.charts = StripCharts(history_seconds=config.gui.plot_history_seconds)
        self.metrics = MetricsPanel()
        self.controls = ControlPanel(config)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(buttons)
        right_layout.addWidget(self.metrics)
        right_layout.addWidget(self.controls, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.viewport, 3)
        left_layout.addWidget(self.charts, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([900, 420])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("ready")

        # Metrics are refreshed on a timer rather than per frame: recomputing the summary on
        # every frame would put O(n) work on the GUI thread at frame rate.
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start(500)

    def _build_config(self) -> AppConfig:
        """Rebuild the configuration from the control panel.

        Returns:
            A validated configuration.

        Raises:
            ConfigError: If the control values violate a specification limit. The spec ranges are
                enforced by ``src.config``, not re-implemented here, so the GUI cannot drift from
                the CLI's validation.
        """
        raw = load_json_document(self.config_path)
        merged = merge_overrides(raw, self.controls.overrides())
        return AppConfig.from_dict(merged, source_path=self.config_path)

    def start_run(self) -> None:
        """Validate the controls and start a run."""
        if self.thread is not None and self.thread.isRunning():
            return
        try:
            config = self._build_config()
        except ConfigError as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return

        self.charts.clear()
        # The mini-map needs the scene size to show how narrow a slice the camera sees.
        self.viewport.set_scene_size(config.scene.width, config.scene.height)
        self.thread = RunnerThread(config, display_rate_hz=config.gui.display_rate_hz)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.finished_run.connect(self._on_finished)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage(f"running ({config.run.mode})")

    def stop_run(self) -> None:
        """Ask the current run to stop."""
        if self.thread is not None:
            self.thread.stop()
            self.statusBar().showMessage("stopping...")

    def _on_frame(self, outcome: FrameOutcome) -> None:
        """Route one frame to the display widgets.

        Args:
            outcome: The processed frame.
        """
        self.viewport.update_frame(outcome)
        self.charts.update_frame(outcome)
        self._last_state = outcome.state.value

    def _refresh_metrics(self) -> None:
        """Refresh the metrics panel from the live run."""
        if self.thread is None or self.thread.runner is None:
            return
        self.metrics.update_summary(self.thread.runner.logger.summary(), self._last_state)

    def _on_finished(self, runner: TrackingRunner) -> None:
        """Write the logs and report, exactly as a headless run would.

        Args:
            runner: The finished runner.
        """
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        summary = runner.logger.summary()
        self.metrics.update_summary(summary, self._last_state)

        outputs = []
        config = runner.config
        if config.telemetry.enabled:
            if config.telemetry.per_frame_csv:
                outputs.append(runner.logger.write_csv())
            if config.telemetry.summary_report:
                outputs.append(write_report(runner.logger, config))

        message = f"finished: {summary.total_frames} frames"
        if summary.below_characterised_envelope:
            message += " - input below characterised envelope, no lock achieved"
        if outputs:
            message += f" - wrote {', '.join(Path(p).name for p in outputs)}"
        self.statusBar().showMessage(message)

    def _on_failed(self, message: str) -> None:
        """Surface a run failure rather than leaving the window silent.

        Args:
            message: Description of the failure.
        """
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(f"failed: {message}")
        QMessageBox.critical(self, "Run failed", message)

    def closeEvent(self, event) -> None:  # noqa: D102, N802 - Qt override
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(2000)
        event.accept()
