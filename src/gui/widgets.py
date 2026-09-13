"""Display widgets: viewport with overlays, strip charts, and the metrics panel.

Pure view code. Every widget receives a :class:`~src.runner.FrameOutcome` or a
:class:`~src.telemetry.metrics.MetricsSummary` and renders it; none decides anything.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap, QColor
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from src.runner import FrameOutcome
from src.telemetry.metrics import MetricsSummary

__all__ = ["ViewportWidget", "StripCharts", "MetricsPanel"]


class ViewportWidget(QWidget):
    """Shows the live camera frame with the estimate and ground truth overlaid."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create an empty viewport."""
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._estimate: Optional[Tuple[float, float]] = None
        self._truth: Optional[Tuple[float, float]] = None
        self._state = ""
        self.setMinimumSize(480, 360)

    def update_frame(self, outcome: FrameOutcome) -> None:
        """Render one processed frame.

        Args:
            outcome: The frame and what the pipeline made of it.
        """
        frame = np.ascontiguousarray(outcome.frame_data.frame)
        height, width = frame.shape
        image = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._estimate = outcome.estimate_xy
        truth = outcome.frame_data.ground_truth
        self._truth = (truth.x, truth.y) if truth is not None else None
        self._state = outcome.state.value
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D102, N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101010"))
        if self._pixmap is None:
            painter.setPen(QPen(QColor("#888")))
            painter.drawText(self.rect(), Qt.AlignCenter, "no run in progress")
            return

        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x0, y0, scaled)

        sx = scaled.width() / self._pixmap.width()
        sy = scaled.height() / self._pixmap.height()

        def marker(point, colour: str, size: int) -> None:
            painter.setPen(QPen(QColor(colour), 2))
            px = x0 + point[0] * sx
            py = y0 + point[1] * sy
            painter.drawLine(int(px - size), int(py), int(px + size), int(py))
            painter.drawLine(int(px), int(py - size), int(px), int(py + size))

        if self._truth is not None:
            marker(self._truth, "#2ca02c", 12)      # ground truth, when available
        if self._estimate is not None:
            marker(self._estimate, "#ff7f0e", 8)    # fused estimate

        painter.setPen(QPen(QColor("#e0e0e0")))
        painter.drawText(x0 + 6, y0 + 18, f"state: {self._state}")
        painter.drawText(x0 + 6, y0 + 34, "green = ground truth   orange = estimate")


class StripCharts(QWidget):
    """Rolling strip charts for centroiding error and processing rate."""

    def __init__(self, history_seconds: float = 30.0, parent: Optional[QWidget] = None) -> None:
        """Create the charts.

        Args:
            history_seconds: Rolling window length.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.history_seconds = history_seconds
        self._t: Deque[float] = deque()
        self._error: Deque[float] = deque()
        self._fps: Deque[float] = deque()

        pg.setConfigOptions(antialias=False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.error_plot = pg.PlotWidget(title="centroiding error (px)")
        self.error_plot.setLogMode(False, True)
        self.error_plot.showGrid(x=True, y=True, alpha=0.3)
        self.error_curve = self.error_plot.plot(pen=pg.mkPen("#ff7f0e", width=2))
        # The graded budget, drawn so the live trace is read against it rather than in isolation.
        self.error_plot.addLine(y=np.log10(10.0), pen=pg.mkPen("#d62728", style=Qt.DashLine))

        self.fps_plot = pg.PlotWidget(title="processing rate (FPS)")
        self.fps_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fps_curve = self.fps_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.fps_plot.addLine(y=20.0, pen=pg.mkPen("#d62728", style=Qt.DashLine))

        layout.addWidget(self.error_plot)
        layout.addWidget(self.fps_plot)

    def update_frame(self, outcome: FrameOutcome) -> None:
        """Append one frame's telemetry and roll the window.

        Args:
            outcome: The processed frame.
        """
        record = outcome.record
        self._t.append(record.timestamp)
        self._error.append(record.centroid_error_px if record.centroid_error_px is not None
                           else float("nan"))
        self._fps.append(1000.0 / record.processing_ms if record.processing_ms else float("nan"))

        while self._t and (self._t[-1] - self._t[0]) > self.history_seconds:
            self._t.popleft()
            self._error.popleft()
            self._fps.popleft()

        times = np.fromiter(self._t, dtype=float)
        self.error_curve.setData(times, np.fromiter(self._error, dtype=float))
        self.fps_curve.setData(times, np.fromiter(self._fps, dtype=float))

    def clear(self) -> None:
        """Discard the history."""
        self._t.clear()
        self._error.clear()
        self._fps.clear()
        self.error_curve.setData([], [])
        self.fps_curve.setData([], [])


class MetricsPanel(QGroupBox):
    """Live metrics shown against their specification targets."""

    #: ``(label, attribute, format, target, lower_is_better)``
    ROWS = (
        ("state", None, "{}", None, True),
        ("centroiding RMSE", "centroid_rmse_px", "{:.3f} px", 10.0, True),
        ("centroiding median", "centroid_median_px", "{:.3f} px", 10.0, True),
        ("pointing RMSE", "pointing_rmse_px", "{:.2f} px", 10.0, True),
        ("lock retention", "lock_retention", "{:.1%}", 0.95, False),
        ("loss rate", "loss_rate", "{:.1%}", 0.05, True),
        ("processing FPS", "processing_fps_mean", "{:.1f}", 20.0, False),
        ("assoc failure", "association_failure_rate", "{:.1%}", None, True),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the panel."""
        super().__init__("live metrics vs spec targets", parent)
        self._values = {}
        grid = QGridLayout(self)
        for row, (label, attribute, _fmt, target, _lower) in enumerate(self.ROWS):
            grid.addWidget(QLabel(label), row, 0)
            value = QLabel("&mdash;")
            value.setTextFormat(Qt.RichText)
            grid.addWidget(value, row, 1)
            grid.addWidget(QLabel("" if target is None else f"target {target:g}"), row, 2)
            self._values[attribute or "state"] = value

    def update_summary(self, summary: MetricsSummary, state: str = "") -> None:
        """Refresh from a metrics summary.

        Args:
            summary: The accumulated run metrics.
            state: Current state machine mode.
        """
        for label, attribute, fmt, target, lower_is_better in self.ROWS:
            widget = self._values[attribute or "state"]
            if attribute is None:
                widget.setText(state or "&mdash;")
                continue
            value = getattr(summary, attribute, None)
            if value is None:
                widget.setText("&mdash;")
                continue
            ok = True if target is None else (
                value <= target if lower_is_better else value >= target)
            colour = "#12632c" if ok else "#94191b"
            widget.setText(f'<span style="color:{colour}">{fmt.format(value)}</span>')
