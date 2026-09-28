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
    """The live camera frame, annotated so an unfamiliar viewer can follow what is happening.

    A bare image with two crosshairs is legible only to someone who already knows what the
    crosshairs mean. Everything drawn here exists to answer a question a newcomer actually asks:

    * *What is the system looking for?* -- the beacon is ringed and labelled.
    * *Did it find it, and how close is it?* -- the estimate is marked, joined to truth by an
      error line carrying the distance in pixels.
    * *Where is it looking?* -- the ROI box shows the small window the pipeline actually
      processes, which is otherwise invisible and is how the throughput requirement is met.
    * *What is it trying to do?* -- the boresight cross marks frame centre, which is where the
      controller is driving the beacon.
    * *Is it working right now?* -- a colour-coded state badge with a plain-English subtitle.
    * *Where are we in the world?* -- a mini-map showing the viewport's slice of the full scene.

    This is presentation only. The widget computes nothing about tracking; it renders what
    :class:`~src.runner.FrameOutcome` already carries.
    """

    #: Plain-English gloss for each tracker state, shown under the badge.
    STATE_TEXT = {
        "search": ("SEARCHING", "#e8a33d", "sweeping the scene for the beacon"),
        "track": ("TRACKING", "#3fb950", "locked on and following the beacon"),
        "coast": ("COASTING", "#e5534b", "beacon lost from view - predicting its path"),
    }

    TRUTH_COLOUR = "#3fb950"
    ESTIMATE_COLOUR = "#ff9f1c"
    ROI_COLOUR = "#58a6ff"
    BORESIGHT_COLOUR = "#8b949e"
    EXTRA_TARGET_COLOUR = "#c792ea"  # purple, distinct from truth (green) and estimate (orange)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create an empty viewport."""
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._estimate: Optional[Tuple[float, float]] = None
        self._truth: Optional[Tuple[float, float]] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._origin: Optional[Tuple[int, int]] = None
        self._scene: Optional[Tuple[int, int]] = None
        self._error_px: Optional[float] = None
        self._state = ""
        self._extra_truths: list = []  # list of (x, y) for extra targets
        # Motion trails — rolling history of the last ~2 s of each target's world position.
        self._truth_trail: Deque[Tuple[float, float]] = deque(maxlen=60)
        self._extra_trails: list = []  # one Deque per extra target, grown/shrunk dynamically
        self.setMinimumSize(560, 420)

    def _in_frame(self, point: Tuple[float, float]) -> bool:
        """Whether a frame-coordinate point actually lies inside the current frame.

        The beacon is routinely outside the viewport during SEARCH. Without this check its
        marker and label were drawn at the clamped screen edge, producing clipped text over the
        image border and implying the tracker could see something it could not.

        Args:
            point: ``(x, y)`` in frame coordinates.

        Returns:
            True when the point is within the frame bounds.
        """
        if self._pixmap is None:
            return False
        return (0 <= point[0] < self._pixmap.width()) and (0 <= point[1] < self._pixmap.height())

    def set_scene_size(self, width: int, height: int) -> None:
        """Tell the widget how large the world is, for the mini-map.

        Args:
            width: Scene width in pixels.
            height: Scene height in pixels.
        """
        self._scene = (int(width), int(height))

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
        if self._truth is not None:
            self._truth_trail.append(self._truth)
        extra_gts = outcome.frame_data.extra_ground_truths or []
        self._extra_truths = [(gt.x, gt.y) for gt in extra_gts]
        # Keep one trail deque per extra target; grow or shrink to match the current count.
        while len(self._extra_trails) < len(self._extra_truths):
            self._extra_trails.append(deque(maxlen=60))
        while len(self._extra_trails) > len(self._extra_truths):
            self._extra_trails.pop()
        for i, pos in enumerate(self._extra_truths):
            self._extra_trails[i].append(pos)
        self._roi = outcome.roi
        self._origin = outcome.origin
        self._error_px = outcome.record.centroid_error_px
        self._state = outcome.state.value
        self.update()

    # -- drawing helpers -------------------------------------------------------------------

    def _draw_badge(self, painter: QPainter, x: int, y: int) -> None:
        """Draw the colour-coded state badge and its plain-English gloss."""
        label, colour, gloss = self.STATE_TEXT.get(
            self._state, (self._state.upper() or "IDLE", "#8b949e", ""))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(label) + 20
        painter.setBrush(QColor(colour))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x, y, width, 26, 13, 13)
        painter.setPen(QPen(QColor("#0d1117")))
        painter.drawText(x + 10, y + 18, label)

        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#c9d1d9")))
        painter.drawText(x + width + 10, y + 18, gloss)

    def _draw_legend(self, painter: QPainter, x: int, y: int) -> None:
        """Draw a colour key naming every overlay in plain words."""
        entries = [
            (self.TRUTH_COLOUR, "where the beacon really is"),
            (self.ESTIMATE_COLOUR, "where the tracker thinks it is"),
            (self.ROI_COLOUR, "the window being processed"),
            (self.BORESIGHT_COLOUR, "camera centre - the aim point"),
        ]
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setBrush(QColor(13, 17, 23, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x - 6, y - 14, 232, len(entries) * 16 + 10, 6, 6)
        for index, (colour, text) in enumerate(entries):
            row = y + index * 16
            painter.setPen(QPen(QColor(colour), 3))
            painter.drawLine(x, row, x + 14, row)
            painter.setPen(QPen(QColor("#c9d1d9")))
            painter.drawText(x + 22, row + 4, text)

    def _draw_minimap(self, painter: QPainter, right: int, top: int) -> None:
        """Draw the viewport's slice of the whole scene, to convey how narrow the view is."""
        if self._scene is None or self._origin is None or self._pixmap is None:
            return
        scene_w, scene_h = self._scene
        box = 96
        scale = box / max(scene_w, scene_h)
        x = right - box - 10
        painter.setBrush(QColor(13, 17, 23, 210))
        painter.setPen(QPen(QColor("#30363d")))
        painter.drawRect(x, top, box, box)

        painter.setPen(QPen(QColor(self.ROI_COLOUR), 2))
        painter.drawRect(int(x + self._origin[0] * scale), int(top + self._origin[1] * scale),
                         max(2, int(self._pixmap.width() * scale)),
                         max(2, int(self._pixmap.height() * scale)))

        # Beacon dots: convert frame-local coords back to world via origin.
        _ox = self._origin[0]
        _oy = self._origin[1]
        if self._truth is not None:
            _wx = self._truth[0] + _ox
            _wy = self._truth[1] + _oy
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self.TRUTH_COLOUR))
            painter.drawEllipse(int(x + _wx * scale - 3), int(top + _wy * scale - 3), 7, 7)
        _mm_extra_cols = [self.EXTRA_TARGET_COLOUR, "#79c0ff", "#ffa657"]
        for _ei, _ep in enumerate(self._extra_truths):
            _wx = _ep[0] + _ox
            _wy = _ep[1] + _oy
            _ec = _mm_extra_cols[_ei % len(_mm_extra_cols)]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(_ec))
            painter.drawEllipse(int(x + _wx * scale - 2), int(top + _wy * scale - 2), 5, 5)
        painter.setBrush(Qt.NoBrush)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#8b949e")))
        caption = f"view in {scene_w}x{scene_h} scene"
        # Right-aligned against the box, not left-aligned from its corner: the caption is wider
        # than the 96 px box and was being clipped off the edge of the widget.
        painter.drawText(x + box - painter.fontMetrics().horizontalAdvance(caption),
                         top - 5, caption)

    def _draw_zoom(self, painter: QPainter, right: int, top: int) -> None:
        """Draw a magnified crop around the tracked point.

        At 640x480 scaled into a widget, a 10 px beacon is a few pixels across and a viewer
        cannot see what is being tracked at all -- the most important object on screen is the
        least visible. This inset magnifies the neighbourhood so the spot, and the marker sitting
        on it, are plainly visible.
        """
        if self._pixmap is None:
            return
        focus = self._estimate or self._truth
        if focus is None or not self._in_frame(focus):
            # During SEARCH the beacon is often outside the viewport entirely. Drawing a "zoom
            # on the beacon" of a region that contains no beacon is worse than drawing nothing:
            # it invites the viewer to squint at noise looking for a target that is not there.
            return
        # span sets the magnification: 120/24 = 5x. At 2.5x a 10 px beacon was still only ~25 px
        # in the inset and no more legible than the main view, which defeats the point of it.
        box, span = 120, 24
        x = right - box - 10
        src_x = int(max(0, min(focus[0] - span / 2, self._pixmap.width() - span)))
        src_y = int(max(0, min(focus[1] - span / 2, self._pixmap.height() - span)))
        crop = self._pixmap.copy(src_x, src_y, span, span).scaled(
            box, box, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(x, top, crop)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#30363d")))
        painter.drawRect(x, top, box, box)

        factor = box / span
        if self._truth is not None:
            painter.setPen(QPen(QColor(self.TRUTH_COLOUR), 2))
            painter.drawEllipse(int(x + (self._truth[0] - src_x) * factor - 11),
                                int(top + (self._truth[1] - src_y) * factor - 11), 22, 22)
        if self._estimate is not None:
            ex = x + (self._estimate[0] - src_x) * factor
            ey = top + (self._estimate[1] - src_y) * factor
            painter.setPen(QPen(QColor(self.ESTIMATE_COLOUR), 2))
            painter.drawLine(int(ex - 7), int(ey), int(ex + 7), int(ey))
            painter.drawLine(int(ex), int(ey - 7), int(ex), int(ey + 7))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#8b949e")))
        caption = f"{int(factor)}x zoom on the beacon"
        painter.drawText(x + box - painter.fontMetrics().horizontalAdvance(caption),
                         top - 5, caption)

    def paintEvent(self, event) -> None:  # noqa: D102, N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1117"))
        if self._pixmap is None:
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No run in progress.\n\nSet parameters on the left, then press Start.")
            return

        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x0, y0, scaled)
        painter.setPen(QPen(QColor("#30363d")))
        painter.drawRect(x0, y0, scaled.width(), scaled.height())

        sx = scaled.width() / self._pixmap.width()
        sy = scaled.height() / self._pixmap.height()

        def to_screen(point):
            return x0 + point[0] * sx, y0 + point[1] * sy

        # Boresight: where the controller is trying to put the beacon.
        cx, cy = x0 + scaled.width() / 2.0, y0 + scaled.height() / 2.0
        pen = QPen(QColor(self.BORESIGHT_COLOUR), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(cx - 14), int(cy), int(cx + 14), int(cy))
        painter.drawLine(int(cx), int(cy - 14), int(cx), int(cy + 14))

        # The window the pipeline actually processed.
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            painter.setPen(QPen(QColor(self.ROI_COLOUR), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(x0 + rx * sx), int(y0 + ry * sy), int(rw * sx), int(rh * sy))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QPen(QColor(self.ROI_COLOUR)))
            painter.drawText(int(x0 + rx * sx), int(y0 + ry * sy) - 4,
                             f"processing window {rw}x{rh}px")

        # Error line between truth and estimate, labelled with the distance.
        if (self._truth is not None and self._estimate is not None
                and self._error_px is not None
                and self._in_frame(self._truth) and self._in_frame(self._estimate)):
            tx, ty = to_screen(self._truth)
            ex, ey = to_screen(self._estimate)
            painter.setPen(QPen(QColor("#f0f6fc"), 1, Qt.DotLine))
            painter.drawLine(int(tx), int(ty), int(ex), int(ey))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#f0f6fc")))
            # Deliberately not beside the markers: at lock the two are sub-pixel apart and any
            # text there overlaps both. Pinned under the state badge, where it is always legible.
            painter.drawText(x0 + 12, y0 + 58,
                             f"tracking error: {self._error_px:.2f} px   (budget 10 px)")

        font = painter.font()
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)

        # Labels are placed on leader lines in fixed, opposite directions. Drawn adjacent to
        # their markers they collided into an unreadable cluster, because at lock the estimate
        # sits within a pixel of truth -- which is exactly the state a viewer looks at longest.
        def labelled(point, colour: str, text: str, dx: int, dy: int) -> None:
            px, py = to_screen(point)
            lx, ly = px + dx, py + dy
            painter.setPen(QPen(QColor(colour), 1))
            painter.drawLine(int(px + (14 if dx > 0 else -14)), int(py), int(lx), int(ly))
            painter.setPen(QPen(QColor(colour)))
            metrics = painter.fontMetrics()
            painter.drawText(int(lx if dx > 0 else lx - metrics.horizontalAdvance(text)),
                             int(ly) + (12 if dy > 0 else -4), text)

        # Motion trails -- fading dots showing the last ~2 s of each target path.
        n_trail = len(self._truth_trail)
        if n_trail > 1:
            painter.setPen(Qt.NoPen)
            for _ti, _pos in enumerate(self._truth_trail):
                if self._in_frame(_pos):
                    _alpha = int(20 + 130 * _ti / (n_trail - 1))
                    _col = QColor(self.TRUTH_COLOUR)
                    _col.setAlpha(_alpha)
                    painter.setBrush(_col)
                    _px, _py = to_screen(_pos)
                    painter.drawEllipse(int(_px - 2), int(_py - 2), 4, 4)
        _xtc = [self.EXTRA_TARGET_COLOUR, "#79c0ff", "#ffa657"]
        for _td, _tc in zip(self._extra_trails, _xtc):
            _n = len(_td)
            if _n > 1:
                painter.setPen(Qt.NoPen)
                for _ti, _pos in enumerate(_td):
                    if self._in_frame(_pos):
                        _a = int(20 + 130 * _ti / (_n - 1))
                        _c = QColor(_tc)
                        _c.setAlpha(_a)
                        painter.setBrush(_c)
                        _px, _py = to_screen(_pos)
                        painter.drawEllipse(int(_px - 2), int(_py - 2), 4, 4)
        painter.setBrush(Qt.NoBrush)

        if self._truth is not None and self._in_frame(self._truth):
            tx, ty = to_screen(self._truth)
            painter.setPen(QPen(QColor(self.TRUTH_COLOUR), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(tx - 13), int(ty - 13), 26, 26)
            labelled(self._truth, self.TRUTH_COLOUR, "BEACON (true position)", -46, -30)
        for idx, extra_pos in enumerate(self._extra_truths):
            if self._in_frame(extra_pos):
                ex, ey = to_screen(extra_pos)
                painter.setPen(QPen(QColor(self.EXTRA_TARGET_COLOUR), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(ex - 13), int(ey - 13), 26, 26)
                labelled(extra_pos, self.EXTRA_TARGET_COLOUR,
                         f"BEACON {idx + 2}", -46, -30 - (idx * 14))

        if self._estimate is not None and self._in_frame(self._estimate):
            ex, ey = to_screen(self._estimate)
            painter.setPen(QPen(QColor(self.ESTIMATE_COLOUR), 2))
            painter.drawLine(int(ex - 9), int(ey), int(ex + 9), int(ey))
            painter.drawLine(int(ex), int(ey - 9), int(ex), int(ey + 9))
            labelled(self._estimate, self.ESTIMATE_COLOUR, "TRACKER estimate", 46, 30)

        if self._truth is not None and not self._in_frame(self._truth):
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(x0 + 12, y0 + 58, "beacon is outside the camera view")

        self._draw_badge(painter, x0 + 10, y0 + 10)
        self._draw_minimap(painter, x0 + scaled.width(), y0 + 46)
        self._draw_zoom(painter, x0 + scaled.width(), y0 + 176)
        self._draw_legend(painter, x0 + 16, y0 + scaled.height() - 66)


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
