"""Qt thread wrapper around the headless :class:`~src.runner.TrackingRunner`.

The only Qt-aware part of the run loop. It owns no logic: it calls the shared runner and
re-emits results as signals, so the GUI and the CLI cannot drift apart. ``CLAUDE.md`` requires
the core modules to stay runnable headless with no business logic in widgets, and the way to
guarantee that is to leave the widgets nothing to decide.

Frames are emitted at most at a display rate rather than on every processed frame: rendering is
slower than processing, and a queue that grows without bound would make the GUI lag further
behind the run the longer it goes on.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.config import AppConfig
from src.runner import FrameOutcome, TrackingRunner

__all__ = ["RunnerThread"]


class RunnerThread(QThread):
    """Runs a tracking session off the GUI thread.

    Attributes:
        config: The configuration this run was started with.
        runner: The underlying headless runner, available after the run starts.
    """

    #: Emitted with each processed frame, throttled to the display rate.
    frame_ready = Signal(object)
    #: Emitted once the run finishes, carrying the :class:`~src.runner.TrackingRunner`.
    finished_run = Signal(object)
    #: Emitted with a human-readable message if the run fails to start or dies.
    failed = Signal(str)

    def __init__(self, config: AppConfig, display_rate_hz: float = 30.0,
                 parent: Optional[object] = None) -> None:
        """Prepare a run.

        Args:
            config: Validated application configuration.
            display_rate_hz: Maximum rate at which frames are emitted to the GUI.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.config = config
        self.runner: Optional[TrackingRunner] = None
        self._display_interval = 1.0 / max(display_rate_hz, 1.0)
        self._last_emit = 0.0
        self._stop = False

    def stop(self) -> None:
        """Ask the run to finish at the next frame boundary."""
        self._stop = True

    def _on_frame(self, outcome: FrameOutcome) -> None:
        """Emit a frame to the GUI, throttled to the display rate.

        Args:
            outcome: The processed frame.
        """
        now = time.perf_counter()
        if now - self._last_emit >= self._display_interval:
            self._last_emit = now
            self.frame_ready.emit(outcome)

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.runner = TrackingRunner(self.config)
            self.runner.run(on_frame=self._on_frame, should_stop=lambda: self._stop)
            self.finished_run.emit(self.runner)
        except Exception as exc:  # pragma: no cover - surfaced in the GUI, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")
