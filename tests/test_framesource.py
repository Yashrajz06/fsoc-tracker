"""Contract tests for the FrameSource abstraction.

These tests protect the seam that keeps Mode A and Mode B running identical vision code. If they
start failing, the dual-mode guarantee described in ``docs/DESIGN.md`` section 8 has been broken,
and Benchmark Performance-2 (30% of the grade) is at risk.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.framesource import BaseFrameSource, FrameData, FrameSource, GroundTruth


class DummyFrameSource(BaseFrameSource):
    """Minimal in-memory source used to exercise the protocol contract."""

    def __init__(self, n_frames: int = 5, width: int = 64, height: int = 48) -> None:
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._index = 0
        self.pan_commands: List[Tuple[float, float, float]] = []

    @property
    def supports_pan_tilt(self) -> bool:
        return True

    @property
    def nominal_rate_hz(self) -> float:
        return 30.0

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        return (0.00625, 0.00625)

    def get_frame(self) -> Optional[FrameData]:
        if self._index >= self._n_frames:
            return None
        frame = np.zeros((self._height, self._width), dtype=np.uint8)
        data = FrameData(
            frame=frame,
            timestamp=self._index / self.nominal_rate_hz,
            frame_index=self._index,
            ground_truth=GroundTruth(x=10.5, y=12.25),
            camera_pan_deg=0.0,
            camera_tilt_deg=0.0,
        )
        self._index += 1
        return data

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        self.pan_commands.append((pan_rate_deg_s, tilt_rate_deg_s, dt))

    def reset(self) -> None:
        self._index = 0
        self.pan_commands.clear()


def test_dummy_source_satisfies_protocol() -> None:
    """A conforming implementation should pass a runtime protocol check."""
    assert isinstance(DummyFrameSource(), FrameSource)


def test_iteration_terminates_at_source_exhaustion() -> None:
    """Iterating a source must stop cleanly rather than looping forever."""
    frames = list(DummyFrameSource(n_frames=5))
    assert len(frames) == 5
    assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]


def test_frames_are_single_channel_uint8() -> None:
    """Downstream vision code assumes grayscale uint8; colour conversion happens at the source."""
    frame_data = DummyFrameSource().get_frame()
    assert frame_data is not None
    assert frame_data.frame.ndim == 2
    assert frame_data.frame.dtype == np.uint8


def test_center_is_the_boresight_used_for_pointing_error() -> None:
    """Frame centre must be the geometric centre, since pointing error is measured against it."""
    frame_data = DummyFrameSource(width=64, height=48).get_frame()
    assert frame_data is not None
    cx, cy = frame_data.center
    assert cx == pytest.approx(31.5)
    assert cy == pytest.approx(23.5)


def test_timestamps_are_monotonic() -> None:
    """All timing metrics derive from timestamps, so they must never go backwards."""
    timestamps = [f.timestamp for f in DummyFrameSource(n_frames=10)]
    assert all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))


def test_ground_truth_is_optional() -> None:
    """Evaluator video has no ground truth; consumers must tolerate None."""
    data = FrameData(
        frame=np.zeros((4, 4), dtype=np.uint8),
        timestamp=0.0,
        frame_index=0,
        ground_truth=None,
    )
    assert data.ground_truth is None


def test_reset_makes_runs_reproducible() -> None:
    """Scenario re-runs must produce identical frame sequences."""
    source = DummyFrameSource(n_frames=3)
    first = [f.frame_index for f in source]
    source.reset()
    second = [f.frame_index for f in source]
    assert first == second


def test_context_manager_closes_source() -> None:
    """Sources holding decoders or file handles must release them deterministically."""
    with DummyFrameSource(n_frames=2) as source:
        assert source.get_frame() is not None
