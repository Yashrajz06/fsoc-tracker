"""Deterministic end-to-end measurement of the SIH re-acquisition metric.

This drives the real ``TrackingRunner`` through source, vision, tracking state machine and
telemetry.  It deliberately uses a clean, fixed beacon so the measured interval means exactly
what Parameter 19 defines: recovery after a declared target loss, not a stochastic detector
failure or an unbounded search.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pytest

from src.config import AppConfig, load_config
from src.framesource import BaseFrameSource, FrameData, GroundTruth
from src.runner import TrackingRunner
from src.sim.beacon import BeaconParams, render_beacon


class DropoutBeaconSource(BaseFrameSource):
    """Fixed in-frame beacon with one deterministic absence interval.

    The source is intentionally non-steerable: it isolates the measured re-acquisition timing
    from search travel while retaining the complete production vision, filter, state-machine and
    telemetry path.  This is the SIH-relevant in-FOV recovery case.
    """

    def __init__(self, width: int, height: int, rate_hz: float,
                 dropout: range, total_frames: int = 48) -> None:
        self.width = width
        self.height = height
        self.rate_hz = rate_hz
        self.dropout = dropout
        self.total_frames = total_frames
        self._index = 0
        self.position = ((width - 1) / 2.0, (height - 1) / 2.0)

    @property
    def supports_pan_tilt(self) -> bool:
        """The fixed frame has no camera to command."""
        return False

    @property
    def nominal_rate_hz(self) -> float:
        """Return the deterministic source cadence."""
        return self.rate_hz

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        """Angular calibration is unavailable for this fixed-frame source."""
        return None

    def get_frame(self) -> Optional[FrameData]:
        """Return the next beacon frame, or a target-free dropout frame."""
        if self._index >= self.total_frames:
            return None
        frame = np.full((self.height, self.width), 10, dtype=np.uint8)
        visible = self._index not in self.dropout
        if visible:
            patch = render_beacon(*self.position, BeaconParams(shape="gaussian", size_px=10,
                                                                 sigma_px=2.5,
                                                                 peak_intensity=220.0))
            y0, x0 = patch.y0, patch.x0
            y1, x1 = y0 + patch.data.shape[0], x0 + patch.data.shape[1]
            frame[y0:y1, x0:x1] = np.clip(
                frame[y0:y1, x0:x1].astype(np.float32) + patch.data, 0, 255).astype(np.uint8)
        data = FrameData(
            frame=frame, timestamp=self._index / self.rate_hz, frame_index=self._index,
            ground_truth=GroundTruth(*self.position, visible=visible))
        self._index += 1
        return data

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """Accept the protocol command; fixed-frame input has no actuator."""

    def reset(self) -> None:
        """Rewind the deterministic sequence."""
        self._index = 0


def test_end_to_end_reacquisition_meets_one_second_requirement() -> None:
    """Parameter 19: a declared in-FOV loss must recover in no more than one second."""
    config: AppConfig = load_config("config/default.json")
    dropout = range(16, 24)  # Eight target-free frames; exceeds the five-frame loss declaration.
    source = DropoutBeaconSource(config.camera.resolution_width, config.camera.resolution_height,
                                 config.camera.update_rate_hz, dropout)
    runner = TrackingRunner(config, source=source)
    runner.run()

    reacquisitions = [event for event in runner.logger.accumulator.acquisitions
                      if event.reacquisition]
    assert len(reacquisitions) == 1, "the dropout must create exactly one recovery event"
    duration = reacquisitions[0].duration_s
    # Loss is declared at frame 20 and the real state machine re-locks at frame 28: 8 frames.
    assert duration == pytest.approx(8.0 / config.camera.update_rate_hz, abs=1e-12)
    assert duration <= config.telemetry.metrics["reacquisition_target_s"]
