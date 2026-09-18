"""Headless tracking runner: the per-frame loop, shared by the CLI and the GUI.

This exists so the GUI can be a genuine view layer. ``CLAUDE.md`` requires the core modules to
stay importable and runnable headless, with no business logic in widgets -- the way to honour
that is not to be careful while writing widgets, but to leave them nothing to be careful about.
Everything that decides anything lives here; the GUI subscribes to the results.

Nothing in this module imports Qt, and nothing in it knows whether a GUI exists.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from src.camera.model import CameraModel
from src.config import AppConfig
from src.control.controller import ControllerParams, PointingController
from src.control.pid import PidParams
from src.control.search import SearchParams, SpiralSearch
from src.control.statemachine import (
    StateMachineParams,
    TrackingStateMachine,
    TrackState,
)
from src.filtering.kalman import KalmanParams
from src.filtering.track import Track, TrackParams
from src.framesource import FrameData, FrameSource
from src.noise.disturbance import JitterParams
from src.sim.source import SimulationFrameSource
from src.telemetry.logger import TelemetryLogger
from src.telemetry.metrics import FrameRecord
from src.video_source import VideoFrameSource
from src.vision.pipeline import Measurement, VisionPipeline
from src.vision.spotscale import ScaleTracker, estimate_scale

__all__ = ["FrameOutcome", "TrackingRunner", "build_frame_source", "LOW_GAIN_THRESHOLD"]

#: Kalman gain below which an accepted measurement is reported as ``low_gain`` rather than
#: ``measured``: it was folded in, but contributed less than this fraction of the state update.
LOW_GAIN_THRESHOLD: float = 0.05


def build_frame_source(config: AppConfig) -> FrameSource:
    """Construct the frame source for the configured run mode.

    The only place in the system that chooses between Mode A and Mode B. Everything downstream
    receives a :class:`~src.framesource.FrameSource` and never learns which concrete class it was
    given.

    Args:
        config: Validated application configuration.

    Returns:
        A frame source implementing the :class:`~src.framesource.FrameSource` protocol.
    """
    if config.run.mode == "video":
        return VideoFrameSource.from_config(config)
    return SimulationFrameSource(config)


@dataclass(frozen=True)
class FrameOutcome:
    """Everything one processed frame produced, for display or logging.

    Attributes:
        frame_data: The source frame and its ground truth.
        measurement: Vision output for the frame.
        record: The telemetry record that was logged.
        estimate_xy: Fused track estimate in frame coordinates, when available.
        roi: Processing window as ``(x, y, w, h)`` in frame coordinates, or ``None`` when the
            frame was processed full-frame. Reported so the GUI can show *where* the pipeline
            actually looked, which is otherwise invisible and is the single least obvious thing
            about how the tracker achieves its throughput.
        origin: Viewport top-left in world coordinates, or ``None`` for a source with no
            steerable camera. Lets the GUI draw where the camera is looking within the scene.
        state: Tracking state machine mode after this frame.
        processing_ms: Wall-clock vision-plus-control time for the frame.
    """

    frame_data: FrameData
    measurement: Measurement
    record: FrameRecord
    estimate_xy: Optional[Tuple[float, float]]
    roi: Optional[Tuple[int, int, int, int]]
    origin: Optional[Tuple[int, int]]
    state: TrackState
    processing_ms: float


class TrackingRunner:
    """Drives one run: source, vision, scale tracking, filtering, control and telemetry.

    Attributes:
        config: Validated application configuration.
        source: The active frame source.
        logger: Telemetry logger accumulating the run.
    """

    def __init__(self, config: AppConfig, source: Optional[FrameSource] = None) -> None:
        """Assemble a run.

        Args:
            config: Validated application configuration.
            source: Frame source. Built from the configuration when omitted.
        """
        self.config = config
        self.source = source if source is not None else build_frame_source(config)

        steerable = bool(self.source.supports_pan_tilt)
        self.steerable = steerable
        in_fov = bool(getattr(self.source, "target_initially_in_fov", False))

        self.logger = TelemetryLogger(config, mode=config.run.mode, steerable_camera=steerable)
        self.logger.accumulator.target_initially_in_fov = in_fov

        self.pipeline = VisionPipeline.from_config(config)
        self.scale_tracker = ScaleTracker(
            fallback_fwhm_px=config.vision.spot_scale.fallback_fwhm_px)

        # Camera jitter is unobservable by the estimator, so it belongs in R -- but only when a
        # camera exists to jitter. With pre-recorded video the frame IS the scene: there is no
        # boresight, nothing is shaken, and inheriting the configured jitter pins sigma_meas at
        # 1.67 px while the detector is delivering 0.068 px, a 25x over-statement of measurement
        # error that no amount of SNR can then correct.
        #
        # Keyed on the source's own capability, never on a mode string -- the same rule the lock
        # criterion follows, and for the same reason.
        jitter = config.noise.camera_jitter
        jitter_sigma = (JitterParams.from_mapping(jitter).sigma_px
                        if (steerable and jitter.get("enabled")) else 0.0)
        self.track = Track(
            TrackParams(
                confirm_m_of_n=tuple(
                    config.filtering.track_management.get("confirm_m_of_n", [3, 5])),
                delete_after_missed=config.filtering.delete_after_missed),
            config.filtering.kalman_params(unobservable_sigma_px=jitter_sigma))

        machine = config.control.state_machine
        self.state_machine = TrackingStateMachine(
            StateMachineParams(
                lock_confirm_frames=int(machine.get("lock_confirm_frames", 3)),
                loss_declare_frames=int(machine.get("loss_declare_frames", 5)),
                coast_timeout_seconds=float(machine.get("coast_timeout_seconds", 1.0)),
                lock_window_px=float(machine.get("lock_window_px", 40.0)),
                unlock_window_px=float(machine.get("unlock_window_px", 80.0)),
                require_pointing_window=steerable),
            target_initially_in_fov=in_fov)

        self.controller: Optional[PointingController] = None
        self.search: Optional[SpiralSearch] = None
        if steerable and hasattr(self.source, "camera"):
            pid = config.control.pid
            self.controller = PointingController(self.source.camera, ControllerParams(
                pid=PidParams(kp=float(pid["kp"]), ki=float(pid["ki"]), kd=float(pid["kd"]),
                              integral_limit=float(pid["integral_limit"]))))
            self.search = SpiralSearch(
                self.source.camera.boresight, config.camera.fov_px,
                scan_speed_px_s=config.camera.max_px_per_frame[0]
                * config.camera.update_rate_hz,
                params=SearchParams(arm_spacing_fov_fraction=float(
                    config.control.search.get("arm_spacing_fov_fraction", 0.9))))

        # Nominal step, used for the first frame and whenever a source supplies no usable
        # timestamp delta. The *live* step comes from the frame timestamps -- see step().
        self.dt = 1.0 / config.camera.update_rate_hz
        self._previous_timestamp: Optional[float] = None
        # Slew-aware ROI, sized from the physics once at construction rather than per frame: it
        # depends on the slew ceiling, the jitter bound and the spot scale, none of which change
        # within a run. See AppConfig.roi_size_unclamped_px.
        self.roi_enabled = bool(config.vision.roi.enabled)
        self.roi_size_px, _clamped = config.roi_size_px(steerable=steerable)
        self.max_roi_size_px = int(config.vision.resolve_geometry(None).max_roi_size_px)
        self.roi_expand_factor = float(config.vision.roi.expand_on_loss_factor)
        self._missed_streak = 0
        self.frames = 0

    def _capture_boresight(self, frame_data: FrameData) -> Optional[Tuple[float, float]]:
        """Reconstruct the boresight at capture time from the angles the frame carries.

        Using the camera's *current* boresight introduces a one-frame skew, because the camera
        slews after the frame is read out.

        Args:
            frame_data: The frame being processed.

        Returns:
            ``(x, y)`` in canvas coordinates, or ``None`` for a source with no camera.
        """
        if (not hasattr(self.source, "camera") or frame_data.camera_pan_deg is None
                or frame_data.camera_tilt_deg is None):
            return None
        camera: CameraModel = self.source.camera
        scene_cx, scene_cy = camera.scene_center
        dpp_x, dpp_y = camera.deg_per_pixel
        return (scene_cx + frame_data.camera_pan_deg / dpp_x,
                scene_cy + frame_data.camera_tilt_deg / dpp_y)

    def _roi_for(self, frame_data: FrameData,
                 origin: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Build the processing window for this frame, or ``None`` for full-frame processing.

        Full-frame processing is used during acquisition, when there is no established track to
        predict from. Once locked, the window is centred on the filter's last position estimate,
        converted to frame coordinates.

        On a run of missed detections the window grows by ``vision.roi.expand_on_loss_factor``
        per miss, capped at ``max_size_*``. This is what makes re-acquisition possible at all: a
        stale prediction after a loss is wrong by a growing amount, and a window sized for the
        locked case cannot contain the target it is trying to recover. Widening trades throughput
        for the chance of recovery, and only while actually lost.

        Args:
            frame_data: The frame being processed.
            origin: Top-left of the viewport in world coordinates, or ``None`` when the source
                has no steerable camera (Mode B), where frame and world coordinates coincide.

        Returns:
            ``(x, y, width, height)`` in frame coordinates, or ``None``.
        """
        if not self.roi_enabled or not self.track.is_established:
            return None
        position = self.track.filter.position
        x = position[0] - origin[0] if origin is not None else position[0]
        y = position[1] - origin[1] if origin is not None else position[1]

        size = self.roi_size_px
        if self._missed_streak:
            size = int(round(size * self.roi_expand_factor ** self._missed_streak))
            size = min(size, self.max_roi_size_px)

        height, width = frame_data.frame.shape[:2]
        if size >= min(width, height):
            # The window has grown to the frame; processing full-frame is cheaper than cropping.
            return None
        left = int(round(x - size / 2.0))
        top = int(round(y - size / 2.0))
        left = max(0, min(left, width - size))
        top = max(0, min(top, height - size))
        return (left, top, size, size)

    def step(self, frame_data: FrameData) -> FrameOutcome:
        """Process one frame end to end.

        Args:
            frame_data: The frame to process.

        Returns:
            The :class:`FrameOutcome`, which has already been recorded in the logger.
        """
        started = time.perf_counter()
        config = self.config

        scale = self.scale_tracker.state

        # The boresight is reconstructed before vision, not after, because the ROI is placed in
        # frame coordinates and the filter predicts in world coordinates.
        capture_boresight = self._capture_boresight(frame_data)
        width, height = config.camera.fov_px
        origin = None
        if capture_boresight is not None:
            origin = (round(capture_boresight[0] - (width - 1) / 2.0),
                      round(capture_boresight[1] - (height - 1) / 2.0))

        roi = self._roi_for(frame_data, origin)
        measurement = self.pipeline.process(
            frame_data.frame, fwhm_px=scale.estimated_fwhm_px,
            from_fallback=scale.estimated_fwhm_px is None, roi=roi)

        # Before adoption this runs every frame: M-of-N counts frames, so gating it on the
        # re-check interval would make adoption take M x interval rather than M frames.
        if not self.scale_tracker.state.adopted or \
                self.scale_tracker.due_for_recheck(frame_data.timestamp):
            self.scale_tracker.update(estimate_scale(frame_data.frame), frame_data.timestamp)

        # The filter models constant velocity in the WORLD, so it is fed canvas coordinates.
        detection = None
        if measurement.found:
            detection = ((measurement.x + origin[0], measurement.y + origin[1])
                         if origin is not None else measurement.position)

        # The filter must advance by the interval the frames actually arrived at, not by the
        # configured camera rate. A pre-recorded clip carries its own frame rate, and
        # VideoFrameSource already timestamps from it; taking dt from config instead would
        # advance a constant-velocity model at the wrong rate on any evaluator video that is not
        # exactly 30 fps, so the prediction would fall behind by a fixed fraction every frame.
        dt = self.dt
        if self._previous_timestamp is not None:
            measured = frame_data.timestamp - self._previous_timestamp
            if measured > 0:
                dt = measured
        self._previous_timestamp = frame_data.timestamp

        update = self.track.update(
            detection, dt, frame_data.timestamp,
            snr_aperture=measurement.snr.snr_aperture if measurement.snr else None,
            fwhm_px=measurement.geometry.fwhm_px if measurement.geometry else 5.887,
            clipped=measurement.clipped, saturated=measurement.saturated)

        self._missed_streak = 0 if update.accepted else self._missed_streak + 1

        centre_x, centre_y = frame_data.center
        pointing_error = math.inf
        estimate_xy: Optional[Tuple[float, float]] = None
        if update.has_estimate:
            boresight = (self.source.camera.boresight if hasattr(self.source, "camera")
                         else (centre_x, centre_y))
            pointing_error = math.hypot(update.position[0] - boresight[0],
                                        update.position[1] - boresight[1])
            estimate_xy = ((update.position[0] - origin[0], update.position[1] - origin[1])
                           if origin is not None else update.position)

        machine = self.state_machine.update(update.accepted, pointing_error,
                                            frame_data.timestamp, dt)

        slew_saturated = False
        if self.controller is not None:
            # Drive the spiral only while there is no estimate. Sweeping past a visible target is
            # how acquisition was lost before.
            if not update.has_estimate:
                if self.search is not None:
                    slew_saturated = self.controller.drive_to(
                        self.search.step(dt), dt).saturated
            else:
                slew_saturated = self.controller.step(
                    update.position, dt, update.velocity).saturated
                if self.search is not None:
                    self.search.recenter(update.position)

        processing_ms = (time.perf_counter() - started) * 1000.0
        truth = frame_data.ground_truth

        detection_error = None
        if measurement.found and truth is not None:
            detection_error = math.hypot(measurement.x - truth.x, measurement.y - truth.y)

        estimate_source = ""
        centroid_error = None
        if update.has_estimate:
            if not update.accepted:
                estimate_source = "predicted"
            elif update.gain is not None and update.gain < LOW_GAIN_THRESHOLD:
                estimate_source = "low_gain"
            else:
                estimate_source = "measured"
            if truth is not None and estimate_xy is not None:
                centroid_error = math.hypot(estimate_xy[0] - truth.x, estimate_xy[1] - truth.y)

        record = FrameRecord(
            frame_index=frame_data.frame_index, timestamp=frame_data.timestamp,
            state=machine.state.value,
            gt_x=truth.x if truth else None, gt_y=truth.y if truth else None,
            est_x=measurement.x if measurement.found else None,
            est_y=measurement.y if measurement.found else None,
            centroid_error_px=centroid_error, detection_error_px=detection_error,
            estimate_source=estimate_source,
            pointing_error_px=(math.hypot(truth.x - centre_x, truth.y - centre_y)
                               if truth is not None else None),
            locked=machine.locked, detected=measurement.found,
            target_present=truth.visible if truth else True,
            snr_aperture=measurement.snr.snr_aperture if measurement.snr else None,
            snr_peak=measurement.snr.snr_peak if measurement.snr else None,
            clipped=measurement.clipped, saturated=measurement.saturated,
            from_fallback=measurement.from_fallback,
            spot_fwhm_px=measurement.geometry.fwhm_px if measurement.geometry else None,
            rho=scale.rho, scale_reason=scale.reason, track_reason=update.reason,
            gated_out=update.gated_out, nis=update.nis, sigma_meas_px=update.sigma_meas_px,
            processing_ms=processing_ms,
            camera_pan_deg=frame_data.camera_pan_deg,
            camera_tilt_deg=frame_data.camera_tilt_deg,
            slew_saturated=slew_saturated)
        self.logger.add(record)
        self.frames += 1

        return FrameOutcome(frame_data=frame_data, measurement=measurement, record=record,
                            estimate_xy=estimate_xy, roi=roi, origin=origin,
                            state=machine.state,
                            processing_ms=processing_ms)

    def run(self, on_frame: Optional[Callable[[FrameOutcome], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> int:
        """Process every frame the source yields.

        Args:
            on_frame: Called with each :class:`FrameOutcome`. The GUI uses this to update; the
                CLI passes nothing.
            should_stop: Polled between frames so a caller can interrupt.

        Returns:
            The number of frames processed.
        """
        with self.source:  # type: ignore[attr-defined]
            for frame_data in self.source:
                outcome = self.step(frame_data)
                if on_frame is not None:
                    on_frame(outcome)
                if should_stop is not None and should_stop():
                    break
        return self.frames
