"""Frame source abstraction — the seam between input mode and the vision pipeline.

This module defines the single contract that both operating modes must satisfy:

* **Mode A (simulation)** — :class:`SimulationFrameSource` renders the virtual scene, applies
  the controller's pan/tilt command, extracts the camera sub-viewport, and degrades it with the
  configured noise and atmospheric models. Ground truth is known exactly.

* **Mode B (benchmark video)** — :class:`VideoFrameSource` ingests a pre-recorded ``.mp4``
  supplied by the evaluators. The pan/tilt loop is bypassed: the video *is* the scene. Ground
  truth is usually unavailable.

Why this abstraction matters
----------------------------
Benchmark Performance-2 is 30% of the project grade and runs on video files we have never seen.
The only defence against overfitting to our own synthetic noise is to guarantee that the vision
pipeline running on evaluator video is *byte-for-byte the same code* that runs in simulation.

That guarantee is structural, not a matter of discipline: the vision, filtering, control and
telemetry modules accept a :class:`FrameSource` and never learn which concrete class they were
given. If you ever find yourself writing ``if mode == "video"`` inside ``src/vision/``, the
abstraction is wrong — fix the abstraction rather than branching on mode.

See ``docs/DESIGN.md`` section 8 for the full dual-mode architecture rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class GroundTruth:
    """Known true state of the beacon for a single frame.

    Only available in simulation mode, or in video mode when the evaluators supply a sidecar
    annotation file. All coordinates are in **frame-local pixel coordinates** (origin at the
    top-left of the returned frame), not world-canvas coordinates, so that consumers need no
    knowledge of the coordinate system of the source.

    Attributes:
        x: True beacon centroid x position, sub-pixel accurate.
        y: True beacon centroid y position, sub-pixel accurate.
        visible: Whether the beacon is actually present within this frame. False during
            occlusion, total beam fade, or when the target has left the viewport.
        world_x: Optional true position in world-canvas coordinates (simulation only).
        world_y: Optional true position in world-canvas coordinates (simulation only).
    """

    x: float
    y: float
    visible: bool = True
    world_x: Optional[float] = None
    world_y: Optional[float] = None


@dataclass(frozen=True)
class FrameData:
    """A single frame plus everything a consumer needs to interpret it.

    Attributes:
        frame: 2-D ``uint8`` array of shape ``(height, width)``. Always single-channel
            grayscale — colour sources are converted at the source boundary so that downstream
            code never has to consider channel count.
        timestamp: Seconds since the run started. Monotonic. Used for all rate and timing
            metrics; never use wall-clock time downstream.
        frame_index: Zero-based sequential frame counter.
        ground_truth: True beacon state, or ``None`` when unavailable (the normal case for
            evaluator video). Consumers must handle ``None`` gracefully and fall back to
            logging estimate-only metrics.
        camera_pan_deg: Camera boresight pan angle at capture time. ``None`` in video mode,
            where there is no virtual camera.
        camera_tilt_deg: Camera boresight tilt angle at capture time. ``None`` in video mode.
        extra_ground_truths: Additional target ground-truth states for targets beyond the
            primary. ``None`` when only one target is active. Frame-local coordinates, same
            convention as ``ground_truth``.
        display_frame: Optional 3-channel BGR array for colour display in the GUI. Always
            ``None`` for simulation frames. When present, the pipeline always uses the
            single-channel ``frame``; this field exists only for rendering.
    """

    frame: np.ndarray
    timestamp: float
    frame_index: int
    ground_truth: Optional[GroundTruth] = None
    camera_pan_deg: Optional[float] = None
    camera_tilt_deg: Optional[float] = None
    extra_ground_truths: Optional[List[GroundTruth]] = None
    display_frame: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Return ``(height, width)`` of the frame."""
        return self.frame.shape[0], self.frame.shape[1]

    @property
    def center(self) -> Tuple[float, float]:
        """Return the frame centre ``(x, y)``, i.e. the camera boresight in frame coordinates.

        Pointing error is measured against this point.
        """
        height, width = self.shape
        return (width - 1) / 2.0, (height - 1) / 2.0


@runtime_checkable
class FrameSource(Protocol):
    """Protocol every input mode must implement.

    Implementations are expected to be iterable so a run loop can simply do::

        for frame_data in source:
            result = vision_pipeline.process(frame_data)

    Implementations must be usable as context managers so that file handles and video decoders
    are released deterministically.
    """

    @property
    def supports_pan_tilt(self) -> bool:
        """Whether this source responds to :meth:`apply_pan_tilt`.

        ``True`` for simulation, ``False`` for pre-recorded video. The control loop uses this to
        decide whether to close the loop or to run in open-loop measurement-only mode. This is
        the *only* place the rest of the system is permitted to care about which mode is active.
        """
        ...

    @property
    def nominal_rate_hz(self) -> float:
        """Nominal frame rate of this source, used to seed timing and Kalman ``dt``.

        Actual per-frame ``dt`` must always be derived from :attr:`FrameData.timestamp` rather
        than assumed from this value.
        """
        ...

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        """Angular resolution as ``(horizontal, vertical)`` degrees per pixel.

        ``None`` when unknown (video mode without calibration metadata), in which case all
        errors are reported in pixels only and angular metrics are omitted from the logs.
        """
        ...

    def get_frame(self) -> Optional[FrameData]:
        """Produce the next frame, or ``None`` when the source is exhausted."""
        ...

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """Command the virtual camera's angular rates for the next ``dt`` seconds.

        Implementations must enforce the configured slew-rate and acceleration limits internally,
        so the controller cannot accidentally exceed the mechanical constraints of the simulated
        gimbal. Video sources implement this as a no-op.

        Args:
            pan_rate_deg_s: Requested pan rate. Will be clamped to the configured maximum.
            tilt_rate_deg_s: Requested tilt rate. Will be clamped to the configured maximum.
            dt: Time step over which the rate applies, in seconds.
        """
        ...

    def reset(self) -> None:
        """Return the source to its initial state so a scenario can be re-run reproducibly."""
        ...

    def close(self) -> None:
        """Release any held resources (file handles, decoders, buffers)."""
        ...

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate frames until the source is exhausted."""
        ...


class BaseFrameSource:
    """Convenience base providing iteration and context-manager behaviour.

    Concrete sources should subclass this and implement :meth:`get_frame`, plus override the
    properties and :meth:`apply_pan_tilt` as appropriate. Subclassing is optional — anything
    structurally matching :class:`FrameSource` is acceptable — but it removes boilerplate.
    """

    def __iter__(self) -> Iterator[FrameData]:
        """Yield frames until :meth:`get_frame` returns ``None``."""
        while True:
            frame_data = self.get_frame()
            if frame_data is None:
                return
            yield frame_data

    def __enter__(self) -> "BaseFrameSource":
        """Enter the context, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release resources on context exit."""
        self.close()

    def get_frame(self) -> Optional[FrameData]:
        """Produce the next frame. Must be implemented by subclasses."""
        raise NotImplementedError

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """No-op by default; simulation sources override this."""
        return None

    def reset(self) -> None:
        """No-op by default; stateful sources override this."""
        return None

    def close(self) -> None:
        """No-op by default; resource-holding sources override this."""
        return None
