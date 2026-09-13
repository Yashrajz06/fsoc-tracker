"""Mode B frame source: pre-recorded video, with the PTZ loop bypassed.

Benchmark Performance-2 hands us ``.mp4`` files we have never seen, at 30 fps, covering a
complete screen with noise and a moving beacon. The video *is* the scene: there is no virtual
camera to steer, so :attr:`supports_pan_tilt` is ``False`` and the control loop runs in
measurement-only mode.

The contract this class exists to honour
----------------------------------------
The vision, filtering and telemetry modules must run **byte-for-byte the same code** on evaluator
video as in simulation. That guarantee is structural, not a matter of discipline: everything
downstream receives a :class:`~src.framesource.FrameSource` and never learns which concrete class
it was given. If anything downstream needs to branch on mode, the abstraction is wrong and the
abstraction is what gets fixed.

Everything that could differ between modes is therefore normalised *here*, at the source
boundary:

* **Resolution is auto-detected.** Never assume 640x480 or 2000x2000.
* **Colour is converted to grayscale here**, so no downstream module ever considers channel count.
  The spec says monochrome; evaluator video may not be.
* **Intensity is normalised per frame** when configured, because we cannot assume anything about
  the brightness range of an unseen file.
* **``deg_per_pixel`` is ``None``.** There is no calibration for evaluator video, so angular
  metrics are *omitted* rather than fabricated from an assumed FOV.
* **``ground_truth`` is ``None``** unless evaluators supply a sidecar CSV, in which case it is
  loaded and reported in full-frame source pixel coordinates under the project's pixel-centre
  convention -- so comparison needs no coordinate negotiation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import AppConfig
from src.framesource import BaseFrameSource, FrameData, GroundTruth

__all__ = ["VideoFrameSource", "load_ground_truth_sidecar"]


def load_ground_truth_sidecar(path: str | Path) -> Dict[int, Tuple[float, float]]:
    """Load a sidecar CSV of true centroids, keyed by frame index.

    Accepts a header row naming the columns, tolerating the obvious spellings, so an evaluator's
    file does not have to match ours exactly.

    Args:
        path: Path to the CSV.

    Returns:
        Mapping of frame index to ``(x, y)`` in full-frame source pixel coordinates.

    Raises:
        ValueError: If the required columns cannot be identified.
    """
    table: Dict[int, Tuple[float, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        if reader.fieldnames is None:
            raise ValueError(f"Ground-truth sidecar {path} has no header row")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}

        def column(*candidates: str) -> str:
            for candidate in candidates:
                if candidate in lookup:
                    return lookup[candidate]
            raise ValueError(
                f"Ground-truth sidecar {path} lacks a column among {candidates}; "
                f"found {reader.fieldnames}")

        frame_col = column("frame", "frame_index", "index", "n")
        x_col = column("x", "gt_x", "centroid_x", "true_x")
        y_col = column("y", "gt_y", "centroid_y", "true_y")

        for row in reader:
            table[int(float(row[frame_col]))] = (float(row[x_col]), float(row[y_col]))
    return table


class VideoFrameSource(BaseFrameSource):
    """Reads frames from a video file, normalising everything at the source boundary.

    Attributes:
        path: Path to the video file.
        width: Detected frame width.
        height: Detected frame height.
        frame_count: Frame count reported by the container, which may be approximate.
    """

    def __init__(self, path: str | Path, force_grayscale: bool = True,
                 normalise_intensity: bool = False,
                 ground_truth_path: Optional[str | Path] = None,
                 nominal_rate_hz: Optional[float] = None) -> None:
        """Open a video file.

        Args:
            path: Path to the ``.mp4``.
            force_grayscale: Convert colour input to single channel here, so downstream code
                never considers channel count.
            normalise_intensity: Rescale each frame to the full 8-bit range. Off by default:
                the vision pipeline already derives every threshold from frame statistics, and
                stretching the range first discards the absolute levels that saturation detection
                depends on.
            ground_truth_path: Optional sidecar CSV of true centroids.
            nominal_rate_hz: Override the container's frame rate, used when it is missing or
                obviously wrong.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be opened as video.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"Could not open {self.path} as video")

        self.force_grayscale = force_grayscale
        self.normalise_intensity = normalise_intensity

        # Auto-detect: never assume a resolution.
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        reported = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._rate = float(nominal_rate_hz or (reported if reported > 0 else 30.0))

        self._ground_truth = (load_ground_truth_sidecar(ground_truth_path)
                              if ground_truth_path else {})
        self._index = 0

    @classmethod
    def from_config(cls, config: AppConfig) -> "VideoFrameSource":
        """Build a video source from an application configuration.

        Args:
            config: Validated application configuration with ``video_input.path`` set.

        Returns:
            The configured :class:`VideoFrameSource`.

        Raises:
            ValueError: If no video path is configured.
        """
        video = config.video_input
        if video.path is None:
            raise ValueError("run.mode='video' requires video_input.path")
        return cls(video.path, force_grayscale=video.force_grayscale,
                   normalise_intensity=video.normalise_intensity,
                   ground_truth_path=video.ground_truth_path)

    # -- FrameSource protocol ----------------------------------------------------------------

    @property
    def supports_pan_tilt(self) -> bool:
        """False: the video *is* the scene, so there is no camera to steer."""
        return False

    @property
    def nominal_rate_hz(self) -> float:
        """Frame rate from the container, or the supplied override."""
        return self._rate

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        """``None``: evaluator video carries no calibration.

        Angular metrics are omitted from the logs rather than computed from an assumed FOV,
        because a fabricated angular scale would be silently wrong in a way no test would catch.
        """
        return None

    @property
    def has_ground_truth(self) -> bool:
        """Whether a sidecar annotation file was supplied."""
        return bool(self._ground_truth)

    @property
    def shape(self) -> Tuple[int, int]:
        """Detected frame shape as ``(height, width)``."""
        return self.height, self.width

    def get_frame(self) -> Optional[FrameData]:
        """Decode, normalise and return the next frame.

        Returns:
            A :class:`FrameData` with a single-channel ``uint8`` frame, or ``None`` at end of
            stream.
        """
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None

        if frame.ndim == 3:
            # Colour is converted here so no downstream module has to consider channel count.
            frame = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.force_grayscale
                     else frame[:, :, 0])
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self.normalise_intensity:
            low, high = int(frame.min()), int(frame.max())
            if high > low:
                frame = (((frame.astype(np.float32) - low) * (255.0 / (high - low)))
                         .astype(np.uint8))

        truth = None
        if self._index in self._ground_truth:
            x, y = self._ground_truth[self._index]
            truth = GroundTruth(x=x, y=y, visible=True)

        data = FrameData(frame=frame, timestamp=self._index / self._rate,
                         frame_index=self._index, ground_truth=truth,
                         camera_pan_deg=None, camera_tilt_deg=None)
        self._index += 1
        return data

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """No-op: there is no virtual camera in Mode B.

        Args:
            pan_rate_deg_s: Ignored.
            tilt_rate_deg_s: Ignored.
            dt: Ignored.
        """
        return None

    def reset(self) -> None:
        """Rewind to the first frame."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._index = 0

    def close(self) -> None:
        """Release the decoder."""
        if self._capture is not None:
            self._capture.release()
