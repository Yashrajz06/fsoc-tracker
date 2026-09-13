"""Unthrottled pipeline-capacity benchmark.

Real-time processing FPS is bounded above by the simulator's frame generation clock. A pipeline
capable of 200 Hz still logs 30 FPS in a 30 Hz run, which both understates the system against the
>=20 FPS requirement and -- more dangerously -- **hides regressions**: throughput could degrade
from 200 Hz to 31 Hz with every log still reading 30 FPS, right up until the frame we miss.

Capacity is therefore measured separately: a pre-generated frame buffer is pushed through the
identical vision path as fast as it will run, with rendering, noise synthesis, GUI and disk I/O
outside the timed region. Warm-up frames are discarded so JIT compilation and first-call
allocation are not charged against throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["CapacityResult", "measure_capacity"]


@dataclass(frozen=True)
class CapacityResult:
    """Outcome of a capacity benchmark.

    Attributes:
        frames: Frames timed, excluding warm-up.
        warmup_frames: Frames discarded before timing began.
        mean_ms: Mean per-frame processing time.
        p50_ms: Median per-frame processing time.
        p95_ms: 95th-percentile per-frame time. The tail matters more than the mean for a
            real-time loop, because it is the tail that drops frames.
        max_ms: Worst single frame.
        max_sustainable_fps: Throughput implied by the mean.
        conservative_fps: Throughput implied by the p95, which is the figure to quote against a
            real-time requirement.
        meets_target: Whether ``conservative_fps`` clears the configured FPS target.
        target_fps: The requirement being checked against.
    """

    frames: int
    warmup_frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    max_sustainable_fps: float
    conservative_fps: float
    meets_target: bool
    target_fps: float

    def as_dict(self) -> Dict[str, object]:
        """Return the result as a plain mapping."""
        return {
            "frames": self.frames, "warmup_frames": self.warmup_frames,
            "mean_ms": self.mean_ms, "p50_ms": self.p50_ms, "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "max_sustainable_fps": self.max_sustainable_fps,
            "conservative_fps": self.conservative_fps,
            "meets_target": self.meets_target, "target_fps": self.target_fps,
        }


def measure_capacity(process: Callable[[np.ndarray], object], frames: Sequence[np.ndarray],
                     warmup_frames: int = 10, target_fps: float = 20.0) -> CapacityResult:
    """Measure unthrottled throughput of a per-frame callable.

    Args:
        process: The function under test. Should perform only the work being measured --
            rendering, noise synthesis, GUI and disk I/O belong outside it.
        frames: Pre-generated frames. Generating them inside the timed region would measure the
            simulator rather than the pipeline.
        warmup_frames: Frames to run before timing starts, discarding JIT compilation and
            first-call allocation costs.
        target_fps: Requirement to check the conservative figure against.

    Returns:
        A :class:`CapacityResult`.

    Raises:
        ValueError: If no frames remain after warm-up.
    """
    if len(frames) <= warmup_frames:
        raise ValueError(
            f"Need more than {warmup_frames} warm-up frames, got {len(frames)}")

    for index in range(warmup_frames):
        process(frames[index % len(frames)])

    timings: List[float] = []
    for index in range(warmup_frames, len(frames)):
        start = time.perf_counter()
        process(frames[index])
        timings.append((time.perf_counter() - start) * 1000.0)

    array = np.asarray(timings, dtype=np.float64)
    mean_ms = float(array.mean())
    p95_ms = float(np.percentile(array, 95))
    conservative = 1000.0 / p95_ms if p95_ms > 0 else float("inf")

    return CapacityResult(
        frames=len(timings), warmup_frames=warmup_frames,
        mean_ms=mean_ms, p50_ms=float(np.median(array)), p95_ms=p95_ms,
        max_ms=float(array.max()),
        max_sustainable_fps=1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        conservative_fps=conservative,
        meets_target=conservative >= target_fps, target_fps=target_fps)
