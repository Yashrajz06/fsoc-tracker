"""Generate H.264 test videos with known ground truth, for Mode B validation.

**Every video here round-trips through real H.264.** Testing against raw or lossless frames would
validate nothing we are actually scored on: evaluator input is compressed ``.mp4``, and a codec
smears a small bright spot across transform blocks, rings around the highest-contrast feature in
the frame (which *is* our beacon), and mangles impulse noise into something quite unlike the
noise model we simulated.

**The characteristics are deliberately unlike our simulator's defaults.** A test video that looks
like our own output proves only that we can track our own output. The whole purpose of Benchmark
Performance-2 is to catch overfitting, so resolution, spot size, brightness, atmospheric preset
and bitrate all vary independently, and several combinations sit well outside anything
``config/default.json`` would produce.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.noise.atmospheric import AtmosphericParams, apply_atmosphere
from src.noise.sensor import (
    GaussianNoiseParams,
    SaltPepperParams,
    add_gaussian_noise,
    add_salt_pepper,
    to_uint8,
)
from src.sim.beacon import BeaconParams, render_beacon

__all__ = ["VideoSpec", "generate_video", "DEFAULT_SPECS"]


@dataclass(frozen=True)
class VideoSpec:
    """One test video's characteristics.

    Attributes:
        name: Output stem.
        width: Frame width.
        height: Frame height.
        frames: Number of frames.
        fps: Frame rate.
        shape: Beacon profile.
        size_px: Beacon size.
        sigma_px: Gaussian sigma for the beacon profile.
        peak: Beacon peak intensity.
        background: Background level.
        gaussian_sigma: Additive noise sigma.
        sp_density: Salt-and-pepper density.
        atmosphere: Optional Koschmieder parameters.
        bitrate_kbps: Target H.264 bitrate. Lower means heavier artifacts.
        motion: ``"linear"``, ``"circular"`` or ``"figure8"``.
        speed_px_s: Target speed.
    """

    name: str
    width: int = 640
    height: int = 480
    frames: int = 150
    fps: float = 30.0
    shape: str = "gaussian"
    size_px: float = 10.0
    sigma_px: float = 2.5
    peak: float = 200.0
    background: float = 25.0
    gaussian_sigma: float = 8.0
    sp_density: float = 0.0
    atmosphere: Optional[AtmosphericParams] = None
    bitrate_kbps: int = 4000
    motion: str = "linear"
    speed_px_s: float = 60.0

    def position(self, frame_index: int) -> Tuple[float, float]:
        """Return the true beacon centre for a frame.

        Args:
            frame_index: Zero-based frame number.

        Returns:
            ``(x, y)`` in full-frame source pixel coordinates.
        """
        t = frame_index / self.fps
        cx, cy = self.width / 2.0, self.height / 2.0
        margin = max(60.0, 4.0 * self.size_px)
        if self.motion == "circular":
            radius = min(cx, cy) - margin
            omega = self.speed_px_s / max(radius, 1.0)
            return cx + radius * math.cos(omega * t), cy + radius * math.sin(omega * t)
        if self.motion == "figure8":
            ax, ay = cx - margin, (cy - margin) * 0.6
            omega = self.speed_px_s / max(ax, 1.0)
            return cx + ax * math.sin(omega * t), cy + ay * math.sin(2.0 * omega * t)
        span = self.width - 2 * margin
        travelled = (self.speed_px_s * t) % (2 * span)
        x = margin + (travelled if travelled <= span else 2 * span - travelled)
        return x, cy + 0.25 * (cy - margin) * math.sin(0.7 * t)


def _render_frame(spec: VideoSpec, frame_index: int,
                  rng: np.random.Generator) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Render one clean-then-degraded frame and its exact ground truth."""
    frame = np.full((spec.height, spec.width), spec.background, dtype=np.float32)
    x, y = spec.position(frame_index)
    patch = render_beacon(x, y, BeaconParams(shape=spec.shape, size_px=spec.size_px,
                                             sigma_px=spec.sigma_px,
                                             peak_intensity=spec.peak))
    ph, pw = patch.data.shape
    y0, x0 = max(0, patch.y0), max(0, patch.x0)
    y1 = min(spec.height, patch.y0 + ph)
    x1 = min(spec.width, patch.x0 + pw)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] += patch.data[y0 - patch.y0:y1 - patch.y0,
                                          x0 - patch.x0:x1 - patch.x0]

    if spec.atmosphere is not None:
        frame = apply_atmosphere(frame, spec.atmosphere, rng)
    out = to_uint8(frame)
    if spec.gaussian_sigma > 0:
        out = to_uint8(add_gaussian_noise(out, GaussianNoiseParams(sigma=spec.gaussian_sigma),
                                          rng))
    if spec.sp_density > 0:
        out = to_uint8(add_salt_pepper(out, SaltPepperParams(enabled=True,
                                                             density=spec.sp_density), rng))
    return out, (x, y)


def _ffmpeg_encoder() -> str:
    """Locate a software H.264 encoder, or explain why validation cannot proceed.

    Returns:
        The ffmpeg encoder name.

    Raises:
        RuntimeError: If no software H.264 encoder is available.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. Mode B validation requires real H.264 compression; a lossless "
            "fallback would validate nothing we are scored on.")
    listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, check=False).stdout
    for name in ("libx264", "libopenh264"):
        if name in listing:
            return name
    raise RuntimeError(
        "No software H.264 encoder (libx264 / libopenh264) in this ffmpeg build. "
        "Mode B validation requires real compression.")


def generate_video(spec: VideoSpec, directory: Path,
                   seed: int = 0) -> Tuple[Path, Path]:
    """Write one H.264 test video and its ground-truth sidecar.

    Frames are piped to ffmpeg's software H.264 encoder rather than written through
    ``cv2.VideoWriter``. That is not a preference: OpenCV's writer here resolves H.264 to the
    hardware ``h264_v4l2m2m`` encoder, fails when no such device exists, and then **silently
    falls back to MPEG-4 Part 2**. The resulting files were 78 KB *per frame* -- essentially
    lossless -- which would have made every compression measurement below meaningless while
    appearing to succeed. Encoder selection is therefore explicit and a fallback is a hard error.

    Args:
        spec: The video's characteristics.
        directory: Output directory.
        seed: RNG seed.

    Returns:
        ``(video_path, sidecar_path)``.

    Raises:
        RuntimeError: If no software H.264 encoder is available, or encoding fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / f"{spec.name}.mp4"
    sidecar_path = directory / f"{spec.name}_truth.csv"
    rng = np.random.default_rng(seed)
    encoder = _ffmpeg_encoder()

    command = [
        shutil.which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{spec.width}x{spec.height}", "-r", str(spec.fps),
        "-i", "-",
        "-an", "-c:v", encoder, "-pix_fmt", "yuv420p",
        "-b:v", f"{spec.bitrate_kbps}k", "-maxrate", f"{spec.bitrate_kbps}k",
        "-bufsize", f"{2 * spec.bitrate_kbps}k",
        # Determinism. Seeding the scene RNG is not enough: x264 slices frames across threads
        # and the slice boundaries depend on thread scheduling, so two encodes of *identical*
        # input produced different pixels -- measured at up to 255 levels of difference on
        # lowlight_impulse (132/150 frames) and 42 levels on baseline_640 (149/150 frames).
        # Salt-and-pepper content is worst affected, because a single flipped impulse near a
        # block edge changes what the detector ranks as the brightest candidate.
        #
        # Without this, a Benchmark-2 number cannot be reproduced from one run to the next, and
        # a regression test over generated video would be measuring the encoder's thread
        # scheduling as much as our tracker.
        "-threads", "1",
        *(["-x264-params", "sliced-threads=0:deterministic=1"]
          if encoder == "libx264" else []),
        # Strip the encoder banner and creation timestamp so the container bytes are stable too,
        # which lets a fixture be checksummed.
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        str(video_path),
    ]

    rows: List[Tuple[int, float, float]] = []
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(spec.frames):
            frame, (x, y) = _render_frame(spec, index, rng)
            process.stdin.write(frame.tobytes())
            rows.append((index, x, y))
        process.stdin.close()
    except BrokenPipeError as exc:  # pragma: no cover - encoder died mid-stream
        raise RuntimeError(f"H.264 encoding failed for {spec.name}") from exc
    if process.wait() != 0:
        stderr = process.stderr.read().decode() if process.stderr else ""
        raise RuntimeError(f"H.264 encoding failed for {spec.name}: {stderr}")

    with sidecar_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# Ground truth in full-frame source pixel coordinates.\n")
        handle.write("# Pixel centres at integer indices; origin top-left; tuples are (x, y).\n")
        out = csv.writer(handle)
        out.writerow(["frame", "x", "y"])
        out.writerows(rows)

    return video_path, sidecar_path


#: Test matrix. Resolution, spot size, brightness, atmosphere and bitrate vary independently, and
#: several rows sit well outside anything our own configuration would produce.
DEFAULT_SPECS: Tuple[VideoSpec, ...] = (
    VideoSpec("baseline_640", width=640, height=480),
    VideoSpec("hd_1280_smallspot", width=1280, height=720, size_px=5.0, sigma_px=1.4,
              peak=140.0, gaussian_sigma=14.0, bitrate_kbps=2500, motion="circular"),
    VideoSpec("fhd_1920_bigspot", width=1920, height=1080, size_px=20.0, sigma_px=6.0,
              peak=240.0, background=60.0, gaussian_sigma=6.0, motion="figure8"),
    VideoSpec("canvas_2000_fog", width=2000, height=2000, frames=90, size_px=14.0,
              sigma_px=4.0, peak=230.0, background=15.0, gaussian_sigma=10.0,
              atmosphere=AtmosphericParams(beta=0.75, airlight=180.0, blur_sigma=1.5),
              motion="circular", speed_px_s=120.0),
    VideoSpec("lowlight_impulse", width=800, height=600, peak=70.0, background=8.0,
              gaussian_sigma=16.0, sp_density=0.08, bitrate_kbps=1200),
    VideoSpec("square_bright", width=720, height=576, shape="square", size_px=12.0,
              peak=250.0, background=40.0, gaussian_sigma=5.0, motion="linear"),
)
