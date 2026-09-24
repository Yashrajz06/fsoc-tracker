"""Mode B: evaluator video ingestion, and the H.264 re-validations it makes possible.

Benchmark Performance-2 is 30% of the grade and the first real test of whether the
:class:`~src.framesource.FrameSource` abstraction held. The central assertion here is that the
vision pipeline runs on video **completely unmodified** -- the same object that processes
simulation frames, with no mode parameter reaching it.

Every video in these tests round-trips through real software H.264. Testing against raw frames
would validate nothing we are scored on.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.video_source import VideoFrameSource, load_ground_truth_sidecar
from src.vision.pipeline import VisionPipeline
from src.vision.spotscale import bootstrap_scale, confirm_scale

pytest.importorskip("cv2")
from scenarios.generate import VideoSpec, generate_video  # noqa: E402

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="Mode B validation requires real H.264; a lossless fallback would validate nothing")


@pytest.fixture(scope="module")
def config():
    """The validated default configuration."""
    return load_config("config/default.json")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A small H.264 clip with a moving beacon and its ground-truth sidecar."""
    directory = tmp_path_factory.mktemp("videos")
    spec = VideoSpec("clip", width=640, height=480, frames=40, peak=200.0,
                     background=25.0, gaussian_sigma=8.0, speed_px_s=50.0)
    return generate_video(spec, directory)


# ------------------------------------------------------------------------------------------
# The abstraction
# ------------------------------------------------------------------------------------------


@requires_ffmpeg
def test_vision_pipeline_runs_on_video_completely_unmodified(config, clip) -> None:
    """The single most important check in this phase.

    The *same* pipeline object processes simulation frames and evaluator video. Nothing in
    ``src/vision`` knows which mode it is in, and nothing takes a mode argument. If this ever
    needs a branch, the abstraction is wrong and the abstraction is what gets fixed.
    """
    video, sidecar = clip
    pipeline = VisionPipeline.from_config(config)
    truth = load_ground_truth_sidecar(sidecar)

    errors = []
    with VideoFrameSource(video, ground_truth_path=sidecar) as source:
        for frame_data in source:
            measurement = pipeline.process(frame_data.frame, fwhm_px=5.887,
                                           from_fallback=False)
            if measurement.found and frame_data.frame_index in truth:
                tx, ty = truth[frame_data.frame_index]
                errors.append(math.hypot(measurement.x - tx, measurement.y - ty))

    assert len(errors) > 30
    assert float(np.median(errors)) < 1.0


@requires_ffmpeg
def test_source_reports_no_pan_tilt_and_no_angular_scale(clip) -> None:
    """The video *is* the scene, and it carries no calibration.

    ``deg_per_pixel`` is ``None`` so angular metrics are omitted rather than fabricated from an
    assumed FOV -- a made-up angular scale would be silently wrong in a way no test would catch.
    """
    video, _ = clip
    with VideoFrameSource(video) as source:
        assert source.supports_pan_tilt is False
        assert source.deg_per_pixel is None
        assert source.apply_pan_tilt(10.0, 10.0, 0.1) is None


@requires_ffmpeg
def test_resolution_is_auto_detected(tmp_path) -> None:
    """Never assume 640x480 or 2000x2000."""
    for width, height in ((320, 240), (800, 600), (1280, 720)):
        spec = VideoSpec(f"res_{width}", width=width, height=height, frames=6)
        video, _ = generate_video(spec, tmp_path)
        with VideoFrameSource(video) as source:
            assert (source.width, source.height) == (width, height)
            frame = source.get_frame()
            assert frame is not None and frame.shape == (height, width)


@requires_ffmpeg
def test_frames_are_single_channel_uint8(clip) -> None:
    """Colour is converted at the source boundary, so no downstream module sees channels."""
    video, _ = clip
    with VideoFrameSource(video) as source:
        frame = source.get_frame()
        assert frame is not None
        assert frame.frame.ndim == 2
        assert frame.frame.dtype == np.uint8


@requires_ffmpeg
def test_ground_truth_is_none_without_a_sidecar(clip) -> None:
    """Evaluator video normally has no annotation; the pipeline must cope with that."""
    video, sidecar = clip
    with VideoFrameSource(video) as source:
        assert source.get_frame().ground_truth is None
        assert not source.has_ground_truth
    with VideoFrameSource(video, ground_truth_path=sidecar) as source:
        assert source.get_frame().ground_truth is not None
        assert source.has_ground_truth


@requires_ffmpeg
def test_timestamps_and_reset(clip) -> None:
    """Timestamps come from the container rate, and reset rewinds for a reproducible re-run."""
    video, _ = clip
    with VideoFrameSource(video) as source:
        first = source.get_frame()
        second = source.get_frame()
        assert second.timestamp - first.timestamp == pytest.approx(1.0 / source.nominal_rate_hz)
        source.reset()
        assert source.get_frame().frame_index == 0


def test_missing_file_fails_loudly(tmp_path) -> None:
    """A missing evaluator file must be an error, not an empty run."""
    with pytest.raises(FileNotFoundError):
        VideoFrameSource(tmp_path / "nope.mp4")


def test_sidecar_accepts_common_column_spellings(tmp_path) -> None:
    """An evaluator's annotation file need not match our column names exactly."""
    path = tmp_path / "truth.csv"
    path.write_text("frame_index,centroid_x,centroid_y\n0,10.5,20.25\n1,11.5,21.25\n",
                    encoding="utf-8")
    table = load_ground_truth_sidecar(path)
    assert table[0] == (10.5, 20.25)

    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a column"):
        load_ground_truth_sidecar(bad)


# ------------------------------------------------------------------------------------------
# H.264 re-validations
# ------------------------------------------------------------------------------------------


@requires_ffmpeg
def test_generated_videos_are_really_compressed(tmp_path) -> None:
    """Guard against the silent lossless fallback that would invalidate everything here.

    OpenCV's ``VideoWriter`` resolves H.264 to a hardware encoder, fails when no such device
    exists, and silently falls back to MPEG-4 Part 2 -- which produced 78 KB *per frame*,
    essentially lossless. Generation now pipes to software libx264 and this test pins the
    outcome: a genuinely compressed grayscale clip is far smaller than that.
    """
    spec = VideoSpec("compressed", width=640, height=480, frames=40, gaussian_sigma=8.0,
                     bitrate_kbps=2000)
    video, _ = generate_video(spec, tmp_path)
    kb_per_frame = video.stat().st_size / 1024 / spec.frames
    assert kb_per_frame < 40.0, f"{kb_per_frame:.1f} KB/frame suggests a lossless fallback"


@requires_ffmpeg
def test_area_gate_survives_correlated_compression_artifacts(tmp_path) -> None:
    """Re-validate the Tier-0 area gate against H.264, not independent impulses.

    The 273-residual-impulse figure assumed *independent* salt-and-pepper. Compression smears
    impulses across transform blocks into correlated multi-pixel artifacts, and the risk
    direction is unfavourable -- artifacts get larger, toward the beacon's size class.

    Measured, and the direction is confirmed: median residual blob area grows 1.0 -> 2.0 px and
    p90 grows 3.0 -> 4.0 px, so gate rejection falls from 96% to 88%. The margin erodes but the
    gate holds: the fraction of artifacts reaching the beacon's size class stays at 0.7%.
    """
    import cv2

    spec = VideoSpec("gate", width=640, height=480, frames=24, peak=200.0, background=25.0,
                     gaussian_sigma=8.0, sp_density=0.08, bitrate_kbps=1200, speed_px_s=0.0)
    video, _ = generate_video(spec, tmp_path)

    areas = []
    with VideoFrameSource(video) as source:
        for frame_data in source:
            if frame_data.frame_index < 5:
                continue
            working = cv2.medianBlur(frame_data.frame.astype(np.float32), 3)
            median = float(np.median(working))
            mad = 1.4826 * float(np.median(np.abs(working - median)))
            level = max(float(np.percentile(working, 99.9)), median + max(mad, 1e-6))
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                (working >= level).astype(np.uint8), connectivity=8)
            if count > 1:
                areas.extend(stats[1:, cv2.CC_STAT_AREA].tolist())

    assert areas
    array = np.asarray(areas)
    from src.vision.spotscale import BootstrapParams
    gate = BootstrapParams().min_area_px
    assert float(np.mean(array < gate)) > 0.75, "area gate is no longer filtering artifacts"
    assert float(np.mean(array >= 20)) < 0.05, "artifacts reached the beacon size class"


@requires_ffmpeg
@pytest.mark.parametrize("shape,expected", [("gaussian", 1.001), ("square", 0.866)])
def test_rho_cluster_centres_are_stable_under_compression(tmp_path, shape, expected) -> None:
    """Re-validate the rho band: compression alters edge sharpness, which is what rho keys on.

    rho gates nothing, but it is logged every frame as a diagnostic, and a shifted band would
    make the trace misleading rather than merely imprecise. Measured shifts are small:
    +0.002 for a Gaussian and -0.001 for a square at 4000 kbps, growing to -0.018 and -0.010 at
    1000 kbps. The cluster centres hold.
    """
    sigma = 14.0 / 2.3548200450309493 if shape == "gaussian" else 2.5
    spec = VideoSpec(f"rho_{shape}", width=640, height=480, frames=24, shape=shape,
                     size_px=14.0, sigma_px=sigma, peak=220.0, background=25.0,
                     gaussian_sigma=6.0, bitrate_kbps=4000, speed_px_s=0.0)
    video, _ = generate_video(spec, tmp_path)

    ratios = []
    with VideoFrameSource(video) as source:
        for frame_data in source:
            tier0 = bootstrap_scale(frame_data.frame)
            if not tier0.succeeded:
                continue
            seed = (int(round(tier0.candidate.x)), int(round(tier0.candidate.y)))
            tier1 = confirm_scale(frame_data.frame, seed)
            if tier1.succeeded:
                ratios.append(tier1.fwhm_px / tier0.fwhm_px)

    assert len(ratios) > 10
    assert float(np.mean(ratios)) == pytest.approx(expected, abs=0.05)


@requires_ffmpeg
def test_compression_alone_costs_about_a_hundredth_of_a_pixel(tmp_path) -> None:
    """Centroid bias attributable to the codec, with no other noise present.

    This bounds the error floor Benchmark-2 can possibly achieve. Measured median error:
    0.0086 px at 4000 kbps, 0.0115 px at 1000 kbps, 0.0181 px at 500 kbps -- comparable to the
    8-bit quantisation floor of ~0.01 px, and two to three orders of magnitude inside the
    10 px budget.
    """
    config = load_config("config/default.json")
    pipeline = VisionPipeline.from_config(config)

    results = {}
    for bitrate in (4000, 500):
        spec = VideoSpec(f"bias_{bitrate}", width=640, height=480, frames=30, peak=200.0,
                         background=25.0, gaussian_sigma=0.0, sp_density=0.0,
                         bitrate_kbps=bitrate, speed_px_s=40.0)
        video, sidecar = generate_video(spec, tmp_path)
        truth = load_ground_truth_sidecar(sidecar)
        errors = []
        with VideoFrameSource(video) as source:
            for frame_data in source:
                measurement = pipeline.process(frame_data.frame, fwhm_px=5.887,
                                               from_fallback=False)
                if measurement.found and frame_data.frame_index in truth:
                    tx, ty = truth[frame_data.frame_index]
                    distance = math.hypot(measurement.x - tx, measurement.y - ty)
                    if distance < 40:
                        errors.append(distance)
        results[bitrate] = float(np.median(errors))

    assert results[4000] < 0.05
    assert results[500] < 0.10
    assert results[500] >= results[4000], "heavier compression should not improve accuracy"


# ------------------------------------------------------------------------------------------
# Full-canvas throughput
# ------------------------------------------------------------------------------------------


@requires_ffmpeg
@pytest.mark.slow
def test_full_canvas_needs_roi_to_meet_the_fps_requirement(tmp_path, config) -> None:
    """The 2000x2000 risk, measured rather than assumed.

    Decode is cheap (6.6 ms/frame, a 151 FPS ceiling). Full-frame *vision* on 4 Mpx is not:
    292 ms mean, giving 2.9 FPS end to end -- well outside the >=20 FPS requirement. ROI
    processing around the prediction is what closes the gap, and it closes it decisively.

    Reported honestly: **steady-state tracking meets the requirement comfortably on a full
    2000x2000 canvas; the acquisition frames, which must run full-frame, do not.**
    """
    from src.telemetry.benchmark import measure_capacity

    spec = VideoSpec("canvas", width=2000, height=2000, frames=24, size_px=14.0, sigma_px=4.0,
                     peak=230.0, background=15.0, gaussian_sigma=10.0, speed_px_s=80.0)
    video, _ = generate_video(spec, tmp_path)
    pipeline = VisionPipeline.from_config(config)

    with VideoFrameSource(video) as source:
        frames = [frame_data.frame for frame_data in source]

    found = bootstrap_scale(frames[5])
    centre = ((int(found.candidate.x), int(found.candidate.y)) if found.candidate
              else (1000, 1000))
    window = (centre[0] - 64, centre[1] - 64, 128, 128)

    full = measure_capacity(lambda f: pipeline.process(f, fwhm_px=14.0, from_fallback=False),
                            frames, warmup_frames=4)
    roi = measure_capacity(
        lambda f: pipeline.process(f, fwhm_px=14.0, from_fallback=False, roi=window),
        frames, warmup_frames=4)

    assert full.conservative_fps < 20.0, "full-frame unexpectedly fast; re-check the claim"
    assert roi.conservative_fps > 20.0
    assert roi.mean_ms < full.mean_ms / 10.0


# ------------------------------------------------------------------------------------------
# Lock criterion: keyed on source capability, never on mode
# ------------------------------------------------------------------------------------------


@requires_ffmpeg
@pytest.mark.slow
def test_centroiding_is_reported_for_a_target_parked_far_from_frame_centre(tmp_path) -> None:
    """A Mode B target that never approaches frame centre must still report centroiding error.

    Regression test for a real defect. Centroid statistics are computed over *locked* frames, and
    the lock criterion used to require pointing error within 40 px of frame centre. In Mode B
    there is no camera to steer, so a target's distance from centre is a property of the file
    rather than of our tracking -- and a clip whose target sat a median 202 px out reported 0%
    lock retention, which made the summary omit centroiding error **entirely**. That is the one
    quantity Benchmark Performance-2 scores.

    The lock criterion is now selected from ``FrameSource.supports_pan_tilt``, never from a mode
    string, so the metrics logic stays free of mode branching exactly as the vision pipeline does.

    This fixture parks the beacon in a corner for the whole clip, so no frame could ever satisfy a
    pointing window.
    """
    import csv as csv_module
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = VideoSpec("far_corner", width=640, height=480, frames=60, peak=200.0,
                     background=25.0, gaussian_sigma=8.0, speed_px_s=0.0)
    video, sidecar = generate_video(spec, tmp_path)

    truth = load_ground_truth_sidecar(sidecar)
    centre = (spec.width / 2.0, spec.height / 2.0)
    offsets = [math.dist(truth[i], centre) for i in sorted(truth)]
    assert min(offsets) > 100.0, "fixture must never approach frame centre"

    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({
        "run": {"mode": "video"},
        "video_input": {"path": str(video), "ground_truth_path": str(sidecar)},
        "ai": {"enabled": False},
    }), encoding="utf-8")

    workdir = tmp_path / "run"
    workdir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--config", str(root / "config" / "default.json"),
         "--scenario", str(scenario), "--headless"],
        cwd=workdir, capture_output=True, text=True, check=False,
        env=dict(os.environ, PYTHONPATH=str(root)))
    assert result.returncode == 0, result.stdout + result.stderr

    rows = list(csv_module.DictReader(
        line for line in (workdir / "logs" / "frames.csv").read_text(
            encoding="utf-8").splitlines() if not line.startswith("#")))
    locked = [r for r in rows if r["locked"] == "True"]
    errors = [float(r["centroid_error_px"]) for r in locked if r["centroid_error_px"]]

    assert locked, "no frames locked: the pointing window is still gating Mode B"
    assert errors, "centroiding error was not reported for any locked frame"
    assert float(np.median(errors)) < 1.0

    # And the report must actually carry the number an evaluator reads.
    report = (workdir / "logs" / "report.html").read_text(encoding="utf-8")
    assert "centroiding RMSE" in report
    assert "&mdash;</td>" not in report.split("centroiding RMSE")[1][:120], \
        "centroiding RMSE rendered as empty in the report"


def test_lock_criterion_follows_source_capability_not_mode() -> None:
    """The switch is ``supports_pan_tilt``; no mode string reaches the metrics logic."""
    from src.control.statemachine import StateMachineParams, TrackingStateMachine, TrackState

    far_from_centre = 500.0

    steerable = TrackingStateMachine(
        StateMachineParams(lock_confirm_frames=2, require_pointing_window=True),
        target_initially_in_fov=True)
    for index in range(5):
        steerable.update(True, far_from_centre, index / 30.0, 1 / 30.0)
    assert steerable.state is TrackState.SEARCH, "pointing window should gate a steerable source"

    fixed = TrackingStateMachine(
        StateMachineParams(lock_confirm_frames=2, require_pointing_window=False),
        target_initially_in_fov=True)
    for index in range(5):
        fixed.update(True, far_from_centre, index / 30.0, 1 / 30.0)
    assert fixed.state is TrackState.TRACK, "no-camera source must not gate on pointing error"


def test_both_lock_criteria_appear_in_the_log_header(config) -> None:
    """A Benchmark-2 log must state on its face why its lock criterion differs from Mode A's.

    Same reasoning as the acquisition clock: an evaluator should never have to infer a definition.
    """
    from src.telemetry.logger import build_header

    steerable = build_header(config, "simulation", steerable_camera=True)
    fixed = build_header(config, "video", steerable_camera=False)

    assert steerable["steerable_camera"] is True
    assert fixed["steerable_camera"] is False

    steerable_text = steerable["metric_definitions"]["lock_criterion"]
    fixed_text = fixed["metric_definitions"]["lock_criterion"]
    assert "pointing error" in steerable_text
    assert "EXCLUDED" in fixed_text
    assert steerable_text != fixed_text

    for header in (steerable, fixed):
        rationale = header["metric_definitions"]["lock_criterion_rationale"]
        assert "supports_pan_tilt" in rationale
        assert "never from a mode string" in rationale


# ------------------------------------------------------------------------------------------
# Reproducibility of the generated fixtures
# ------------------------------------------------------------------------------------------


@requires_ffmpeg
def test_generated_video_is_bit_reproducible(tmp_path) -> None:
    """The same spec must produce the same bytes twice.

    Seeding the scene RNG is necessary but not sufficient. x264 slices frames across threads and
    the slice boundaries follow thread scheduling, so two encodes of byte-identical input
    produced *different decoded pixels* -- up to 255 levels on the salt-and-pepper clip, and
    differing on 149 of 150 frames on the baseline clip.

    This is not a cosmetic concern. It means a Benchmark-2 figure cannot be reproduced from one
    run to the next, and any regression test over generated video is partly measuring the
    encoder's thread scheduling. The generator therefore pins ``-threads 1`` and x264's
    ``deterministic=1``, and this test is what keeps those flags from being dropped.
    """
    spec = VideoSpec("repro", width=320, height=240, frames=12, sp_density=0.08,
                     bitrate_kbps=1000)
    first, first_truth = generate_video(spec, tmp_path / "a")
    second, second_truth = generate_video(spec, tmp_path / "b")

    assert first.read_bytes() == second.read_bytes(), (
        "regenerating the same spec produced different video bytes; results over generated "
        "fixtures are not reproducible run to run")
    assert first_truth.read_text() == second_truth.read_text()
