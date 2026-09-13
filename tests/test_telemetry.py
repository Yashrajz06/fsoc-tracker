"""Tests for metrics, logging, capacity benchmarking and the summary report.

Metric definitions are tested against synthetic sequences with *known* answers rather than
plausible ones. A run that locks at frame 30 at 30 Hz must report exactly 1.0 s -- not 0.97, not
1.03 -- because the acquisition clock's start and stop are a scoring ambiguity and an
off-by-one-frame convention would be invisible in any test that only checked "about a second".
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.telemetry.benchmark import measure_capacity
from src.telemetry.logger import TelemetryLogger, build_header, known_limits
from src.telemetry.metrics import FrameRecord, MetricsAccumulator
from src.telemetry.report import write_report

RATE = 30.0


@pytest.fixture(scope="module")
def config():
    """The validated default configuration."""
    return load_config("config/default.json")


def _feed(accumulator: MetricsAccumulator, locked_from: int, frames: int = 120,
          detected_from: int = 0, **kwargs) -> None:
    """Feed a synthetic sequence with a known lock onset."""
    for index in range(frames):
        detected = index >= detected_from
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=detected,
                                    locked=detected and index >= locked_from, **kwargs))


# ------------------------------------------------------------------------------------------
# Acquisition timing -- exact, not approximate
# ------------------------------------------------------------------------------------------


def test_acquisition_at_frame_30_reports_exactly_one_second() -> None:
    """K=3 consecutive locks ending at frame 30, at 30 Hz, is exactly 1.0 s.

    Exactness matters: the clock starts at the first frame of the run and stops on the frame the
    criterion completes, so an off-by-one in either bound shifts every acquisition number by
    33 ms. A tolerance-based test would not notice.
    """
    accumulator = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=True)
    _feed(accumulator, locked_from=28)
    event = accumulator.acquisitions[0]
    assert event.frame_index == 30
    assert event.duration_s == 1.0
    assert accumulator.summary().acquisition_in_fov_s == 1.0


@pytest.mark.parametrize("lock_frame,expected_s", [(3, 0.1), (15, 0.5), (30, 1.0), (60, 2.0)])
def test_acquisition_timing_is_exact_at_several_points(lock_frame: int,
                                                       expected_s: float) -> None:
    """The clock must be exact across the range, not merely at one convenient point."""
    accumulator = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=True)
    _feed(accumulator, locked_from=lock_frame - 2)
    assert accumulator.acquisitions[0].duration_s == pytest.approx(expected_s, abs=1e-12)


def test_acquisition_requires_k_consecutive_frames() -> None:
    """Intermittent locks must not satisfy the criterion."""
    accumulator = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=True)
    for index in range(60):
        locked = index % 2 == 0          # never three in a row
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=locked))
    assert not accumulator.acquisitions


def test_acquisition_populations_are_never_pooled() -> None:
    """In-FOV and search-limited are reported as separate fields.

    Pooling them hides a physical limit behind an initial-condition lottery: the search-limited
    case is bounded below by the slew ceiling at 11.6 s for the default canvas.
    """
    in_fov = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=True)
    _feed(in_fov, locked_from=28)
    summary = in_fov.summary()
    assert summary.acquisition_in_fov_s == 1.0
    assert summary.acquisition_search_limited_s is None

    searched = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=False)
    _feed(searched, locked_from=28)
    summary = searched.summary()
    assert summary.acquisition_search_limited_s == 1.0
    assert summary.acquisition_in_fov_s is None


def test_reacquisition_clock_starts_at_declared_loss() -> None:
    """The clock starts after N consecutive misses, and stops when the criterion is re-met."""
    accumulator = MetricsAccumulator(lock_confirm_frames=3, loss_declare_frames=5,
                                     target_initially_in_fov=True)
    for index in range(20):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=True))
    # Frames 20-29 missed; loss declared on frame 24 (the fifth consecutive miss).
    for index in range(20, 30):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=False, locked=False))
    for index in range(30, 40):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=True))

    reacqs = [a for a in accumulator.acquisitions if a.reacquisition]
    assert len(reacqs) == 1
    # Loss declared at frame 24, re-lock completes at frame 32: 8 frames = 0.2666... s.
    assert reacqs[0].frame_index == 32
    assert reacqs[0].duration_s == pytest.approx(8.0 / RATE, abs=1e-12)


def test_reacquisition_is_never_search_limited() -> None:
    """After a loss the last position is known, so re-acquisition is local by definition."""
    accumulator = MetricsAccumulator(lock_confirm_frames=3, loss_declare_frames=3,
                                     target_initially_in_fov=False)
    for index in range(10):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=True))
    for index in range(10, 16):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=False, locked=False))
    for index in range(16, 24):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=True))
    reacqs = [a for a in accumulator.acquisitions if a.reacquisition]
    assert reacqs and all(a.population == "in_fov" for a in reacqs)


# ------------------------------------------------------------------------------------------
# Error and rate metrics
# ------------------------------------------------------------------------------------------


def test_rmse_matches_its_definition_exactly() -> None:
    """RMSE is sqrt(mean(error^2)) over frames where lock was held."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    errors = [3.0, 4.0, 0.0, 5.0]
    for index, error in enumerate(errors):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=True, centroid_error_px=error))
    summary = accumulator.summary()
    assert summary.centroid_rmse_px == pytest.approx(math.sqrt(sum(e * e for e in errors) / 4))
    assert summary.centroid_mean_px == pytest.approx(3.0)
    assert summary.centroid_max_px == pytest.approx(5.0)


def test_unlocked_frames_are_excluded_from_error_statistics() -> None:
    """The RMSE definition is explicit that it covers frames where lock was held."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    accumulator.add(FrameRecord(frame_index=0, timestamp=0.0, detected=True, locked=True,
                                centroid_error_px=1.0))
    accumulator.add(FrameRecord(frame_index=1, timestamp=1 / RATE, detected=True, locked=False,
                                centroid_error_px=99.0))
    assert accumulator.summary().centroid_max_px == pytest.approx(1.0)


def test_centroiding_and_pointing_are_reported_separately() -> None:
    """The spec's "tracking error" is ambiguous between them, so both are kept and labelled."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(10):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, centroid_error_px=0.5,
                                    pointing_error_px=7.0))
    summary = accumulator.summary()
    assert summary.centroid_mean_px == pytest.approx(0.5)
    assert summary.pointing_mean_px == pytest.approx(7.0)
    assert summary.centroid_mean_px != summary.pointing_mean_px


def test_loss_rate_matches_its_definition() -> None:
    """Frames without valid lock divided by frames where the target was present."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(10):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE,
                                    detected=True, locked=index < 8, target_present=True))
    summary = accumulator.summary()
    assert summary.loss_rate == pytest.approx(0.2)
    assert summary.lock_retention == pytest.approx(0.8)


def test_frames_without_a_target_are_excluded_from_loss_rate() -> None:
    """The denominator is frames where the target was present, by definition."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(10):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, target_present=index < 5))
    assert accumulator.summary().loss_rate == pytest.approx(0.0)


def test_three_clocks_are_kept_separate() -> None:
    """Camera rate, control rate and processing FPS are three different quantities."""
    accumulator = MetricsAccumulator(camera_rate_hz=30.0, control_rate_hz=20.0,
                                     lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(20):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, processing_ms=5.0))
    summary = accumulator.summary()
    assert summary.camera_rate_hz == 30.0
    assert summary.control_rate_hz == 20.0
    assert summary.processing_fps_mean == pytest.approx(200.0)


def test_attribution_columns_are_totalled() -> None:
    """Every attribution flag must reach the summary; they are the diagnosis path."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(10):
        accumulator.add(FrameRecord(
            frame_index=index, timestamp=index / RATE, detected=True, locked=True,
            clipped=index < 3, saturated=index < 2, from_fallback=index < 4,
            gated_out=index < 5, slew_saturated=index < 6,
            track_reason="lockout_reinitiated" if index < 1 else "updated", nis=2.0))
    summary = accumulator.summary()
    assert (summary.frames_clipped, summary.frames_saturated) == (3, 2)
    assert (summary.frames_from_fallback, summary.frames_gated_out) == (4, 5)
    assert summary.frames_slew_saturated == 6
    assert summary.lockout_events == 1
    assert summary.nis_mean == pytest.approx(2.0)


def test_empty_run_summarises_without_error() -> None:
    """A run that produced nothing must still summarise, not raise."""
    summary = MetricsAccumulator().summary()
    assert summary.total_frames == 0
    assert summary.centroid_rmse_px is None


# ------------------------------------------------------------------------------------------
# Log header
# ------------------------------------------------------------------------------------------


def test_header_carries_definitions_and_convention(config) -> None:
    """The ambiguities the spec leaves open must be closed in the log itself."""
    header = build_header(config)
    assert "pixel centres at integer indices" in header["coordinate_convention"].lower()
    definitions = header["metric_definitions"]
    for key in ("acquisition_time_s", "reacquisition_time_s", "centroid_error_px",
                "pointing_error_px", "lock_criterion", "loss_rate"):
        assert key in definitions and definitions[key].strip()
    assert "Never pooled" in definitions["acquisition_time_s"]


def test_header_separates_the_three_clocks(config) -> None:
    """Conflating them is the specific failure the header exists to prevent."""
    clocks = build_header(config)["clocks"]
    assert clocks["camera_update_hz"] == config.camera.update_rate_hz
    assert clocks["control_update_hz"] == config.control.update_rate_hz
    assert "understates" in clocks["processing_fps"]


def test_known_limits_carry_provenance(config) -> None:
    """An envelope figure must never be mistakable for a measurement from this run.

    Values computed from the active configuration say so; values measured elsewhere name the
    configuration they were measured at, because they do not automatically transfer.
    """
    limits = known_limits(config)
    for name, entry in limits.items():
        assert entry["provenance"], name
        assert entry["note"], name
    assert limits["search_coverage_time_s"]["value"] == pytest.approx(11.6, abs=0.2)
    assert "computed for this configuration" in limits["search_coverage_time_s"]["provenance"]
    assert "default configuration" in limits["max_trackable_velocity_px_s"]["provenance"]


def test_search_coverage_limit_tracks_the_active_configuration(config) -> None:
    """The computed limits must follow the config, not be stale constants."""
    from dataclasses import replace

    bigger = replace(config, scene=replace(config.scene, width=4000, height=4000))
    assert known_limits(bigger)["search_coverage_time_s"]["value"] > \
        known_limits(config)["search_coverage_time_s"]["value"]


# ------------------------------------------------------------------------------------------
# Output files
# ------------------------------------------------------------------------------------------


def _populated_logger(config, tmp_path: Path) -> TelemetryLogger:
    """Build a logger with a short synthetic run in it."""
    logger = TelemetryLogger(config, output_dir=tmp_path)
    for index in range(60):
        locked = index >= 28
        logger.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=locked,
                               locked=locked, state="track" if locked else "search",
                               centroid_error_px=0.4 if locked else None,
                               pointing_error_px=5.0 if locked else None,
                               processing_ms=6.0, snr_aperture=55.0, rho=1.0,
                               track_reason="updated" if locked else "initiating"))
    return logger


def test_csv_has_every_attribution_column(config, tmp_path: Path) -> None:
    """The trace is only useful if the columns are actually there."""
    logger = _populated_logger(config, tmp_path)
    path = logger.write_csv()
    rows = [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]
    columns = rows[0].split(",")
    for name in ("clipped", "saturated", "from_fallback", "rho", "scale_reason",
                 "track_reason", "gated_out", "nis", "centroid_error_px",
                 "pointing_error_px", "slew_saturated"):
        assert name in columns, name
    assert len(rows) == 61  # header plus 60 frames


def test_csv_header_is_machine_readable_json(config, tmp_path: Path) -> None:
    """Comment-prefixed JSON, so a plain CSV reader skips it but nothing is lost."""
    logger = _populated_logger(config, tmp_path)
    text = logger.write_csv().read_text(encoding="utf-8")
    comment = "\n".join(line[2:] for line in text.splitlines() if line.startswith("# "))
    header = json.loads(comment)
    assert "metric_definitions" in header and "known_limits" in header


def test_json_run_document_round_trips(config, tmp_path: Path) -> None:
    """The JSON log must carry header, summary, acquisitions and frames together."""
    logger = _populated_logger(config, tmp_path)
    document = json.loads(logger.write_json().read_text(encoding="utf-8"))
    assert set(document) == {"header", "summary", "acquisitions", "frames"}
    assert len(document["frames"]) == 60
    assert document["summary"]["acquisition_in_fov_s"] is not None or \
        document["summary"]["acquisition_search_limited_s"] is not None


def test_report_separates_results_from_known_limits(config, tmp_path: Path) -> None:
    """Section 3 must be structurally distinct, so an envelope cannot read as a measurement."""
    logger = _populated_logger(config, tmp_path)
    text = write_report(logger, config).read_text(encoding="utf-8")
    assert "1. Results measured in this run" in text
    assert "2. Attribution" in text
    assert "3. Known limits of this system" in text
    assert "are <b>not</b> measurements from this run" in text
    assert "4. Metric definitions in force" in text
    assert "5. Configuration snapshot" in text


def test_report_states_both_error_definitions(config, tmp_path: Path) -> None:
    """Benchmark-2 scores centroiding; the report must say so rather than leave it implied."""
    logger = _populated_logger(config, tmp_path)
    text = write_report(logger, config).read_text(encoding="utf-8")
    assert "centroiding" in text.lower() and "pointing" in text.lower()
    assert "Benchmark-2" in text


def test_report_is_generated_with_no_manual_steps(config, tmp_path: Path) -> None:
    """A mandatory deliverable: the report must fall out of a run automatically."""
    logger = _populated_logger(config, tmp_path)
    path = write_report(logger, config)
    assert path.exists() and path.stat().st_size > 2000


# ------------------------------------------------------------------------------------------
# Capacity benchmark
# ------------------------------------------------------------------------------------------


def test_capacity_measures_throughput_not_the_real_time_rate() -> None:
    """Capacity must reflect the work done, not the clock a producer happens to run at."""
    frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(60)]
    calls = {"n": 0}

    def work(_frame):
        calls["n"] += 1

    result = measure_capacity(work, frames, warmup_frames=10, target_fps=20.0)
    assert result.frames == 50
    assert calls["n"] == 60          # warm-up ran too
    assert result.max_sustainable_fps > 1000.0
    assert result.meets_target


def test_capacity_reports_the_tail_not_just_the_mean() -> None:
    """The p95 is what a real-time loop lives or dies by; the mean hides the frames it drops."""
    frames = [np.zeros((4, 4), dtype=np.uint8) for _ in range(40)]
    state = {"n": 0}

    def work(_frame):
        state["n"] += 1
        if state["n"] % 10 == 0:
            time_sink = np.zeros((300, 300))
            time_sink @ time_sink.T

    result = measure_capacity(work, frames, warmup_frames=5)
    assert result.p95_ms >= result.p50_ms
    assert result.max_ms >= result.p95_ms
    assert result.conservative_fps <= result.max_sustainable_fps


def test_capacity_requires_frames_beyond_warmup() -> None:
    """Timing nothing is a caller error, not an infinite frame rate."""
    with pytest.raises(ValueError, match="warm-up"):
        measure_capacity(lambda f: None, [np.zeros((4, 4), np.uint8)], warmup_frames=10)


def test_capacity_result_serialises(config) -> None:
    """The result goes into the report and the JSON log."""
    frames = [np.zeros((4, 4), dtype=np.uint8) for _ in range(30)]
    payload = measure_capacity(lambda f: None, frames, warmup_frames=5).as_dict()
    assert payload["frames"] == 25
    assert "conservative_fps" in payload and "meets_target" in payload


# ------------------------------------------------------------------------------------------
# Median / p95 / RMSE together, and independent verifiability
# ------------------------------------------------------------------------------------------


def test_summary_reports_median_p95_and_rmse_together() -> None:
    """Each statistic answers a different question, so quoting one alone misleads.

    With a sub-pixel body and a 5% catastrophic tail: median 0.100 px (precision while locked),
    p95 10.1 px (where the tail starts), RMSE 44.7 px (failures exist). Reporting only RMSE
    understates the system; reporting only median hides real losses.
    """
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    errors = [0.1] * 95 + [200.0] * 5
    for index, error in enumerate(errors):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, centroid_error_px=error))
    summary = accumulator.summary()

    assert summary.centroid_median_px == pytest.approx(0.1)
    assert summary.centroid_p95_px == pytest.approx(10.095, abs=0.01)
    assert summary.centroid_rmse_px == pytest.approx(44.7, abs=0.1)
    assert summary.centroid_median_px < summary.centroid_p95_px < summary.centroid_rmse_px


def test_pointing_error_also_carries_median_and_p95() -> None:
    """Both error definitions get the same treatment, since both are logged and labelled."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index, error in enumerate([1.0] * 90 + [50.0] * 10):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, pointing_error_px=error))
    summary = accumulator.summary()
    assert summary.pointing_median_px == pytest.approx(1.0)
    assert summary.pointing_p95_px is not None and summary.pointing_p95_px > 1.0


def test_csv_permits_independent_recomputation_of_every_statistic(config,
                                                                  tmp_path: Path) -> None:
    """An evaluator must be able to verify the summary rather than trust it.

    The report states that all three statistics can be recomputed from ``frames.csv``. This test
    performs exactly that recomputation from the file alone -- no access to the accumulator --
    and requires an exact match, so the claim cannot quietly become false.
    """
    logger = TelemetryLogger(config, output_dir=tmp_path)
    rng = np.random.default_rng(5)
    for index in range(200):
        locked = index >= 10
        logger.add(FrameRecord(
            frame_index=index, timestamp=index / RATE, detected=locked, locked=locked,
            target_present=True,
            centroid_error_px=(float(abs(rng.normal(0, 0.2))) if index % 40 else 90.0)
            if locked else None,
            pointing_error_px=float(abs(rng.normal(5, 1))) if locked else None))
    path = logger.write_csv()
    summary = logger.summary()

    rows = list(csv.DictReader(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")))
    # Exactly the rule the report states: statistics are over frames where locked is true.
    errors = np.asarray([float(r["centroid_error_px"]) for r in rows
                         if r["locked"] == "True" and r["centroid_error_px"]])
    assert errors.size > 100

    assert float(np.median(errors)) == pytest.approx(summary.centroid_median_px)
    assert float(np.percentile(errors, 95)) == pytest.approx(summary.centroid_p95_px)
    assert float(errors.mean()) == pytest.approx(summary.centroid_mean_px)
    assert float(errors.max()) == pytest.approx(summary.centroid_max_px)
    assert float(np.sqrt((errors ** 2).mean())) == pytest.approx(summary.centroid_rmse_px)


def test_report_explains_the_median_rmse_divergence(config, tmp_path: Path) -> None:
    """An evaluator comparing teams' RMSE figures should not have to derive why ours diverges."""
    logger = TelemetryLogger(config, output_dir=tmp_path)
    for index in range(100):
        error = 0.1 if index % 20 else 150.0
        logger.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                               locked=True, centroid_error_px=error,
                               pointing_error_px=4.0, processing_ms=6.0))
    text = write_report(logger, config).read_text(encoding="utf-8")

    assert "centroiding median (system output)" in text
    assert "centroiding p95" in text
    assert "Reading the spread" in text
    assert "Why there are two error columns" in text
    assert "&times; the median" in text, "divergence should be quantified, not just described"
    assert "lock retention" in text
    assert "frames.csv" in text, "the report must say where to verify the numbers"


def test_report_note_adapts_when_there_is_no_tail(config, tmp_path: Path) -> None:
    """A clean run should not carry a warning about a tail it does not have."""
    logger = TelemetryLogger(config, output_dir=tmp_path)
    for index in range(100):
        logger.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                               locked=True, centroid_error_px=0.1, pointing_error_px=4.0,
                               processing_ms=6.0))
    text = write_report(logger, config).read_text(encoding="utf-8")
    assert "no significant" in text
    assert "&times; the median" not in text


# ------------------------------------------------------------------------------------------
# Two error columns, doing separate jobs
# ------------------------------------------------------------------------------------------


def test_injected_association_failures_are_reported_not_absorbed() -> None:
    """The fused output stays sub-pixel while the association-failure rate reports the truth.

    Pins both columns doing their separate jobs. A synthetic run injects association failures at
    a known rate: on those frames the raw detection lands far away and the validation gate
    rejects it, so the filter coasts and its prediction remains accurate.

    This is the exact shape of the real defect it guards against. On a low-SNR clip, scoring the
    *raw* measurement as the system's output reported 295 px RMSE where the true output error was
    0.27 px, because 34 of 131 "locked" frames carried a measurement the gate had rejected with a
    median NIS of 56,000 against a threshold of 9.21.
    """
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    total, every_nth = 200, 5          # a 20% injected association-failure rate
    for index in range(total):
        failed = index % every_nth == 0
        accumulator.add(FrameRecord(
            frame_index=index, timestamp=index / RATE, detected=True, locked=True,
            gated_out=failed,
            estimate_source="predicted" if failed else "measured",
            # The raw measurement is catastrophically wrong on a failed association...
            detection_error_px=533.0 if failed else 0.20,
            # ...while the filter's output stays accurate, because it rejected that measurement.
            centroid_error_px=0.35 if failed else 0.20,
            nis=56_000.0 if failed else 0.1))

    summary = accumulator.summary()

    # The rate is reported exactly, and is computable without ground truth.
    assert summary.association_failure_rate == pytest.approx(1.0 / every_nth)
    assert summary.frames_predicted == total // every_nth
    assert summary.frames_measured == total - total // every_nth

    # The system's output stays sub-pixel despite a fifth of detections being unusable.
    assert summary.centroid_median_px == pytest.approx(0.20)
    assert summary.centroid_p95_px is not None and summary.centroid_p95_px < 1.0
    assert summary.centroid_rmse_px is not None and summary.centroid_rmse_px < 1.0

    # And the failures remain visible in the raw column rather than being smoothed away.
    assert summary.detection_rmse_px is not None and summary.detection_rmse_px > 100.0
    assert summary.detection_p95_px is not None and summary.detection_p95_px > 100.0


def test_estimate_source_separates_measured_from_predicted_frames(config,
                                                                  tmp_path: Path) -> None:
    """A coasted frame scoring its prediction is correct, but must be identifiable as such.

    That separation is what makes the fused number verifiable rather than merely better-looking.
    """
    logger = TelemetryLogger(config, output_dir=tmp_path)
    for index in range(50):
        predicted = index % 10 == 0
        logger.add(FrameRecord(
            frame_index=index, timestamp=index / RATE, detected=True, locked=True,
            gated_out=predicted,
            estimate_source="predicted" if predicted else "measured",
            centroid_error_px=0.3, detection_error_px=400.0 if predicted else 0.3))

    rows = list(csv.DictReader(
        line for line in logger.write_csv().read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")))
    assert "estimate_source" in rows[0]
    assert "detection_error_px" in rows[0]

    predicted_rows = [r for r in rows if r["estimate_source"] == "predicted"]
    measured_rows = [r for r in rows if r["estimate_source"] == "measured"]
    assert len(predicted_rows) == 5 and len(measured_rows) == 45
    # An evaluator can recompute the fused statistic over measured frames only.
    assert all(float(r["detection_error_px"]) > 100.0 for r in predicted_rows)


def test_report_explains_why_two_error_columns_exist(config, tmp_path: Path) -> None:
    """Without the note, two error columns look like we picked the flattering one."""
    logger = TelemetryLogger(config, output_dir=tmp_path)
    for index in range(100):
        failed = index % 5 == 0
        logger.add(FrameRecord(
            frame_index=index, timestamp=index / RATE, detected=True, locked=True,
            gated_out=failed, estimate_source="predicted" if failed else "measured",
            centroid_error_px=0.3, detection_error_px=533.0 if failed else 0.3,
            pointing_error_px=4.0, processing_ms=6.0))
    raw = write_report(logger, config).read_text(encoding="utf-8")
    # Normalise whitespace: the template wraps, so phrases span line breaks in the output.
    text = " ".join(raw.split())

    assert "Why there are two error columns" in text
    assert "association failure rate" in text
    assert "correctly rejected" in text
    assert "estimate_source" in text
    assert "&times; the system output" in text, "the gap should be quantified"


# ------------------------------------------------------------------------------------------
# Zero-lock runs
# ------------------------------------------------------------------------------------------


def test_zero_lock_run_states_why_rather_than_rendering_blanks(config,
                                                               tmp_path: Path) -> None:
    """A run that never locked must explain itself, not present an empty table.

    "No centroiding statistics" and "the software produced nothing" look identical in a report
    full of blanks, and only one of them is a defect. On the clip that motivated this, 149 of 150
    detections were off-target -- compression artifacts outranking a dim beacon -- so no-lock was
    the honest outcome, but the report said nothing at all.
    """
    logger = TelemetryLogger(config, output_dir=tmp_path, mode="video",
                             steerable_camera=False)
    for index in range(150):
        logger.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                               locked=False, target_present=True,
                               gated_out=index % 100 != 0, snr_aperture=15.9,
                               detection_error_px=282.0, estimate_source="predicted",
                               processing_ms=6.0))
    summary = logger.summary()
    assert summary.below_characterised_envelope

    note = summary.envelope_note(15.9)
    assert note is not None
    assert "No frames met the lock criterion" in note
    assert "association failure rate" in note
    assert "15.9" in note and "SNR 10 detection-limited boundary" in note
    assert "below the envelope this system is characterised for" in note

    text = write_report(logger, config).read_text(encoding="utf-8")
    assert "Input below characterised envelope" in text


def test_envelope_note_warns_when_snr_is_measured_at_an_artifact(config) -> None:
    """A high SNR on a failing run is the artifact's, not the target's, and must say so.

    SNR is measured at the *detected* position. When the detector is on a bright compression
    artifact the figure describes the artifact, and reading it as evidence the target was
    detectable inverts the diagnosis.
    """
    accumulator = MetricsAccumulator(lock_confirm_frames=3, target_initially_in_fov=False)
    for index in range(60):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=False, target_present=True, gated_out=True))
    note = accumulator.summary().envelope_note(median_snr=15.9)
    assert note is not None and "describes the artifact" in note

    low = accumulator.summary().envelope_note(median_snr=4.0)
    assert low is not None and "describes the artifact" not in low
    assert "below the SNR 10" in low


def test_a_run_that_locks_gets_no_envelope_note(config) -> None:
    """The banner must not appear on a healthy run."""
    accumulator = MetricsAccumulator(lock_confirm_frames=1, target_initially_in_fov=True)
    for index in range(20):
        accumulator.add(FrameRecord(frame_index=index, timestamp=index / RATE, detected=True,
                                    locked=True, target_present=True, centroid_error_px=0.3))
    summary = accumulator.summary()
    assert not summary.below_characterised_envelope
    assert summary.envelope_note(50.0) is None
