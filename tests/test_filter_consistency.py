"""The filter must not be worse than the measurements it smooths.

Every filtering test before this one checked the filter against *itself*: that it converged, that
it gated outliers, that its NIS was plausible. All of them passed while the filter was degrading
its own input by a factor of 17 on the cleanest scenario we have, because none of them ever
compared the fused output to the raw measurement stream.

The defect was process noise set an order of magnitude too low (``q = 50``, implying a target
manoeuvring at ~7 px/s^2, against a default circular trajectory that manoeuvres at 36 px/s^2).
Too small a ``q`` does not simply make the filter sluggish. It shrinks ``P``, which shrinks the
innovation covariance, which closes the Mahalanobis gate -- so when a detection dropout made the
constant-velocity prediction drift along the curve, the returning detections were rejected as
outliers and the filter coasted further instead of correcting. Fused RMSE 0.95 px against a raw
measurement RMSE of 0.056 px.

The lesson these tests encode: a state estimator's output must be scored against its input, not
only against its own internal consistency statistics. Note that this fails *loudest* on clean,
high-SNR inputs, which is exactly where a filter is least expected to be scrutinised.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.runner import TrackingRunner


def _run(tmp_path: Path, **overrides) -> list:
    """Run a headless Mode A scenario and return the per-frame telemetry records.

    The duration matters. This failure is event-driven rather than steady-state: it needs a
    detection dropout to occur on a curved segment, so the prediction has something to diverge
    along. Measured at the defective ``q = 50``, a 6 s window showed a harmless-looking ratio of
    1.1 and a 8 s window showed 16.9. A short test would have passed against a badly broken
    filter, which is close to how this survived undetected in the first place.
    """
    base = {"run": {"duration_seconds": 10.0},
            "target": {"initial_position": "center"},
            "noise": {"camera_jitter": {"enabled": False}}}
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps(base))
    config = load_config("config/default.json", overrides=[str(override_path)])
    runner = TrackingRunner(config)
    records = []
    runner.run(on_frame=lambda outcome: records.append(outcome.record),
               should_stop=lambda: False)
    return records


def _rmse(values) -> float:
    return math.sqrt(float(np.mean(np.square(values)))) if values else float("nan")


def _errors(records):
    fused = [r.centroid_error_px for r in records if r.centroid_error_px is not None]
    raw = [r.detection_error_px for r in records if r.detection_error_px is not None]
    return raw, fused


def test_fused_estimate_is_not_worse_than_the_raw_measurement(tmp_path):
    """On a clean run the filter must improve on its input, or at worst roughly match it.

    This is the assertion that was missing. At ``q = 50`` the ratio was 17; the tolerance below
    is deliberately loose, because the point is to catch a filter that has become actively
    harmful, not to pin a specific accuracy.
    """
    raw, fused = _errors(_run(tmp_path))
    assert raw and fused, "scenario produced no scored frames"
    raw_rmse, fused_rmse = _rmse(raw), _rmse(fused)
    assert fused_rmse < 1.5 * raw_rmse, (
        f"filter degraded its own input: fused RMSE {fused_rmse:.4f} px vs raw measurement "
        f"RMSE {raw_rmse:.4f} px (ratio {fused_rmse / raw_rmse:.1f})")


def test_clean_run_does_not_gate_out_good_measurements(tmp_path):
    """A high-SNR run with no jitter should reject almost nothing.

    The gate exists to reject impulse false positives. On a clean scenario there are none, so
    sustained rejections mean the gate is closing on correct measurements -- the mechanism by
    which low process noise turns into unbounded coasting error.
    """
    records = _run(tmp_path)
    gated = sum(1 for r in records if r.gated_out)
    assert gated / len(records) < 0.05, (
        f"{gated}/{len(records)} frames gated out on a clean run; the validation gate is "
        f"rejecting correct measurements")


def test_process_noise_is_wired_from_configuration(tmp_path):
    """``filtering.kalman`` must reach the filter.

    It did not: ``process_noise_psd`` was validated by config and read by nobody, so the filter
    used the dataclass default. It agreed with the JSON only by coincidence, and editing the
    scenario file -- what an evaluator does -- changed nothing.
    """
    override_path = tmp_path / "q.json"
    override_path.write_text(json.dumps({"filtering": {"kalman": {"process_noise_psd": 777.0}}}))
    config = load_config("config/default.json", overrides=[str(override_path)])
    assert config.filtering.kalman_params().process_noise_psd == pytest.approx(777.0)
    assert TrackingRunner(config).track.filter.params.process_noise_psd == pytest.approx(777.0)


# ------------------------------------------------------------------------------------------
# Wiring: every filtering knob must reach the filter and change its behaviour
# ------------------------------------------------------------------------------------------
#
# These are mutation tests, not reference-count checks. Each sets a knob to a value that *must*
# change what the filter does and asserts the change is observable. A static search cannot prove
# this: config binds JSON keys to dataclass fields by name, so a live knob's literal may never
# appear in the source, and a dead one can still be validated on load. ``process_noise_psd`` was
# validated by config and read by nobody, and agreed with the filter only by coincidence.


def _params(**kalman):
    """Build filter params from a config whose ``filtering`` block carries ``kalman``."""
    from src.config import AppConfig

    raw = json.loads(Path("config/default.json").read_text())
    raw["filtering"]["kalman"].update(
        {k: v for k, v in kalman.items() if k != "mahalanobis_threshold"})
    if "mahalanobis_threshold" in kalman:
        raw["filtering"]["gating"]["mahalanobis_threshold"] = kalman["mahalanobis_threshold"]
    return AppConfig.from_dict(raw).filtering.kalman_params()


def test_mahalanobis_threshold_changes_what_the_gate_accepts():
    """The gate threshold must reach the gate.

    This is the knob most likely to be edited in an evaluator scenario -- it trades false
    detections against rejected good measurements -- and it is the mechanism behind the worst
    defect found so far, where a too-tight effective gate rejected 22 of 240 correct
    measurements. A scenario loosening it must actually loosen it.
    """
    from src.filtering.kalman import KalmanFilterCV

    def accepts(threshold: float) -> bool:
        # The initial covariance has to be tightened deliberately. At the shipped defaults
        # P[0, 0] is 2500 px^2, so a 3 px innovation scores NIS 0.004 and every threshold in
        # range accepts it -- a test built on the defaults would pass whatever the gate did.
        kf = KalmanFilterCV(_params(mahalanobis_threshold=threshold,
                                    initial_position_uncertainty_px=1.0,
                                    initial_velocity_uncertainty_px_s=1.0,
                                    process_noise_psd=1.0))
        kf.initialise(100.0, 100.0, 0.0, 0.0)
        kf.predict(1.0 / 30.0)
        return kf.gate((103.0, 100.0), sigma_meas_px=1.0).accepted

    assert not accepts(0.5), "a very tight gate accepted a 3-sigma-plus innovation"
    assert accepts(500.0), "a very loose gate still rejected the same innovation"


def test_initial_uncertainty_knobs_change_the_starting_covariance():
    """``initial_*_uncertainty`` must set the initial covariance, not decorate the JSON."""
    from src.filtering.kalman import KalmanFilterCV

    def covariance(pos: float, vel: float):
        kf = KalmanFilterCV(_params(initial_position_uncertainty_px=pos,
                                    initial_velocity_uncertainty_px_s=vel))
        kf.initialise(0.0, 0.0, 0.0, 0.0)
        return float(kf.covariance[0, 0]), float(kf.covariance[2, 2])

    small_pos, small_vel = covariance(1.0, 1.0)
    large_pos, large_vel = covariance(100.0, 100.0)
    assert large_pos > small_pos * 100, "initial_position_uncertainty_px does not reach P"
    assert large_vel > small_vel * 100, "initial_velocity_uncertainty_px_s does not reach P"


def test_process_noise_changes_how_fast_covariance_grows():
    """``process_noise_psd`` must drive covariance growth during prediction."""
    from src.filtering.kalman import KalmanFilterCV

    def growth(psd: float) -> float:
        # Velocity uncertainty is pinned near zero: position variance grows by
        # P[2, 2] * dt^2 as well as by q, and at the default 200 px/s that term is 10000 px^2
        # over half a second -- three orders above the q contribution being measured.
        kf = KalmanFilterCV(_params(process_noise_psd=psd,
                                    initial_position_uncertainty_px=0.01,
                                    initial_velocity_uncertainty_px_s=0.01))
        kf.initialise(0.0, 0.0, 0.0, 0.0)
        before = float(kf.covariance[0, 0])
        kf.predict(0.5)
        return float(kf.covariance[0, 0]) - before

    assert growth(10_000.0) > growth(10.0) * 10, "process_noise_psd does not reach the filter"
