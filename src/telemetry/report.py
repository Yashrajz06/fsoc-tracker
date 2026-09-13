"""Auto-generated run summary report.

Produced with zero manual steps at the end of a headless run, because "the software must be
capable of automatically generating a performance report" is a mandatory deliverable.

The report has three clearly separated sections, and the separation is deliberate:

1. **Results measured in this run.**
2. **Attribution** -- the per-frame flags aggregated, so an anomalous result can be explained
   from the report itself rather than by re-running anything.
3. **Known limits of the system**, with provenance on every figure. Values computed from the
   active configuration are marked as such; values measured elsewhere carry the configuration
   they were measured at, because they do not automatically transfer.

The third section is kept visually and structurally distinct from the first so that an envelope
figure can never be mistaken for a measurement from this run.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.config import AppConfig
from src.telemetry.benchmark import CapacityResult
from src.telemetry.logger import TelemetryLogger
from src.telemetry.metrics import AcquisitionRecord, MetricsSummary

__all__ = ["write_report"]


def _fmt(value: Optional[float], suffix: str = "", digits: int = 3) -> str:
    """Format an optional number for display."""
    if value is None:
        return "&mdash;"
    return f"{value:.{digits}f}{suffix}"


def _verdict(actual: Optional[float], target: float, lower_is_better: bool = True) -> str:
    """Render a pass/fail chip against a specification target."""
    if actual is None:
        return '<span class="chip na">no data</span>'
    ok = (actual <= target) if lower_is_better else (actual >= target)
    label = "meets" if ok else "EXCEEDS" if lower_is_better else "BELOW"
    return f'<span class="chip {"ok" if ok else "bad"}">{label} {target:g}</span>'


def _rows(pairs: List[tuple]) -> str:
    """Render a list of ``(label, value)`` pairs as table rows."""
    return "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{v}</td></tr>" for k, v in pairs)


def write_report(logger: TelemetryLogger, config: AppConfig,
                 path: Optional[Path] = None,
                 capacity: Optional[CapacityResult] = None) -> Path:
    """Write the HTML summary report.

    Args:
        logger: The telemetry logger holding this run's records.
        config: Validated application configuration.
        path: Destination file. Defaults to ``<output_dir>/report.html``.
        capacity: Optional unthrottled capacity result. Reported separately from the real-time
            rate, never merged with it.

    Returns:
        The path written.
    """
    summary: MetricsSummary = logger.summary()
    targets = config.telemetry.metrics
    path = Path(path or logger.output_dir / "report.html")
    path.parent.mkdir(parents=True, exist_ok=True)

    acquisitions: List[AcquisitionRecord] = logger.accumulator.acquisitions
    initial = [a for a in acquisitions if not a.reacquisition]
    reacqs = [a for a in acquisitions if a.reacquisition]

    acquisition_rows = []
    for event in initial:
        label = ("initial acquisition (target initially IN FOV)"
                 if event.population == "in_fov"
                 else "initial acquisition (SEARCH-LIMITED)")
        chip = (_verdict(event.duration_s, float(targets.get("acquisition_target_s", 2.0)))
                if event.population == "in_fov"
                else '<span class="chip na">not comparable &mdash; see limits</span>')
        acquisition_rows.append((label, f"{_fmt(event.duration_s, ' s')} {chip}"))
    if not initial:
        acquisition_rows.append(("initial acquisition", "&mdash; never acquired"))

    for index, event in enumerate(reacqs, start=1):
        acquisition_rows.append((
            f"re-acquisition {index} (frame {event.frame_index})",
            f"{_fmt(event.duration_s, ' s')} "
            f"{_verdict(event.duration_s, float(targets.get('reacquisition_target_s', 1.0)))}"))

    error_target = float(targets.get("tracking_error_target_px", 10.0))
    fps_target = float(targets.get("fps_target", 20.0))

    # Quantify the association gap rather than leaving two columns looking like a choice of the
    # flattering one.
    association_note = ""
    if (summary.detection_rmse_px is not None and summary.centroid_rmse_px is not None
            and summary.centroid_rmse_px > 0):
        gap = summary.detection_rmse_px / summary.centroid_rmse_px
        if gap >= 5.0:
            association_note = (
                f"Raw detection RMSE is {gap:.0f}&times; the system output's, and "
                f"{100.0 * summary.association_failure_rate:.1f}% of detections were rejected by "
                f"the validation gate. ")
        else:
            association_note = (
                f"The two agree closely here, with "
                f"{100.0 * summary.association_failure_rate:.1f}% of detections rejected. ")

    # Quantify the divergence in words rather than leaving an evaluator to work it out.
    divergence_note = ""
    if (summary.centroid_median_px is not None and summary.centroid_rmse_px is not None
            and summary.centroid_median_px > 0):
        ratio = summary.centroid_rmse_px / summary.centroid_median_px
        if ratio >= 5.0:
            divergence_note = (
                f"RMSE here is {ratio:.0f}&times; the median, so the figure is set by a minority "
                f"of frames rather than by typical accuracy. ")
        else:
            divergence_note = (
                "RMSE and median are close here, so the error distribution has no significant "
                "tail. ")

    # A run that never locked must say so, and why, rather than rendering a table of blanks.
    median_snr = None
    snrs = [r.snr_aperture for r in logger.accumulator.records if r.snr_aperture is not None]
    if snrs:
        median_snr = float(sorted(snrs)[len(snrs) // 2])
    envelope_text = summary.envelope_note(median_snr)
    envelope_block = ""
    if envelope_text:
        envelope_block = (
            '<p class="note" style="background:#fdf3e7;border:1px solid #e8d5b5;'
            'padding:0.6rem 0.9rem;border-radius:0.3rem;">'
            '<b>Input below characterised envelope &mdash; no lock achieved.</b> '
            + html.escape(envelope_text) + '</p>')

    limits = logger.header["known_limits"]
    limit_rows = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td><b>{entry['value']}</b></td>"
        f"<td class='prov'>{html.escape(str(entry['provenance']))}</td>"
        f"<td class='note'>{html.escape(str(entry['note']))}</td></tr>"
        for name, entry in limits.items())

    definitions = logger.header["metric_definitions"]
    definition_rows = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td class='note'>{html.escape(str(text))}</td></tr>"
        for name, text in definitions.items())

    capacity_block = "<p class='note'>Not measured for this run.</p>"
    if capacity is not None:
        capacity_block = f"""<table>{_rows([
            ("frames timed", f"{capacity.frames} (after {capacity.warmup_frames} warm-up)"),
            ("mean / p50 / p95 / max", f"{capacity.mean_ms:.2f} / {capacity.p50_ms:.2f} / "
                                       f"{capacity.p95_ms:.2f} / {capacity.max_ms:.2f} ms"),
            ("max sustainable (mean)", _fmt(capacity.max_sustainable_fps, " FPS", 1)),
            ("conservative (p95)", f"{_fmt(capacity.conservative_fps, ' FPS', 1)} "
                                   f"{_verdict(capacity.conservative_fps, capacity.target_fps, False)}"),
        ])}</table>"""

    config_json = json.dumps({
        "scene": [config.scene.width, config.scene.height],
        "resolution": list(config.camera.fov_px),
        "fov_deg": [config.camera.fov_horizontal_deg, config.camera.fov_vertical_deg],
        "deg_per_pixel": list(config.camera.deg_per_pixel),
        "max_slew_deg_s": [config.camera.max_pan_speed_deg_s,
                           config.camera.max_tilt_speed_deg_s],
        "pid": {k: v for k, v in config.control.pid.items() if not k.startswith("_")},
        "motion": config.target.motion_type,
        "noise_enabled": config.noise.enabled,
        "threshold_method": config.vision.detection.threshold_method,
        "centroid_method": config.vision.centroid.method,
        "random_seed": config.run.random_seed,
    }, indent=2)

    document = f"""<!doctype html>
<meta charset="utf-8">
<title>FSOC tracking performance report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 60rem; color: #1a1a1a; line-height: 1.45; }}
 h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
 h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 2px solid #e3e3e3;
       padding-bottom: 0.3rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: 0.6rem 0; }}
 th, td {{ text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #ececec;
           vertical-align: top; }}
 th {{ width: 22rem; font-weight: 600; color: #333; }}
 .chip {{ font-size: 0.75rem; padding: 0.1rem 0.45rem; border-radius: 0.6rem;
          margin-left: 0.4rem; white-space: nowrap; }}
 .ok {{ background: #e4f5e9; color: #12632c; }}
 .bad {{ background: #fdeaea; color: #94191b; }}
 .na {{ background: #eee; color: #555; }}
 .note, .prov {{ font-size: 0.82rem; color: #555; }}
 .limits {{ background: #fbfaf5; border: 1px solid #e8e2cf; padding: 0.4rem 1rem 1rem;
            border-radius: 0.4rem; }}
 .limits h2 {{ border-bottom-color: #e0d8bf; }}
 code {{ background: #f4f4f4; padding: 0.05rem 0.3rem; border-radius: 0.2rem; }}
</style>
<h1>FSOC coarse-alignment tracking &mdash; performance report</h1>
<p class="note">Generated {html.escape(datetime.now(timezone.utc).isoformat(timespec='seconds'))}
 &middot; mode <code>{html.escape(str(logger.mode))}</code>
 &middot; config <code>{html.escape(str(config.source_path))}</code></p>

<h2>1. Results measured in this run</h2>
<table>{_rows([
    ("simulation duration", _fmt(summary.duration_s, " s")),
    ("total frames", summary.total_frames),
    ("frames with target present", summary.frames_with_target),
    ("frames locked", summary.frames_locked),
])}</table>

<h3>Acquisition</h3>
<p class="note">Split into two populations and never pooled. The search-limited case is bounded
below by the slew ceiling and is not comparable against the in-FOV budget &mdash; see section 3.</p>
<table>{_rows(acquisition_rows)}</table>

{envelope_block}
<h3>Error &mdash; centroiding and pointing reported separately</h3>
<p class="note">Benchmark-2 scores <b>centroiding</b> error. Both are given because the
specification's "tracking error &le; 10 px" is ambiguous between them.</p>
<table>{_rows([
    ("centroiding median (system output)",
     f"{_fmt(summary.centroid_median_px, ' px')} "
     f"{_verdict(summary.centroid_median_px, error_target)}"),
    ("centroiding p95", _fmt(summary.centroid_p95_px, " px")),
    ("centroiding mean / max", f"{_fmt(summary.centroid_mean_px, ' px')} / "
                               f"{_fmt(summary.centroid_max_px, ' px')}"),
    ("centroiding RMSE", f"{_fmt(summary.centroid_rmse_px, ' px')} "
                         f"{_verdict(summary.centroid_rmse_px, error_target)}"),
    ("detection median / p95 (raw measurement)",
     f"{_fmt(summary.detection_median_px, ' px')} / "
     f"{_fmt(summary.detection_p95_px, ' px')}"),
    ("detection RMSE (raw measurement)", _fmt(summary.detection_rmse_px, " px")),
    ("association failure rate",
     f"{_fmt(100.0 * summary.association_failure_rate, ' %', 1)} "
     f"({summary.frames_measured} measured / {summary.frames_predicted} predicted frames)"),
    ("pointing median / p95", f"{_fmt(summary.pointing_median_px, ' px')} / "
                              f"{_fmt(summary.pointing_p95_px, ' px')}"),
    ("pointing mean / max", f"{_fmt(summary.pointing_mean_px, ' px')} / "
                            f"{_fmt(summary.pointing_max_px, ' px')}"),
    ("pointing RMSE", _fmt(summary.pointing_rmse_px, " px")),
    ("lock retention", _fmt(100.0 * summary.lock_retention, " %", 1)),
    ("target loss rate", f"{_fmt(100.0 * summary.loss_rate, ' %', 1)} "
                         f"{_verdict(summary.loss_rate, float(targets.get('loss_rate_target', 0.05)))}"),
])}</table>
<p class="note"><b>Why there are two error columns.</b> {association_note}
<b>Detection error</b> measures the raw vision measurement, including associations the filter
went on to reject; <b>centroid error</b> measures the system's actual output, which is the fused
track estimate. A large gap between them indicates low-SNR association failures being correctly
rejected &mdash; the tracker discarding bad measurements is the system working, not failing. The
<code>estimate_source</code> column records each frame's real provenance:
<code>measured</code> (the measurement was accepted and materially moved the state),
<code>low_gain</code> (accepted, but with a Kalman gain under 5% &mdash; folded in while
contributing almost nothing, so the estimate is effectively the model's prediction), or
<code>predicted</code> (rejected or absent, running on the filter). The
<code>low_gain</code> category exists because "measured" with near-zero weight is <i>worse</i>
than "predicted": at least a predicted frame is honest about running on the model. A run showing
many <code>low_gain</code> frames is one where measurements are reaching the filter but not
correcting it, which is how a drifting estimate can otherwise masquerade as a tracked one.</p>
<p class="note"><b>Reading the spread.</b> {divergence_note}
Median reflects precision while locked; p95 marks where the body of the distribution ends and the
tail begins; RMSE covers all scored frames including momentary losses, so it is dominated by the
tail. A large median-to-RMSE gap means losses exist rather than that the estimator is imprecise
&mdash; cross-read it against lock retention above. Every per-frame value needed to recompute all
three independently is in <code>frames.csv</code> (<code>centroid_error_px</code>,
<code>pointing_error_px</code>, <code>locked</code>, <code>target_present</code>); the statistics
here are taken over frames where <code>locked</code> is true.</p>

<h3>Three clocks, never conflated</h3>
<table>{_rows([
    ("camera update rate (configured)", _fmt(summary.camera_rate_hz, " Hz", 1)),
    ("control update rate (configured)", _fmt(summary.control_rate_hz, " Hz", 1)),
    ("processing FPS, real-time (mean / min / max)",
     f"{_fmt(summary.processing_fps_mean, '', 1)} / {_fmt(summary.processing_fps_min, '', 1)} / "
     f"{_fmt(summary.processing_fps_max, '', 1)}"),
    ("mean processing time per frame", _fmt(summary.processing_ms_mean, " ms")),
])}</table>
<p class="note">Real-time processing FPS is bounded above by the camera update clock, so it
understates the pipeline and would hide a throughput regression until frames start dropping.
Capacity is measured separately, unthrottled:</p>
{capacity_block}

<h2>2. Attribution</h2>
<p class="note">Per-frame flags aggregated. These exist because in every development phase a
summary number looked healthy while something underneath was wrong &mdash; the frame trace is what
exposed a Kalman gate rejecting perfect measurements. An anomaly here should be explainable from
this report without re-running anything.</p>
<table>{_rows([
    ("frames clipped (centroid biased inward, up to ~2 px)", summary.frames_clipped),
    ("frames saturated (centroid biased ~0.19 px)", summary.frames_saturated),
    ("frames on fallback geometry (spot scale not measured)", summary.frames_from_fallback),
    ("measurements rejected by the Kalman gate", summary.frames_gated_out),
    ("frames with the rate command slew-saturated", summary.frames_slew_saturated),
    ("track lockout re-initiations", summary.lockout_events),
    ("mean NIS (filter consistency; ~2.0 for 2 DOF)", _fmt(summary.nis_mean, "", 2)),
])}</table>

<div class="limits">
<h2>3. Known limits of this system</h2>
<p class="note">These are <b>not</b> measurements from this run. They are the characterised
envelope, included so the results above can be interpreted correctly rather than against an
envelope we never claimed. Provenance is given for each, because figures measured at the default
configuration do not automatically transfer to a different scenario.</p>
<table>
<tr><th>limit</th><td><b>value</b></td><td class="prov">provenance</td><td class="note">note</td></tr>
{limit_rows}
</table>
</div>

<h2>4. Metric definitions in force</h2>
<p class="note">Reproduced in every log this run produced, so no definition has to be inferred.</p>
<table>{definition_rows}</table>

<h2>5. Configuration snapshot</h2>
<pre class="note">{html.escape(config_json)}</pre>
"""
    path.write_text(document, encoding="utf-8")
    return path
