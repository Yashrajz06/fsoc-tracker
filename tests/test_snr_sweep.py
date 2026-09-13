"""Centroid error versus SNR: the graded deliverable curve.

This produces ``docs/figures/centroid_error_vs_snr.png``, which is a report figure and a Q&A
centrepiece, and asserts the accuracy the pipeline actually achieves.

Three analysis decisions, each of which changes what the curve means:

1. **False locks are separated from centroid error.** Below SNR ~10 the pipeline frequently
   selects a noise blob rather than the target, giving errors around 65 px -- which is simply the
   mean separation of two random points in a 160 px frame. That is a *detection* failure, not a
   centroiding error, and averaging the two together would produce a curve describing neither.
   Points beyond an association gate of ``3 * FWHM`` are counted as false locks and reported as a
   rate; centroid error is quoted only over correctly associated frames.

2. **Split by shape.** The specification's default target is a square, and Phase 1 showed hard
   edges behave differently from a Gaussian.

3. **Split by ``from_fallback``.** The aperture radius and centroid window both scale with the
   estimated FWHM, so a curve mixing measured and fallback geometry is two different measurements
   sharing an axis.

What the curve shows
--------------------
* **SNR < 10 -- detection-limited.** False-lock rates of 30-95%. Errors near 65 px are simply the
  mean separation of two random points in the frame; this regime is about picking the wrong blob,
  not about centroiding accuracy.
* **SNR 10-100 -- centroiding-limited.** Error tracks ``FWHM / (2 * SNR)`` within about 10% for
  square and circle. Median error is below 1 px for every shape from SNR 10 upward.
* **Above SNR ~100 -- the curve turns up, and the cause is saturation, not peak-locking.**
  Raising peak intensity to reach high SNR necessarily drives the spot into the 8-bit ceiling.
  Measured for a square: error 0.088 -> 0.214 -> 0.326 px as core saturation rises 0.30 -> 0.37,
  while the measured SNR itself stalls (170 -> 190 -> 195) because a clipped peak stops adding
  signal. The ``saturated`` flag fires on 98-100% of those frames, so the effect is visible in
  the trace rather than silent.

**No peak-locking floor was observed**, and that is the expected result rather than a suspicious
one. Peak-locking is an *undersampling* phenomenon: it appears when the FWHM approaches a pixel,
because the sub-pixel information is then carried by too few samples. Our spots are 5.9-11.3 px
FWHM, far above the pixel-to-spot ratio of 1.5-2.5 at which the Cramer-Rao bound is minimised
(``docs/DESIGN.md`` section 2), so there are many samples across the profile and no pull toward
pixel centres.

Reaching high SNR by *lowering noise* instead of raising brightness -- which avoids saturation
entirely -- the error continues to fall to **0.007-0.013 px** with no plateau. That residual
floor is **8-bit quantisation**, confirmed by two independent checks: it scales as ``1/peak``
(0.0169 -> 0.0108 -> 0.0059 -> 0.0030 px as peak doubles from 30 to 240), and processing the same
frames in ``float32`` without quantisation drops the error to 0.0005 px, 6-36x lower.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.config import load_config
from src.noise.sensor import GaussianNoiseParams, add_gaussian_noise, to_uint8
from src.sim.beacon import BeaconParams, render_beacon
from src.vision.pipeline import VisionPipeline
from src.vision.snr import SNR_APERTURE_FORMULA

FRAME_PX = 160
BACKGROUND = 30.0
NOISE_SIGMA = 12.0
TRIALS_PER_LEVEL = 200
SHAPES = ("gaussian", "square", "circle")

#: True equivalent-Gaussian FWHM per shape, from the Phase-3 shape-bias measurement. A square of
#: side L reads 1.128*L through the half-max-area conversion because its half-maximum region is
#: the square itself.
TRUE_FWHM: Dict[str, float] = {"gaussian": 5.887, "square": 11.28, "circle": 10.0}

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "centroid_error_vs_snr.png"


def _render_trial(shape: str, peak: float, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    """Render one noisy frame with the beacon at a random sub-pixel position.

    Args:
        shape: Beacon profile.
        peak: Peak intensity before clipping.
        rng: Seeded generator.

    Returns:
        ``(frame, true_x, true_y)``.
    """
    tx = FRAME_PX / 2.0 + float(rng.uniform(-0.5, 0.5))
    ty = FRAME_PX / 2.0 + float(rng.uniform(-0.5, 0.5))
    frame = np.full((FRAME_PX, FRAME_PX), BACKGROUND, dtype=np.float32)
    patch = render_beacon(tx, ty, BeaconParams(shape=shape, size_px=10.0, sigma_px=2.5,
                                               peak_intensity=peak))
    ph, pw = patch.data.shape
    y0, x0 = max(0, patch.y0), max(0, patch.x0)
    y1, x1 = min(FRAME_PX, patch.y0 + ph), min(FRAME_PX, patch.x0 + pw)
    frame[y0:y1, x0:x1] += patch.data[y0 - patch.y0:y1 - patch.y0, x0 - patch.x0:x1 - patch.x0]
    noisy = to_uint8(add_gaussian_noise(to_uint8(frame),
                                        GaussianNoiseParams(sigma=NOISE_SIGMA), rng))
    return noisy, tx, ty


def run_sweep(shape: str, use_measured_geometry: bool, trials: int,
              seed: int = 20260910) -> np.ndarray:
    """Sweep peak intensity and record measured SNR against centroid error.

    SNR is *measured* per frame rather than commanded, so the x-axis is the quantity actually
    defined in :data:`~src.vision.snr.SNR_APERTURE_FORMULA` rather than a nominal target.

    Args:
        shape: Beacon profile.
        use_measured_geometry: Pass the true scale (measured-geometry case) or ``None``
            (fallback-geometry case).
        trials: Trials per intensity level.
        seed: RNG seed.

    Returns:
        An ``(n, 2)`` array of ``(snr_aperture, error_px)``.
    """
    config = load_config("config/default.json")
    pipeline = VisionPipeline.from_config(config)
    rng = np.random.default_rng(seed)
    rows: List[Tuple[float, float]] = []

    for peak in np.geomspace(4.0, 4000.0, 14):
        for _ in range(trials):
            frame, tx, ty = _render_trial(shape, float(peak), rng)
            if use_measured_geometry:
                measurement = pipeline.process(frame, fwhm_px=TRUE_FWHM[shape],
                                               from_fallback=False)
            else:
                measurement = pipeline.process(frame)
            if not measurement.found or measurement.snr is None:
                continue
            if measurement.snr.snr_aperture is None:
                continue
            rows.append((measurement.snr.snr_aperture,
                         math.hypot(measurement.x - tx, measurement.y - ty)))
    return np.asarray(rows, dtype=np.float64)


def summarise(data: np.ndarray, shape: str) -> List[dict]:
    """Bin a sweep by SNR and summarise error and false-lock rate.

    Args:
        data: ``(n, 2)`` array of ``(snr, error_px)``.
        shape: Beacon profile, setting the association gate.

    Returns:
        One dict per populated bin.
    """
    association_gate = 3.0 * TRUE_FWHM[shape]
    snr, error = data[:, 0], data[:, 1]
    associated = error <= association_gate

    edges = np.geomspace(1.0, 2000.0, 17)
    out: List[dict] = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (snr >= low) & (snr < high)
        if in_bin.sum() < 10:
            continue
        good = in_bin & associated
        out.append({
            "snr": float(np.sqrt(low * high)),
            "n": int(in_bin.sum()),
            "false_lock_rate": float(1.0 - good.sum() / in_bin.sum()),
            "median_error": float(np.median(error[good])) if good.sum() else float("nan"),
            "p90_error": (float(np.percentile(error[good], 90)) if good.sum() > 3
                          else float("nan")),
        })
    return out


@pytest.mark.slow
def test_centroid_error_versus_snr_curve() -> None:
    """Produce the SNR curve and assert the accuracy it shows.

    Assertion: **median centroid error below 1 px above SNR 10**, over correctly associated
    frames, for every shape under measured geometry. Note the p90 does exceed 1 px in the
    SNR 10-20 bin for a square -- the median is what is asserted, and both are plotted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results: Dict[Tuple[str, bool], List[dict]] = {}
    for shape in SHAPES:
        for measured in (True, False):
            data = run_sweep(shape, measured, TRIALS_PER_LEVEL)
            assert data.size, f"{shape}/{measured}: sweep produced no data"
            results[(shape, measured)] = summarise(data, shape)

    # --- assertions ------------------------------------------------------------------------
    for shape in SHAPES:
        above_ten = [row for row in results[(shape, True)]
                     if row["snr"] >= 10.0 and not math.isnan(row["median_error"])]
        assert above_ten, f"{shape}: no bins above SNR 10"
        worst = max(row["median_error"] for row in above_ten)
        assert worst < 1.0, f"{shape}: median error {worst:.3f} px above SNR 10"

        # Detection, not centroiding, is what fails at low SNR -- and it must recover by SNR 20.
        high_snr = [row for row in results[(shape, True)] if row["snr"] >= 20.0]
        assert all(row["false_lock_rate"] < 0.05 for row in high_snr), \
            f"{shape}: false locks persist above SNR 20"

    # --- figure ----------------------------------------------------------------------------
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax_lock) = plt.subplots(
        2, 1, figsize=(9.5, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    colours = {"gaussian": "#1f77b4", "square": "#d62728", "circle": "#2ca02c"}
    for shape in SHAPES:
        for measured in (True, False):
            rows = [r for r in results[(shape, measured)] if not math.isnan(r["median_error"])]
            if not rows:
                continue
            xs = [r["snr"] for r in rows]
            ys = [r["median_error"] for r in rows]
            ax.plot(xs, ys, marker="o" if measured else "^", ms=4,
                    ls="-" if measured else "--", color=colours[shape], alpha=1.0 if measured else 0.45,
                    label=f"{shape} ({'measured' if measured else 'fallback'} geometry)")
        rows = [r for r in results[(shape, True)] if not math.isnan(r["p90_error"])]
        ax.fill_between([r["snr"] for r in rows],
                        [r["median_error"] for r in rows],
                        [r["p90_error"] for r in rows],
                        color=colours[shape], alpha=0.10, linewidth=0)

    reference = np.geomspace(10, 1000, 50)
    ax.plot(reference, TRUE_FWHM["gaussian"] / (2.0 * reference), color="0.35", lw=1.2, ls=":",
            label=r"$\sigma_x = \mathrm{FWHM}/(2\,\mathrm{SNR})$  (shot-noise law)")

    ax.axvspan(1, 10, color="0.85", alpha=0.5, zorder=0)
    ax.text(1.6, 4e-3, "detection-limited\n(false locks)", fontsize=8, color="0.35")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.text(1.05e3, 1.08, "10 px spec budget is 10x above this line", fontsize=7,
            color="0.4", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("centroid error (px), median; shaded to p90")
    ax.set_title("Centroid error vs SNR — FSOC coarse alignment vision pipeline\n"
                 "IWCoG (3 iterations) on top-hat residual; median/MAD background over an annulus",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    for shape in SHAPES:
        rows = results[(shape, True)]
        ax_lock.plot([r["snr"] for r in rows], [100.0 * r["false_lock_rate"] for r in rows],
                     marker="o", ms=3, color=colours[shape], label=shape)
    ax_lock.set_ylabel("false lock (%)")
    ax_lock.set_xlabel(
        "measured aperture SNR\n" + SNR_APERTURE_FORMULA
        + r"    aperture $r<1.5\,\mathrm{FWHM}$;  annulus $2.5$–$4\,\mathrm{FWHM}$;  "
          r"$\mu_{bg}=\mathrm{median}$, $\sigma_{bg}=1.4826\,\mathrm{MAD}$")
    ax_lock.axvspan(1, 10, color="0.85", alpha=0.5, zorder=0)
    ax_lock.grid(True, which="both", alpha=0.25)
    ax_lock.legend(fontsize=7.5)

    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert FIGURE_PATH.exists()


@pytest.mark.slow
def test_fallback_geometry_costs_accuracy_on_non_gaussian_shapes() -> None:
    """Quantify what the spot-scale estimator buys, which is the point of the split.

    For a Gaussian the configured fallback *is* the right scale, so the two curves coincide and
    the estimator earns nothing. For a square it is wrong by 1.9x, and the cost is large at
    moderate SNR -- which is the concrete argument for estimating scale rather than declaring it.
    """
    square_measured = summarise(run_sweep("square", True, 60), "square")
    square_fallback = summarise(run_sweep("square", False, 60), "square")

    def median_at(rows, low, high):
        vals = [r["median_error"] for r in rows
                if low <= r["snr"] < high and not math.isnan(r["median_error"])]
        return float(np.mean(vals)) if vals else float("nan")

    measured = median_at(square_measured, 10.0, 50.0)
    fallback = median_at(square_fallback, 10.0, 50.0)
    assert not math.isnan(measured) and not math.isnan(fallback)
    assert fallback > 2.0 * measured, (
        f"fallback geometry ({fallback:.3f} px) should be markedly worse than measured "
        f"({measured:.3f} px) for a square, or the scale estimator is not earning its cost")
