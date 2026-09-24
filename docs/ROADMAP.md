# Build Roadmap

Work **one module per session**. Write the module, write its test, tick the box, move on.
Update this file as phases complete so context carries between sessions.

Current phase: **Phase 7** (Phases 0-6 complete)

---

## Phase 0 — Foundation

Everything depends on these. Get them right before writing any algorithm.

- [x] `src/framesource.py` — `FrameSource` protocol + `FrameData` dataclass (skeleton provided)
- [x] `src/config.py` — load `config/default.json` into typed dataclasses, validate ranges.
      Specifically must include:
      - **Scale-relative vision parameters.** Top-hat kernel, blob-area gates, centroid window
        and ROI size are expressed as multiples of a runtime-estimated spot scale. The absolute
        pixel values are *fallbacks only*, used when spot-scale estimation fails. This is the
        single highest-value robustness item for Benchmark-2.
      - **Derived properties, not stored literals:** `deg_per_pixel`, `max_px_per_frame`,
        search arm spacing (FOV-derived).
      - **Worst-case spiral search time** as a computed property, with a startup warning when it
        exceeds the ≤2 s acquisition budget (see DESIGN §7.5).
      - Range validation against the spec: σ ≤ 20, jitter ≤ 20 px/frame, platform motion
        ≤ 20 px/frame, slew 5–10 °/s, target 5–20 px, S&P density ≤ 0.10.
      - **Partial-override merging** (`merge_overrides`, `load_config(path, overrides)`, and the
        `--scenario` flag) so evaluator scenarios are small files naming only what differs.
        Nested blocks merge recursively, lists replace wholesale, and validation runs *after*
        merging so a scenario can never push us out of spec. Override paths are recorded in the
        log header — without that a benchmark number cannot be reproduced from its log alone.
- [x] `requirements.txt` pinned, virtualenv working
- [x] `pytest` runs and collects
- [x] `src/main.py` — headless entry point that can run a scenario end-to-end (stub modules ok)

**Done when:** `python -m src.main --config config/default.json --headless` runs without error,
even if it does nothing useful yet.

---

## Phase 1 — Simulation Engine

- [x] `src/sim/canvas.py` — 2000×2000 world buffer, allocated once, reused
- [x] `src/sim/beacon.py` — Gaussian beacon renderer, supersampled 4× then downsampled,
      shapes: gaussian / square / circle, size 5–20 px
- [x] `src/sim/trajectories.py` — line, circular, figure-8, random (mandatory);
      spiral, sinusoidal, OU process (optional)
- [x] `src/camera/viewport.py` — 640×480 sub-viewport extraction with pan/tilt position
- [x] `src/camera/model.py` — FOV ↔ pixel mapping, deg_per_pixel, slew rate limits
- [x] Tests: ground-truth centroid recoverable to **<0.05 px** from a clean render.
      **Must explicitly assert the pixel-centre convention** (`CLAUDE.md` → Coordinate
      conventions): pixel centres at integer indices, frame centre at `((W-1)/2, (H-1)/2)`.

- [x] **Separate test: the supersample→downsample coordinate mapping in isolation, to <0.01 px.**
      This is a *distinct* test from the <0.05 px full-pipeline one above, and both must exist.

      **Why it must be separate.** The downsample mapping is the most likely place for a
      sub-pixel bias to hide. At S=4 a half-subpixel error is `0.5/4 = 0.125 px` — big enough to
      invalidate the centroid-error-vs-SNR curve, which is a report deliverable and a Q&A
      centrepiece. Three reasons an end-to-end test is not sufficient cover:
      - **Tolerance.** The mapping is exact arithmetic with no noise in it, so it should hold to
        <0.01 px. The end-to-end test cannot be tightened that far, because the centroid
        estimator has its own finite accuracy. A bias of 0.02–0.04 px therefore sits inside the
        end-to-end tolerance permanently while being a genuine renderer defect.
      - **Localisation.** When the end-to-end test fails, it does not say whether the fault is in
        the Gaussian profile, the downsample mapping, or the estimator. The isolated test names
        the culprit directly.
      - **Masking.** A symmetric estimator can partially absorb a symmetric bias, so the
        end-to-end number can look acceptable while the mapping is wrong. Testing the transform
        alone separates "the renderer is correct" from "the renderer is wrong in a way the
        estimator happens to absorb today".

      **What to assert.** Render at 4x, then verify that the mapping from supersampled
      coordinates to output coordinates preserves centroid position to <0.01 px — testing the
      transform itself, not the end-to-end render. Recall the mapping from `CLAUDE.md`:
      supersample pixel `[R, C]` has centre `((C + 0.5)/S - 0.5, (R + 0.5)/S - 0.5)` in output
      coordinates. A plain `C/S` (the natural-looking wrong answer) yields the `(S-1)/(2S)` px
      bias = 0.375 px at S=4.

- [x] **Both tests must cover these three position classes**, because the first two can each
      accidentally pass under a wrong convention that fails everywhere else:
      - **exact integer** (e.g. 100.0) — passes under several wrong conventions by symmetry;
      - **exact half-integer** (e.g. 100.5) — also symmetric, catches a different subset;
      - **irrational-ish offset** (e.g. 100.37) — asymmetric, and the only one of the three that
        reliably fails a wrong mapping.
      A test suite containing only the first two is the specific trap here: it looks thorough,
      and it is exactly what a `C/S` mapping would survive.

- [x] `src/sim/scene.py` — canvas + beacon + trajectory on one clock, emitting `SceneState`
      with exact sub-pixel ground truth per frame (added; not in the original plan).

**Done when:** you can produce a clean video of a beacon moving in all four mandatory patterns,
with exact known ground truth per frame. **MET** — all four motions render to `.mp4` with a
per-frame ground-truth CSV; worst centroid error over fully-inside frames is 0.007 px,
7× inside the 0.05 px budget.

> **Finding carried into later phases.** Hard-edged shapes needed *area* sampling, not point
> sampling, to reach sub-pixel accuracy (DESIGN §2). And edge-clipped spots are biased inward
> by up to ~2 px with no noise present — Phase 3/4 must down-weight detections near the frame
> boundary rather than trusting them equally.

---

## Phase 2 — Noise and Disturbance Pipeline

- [x] `src/noise/sensor.py` — Gaussian, Poisson, salt & pepper (composable pure functions)
- [x] `src/noise/atmospheric.py` — Koschmieder model; clear/haze/fog/rain/low-light presets
- [x] `src/noise/turbulence.py` — log-normal / Gamma–Gamma scintillation, beam wander
- [x] `src/noise/disturbance.py` — camera jitter (±20 px/frame), platform motion (±20 px/frame)
- [x] `src/noise/pipeline.py` — configurable ordered composition of the above
- [x] Tests: statistical validation (measured mean/variance matches requested sigma;
      S&P density within tolerance of requested density)
- [x] **Saturation bias test: quantify centroid bias vs. saturation fraction.**
      Saturation is the sibling of the Phase 1 edge-clipping finding and has the same failure
      shape: once atmospheric brightening or scintillation pushes peak intensity to 255, the
      spot's top is flattened and the intensity-weighted centroid biases toward the *geometric*
      centre of the saturated region. Like clipping it is smooth, systematic and noise-free, so
      it looks exactly like tracker lag and is easy to misdiagnose as a control problem.
      Sweep saturation fraction (fraction of spot pixels at 255) from 0 to heavily clipped, at
      several sub-pixel phases, and record where bias crosses 0.05 px, 0.5 px and 10 px. That
      curve tells Phase 3/4 where the `saturated` flag has to start firing, and belongs in the
      technical report next to the SNR curve.

**Done when:** every noise type is independently toggleable from config and statistically
correct. **MET** — all stages validated statistically (Gaussian variance, Poisson
variance-equals-mean, exact S&P density, Koschmieder contrast == transmission,
scintillation index against closed form, OU stationary sigma, jitter bounds, platform
per-frame cap). All five atmospheric presets run end-to-end with the scene.

> **Findings.** (a) The Gamma-Gamma alpha/beta figures previously quoted in DESIGN §4.3 and
> `config/default.json` did not follow from the plane-wave relations at any
> parameterisation; corrected to computed values, with the formula now the reference.
> (b) The saturation-bias curve shows saturation costs **precision, not lock** (~0.19 px
> worst case vs a 10 px budget), so the Phase 3 `saturated` flag should inflate `R` rather
> than reject the detection — unlike an impulse false positive, which must be rejected.

---

## Phase 3 — Vision Pipeline

This is the scoring core. Spend the most care here.

- [x] `src/vision/spotscale.py` — **Tier 0** scale-free bootstrap (median → percentile →
      connected components → area gate → half-maximum-area FWHM), with a 5-sigma peak
      significance floor so it fails closed. Recovers scale to 0.4–10.7% across a 6× spot-size
      range, four resolutions, 10% salt-and-pepper, and a beacon at peak 45 — all inside the
      30% Tier-0/Tier-1 agreement band. See DESIGN §5.1.2.
- [x] `src/vision/spotscale.py` — **Tier 1**: scale-normalised LoG confirmation on the Tier-0
      ROI, parabola-fit refinement, curvature as the confidence measure. **Not top-hat** — see
      the DESIGN §5.1.2 measurement showing top-hat has no interior maximum over scale.
      Agreement uses a **±30% band on `rho = Tier1/Tier0`**. Cluster-membership gating was tried
      and refuted — a wider sweep showed the profile clusters overlap heavily (raw gap −0.204),
      and smoothing does not help because the spread is systematic in spot size. `rho` is logged
      per frame as a diagnostic but never gates. Tier 0 supplies the reported scale; Tier 1 is
      the cross-check. **Saturation >10% blocks scale resolution outright** (measured +173%
      inflation, which the `max_fwhm` clamp does not catch on small beacons).
      Runtime measured: Tier 0 18/122/253 ms at 640×480 / 1920×1080 / 2000×2000; Tier 1 is
      ROI-cropped and frame-size independent at ~1 ms.
- [x] `src/vision/spotscale.py` — **Tier 2**: `ScaleTracker` with EMA refinement, M-of-N
      anti-poisoning, integer-kernel deadband, and frame-size-aware re-check scheduling derived
      from measured cost (`recommended_recheck_interval_s`).
      **ROI re-checks** (`bootstrap_scale_roi`) collapse steady-state cost to ~2.5 ms at every
      frame size — 7×/52×/96× faster than full-frame at 640×480 / 1920×1080 / 2000×2000.
      Full-frame is reserved for cold bootstrap and post-loss re-acquisition.
      **Ratchet defence:** persistent estimate (valid measurements only) is kept separate from a
      transient search scale (inflates while blind, capped 3×, resets on any candidate). Blocked
      frames — saturation, out-of-band rho, low curvature — *hold* rather than grow, closing the
      second path into the ratchet.
- [ ] Optional: **conditional** fallback A/B gated on Tier-0/Tier-1 disagreement, if measurement
      shows it earns its cost.
- [x] `src/vision/snr.py` — `snr_aperture` (primary, SNR-curve x-axis) and `snr_peak`, both with
      robust median/MAD background over an annulus. Definitions are settled in `CLAUDE.md` →
      Metric definitions and must appear in the log header and on the plot axis.
- [x] `src/vision/preprocess.py` — median filter, top-hat morphology, normalisation
- [x] `src/vision/detect.py` — adaptive threshold, connected components, size/shape gating.
      Default operator is `mean + k*sigma` on the **top-hat residual** (DESIGN §5.1.1); `otsu`,
      `adaptive_gaussian` and `percentile` stay selectable. Benchmark all four on the SNR sweep
      and put the comparison table in the report.
      All size/shape gates derive from the runtime spot-scale estimate, not absolute pixels.
- [x] `src/vision/centroid.py` — thresholded CoG, IWCoG, optional Gaussian-fit refinement
- [ ] **Detection result carries explicit `clipped` and `saturated` flags.**
      - `clipped`: set when the candidate blob's bounding box touches the frame edge.
      - `saturated`: set when a material fraction of the blob sits at the maximum grey level
        (threshold set by the Phase 2 bias curve, not guessed).
      Both describe conditions under which the centroid is biased by a *known, systematic,
      noise-free* mechanism rather than by noise. Inflating Kalman `R` is the response
      (Phase 4), but **the flag is what makes the condition diagnosable**: without it, a
      Phase 4 or Phase 6 error excursion has to be re-derived from scratch to work out whether
      it was clipping, saturation, or genuine tracker lag. With it, the attribution is one
      column of the log away. Exactly the same reasoning as `ResolvedVisionGeometry.from_fallback`
      in Phase 0 — record *why* a number is untrustworthy at the moment we know it, not later.
- [x] `src/vision/pipeline.py` — full chain, ROI-aware (accepts an optional ROI window)
- [x] **SNR sweep test** producing the centroid-error-vs-SNR curve → save plot to `docs/figures/`
- [x] Tests: accuracy vs ground truth across SNR 1–100; false-positive rate under 10% S&P

**Done when:** you have a plot of centroid error vs SNR and know exactly where the pipeline
breaks down. This plot is a report deliverable and a Q&A centrepiece.
**MET** — `docs/figures/centroid_error_vs_snr.png`, split by shape and by `from_fallback`, with a
false-lock panel. Three regimes identified: detection-limited below SNR 10; centroiding-limited
from 10 to ~100 tracking `FWHM/(2·SNR)`; saturation-driven upturn above that. **No peak-locking
floor** — expected, since our 5.9–11.3 px FWHM spots are far from the undersampled regime where
it appears. The true floor is 8-bit quantisation at 0.007–0.013 px, confirmed by `1/peak` scaling
and by a float32 path measuring 6–36× lower.

**Reminder: no hardcoded thresholds. Anything numeric must derive from frame statistics.**

> **Detection-path findings (four defects, all found by measurement).**
> 1. **No peak-significance gate.** On a target-free frame the pipeline returned a confident
>    detection on **30 of 30** frames. A 5σ floor, measured on the *denoised* frame (the residual's
>    rectified noise makes its MAD understate the tail), cuts that to ~10/40 while holding 20/20
>    retention down to aperture SNR ~15. It cannot reach zero on a single frame — that is the
>    K=3 lock criterion's and the Kalman gate's job.
> 2. **`mean + k·σ` degenerates when the robust σ is zero** — a flat or noiseless residual, as
>    heavy compression produces. Threshold collapses to the median and the mask swallows the frame.
> 3. **The percentile fallback is degenerate too** when the target is smaller than the tail the
>    percentile keeps (a 100 px beacon is 0.39% of a 25600 px frame, below the 0.5% kept by the
>    99.5th percentile). Final guard: midpoint between background median and residual peak — still
>    frame-statistical, never a fixed intensity.
> 4. **Otsu-on-residual is not "much better behaved"** as DESIGN previously claimed; it fails
>    completely at noise σ ≥ 5. Corrected, with the measured table now serving as the report's
>    comparison row.

> **Tier-0 findings (three real bugs, all found by measurement rather than review).**
> 1. **Capping candidates by label index silently discards the target.** Connected-component
>    labels run in raster order, so on a noisy frame with hundreds of spurious blobs the beacon
>    near frame centre got a high label and was never examined. Any cap must be applied *after*
>    ranking by strength.
> 2. **Ranking candidates by peak intensity favours salt.** Impulses sit at the 8-bit ceiling
>    and out-rank any beacon dimmer than 255. Integrated flux separates them decisively.
> 3. **A thresholded second moment is the wrong FWHM estimator here.** It weights by distance²
>    and clipping the residual at zero rectifies zero-mean noise into a positive pedestal, so
>    error ran 21–99%. Half-maximum contour area, seeded from the flux-ranked blob's own peak,
>    holds 0.4–10.7%.
> 4. **Frame-fraction gates must be sized from the full frame, not from a crop.** On a 128 px
>    ROI the `max_fwhm` ceiling collapsed to 6.4 px and rejected a legitimate 14 px beacon.
> 5. **Tier 1 was not actually ROI-limited** despite the design saying so — the blur ladder ran
>    full-frame and only the sampling was windowed. Cropping first: 121 ms → 0.95 ms at
>    2000×2000, identical answers.
>
> **Shape dependence resolved as (a), absorbable.** Square and circle spread 10.2–13.2% against
> Gaussian — systematic and analytic (a square's half-max region is the square, giving the
> `2/sqrt(pi) = 1.128` bias, measured 1.125). Scale estimate is *completely* contrast-invariant,
> which is the Mode B property that matters most. Counterintuitively the bias is **best** in
> clear conditions: blur makes shape disagreement worse, not better, because it genuinely widens
> the Gaussian. A shape-agnostic joint flux+area estimator was tested and rejected — it does not
> clearly help and reintroduces the noise sensitivity that killed the second moment.

---

## Phase 4 — Filtering, Control, State Machine

- [x] `src/filtering/kalman.py` — CV model, predict/update, Mahalanobis validation gate.
      **Adaptive measurement noise `R` derived per frame from
      detection SNR** via `sigma ≈ FWHM / (2·SNR)` (DESIGN §6), floored/capped. Validate with a
      NIS consistency check across the SNR sweep. Flag this in the report as an innovation point.
      **`R` must also be inflated when the detection carries `clipped` or `saturated`** (Phase 3).
      These are bias sources, not variance sources, so the honest treatment is to widen the
      measurement uncertainty and let the filter lean on its prediction, rather than trusting a
      centroid we already know is systematically displaced.
- [x] `src/filtering/track.py` — track management, M-of-N confirmation/deletion, coast logic,
      **two-point initiation and gate-lockout recovery** (DESIGN §6.2)
- [x] `src/control/pid.py` — PID with anti-windup, slew-rate limiting
- [x] `src/control/controller.py` — pixel error → angular rate command + Kalman feedforward.
      **Consumes the Kalman-smoothed state only, never the raw per-frame centroid** (DESIGN §7.2);
      derivative term taken from the filter's velocity state rather than by differencing
      measurements. Jitter at frame rate is above loop bandwidth and cannot be rejected — feeding
      it to the PID injects it rather than attenuating it.
- [x] `src/control/statemachine.py` — SEARCH / TRACK / COAST with hysteresis.
      Tags each acquisition event as **in-FOV** or **search-limited** (DESIGN §7.5).
- [x] `src/control/search.py` — Archimedean spiral acquisition scan.
      Arm spacing is **FOV-derived** (`0.9 · min(fov_w, fov_h)`), never a stored constant.
- [x] Tests: controller step response; max-trackable-velocity sweep → save plot

> **HARD GATE — LIFTED.** The PID gains were PROVISIONAL pending a step-response test. That test
> now exists in `tests/test_control.py`, runs **with the real Kalman filter in the loop**, and
> passes. Gains updated from the analytic (8, 0.5, 1.5) to the measured **(12, 1.0, 0.0)**:
> Kp raised because the filter's lag inside the loop made the analytic value settle in 1.07 s
> rather than 0.30 s; Ki raised because platform-drift rejection at 150 px/s needs it (19.5 px at
> ki=0, outside budget); Kd measured as zero because Kalman feedforward already does the velocity
> cancellation and the derivative only amplifies estimator noise.
> `config/default.json` now carries `_VALIDATED` in place of `_PROVISIONAL`, and the startup
> warning no longer fires. `tests/test_config.py` still covers the warning *mechanism* against
> the next set of untrusted gains.

**Done when:** closed loop holds lock on all four mandatory motions, and you can state the
maximum trackable target velocity with evidence.
**MET for the velocity envelope** — `docs/figures/trackable_velocity.png`. Max trackable speed is
**279 px/s (1.75 deg/s)** against a 10 px budget, which is **35% of the 800 px/s slew ceiling**:
the loop bandwidth binds long before the mechanism does, so quoting the slew ceiling would
overstate the envelope threefold. The sweep continues past failure to 1200 px/s.

> **Control findings.**
> 1. **Derivative double-count.** The derivative was fed the target velocity rather than the
>    *error* rate, making `kd` a second feedforward. Cost 2-4x in pointing error and presented as
>    "kd is harmful". Fixed; `kd=0` remains optimal afterwards.
> 2. **Feedforward and integral are partially redundant** on constant-velocity targets, so the
>    obvious test measures nothing. Isolated with ki=0; feedforward's genuine advantage is on
>    manoeuvring targets.
> 3. **The FOV trade-off inverts in a pixel-defined canvas** — a wider FOV searches *slower*
>    (23.1 s vs 5.8 s), because arm spacing in pixels is fixed by resolution while deg/px
>    coarsens. Worse on both counts, contrary to the standard argument.
> 4. Re-acquisition was checked for the Phase-4 lockout at 150-900 px/s: velocity is re-derived
>    by fresh two-point initiation (900.9 vs true 900.0), no lockout.

> **Filtering findings.**
> 1. **The theoretical accuracy law understates our measured error by ~1.32x.** Setting `R` from
>    the raw law would tighten the gate by a third and reject good detections. Calibrated against
>    the Phase 3 sweep.
> 2. **The validation gate can lock itself out, silently.** Single-detection initiation assumes
>    zero velocity; against a moving target the prediction lags, the collapsed covariance makes
>    the gate tighter than the lag, and rejection prevents the velocity ever being learned. Error
>    grew to 27 px with the detector working perfectly. Fixed by two-point initiation + ungated
>    confirmation + lockout re-initiation: track loss **3/12 → 0/12**.
> 3. A first single-run comparison suggested the R calibration *caused* losses; across 12 seeds
>    the loss rate was identical (3/12 both) and calibration gave lower error whenever the track
>    held. Single runs are not evidence here.

---

## Phase 5 — Telemetry and Reporting

- [x] `src/telemetry/metrics.py` — acquisition/re-acquisition timers, error stats, loss rate, FPS.
      Acquisition is reported as **two separate populations** (in-FOV vs search-limited, DESIGN
      §7.5), each with count, mean and worst case. Never pooled.
- [x] `src/telemetry/logger.py` — per-frame CSV + JSON, with metric definitions **and the
      coordinate convention** in the header.
      **Per-frame columns must include `clipped`, `saturated` and `spot_scale_from_fallback`.**
      These are the attribution columns: an error excursion in a Benchmark-2 run should be
      explainable directly from the log, without re-running anything.
- [x] `src/telemetry/benchmark.py` — **unthrottled capacity mode** (DESIGN §9.1). Feeds a
      pre-generated frame buffer through the identical vision + filtering + control chain as fast
      as it will run, with rendering, noise synthesis, GUI and disk I/O outside the timed region.
      Reports mean/p50/p95/max per-frame processing time and max sustainable FPS.
      **Rationale:** in a real-time run the 30 Hz producer caps the logged FPS at 30, so a
      pipeline that silently degrades from 200 Hz to 31 Hz still logs 30 FPS — right up until
      the frame we miss. Capacity and real-time rate are reported as two distinct figures, clearly
      labelled, and capacity is what substantiates the ≥20 FPS claim with headroom.
- [x] `src/telemetry/report.py` — auto-generated HTML/PDF summary with plots. Must show real-time
      rate and pipeline capacity separately, and never conflate them.
- [x] Tests: metric definitions produce expected values on synthetic known-answer sequences.
      Run the capacity benchmark as a CI-style regression check so throughput loss is caught the
      day it lands, not during the demo.

**Done when:** a headless run automatically drops a complete, evaluator-readable performance
report on disk with zero manual steps. **MET** — `python -m src.main --headless` writes
`logs/frames.csv` and `logs/report.html` with no manual step. Also wired `src/sim/source.py`
(`SimulationFrameSource`) and closed the control loop in `src/main.py`, so the pipeline now runs
end to end: in-FOV case locks at frame 4 with **97.8% retention, 0.063 px centroiding RMSE**;
search-limited circular case (radius 400, never in the initial viewport) acquires in **2.27 s**
with **98.0% retention**.

> **Integration findings — four coordinate/observability bugs, all found from the frame trace.**
> 1. **Loss rate could go negative.** Retention counted all locked frames against a
>    target-present denominator, giving retention 2.0 and loss rate −1.0 whenever any frame
>    lacked a target (occlusion, fade, beacon off-canvas).
> 2. **The Kalman was fed frame-local coordinates** while the camera slewed, so the target
>    appeared to jump and every measurement was gated out — detection perfect at 0.02–0.08 px
>    while pointing error grew without bound. The filter models constant velocity in the *world*.
> 3. **Camera jitter is unobservable and must be inside `R`.** The viewport is extracted at
>    boresight + jitter while the reported angle excludes it, so frame→canvas conversion carries
>    the jitter as error. Without it, sigma was 0.028 px against 1–2 px innovations: NIS in the
>    thousands. `KalmanParams.unobservable_sigma_px` now carries it, which took retention from
>    36% to 98%.
> 4. **SEARCH kept sweeping past a visible target.** The beacon was visible for 24 consecutive
>    frames — ample for M-of-N plus K — but the camera kept driving the spiral, so the pointing
>    error never entered the 40 px lock window. SEARCH means "no estimate", not "ignore the
>    estimate we have".

---

## Phase 6 — Mode B (Benchmark video ingestion)

**Do this earlier than feels natural — it validates the whole abstraction.**

- [x] `src/video_source.py` — `VideoFrameSource`, auto-detect resolution, colour→gray, native rate
- [x] Verify the vision pipeline runs unmodified on video input
- [x] Self-generated `.mp4` test files with known ground truth in `scenarios/`.
      **HARD REQUIREMENT: every test video must round-trip through real H.264 at a realistic
      bitrate.** Raw or lossless frames are not acceptable as a validation input — evaluator
      input is compressed `.mp4`, and a codec smears a small bright spot across transform blocks,
      rings around the highest-contrast feature in the frame (which is our beacon), and mangles
      impulse noise into something quite unlike the noise model we simulated. Validating against
      lossless video validates nothing we are actually scored on.
- [x] **Measure the centroid bias attributable to compression alone**: run the identical frame
      sequence through the pipeline lossless vs H.264-encoded and difference the results. That
      delta is a report figure and bounds the error floor Benchmark-2 can possibly achieve.
      Sweep it across at least two bitrates so the trend is visible.
- [x] Centroiding-error log format validated end-to-end, in **full-frame source pixel
      coordinates** under the documented pixel-centre convention, stated in the log header.
- [x] Robustness pass: vary brightness, contrast, noise, resolution, bitrate and colour/grayscale
      in test videos. Include at least one case at a resolution and spot size far from our
      defaults — this is what exercises the scale-relative parameterisation from Phase 0.

> **RISK — full-canvas video throughput.** The spec says evaluator video covers "a complete
> screen", which may mean the full 2000×2000 canvas rather than a 640×480 viewport. That hits the
> budget twice: decode cost for 4 Mpx frames at 30 fps, and full-frame acquisition search over
> 4 Mpx before lock. ROI processing only helps *after* lock. Measure decode and first-lock cost on
> full-canvas video **early in this phase, not at the end**. If it will not hold ≥20 FPS, the fix
> is coarse-to-fine search (downsampled full-frame scan to localise, full-resolution ROI to
> centroid) — not a threshold hack.

**Done when:** you can hand the system an arbitrary `.mp4` it has never seen and get a valid
centroiding-error log out. **MET** — `src/video_source.py` plus `scenarios/generate.py`.
The vision pipeline runs **completely unmodified**: the same `VisionPipeline` object processes
simulation frames and evaluator video, and no mode argument reaches `src/vision/` at all.

> **Mode B findings.**
> 1. **OpenCV's H.264 writer silently falls back to MPEG-4 Part 2** when no hardware encoder
>    exists — 78 KB/frame, essentially lossless. That would have invalidated every compression
>    measurement while appearing to succeed. Generation now pipes to software libx264 (4.3–17.9
>    KB/frame) and a fallback is a hard error. A test pins the file size.
> 2. **Area gate re-validated under H.264, risk direction confirmed.** Compression grows
>    residual artifacts (median 1.0 → 2.0 px, p90 3.0 → 4.0 px) and gate rejection falls
>    96% → 88%. Margin erodes but the gate holds: artifacts reaching the beacon's size class stay
>    at 0.7%.
> 3. **rho band re-validated; cluster centres hold.** Shifts are +0.002 (gaussian) and −0.001
>    (square) at 4000 kbps, growing to −0.018 / −0.010 at 1000 kbps. The diagnostic stays honest.
> 4. **Compression-only centroid bias is ~0.01 px** — 0.0086 px at 4000 kbps, 0.0181 px at
>    500 kbps. Comparable to the 8-bit quantisation floor and 500× inside the 10 px budget.
> 5. **Full-canvas 2000×2000 misses 20 FPS full-frame (2.9 FPS) and meets it comfortably with
>    ROI** (63/91/118 FPS at 256/128/64 px). Decode is cheap at 6.6 ms. So steady-state tracking
>    meets the requirement on a full canvas; the acquisition frames, which must run full-frame,
>    do not.
> 6. **`lowlight_impulse` never locks, and the cause is detection, not filtering.** 149 of 150
>    detections are off-target with a median error of 282 px against a 367 px mean random-point
>    separation — compression artifacts at 1200 kbps consistently outrank a dim beacon in the
>    flux ranking. Bisecting against the MAD floor and the track contest showed neither is
>    responsible (all four combinations give 98%), and spot scale barely matters (97.3–98.0%
>    across a 2× range). Reported via the zero-lock envelope note rather than as blanks.
>    Note the measured SNR of 15.9 is the *artifact's*, not the target's — the beacon reads 62.6
>    on the single frame it is found — so a high SNR on a failing run must not be read as
>    evidence the target was detectable.
> 7. A low-light heavily-compressed clip (SNR ≈ 7.9) tracks poorly — consistent with the Phase 3
>    SNR curve, which put the detection-limited boundary at SNR ≈ 10. Not a new failure; a
>    confirmation of the characterised envelope.

---

## Phase 7 — GUI

Last, deliberately. A headless system with great logs scores better on the 60% benchmark stages
than a pretty GUI around a fragile tracker.

- [x] `src/gui/main_window.py` — PySide6 shell
- [x] `src/gui/viewport_widget.py` — live camera view with estimated/true centroid overlay (implemented in `widgets.py`)
- [x] `src/gui/plots.py` — pyqtgraph live error and FPS strip charts (implemented in `widgets.py`)
- [x] `src/gui/controls.py` — motion selector, noise selectors, atmospheric preset, parameter entry
- [x] `src/gui/mode_panel.py` — Mode A / Mode B switch, video file picker (implemented in `controls.py`)

**Done when:** every mandatory function from the spec is visibly demonstrable in the GUI within
a 10–15 minute demo.

---

## Phase 8 — Packaging and Documentation

**Start the PyInstaller build at ~50% overall completion, not here.** This phase is for
finishing, not for discovering problems.

> **Packaging done early, before Qt (Phase 8 work pulled forward).** Headless build only, so the
> Numba/llvmlite and OpenCV problems are isolated from the Qt plugin problems.
> Lean build **103 MB**, `--with-numba` **181 MB**. `scripts/build.sh`, `docs/PACKAGING.md`.
>
> **Findings.**
> 1. **Numba/llvmlite are not actually imported anywhere in `src/`** — only a config flag exists.
>    `libllvmlite.so` alone is 179 MB, so bundling it now would nearly double the binary for code
>    never called. Off by default, but the mechanism is implemented and **verified working**
>    (frozen `--selftest` reports "llvmlite loaded and compiled"), so a future hotspot only needs
>    the flag flipped.
> 2. **The bundled config was unreachable when frozen.** A relative default path resolves against
>    the working directory, not `sys._MEIPASS`, so `./fsoc-tracker --headless` from a user's home
>    directory failed. Fixed with bundle-aware resolution.
> 3. **`build_frame_source` never got wired to `VideoFrameSource`** — video mode silently ran the
>    Phase-0 placeholder, reporting 1800 frames for a 150-frame file. Mode B is 30% of the grade
>    and the executable could not run it. Now verified frozen: identical to source at
>    0.1143 px centroiding RMSE.
> 4. **ffmpeg is development-only.** It generates test video; it is never invoked by the shipped
>    application, which decodes through OpenCV's bundled libraries. A test enforces that `src/`
>    never shells out.
> 5. Frozen and source outputs are **byte-identical on all 28 non-timing CSV columns** across
>    180 frames.

- [x] PyInstaller spec file, which **must be named `fsoc-tracker.spec`**; bundle Numba
      llvmlite binary, Qt plugins, OpenCV.
      The name is load-bearing: `.gitignore` ignores `*.spec` but carries an explicit
      `!fsoc-tracker.spec` negation so this file is version-controlled. The standalone
      executable is a mandatory graded deliverable, so its spec is source, not build
      output. Any other filename silently falls back into the ignore rule and the spec is
      one `git clean` away from being lost.
- [ ] Test the frozen executable on a **clean machine with no Python installed** — procedure
      written up in `docs/PACKAGING.md`; **not yet executed**, needs a container or VM
- [x] Technical report (10–15 pages) — reuse `docs/DESIGN.md` as the backbone
- [x] User manual — installation, operation, parameter configuration, GUI description
- [ ] 3–5 minute demo video (optional deliverable, worth doing)

---

## Optional / bonus (only after everything above)

- [x] Lightweight CNN spot validator via ONNX Runtime (the explicit "AI" component)
- [ ] Multiple simultaneous targets + data association
- [ ] IMM filter for maneuvering targets
- [ ] Additional motion patterns (spiral, sinusoidal, user-defined)
- [ ] Colour camera mode
- [ ] OpenCV tracker comparison (KCF/CSRT/MOSSE) as a report benchmark table

---

## Standing priorities (from the grading rubric)

1. **Benchmark-1 + Benchmark-2 = 60%.** Vision core + automatic logs + unseen-input robustness.
2. **Functional Verification = 20%.** Every mandatory function visibly working in the demo.
3. **Technical Evaluation = 20%.** The classical-vs-AI argument, the SNR curve, the slew-rate
   feasibility analysis.
