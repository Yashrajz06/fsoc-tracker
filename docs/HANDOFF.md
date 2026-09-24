# Handoff — large-spot association failure

> **PHASE A UPDATE (see "Phase A findings" at the foot of this file): the threshold hypothesis in
> §2 is FALSIFIED. The mechanism is gate lockout with a near-perfect detector, not detection.
> §2 and §3 are retained as written for the record; read the Phase A section for what is true.**

Written mid-investigation, at low context, immediately after retracting a wrong conclusion. The
retraction is first because the next session's most likely failure mode is re-deriving it.

**Suite is green at 662 passed (`pytest tests/ -m "not slow"`).** Nothing here is a broken build.

---

## 1. RETRACTED: the spot-scale bootstrap was NOT the cause

A previous session concluded that large-spot association failure was caused by the spot-scale
estimator under-reporting, which shrank `max_blob_area_px` and gated the true beacon out.

**That conclusion is wrong.** Measured on the `sz20` clip (true FWHM 14.1 px):

```
f 0  est 16.27 (ok)   tracker  nan
f 2  est 15.84 (ok)   tracker 15.84
f 8  est 15.84 (ok)   tracker 15.94
```

The estimator reports **15.8–16.3 against a true 14.1** — slightly *over*, accurate to ~12%,
flagged `ok` from frame 0, adopted by frame 2, stable thereafter. There is no under-reading and
no ratchet. The Phase 3 validation of the half-max contour estimator (0.4–10.7% error across
eight cases, including sigma=6 / FWHM 14.13 at 0.4%) holds in pipeline context as well.

### Why the earlier trace was wrong — do not repeat this

The trace that produced the wrong conclusion called the pipeline like this:

```python
m = pipe.process(frame, fwhm_px=5.887, from_fallback=True)
```

It **forced the fallback scale**, which is a configuration `src/runner.py` never uses — the runner
passes `scale.estimated_fwhm_px` from the live `ScaleTracker`. Under that forced fallback the area
gate is `[4, 900]` and the beacon survives 0/26 frames; with the true scale the gate is
`[23, 5153]` and it survives 20/26. Both numbers are real, but the 0/26 describes a hypothetical
system, not the shipped one, and it was reported as though it described the shipped one.

This is the second conclusion lost to the same mistake (the first: Phase 6 measured ROI vs
full-frame vision cost in isolation and reported the speed-up as if the run loop used ROI — it did
not; the ROI path was unwired until this session). A standing rule has been added to `CLAUDE.md`.

---

## 2. Current leading hypothesis: the adaptive threshold degenerates

Not yet confirmed. Stated with its evidence so it can be attacked rather than assumed.

Measured on a **clean** `sz20` frame (1920x1080, bitrate 4000, gaussian_sigma 6):

```
thr 0.02 | residual median 0.00  p99.9 0.03  max 1.00
         | px > thr 9818 (0.47% of 2.07M)
         | 2856 blobs pre-gate -> 1890 post-gate
```

`vision.preprocess.normalise_per_frame` scales the top-hat residual to `[0, 1]`. The resulting
distribution is extremely peaked at zero — median 0.00, p99.9 only 0.03 — so
`mean + k*sigma` lands at **0.02** and admits 0.47% of two million pixels. That is where the
1139–2856 candidates on a visually clean clip come from.

**Proposed causal chain:** degenerate threshold → thousands of spurious candidates → the true
beacon must out-rank ~1890 noise blobs by flux, and intermittently loses.

**Supporting evidence:** the association-failure rate across the size sweep is **non-monotonic**
— 0% / 0% / 39.8% / 13.6% at 5 / 10 / 15 / 20 px. A lottery against a large noise population
explains a non-monotonic rate; a smooth function of spot size does not. Spot size still gates the
effect (0% at 5 and 10 px), so the two interact, but size is not the mechanism.

**Falsification test:** the size sweep clips are in the scratchpad `sweep/` directory and are
regenerable from `scripts/`-style specs (1920x1080, frames=120, peak 240, background 60,
gaussian_sigma 6, bitrate 4000, motion figure8, only `size_px`/`sigma_px` varying). Vary the
threshold method, hold size constant. If association failure tracks the threshold rather than the
size, the hypothesis is confirmed.

---

## 3. The exact next question

**Why does `_separating_threshold()`'s midpoint guard not engage on a residual this peaked?**

That guard was added in Phase 3 specifically for the case where `mean + k*sigma` and the
percentile method both degenerate. A residual with median 0.00 and p99.9 0.03 is squarely the
degenerate case it exists to catch. Either its trigger condition does not fire on this
distribution, or it fires and its midpoint is itself near zero. Find out which before changing
anything — `src/vision/detect.py`, `adaptive_threshold()` and `_separating_threshold()`.

Note `vision.detection.threshold_method` is `mean_plus_k_sigma` and `percentile` is 99.5. At
0.47% of pixels above threshold, the percentile method would cut at almost exactly the same
place, which is itself worth checking.

---

## 4. Open items, in order

1. **Midpoint guard diagnosis** (above). Everything else waits on this.
2. **Area-gate asymmetry policy.** The Phase 3 argument — underestimating spot scale is
   catastrophic, overestimating is merely lossy — currently applies only to the top-hat
   morphology kernel. It applies just as directly to the **upper blob-area bound**, which should
   tolerate an overestimated scale rather than discarding a too-large blob.
   **This is a correctness fix on its own merits. It is NOT a fix for the bug in §2**, and must
   not be reported as one — see the retraction above for what happens when the two are conflated.
3. **Re-run the size sweep** (5 / 10 / 15 / 20 px) to confirm whatever fix lands.
4. **Re-run the six-clip table**, ROI on and off. `fhd_1920_bigspot` is the clip to watch: it has
   now broken two independent mechanisms (the AI discriminator, and ROI), both association-related,
   and both plausibly downstream of §2.
5. **Re-acquisition measurement under ROI** (§5).

---

## 5. Re-acquisition prediction — on record, unmeasured

Written before measuring, deliberately, because predictions have done badly in this project and a
recorded prediction is informative either way:

> `expand_on_loss_factor` will restore lock after a **genuine loss** (detections stop,
> `_missed_streak` increments, the window grows geometrically until the target is recovered). It
> will **not** help the wrong-lock case, because the detector keeps confidently finding something
> every frame, `_missed_streak` never increments, and the window never grows.

Stated confidence: **moderate** on the first half, **high** on the second. The second is close to
mechanical from the code path. The first could fail if the boresight has slewed far during the
loss and the window, capped at `max_size_*`, still cannot reach the target.

---

## 6. Frozen state — do not finalise these

- **`docs/TECHNICAL_REPORT.md` §7 is unfinalised.** The ROI path is now wired and sized from the
  slew ceiling, and measured at 294.7 / 115.9 / 73.9 FPS mean (640x480 / 1920x1080 / 2000x2000),
  all clearing 20 FPS at p95, with retention and RMSE unchanged from full-frame to three digits.
  But §7 must also state plainly that the 3.4–18.8 FPS Mode B figures elsewhere in the report are
  **full-frame, no-lock, no-ROI** measurements. Two unexplained FPS numbers in one report read as
  a contradiction — handle it the way the pooled/excluded RMSE convention is handled, by naming
  the measurement condition next to every number.
- **Slide 15 is out of the deck** pending this investigation.
- **The ROI shipping decision is undecided.** ROI is neutral-or-better on four of five clips and
  transforms throughput, but converts `fhd_1920_bigspot`'s 10.8% association failure into 100%,
  because a wrong lock is self-reinforcing once the window follows it. Do not decide until §2 is
  resolved — if the threshold is the cause, the regression may disappear.
- **The AI discriminator ships disabled** (`ai.enabled: false`), failing its gate on the same
  clip. Same caveat: may be downstream of §2.
- **Suite green at 662 passed**, 8 deselected (slow).


---

# Phase A findings — root cause located

Phase A is **diagnostically complete**. The threshold hypothesis is dead, the root cause is
identified with a frame trace, and three unrelated leaks were found and fixed on the way. The
two fixes the root cause implies are **proposed, not implemented** — see §P5.

---

## P1. The threshold hypothesis is falsified

Re-measured with current code (ROI wired, q=1300, square default). Association failure with
median candidates per frame:

| clip | ROI off | ROI on | candidates/frame (ROI off) |
|---|---|---|---|
| sz5  | 0.0 % | 0.0 % | 2478 |
| sz10 | 0.0 % | 0.0 % | 1517 |
| sz15 | 39.8 % | 99.2 % | 2691 |
| sz20 | 13.6 % | 99.1 % | 1106 |

**sz5 carries 2478 candidates per frame with zero failures; sz20 carries 1106 with 13.6 %.**
Candidate count is therefore not the mechanism, and the "degenerate threshold → thousands of
blobs → flux lottery" chain in §2 is dead.

`normalise_per_frame` was never touched and the Phase A constraint on preserving it never bound,
because no threshold change was proposed or made.

## P2. The §3 question, answered: the guard's trigger never fires

`_separating_threshold()` returns the computed level unchanged unless `level <= median_level`.
Observed on the shipped path across 25 consecutive frames:

* `level > median_level` on **25/25** calls
* guard fired on **0/25**
* the level sat at **4.07 % of the way from median to peak**

It catches *total* collapse only. A threshold that is nominally above background yet admits
almost everything passes it untouched. **This is a real weakness and worth fixing on its own
merits** — it is simply not the cause of this bug.

## P3. Root cause: one spurious first detection, unbounded initiation velocity

`sz15`, ROI off, from run start. `detErr` is the detector against truth; `fusedErr` is the track.

```
  f   detErr  fusedErr     source         reason      NIS   |v|est
  0  499.907        -               initiating          -        -
  1    0.146        -               initiating          -        -
  2    0.162     0.16   measured      confirmed          -   7550.6
  3    0.166   250.18  predicted      gated_out      24.63   7550.6
  4    0.061   500.21  predicted      gated_out      93.41   7550.6
  5    0.237   750.24  predicted      gated_out     193.94   7550.6
  ...
 18    0.049  4000.64  predicted  contest_incumbent_held  1147.83  7550.6
  ...
 31    0.178  7251.05  predicted      gated_out    1305.59   7550.6
```

The **first detection is 499.9 px wrong**. `Track._initiate` derives velocity from that spurious
point and frame 1's correct one, yielding **7550.6 px/s against a true target speed of
71.1 px/s** — 106x too fast. The track then flies in a straight line at exactly 251.7 px/frame
and never returns. Every subsequent detection is correct to ~0.06 px and every one is gated out.

The gate rejections, the NIS explosion, the ROI amplification to 99 % and the whole "large-spot
association failure" are all downstream of that single unvalidated velocity.

`_initiate` computes `vx = (last_x - first_x) / span` with **no sanity check of any kind**. Any
spurious first detection becomes a permanent velocity.

### Why this misled us for three sessions

sz5 and sz10 do not fail because their first detection happens to be correct. It is an
**initiation lottery**, not a property of spot size — larger spots merely make a spurious first
detection more likely during acquisition, before the scale estimate is adopted. That is also the
true explanation of the **non-monotonic 0 / 0 / 39.8 / 13.6 %** rate that made a "lottery against
noise" hypothesis look plausible. The lottery is real; it is at *initiation*, not at ranking.

## P4. Three leaks fixed (commit `1d7fcfe`)

All three fed camera/platform-derived quantities into filtering or ROI sizing regardless of
whether the source has a camera. All are now keyed on `FrameSource.supports_pan_tilt`. This is
the third, fourth and fifth instance of this shape — after `gate_shape` computing frame-fraction
gates from a crop, and pointing error in the lock criterion. **Assume a sixth until checked.**

| leak | effect | status |
|---|---|---|
| **A** camera jitter → `R` | `sigma_meas` pinned at 1.6668 px on every clip, across spot 5–20 px and SNR 685–5242 | fixed |
| **B** `dt` from config, not the source timebase | filter advanced at `camera.update_rate_hz` while `VideoFrameSource` timestamped from the clip's real rate | fixed |
| **C** ROI sized for slew and jitter a fixed camera lacks | 75 px where 64 px is needed | fixed |

**Leak B is a live Benchmark-2 risk in its own right.** An evaluator clip at 25 or 60 fps would
have advanced the constant-velocity model at the wrong rate on every frame, silently, with the
prediction falling behind by a fixed fraction each time. Nothing would have looked broken.

`sigma_meas` before/after: `1.6668` on all four clips → `0.0200` full-frame (floored) and
`0.1704 / 0.2615` on the large-spot ROI cases. Association failure **unchanged**, as expected:
inflating `R` makes the gate *more* permissive, so the leak could never have been the cause.

## P5. Two proposed fixes — BOTH IMPLEMENTED

These landed as separate logical changes within src/filtering/track.py.

**1. Bound the initiation velocity.** `TrackParams.max_initiation_velocity_px_s` (default
4000 px/s = 5× the slew ceiling). Any two-point pair implying a higher speed causes `_initiate`
to discard the oldest pending detection and wait for a fresh pair. 4000 px/s sits comfortably
above any legitimate target speed (900 px/s is the highest in the test suite) but far below
the 7550 px/s phantom. Setting the bound at the slew ceiling (800) was tried and rejected: it
broke re-acquisition at 900 px/s, which is a legitimate speed even if untrackable. Tested in
`tests/test_filtering.py::test_spurious_first_detection_does_not_produce_a_phantom_track`.

**2. Ask why the contest let an incumbent carrying NIS 1147 hold.** Fixed via
`TrackParams.incumbent_max_mean_nis` (default 100). An incumbent whose mean NIS over the contest
window exceeds this ceiling loses unconditionally — a hypothesis explaining nothing should never
win. Well below 1147 (where the fix fires), well above 2 (the expected mean NIS for a consistent
2-DOF filter). Tested in
`tests/test_filtering.py::test_contest_rejects_an_incumbent_with_catastrophic_nis`.

## P6. Phase B/C verification update

The post-fix measurements below are from the shipped `TrackingRunner` path, not a forced vision
diagnostic.  They supersede the frozen-decision note above.

- **Area-gate asymmetry is implemented.** The dynamic upper area gate is `2 ×
  max_blob_area_spot_multiple` (66× nominal area), while the lower gate is unchanged.  At the
  5.89 px calibration scale this is approximately 1800 px² rather than 900 px².  A fixed fallback
  maximum was rejected because it would again cap an otherwise valid large spot.  Configuration
  tests pin this policy.
- **Size sweep, ROI on:** the reproducible 1920×1080 H.264 clips measure 1.7% / 0.8% / 1.7% /
  0.0% association failure for 5 / 10 / 15 / 20 px spots respectively (120 frames each).  The
  old 39.8% / 13.6% large-spot failure is gone, but the stronger all-zero gate is not met.
  `scenarios.generate.SIZE_SWEEP_SPECS` defines and regenerates all four fixtures.
- **ROI decision remains conditional.** `fhd_1920_bigspot` with ROI on now measures 0.7%
  association failure and 95.3% lock retention, instead of the former near-total regression.
  This is strong evidence for ROI, but does not justify claiming a zero-failure shipping gate.
- **AI remains disabled.** On the same large-spot ROI run, the current ONNX model measures 1.4%
  association failure and 86.7% retention, worse than the classical path.
- **Phase C is measured and passes in the deterministic in-FOV recovery case.**
  `tests/test_reacquisition_benchmark.py` drives source → vision → filtering → state machine →
  telemetry through an eight-frame dropout. Loss is declared at frame 20 and lock returns at
  frame 28: **0.2667 s**, within Parameter 19's ≤1 s requirement. Telemetry now consumes the
  authoritative `TRACK → COAST → TRACK` lifecycle transition, preventing a second delayed miss
  counter from suppressing a valid re-acquisition event.

## P7. Remaining handoff state

- ROI can be enabled for demonstrations with its measured residual-risk caveat; its formal
  zero-failure shipping decision remains open.
- The AI discriminator ships disabled (`ai.enabled: false`).
- **The slide deck does not exist.** There is no `docs/ppt/` directory; slide content is to be
  generated at Phase F rather than updated.
- Report §7 needs its now-stale `fhd_1920_bigspot` caveat updated before submission.
