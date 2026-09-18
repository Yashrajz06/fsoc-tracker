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

# Phase A findings — the threshold hypothesis is falsified

## The §3 question is answered: the guard's trigger never fires

`_separating_threshold()` returns the computed level unchanged unless `level <= median_level`.
Observed on the shipped path over 25 consecutive frames: `level > median_level` on **25/25**
calls, and the guard fired on **0/25**. The level sat at **4.07 % of the way from median to
peak** — nominally above background, nowhere near separating. The guard only catches *total*
collapse, so a weak-but-positive threshold passes it untouched.

That is the answer to §3. It is **not** "fires with a near-zero midpoint".

## But the threshold is not the cause

Re-measured with current code (ROI wired, q=1300, square default). Association failure, ROI off
versus ROI on, with median candidates per frame:

| clip | ROI off | ROI on | candidates (ROI off) |
|---|---|---|---|
| sz5  | 0.0 % | 0.0 % | 2478 |
| sz10 | 0.0 % | 0.0 % | 1517 |
| sz15 | 39.8 % | 99.2 % | 2691 |
| sz20 | 13.6 % | 99.1 % | 1106 |

ROI off reproduces the original 39.8 / 13.6 figures exactly. **sz5 has more candidates (2478)
than sz20 (1106) and zero failures**, so candidate count cannot be the mechanism — which
falsifies the causal chain in §2.

## The real mechanism: gate lockout on a near-perfect detector

On the failing frames (ROI off), where the fused error exceeds 40 px:

| clip | detection error (median) | estimate_source | gated_out |
|---|---|---|---|
| sz15 | **0.068 px** | 100 % `predicted` | **47/47** |
| sz20 | **0.068 px** | 94 % `predicted` | **15/16** |

The detector is essentially perfect and **every correct measurement is rejected by the
Mahalanobis gate**, after which the filter coasts away on prediction. Confirmed independently at
the blob level: on failing frames the beacon is a pre-gate blob in 47/47 cases, passes the area
gate in 47/47, and is the flux winner — winner and beacon are the same blob to 1.00x on area,
flux and peak.

With ROI enabled the detection error on failing frames becomes 604–1007 px, because the window
follows the diverged track. That is the self-reinforcement already described in report §7.5, and
it is an amplifier, not the cause.

## A concrete defect found along the way

`sigma_meas_px` is **1.6668 px on every clip**, identical across spot sizes 5–20 px and aperture
SNR 685–5242. Adaptive R is not adapting at all in Mode B. The cause: the runner sets
`unobservable_sigma_px` from `noise.camera_jitter.max_px_per_frame / 3` = 1.6667, and **Mode B
inherits it even though Mode B has no camera and applies no jitter** — the video is the scene.
So R is pinned at 1.67 px while the detector delivers 0.068 px.

This must be fixed on its own merits. Whether it is the whole of the lockout is **not yet
established** — inflating R makes the gate more permissive, so it does not by itself explain
rejection, and NIS is nonetheless enormous (median 193.8 at sz15 against a 9.21 gate), implying
the track had already diverged by hundreds of pixels.

## Where to go next

The fix is in **filtering**, not in the threshold. The Phase A constraint about preserving
`normalise_per_frame` therefore does not bind — no threshold change is proposed.

Open question: why the track diverges far enough to produce NIS in the hundreds while the
detector is reporting 0.068 px, and why `track.py`'s lockout and competing-hypothesis machinery
— written for exactly this failure — does not recover. `contest_opened` appears only 3 times in
47 failing frames, and where a contest ran the **incumbent held**.
