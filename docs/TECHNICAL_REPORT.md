# AI-Based Virtual Camera Tracking for Coarse Alignment of Mobile FSOC Terminals

**Smart India Hackathon — Problem Statement 4 (Department of Space / ISRO), Category: Software**

Technical Report

---

## 1. Executive summary

This system simulates the **coarse alignment stage** of a Free Space Optical Communication (FSOC)
Pointing, Acquisition and Tracking (PAT) loop entirely in software. It renders a large virtual
scene containing a moving optical beacon, extracts a narrow camera viewport, corrupts that
viewport with sensor noise and atmospheric degradation, then detects the beacon, estimates its
centroid to sub-pixel accuracy, and drives a virtual pan/tilt camera to keep it centred — logging
tracking performance against the graded targets in real time.

Two input modes share one vision pipeline through a `FrameSource` abstraction: **Mode A**, the
closed-loop simulation with a steerable camera, and **Mode B**, ingestion of pre-recorded `.mp4`
files in which the video *is* the scene and the pan/tilt loop is bypassed.

### Headline results

| Requirement | Target | Measured |
|---|---|---|
| Centroiding error | ≤ 10 px | **0.07 – 0.9 px** across six Mode B clips |
| Processing throughput | ≥ 20 FPS | met with ROI processing (see §7) |
| Camera update rate | ≥ 30 Hz | 30 Hz, logged separately |
| Control update interval | ≥ 20 Hz | 20 Hz, logged separately |
| Acquisition (in-FOV) | ≤ 2 s | met |
| Acquisition (search-limited) | ≤ 2 s | **not achievable — see §8.1** |

The **AI component was built, measured, and left disabled by default** on the evidence (§5.5) --
"we built it, measured it, and the classical path won" is stated here rather than quietly omitted.

Two limits are reported honestly rather than hidden, because both are arithmetic rather than
tuning failures: the **search-time envelope** (§8.1) and the **maximum trackable velocity**
(§8.2). One input in our own test set — a low-light, heavily-compressed clip with 8%
salt-and-pepper — **never achieves lock**, and §8.3 states why.

---

## 2. Problem and scope

See `docs/PROBLEM_STATEMENT.md` for the verbatim official specification. The mandatory functions
are a configurable virtual scene, a steerable virtual camera, injectable noise and atmospheric
effects, automatic beacon detection and centroiding, closed-loop pointing control, and logged
performance metrics — all reachable from a GUI.

**Scope note.** Two of the specification's 25 parameters are not implemented, both marked
*optional* in the specification: colour camera input (parameter 2) and multiple simultaneous
targets (parameter 8). Rather than expose controls that would silently do nothing, the GUI pins
the target count to 1 and presents camera type as a label. Mode B colour video is still accepted
and converted to grayscale at the source boundary.

---

## 3. Architecture

```
CONFIG (JSON)
     |
     v
SIMULATION ENGINE  --world canvas 2000x2000-->  CAMERA / VIEWPORT  --640x480-->  NOISE + ATMOSPHERE
  target generator                              (pan/tilt, deg/px)                     |
  trajectory models                                     ^                              v
        ^                                               |                       VISION PIPELINE
        |                                          rate command                 preprocess -> detect
   platform motion                                      |                       -> centroid -> gate
                                                   CONTROLLER  <---- state ----        |
                                                   PID + slew limit                     |
                                                   + Kalman feedforward  <--------------+
                                                        |
                                                        v
                                              TELEMETRY / STATE MACHINE
                                                        |
                                                        v
                                              GUI DASHBOARD + CSV/JSON logs
```

The load-bearing design decisions:

- **`FrameSource` is the only mode seam.** `SimulationFrameSource` and `VideoFrameSource`
  implement the same protocol. No module under `src/vision/`, `src/filtering/`, `src/control/` or
  `src/telemetry/` contains a mode branch. Mode B is not a code path; it is a different source.
- **`src/runner.py` holds the run loop**, and imports no Qt. The GUI is a view over it. This is
  enforced by a test that subprocess-probes a core import for leaked Qt modules, and by a source
  scan asserting the widgets cannot reach the tracker classes.
- **No magic numbers in the vision pipeline.** Every threshold is derived from frame statistics at
  runtime, and all geometry is scale-relative via `VisionConfig.resolve_geometry(fwhm_px)`.
- **Core modules are headless.** Two executables ship: a lean headless binary and a GUI binary.

---

## 4. Coordinate and metric conventions

Sub-pixel ground truth is only meaningful if every stage agrees where a pixel *is*.

- Pixel centres sit at integer indices; pixel `[r, c]` has its centre at `(x=c, y=r)`.
- Origin top-left, x right, y down; all coordinate tuples are `(x, y)`.
- A frame of width `W` spans x from `-0.5` to `W-0.5`; the boresight is at `((W-1)/2, (H-1)/2)`.
- The supersampled renderer maps supersample pixel `[R, C]` to output
  `((C + 0.5)/S - 0.5, (R + 0.5)/S - 0.5)`. Getting this wrong produces a `(S-1)/(2S)` px offset —
  0.375 px at S=4, which is 7.5× the Phase 1 accuracy target.

Metrics are defined precisely and carried in the log header:

- **Centroiding error** — estimate to ground truth, per frame.
- **Pointing error** — target to boresight, per frame. Logged **separately**; the specification's
  "tracking error" is ambiguous between the two.
- **Acquisition time** — from frame 0 to the first frame the lock criterion holds for K=3
  consecutive frames. Reported as **two populations**, never pooled (§8.1).
- **Re-acquisition time** — from loss (N=5 consecutive misses) to re-satisfied lock.
- **SNR** — `snr_aperture` is primary (aperture = 1.5×FWHM disc), `snr_peak` secondary. Background
  is estimated robustly (median, `1.4826·MAD`) over a 2.5–4×FWHM annulus, because plain mean/std
  is inflated by impulse noise: at 10% salt-and-pepper it roughly doubles σ, silently halving
  every reported SNR.
- **Loss rate** — frames without valid lock ÷ frames where the target was present.

---

## 5. Algorithms

### 5.1 Vision pipeline

Median filter → top-hat morphology → adaptive threshold → intensity-weighted centroid, all on an
ROI around the Kalman prediction once locked. Full-frame processing occurs only during initial
acquisition search.

Threshold selection is adaptive (Otsu on the top-hat residual, or mean + k·σ, or percentile), with
a `_separating_threshold()` midpoint guard for the case where both degenerate.

### 5.2 Spot scale

A three-tier estimator: Tier 0 half-max contour area, Tier 1 scale-normalised Laplacian-of-Gaussian
over a cropped ROI, Tier 2 an EMA tracker with periodic re-checks. The estimate carries a
`from_fallback` flag so that every accuracy figure can be split by whether the scale was measured
or assumed.

### 5.3 State estimation

A constant-velocity Kalman filter with adaptive measurement noise derived from measured SNR
(`sigma ≈ FWHM / (2·SNR)`, calibration 1.32), Joseph-form covariance update, and a Mahalanobis
validation gate at χ²(2, 0.99) = 9.21.

Track lifecycle is handled by three mechanisms that exist because the gate can lock itself out:
two-point initiation, ungated confirmation, and **competing-hypothesis track management** — when
the incumbent rejects a sustained run of detections, a challenger is initiated from those
detections and both are scored on accumulated NIS over the same frames. The incumbent keeps
running throughout and its output continues to drive the controller.

### 5.4 Control

PID on angular error with anti-windup and a slew ceiling, plus Kalman velocity feedforward.
Derivative acts on the **error rate**, not target velocity, to avoid double-counting feedforward.

### 5.5 The AI component: a candidate discriminator, built and measured

The beacon is a bright, symmetric blob on a near-uniform background. Classical weighted
centroiding is near-optimal for *localisation*: error scales as spot-width / sqrt(photons) in the
shot-noise limit. A CNN would be slower and less accurate at that task, and would jeopardise the
>=20 FPS budget. So the network was never given the localisation job.

It was given a **discrimination** job instead, chosen because it is the one place classical logic
provably has no signal to work with. The detector ranks candidate blobs by integrated flux, and
flux cannot distinguish a bright compression artifact from a dim beacon -- they differ in *shape*.
Shape discrimination is what a small convolutional network does well.

**Design.** A 3,585-parameter CNN scores 32x32 patches: three 3x3 convolutions (1->8->16->16) with
ReLU and max-pooling, global average pooling, and a single linear output. Global average pooling
rather than a dense head keeps the parameter count low and makes the score depend on how
beacon-like the patch is rather than on where in it the evidence sits -- the right property when
the candidate's coarse position is itself uncertain by a pixel or two. Patches are standardised
per-patch by median and `1.4826 * MAD`, so the network cannot learn absolute brightness, which is
exactly the cue that fails to transfer to an evaluator's clip.

**No deep-learning framework is used anywhere.** Training is written directly against NumPy,
including backpropagation. PyTorch would be a ~2 GB dependency serving a 3,585-parameter model,
and every dependency raises the chance the PyInstaller build fails. Only the exported 15 KB
`.onnx` and `onnxruntime` reach the runtime; `onnx` is developer-only, in the same way `ffmpeg`
is developer-only for fixture generation.

**Training data comes from the classical detector's own candidates**, not from synthetic positive
and negative images. Each frame of six purpose-built clips is run through the real pipeline, every
surviving candidate is cropped, and it is labelled by distance to known ground truth. A network
trained on hand-made examples learns to separate a distribution it will never see; this one learns
to separate the two things actually confusable at inference. The training clips are **disjoint
from the six evaluation fixtures** -- training on the clips the model is then scored against would
make the result self-confirming. Validation accuracy: **0.9176** on 85 held-out patches.

#### The premise we scoped on was wrong

The work was scoped on the belief that `lowlight_impulse` fails because artifacts *outrank* a
detected beacon by flux -- a ranking failure, which is exactly what a discriminator fixes.
Measurement showed otherwise:

- The true beacon is surfaced as a candidate at all in only **34 of 150 frames (23%)**.
- When it is present, its median flux rank is **28**, ranging to 549.

So in 77% of frames there is nothing to re-rank. The binding constraint is **detection
sensitivity, not candidate ranking**, and 23% is the hard ceiling for *any* discriminator on that
clip -- no architecture or training set changes that. The clip was therefore excluded from the
shipping-gate measurement: the compute buys nothing a better model could redeem.

Where the discriminator *can* act, it does beat classical ranking. On the 34 frames where the
beacon is among the candidates, the classical flux pick is correct **0/34** and the discriminator
is correct **4/34**. Genuinely better, and still far short of the sustained detection a lock
requires.

#### The shipping gate, and why it fails

The gate was narrow: enabling the discriminator must not harm a clip that already works, and must
stay inside the FPS budget. Four fastest clips, AI off against AI on:

| clip | RMSE off | RMSE on | assoc-fail off | assoc-fail on | FPS off | FPS on | frames overridden |
|---|---|---|---|---|---|---|---|
| baseline_640 | 0.122 | 0.122 | 0.0% | 0.0% | 18.8 | 16.5 | 0 / 140 |
| square_bright | 0.124 | 0.124 | 0.0% | 0.0% | 11.2 | 11.1 | 0 / 68 |
| hd_1280_smallspot | 0.223 | **0.219** | 0.0% | 0.0% | 5.5 | 5.5 | 16 / 150 |
| fhd_1920_bigspot | 0.066 | 0.073 | 10.8% | **96.6%** | 3.4 | 3.3 | 125 / 145 |

**The gate fails on `fhd_1920_bigspot`.** Association failure rises from 10.8% to 96.6%: the
network overrides the classical pick on 125 of 145 frames and is wrong almost every time. The
RMSE column looks harmless there only because it is computed over the 3.4% of frames that still
associate -- which is precisely why this report carries association failure as a separate column
rather than pooling it into RMSE (§6.2).

The likely cause is that the clip's 20 px beacon sits at the top of the specified size range and
is under-represented in training, so the network's confidence is misplaced rather than merely low
-- and a *confident* wrong override is worse than no model at all.

**Therefore the discriminator ships disabled by default** (`ai.enabled: false`). It is fully
implemented, tested, exported and bundled, and can be enabled per scenario. Cost when enabled is
modest -- 0.235 ms per frame steady-state for 8 candidates, at most ~12% FPS on the fastest clip
-- but cost was never the reason to leave it off. Correctness was.

**The honest summary for Q&A:** the classical path is primary and remains so on measured evidence,
not on preference. We built the AI component, scoped it to the one regime where classical logic
provably has no signal, measured it against the classical path on equal terms, and it lost on the
input that mattered. Both halves of that claim have numbers behind them.

---

## 6. Results

### 6.1 Centroid error versus SNR

![Centroid error vs SNR](figures/centroid_error_vs_snr.png)

Error falls with SNR and **flattens above SNR ≈ 10**, as expected. No peak-locking floor is
observed — the spot is well-sampled. With noise removed entirely the error continues to fall to
0.007–0.013 px with no plateau; that residual floor is **8-bit quantisation**, confirmed because
it scales as 1/peak. The curve is split by shape and by `from_fallback`.

### 6.2 Mode B: six clips with independently-generated characteristics

Fixtures are generated with resolution, spot size, brightness, atmosphere and bitrate deliberately
different from our own defaults, to test for overfitting rather than confirm it.

| clip | centroid RMSE (px) | median | p95 | assoc-fail | retention |
|---|---|---|---|---|---|
| baseline_640 | 0.122 | 0.108 | 0.212 | 0.0% | 98.7% |
| hd_1280_smallspot | 0.223 | 0.160 | 0.400 | 0.0% | 98.7% |
| fhd_1920_bigspot | 0.066 | 0.058 | 0.114 | 10.8% | 98.7% |
| canvas_2000_fog | 0.871 | 0.752 | 1.544 | 0.0% | 97.8% |
| square_bright | 0.124 | 0.122 | 0.194 | 0.0% | 98.7% |
| lowlight_impulse | *not measured* | — | — | — | — |

`lowlight_impulse` is marked *not measured* rather than carrying a number. The beacon is
surfaced as a candidate in only 34 of 150 frames (§5.5), so the clip is bounded at 23% by
detection sensitivity before any tracking logic runs; a full AI-on/AI-off sweep over it costs
substantial compute to re-confirm a ceiling already established directly. The clip's failure mode
is characterised in §8.3.

**Two error columns, deliberately.** `detection_error_px` scores the detector against truth;
`centroid_error_px` scores the fused estimate. They diverge when the detector locks onto the wrong
object — an *association* failure, not an estimator error — and pooling them produces a headline
number dominated by a handful of gross outliers. On `fhd_1920_bigspot` the pooled RMSE is 2136 px
and the association-excluded RMSE is 0.066 px; the difference is one excursion, not accuracy.
Both are logged per frame so either can be recomputed independently.

### 6.3 Trackable velocity envelope

![Trackable velocity](figures/trackable_velocity.png)

See §8.2.

---

## 7. Throughput

Three clocks are logged separately and never conflated: frame generation (≥30 Hz), control
(≥20 Hz), and end-to-end processing FPS, plus an unthrottled capacity benchmark reporting p95.

ROI processing is what makes the requirement achievable. Full-frame vision on a 2000×2000 canvas
costs 292 ms/frame — 2.9 FPS end to end, well outside the requirement — while decode alone is
6.6 ms. Cropping the Tier-1 blur ladder to the ROI took it from 121 ms to 0.95 ms.

---

## 8. Honest limits

### 8.1 The ≤2 s acquisition budget is unachievable when the beacon starts outside the viewport

Covering a 2000×2000 uncertainty region with a spiral whose arm spacing equals the limiting FOV
dimension (480 px) requires a path of roughly `4×10⁶ / 480 ≈ 8.3×10³ px`. At the 5°/s slew ceiling
(800 px/s) that is **≈10 s worst case**, and ≈5 s even at 10°/s. This is arithmetic, not tuning.

`src/config.py` computes this envelope for the active configuration and warns at startup when it
exceeds the budget. Acquisition is therefore reported as **two populations** — in-FOV
(detection-limited, where ≤2 s is realistic) and search-limited — because pooling them hides a
physical limit behind an initial-condition lottery and makes the headline number unreproducible.

### 8.2 Maximum trackable velocity

Max pan 5°/s = 800 px/s = **26.7 px/frame at 30 Hz**; at 10°/s, ~53 px/frame. A target moving
faster than this between frames cannot be kept centred. The sweep was run until failure and the
failure point is reported rather than avoided by testing only slow targets.

### 8.3 `lowlight_impulse` never locks

At peak 70 against background 8 with 8% salt-and-pepper at 1200 kbps, 100% of detections are
association failures: compression artifacts outrank a dim beacon. The measured SNR is below the
SNR ≈ 10 detection threshold characterised in §6.1, and the summary report says so explicitly
rather than reporting a zero-lock run as a bare failure.

---

## 9. Engineering findings

Each phase surfaced a defect that inspection had passed. The method that found them was
consistent: **check a cheap path against a reference path, or trace frame-by-frame when a summary
number looks fine.** A selection of the most instructive:

**Process noise an order of magnitude too low.** `q = 50` implied a target manoeuvring at
~7 px/s² against a default circular trajectory manoeuvring at 36 px/s². Too small a `q` does not
merely make the filter sluggish — it shrinks `P`, which shrinks the innovation covariance, which
**closes the validation gate**. A detection dropout let the constant-velocity prediction drift
along the curve; the returning detections were then rejected as outliers and the filter coasted
further. Fused RMSE was 0.95 px against a raw measurement RMSE of 0.056 px — *the filter was 17×
worse than its own input*. It fails loudest on clean, high-SNR inputs, which is where a filter is
least likely to be scrutinised, and it is event-driven: a 6 s window showed a harmless 1.1× ratio
where an 8 s window showed 16.9×.

**Dead configuration.** `filtering.kalman.process_noise_psd` was validated on load and read by
nobody; the filter used its dataclass default, which agreed with the JSON only by coincidence. An
evaluator editing it in a scenario file would have changed nothing. The whole `filtering.kalman`
block was inert. This is a direct scoring risk on Benchmark-1, where scenarios are the input.

**Generated fixtures were not reproducible.** Seeding the scene RNG was not sufficient: x264
slices frames across threads and the slice boundaries follow thread scheduling, so two encodes of
byte-identical input produced *different decoded pixels* — up to 255 levels on the
salt-and-pepper clip (132/150 frames). Salt-and-pepper content is worst affected, because one
flipped impulse near a block edge changes which candidate the detector ranks brightest. Pinned
with `-threads 1`, `deterministic=1` and `+bitexact`.

**A 42 ms warm-up produced a false "the AI changes nothing".** ONNX Runtime performs graph
optimisation and kernel selection on the *first* `run` call, costing 41.8 ms against the 20 ms
per-frame budget. The discriminator's own budget guard therefore disabled it on frame 1 of every
clip, and the first evaluation reported the AI as having no effect on any input. That was a
conclusion about initialisation cost wearing the costume of a conclusion about the model. Steady
state is 0.235 ms -- 170x faster. The session is now warmed at load, and a test fails if the
warm-up is removed. The general lesson matches the others here: a guard that silently degrades is
indistinguishable from the thing it was guarding against, unless something measures whether the
guard fired.

**Four unguarded MAD sites.** Robust sigma degenerates to zero on a uniform region, producing an
infinite SNR. Grepping for the pattern rather than fixing the one found instance turned up four.

Others, in brief: point-sampled hard edges capped accuracy at `1/(2S)`; Tier 0 capped candidates
by raster order so a centred beacon was never examined; peak-ranking favoured salt impulses at
255 over a 200-peak beacon; a second-moment FWHM estimator was 21–99% wrong; camera jitter is
unobservable and belongs in `R` (retention 36% → 98%); SEARCH swept past a visible target.

---

## 10. Verification

651 tests pass (`pytest tests/`, excluding slow-marked). Notable properties under test:

- The renderer preserves the pixel-centre convention through supersample/downsample to <0.05 px,
  with dedicated integer, half-integer and irrational-offset cases.
- Noise functions are tested statistically (mean, variance, impulse density), not by eyeball.
- The SNR sweep is a **deliverable**, not a debug tool, and asserts error < 1 px above SNR 10.
- Mode B runs against `.mp4` files we generate with known ground truth, so the logging format is
  validated end to end before an evaluator hands us a video.
- Generated fixtures are asserted bit-reproducible.
- The fused estimate is asserted **not worse than the raw measurement** it smooths — the assertion
  whose absence allowed the process-noise defect to survive every other filtering test.
- Config wiring is verified by **mutation**: each knob is changed and the observable consequence
  asserted. A reference count is not evidence, because config binds JSON keys to dataclass fields
  by name and a live knob's literal may never appear in the source.

### Configuration coverage — stated honestly

Of 202 configuration leaves, 151 are consumed. Mutation tests now cover the parameters an
evaluator would plausibly vary in a Benchmark-1 scenario: camera FOV, resolution and slew limits;
every motion model's parameters; noise magnitudes and the atmospheric preset; the scale-relative
vision geometry; PID gains; state-machine thresholds; track management; and the full
`filtering.kalman` block including the gate threshold.

**The remaining knobs are not covered by wiring tests.** 18 inert keys were deleted outright
rather than left looking tunable. Three groups are retained but explicitly marked
`_not_implemented` in `config/default.json`: `control.search.pattern` and `scan_speed_deg_s`,
`vision.roi.expand_on_loss_factor`, and the `ai.*` block.

---

## 11. Packaging

Two executables are built and maintained side by side (see `docs/PACKAGING.md`): a headless binary
(~99 MB) and a GUI binary (~318 MB). They are kept separate deliberately — Qt plugin failures
occur at runtime on the target machine with opaque messages, so a headless executable that passes
its own clean-machine test means a Qt problem never costs the deliverable.

That separation earned itself immediately: `opencv-python` ships its own Qt plugins under
`cv2/qt/plugins`, which collide with PySide6's. PyInstaller failed to extract the duplicate and
the GUI died at startup before any window appeared. The headless build was unaffected, so the
failure was unambiguously a Qt problem rather than a base-packaging one.

---

## 12. Open items

- The candidate discriminator (§5.5) is implemented, trained, tested and bundled, but ships
  **disabled by default**: it fails the shipping gate on `fhd_1920_bigspot`, where it raises
  association failure from 10.8% to 96.6%. Re-training with large-spot examples better
  represented is the obvious next step, but it is not on the critical path.
- `lowlight_impulse` never locks (§8.3), and this is now known to be a detection-sensitivity
  limit rather than a ranking one -- the beacon is surfaced as a candidate in only 34/150 frames.
- Clean-machine container verification: distributions verified and unverified are listed in
  `docs/PACKAGING.md`.
- Tests write into the project `logs/` directory, overwriting run outputs.
