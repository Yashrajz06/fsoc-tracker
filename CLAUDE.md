# CLAUDE.md — Project Context

## What this project is

An **AI-based virtual camera tracking system for coarse alignment of mobile Free Space Optical
Communication (FSOC) terminals**. This is a Smart India Hackathon problem statement issued by the
Department of Space / ISRO (Problem Statement 4, Category: Software).

We simulate the **coarse alignment stage** of a laser-communication Pointing, Acquisition and
Tracking (PAT) system entirely in software: render a large virtual scene containing a moving
optical beacon, extract a narrow camera viewport from it, corrupt that viewport with realistic
sensor noise and atmospheric degradation, then automatically detect the beacon, estimate its
centroid, and drive a virtual pan/tilt camera to keep the beacon centred — while logging
tracking performance in real time.

**Read `docs/PROBLEM_STATEMENT.md` for the verbatim official spec.**
**Read `docs/DESIGN.md` for the architecture, algorithms and mathematical models.**
**Read `docs/ROADMAP.md` for the build order and current phase.**

---

## Hard performance targets (these are graded)

| Metric | Requirement |
|---|---|
| Tracking / centroiding error | **≤ 10 pixels** |
| Acquisition time | **≤ 2 seconds** |
| Re-acquisition time | **≤ 1 second** |
| Target loss rate | **< 5 %** |
| Processing throughput | **≥ 20 FPS** |
| Camera update rate | **≥ 30 Hz** |
| Control update interval | **≥ 20 Hz** |

Note these are **three separate clocks**: frame generation (≥30 Hz), control loop (≥20 Hz), and
end-to-end processing throughput (≥20 FPS). Log all three separately. Never conflate them.

---

## Non-negotiable engineering constraints

These exist because of how the project is scored. Do not violate them, and flag it if a request
seems to require violating them.

1. **NO MAGIC NUMBERS IN THE VISION PIPELINE.** Every threshold must be derived from frame
   statistics at runtime (Otsu, adaptive thresholding, mean + k·sigma, percentile). Never write
   `if pixel > 200`. 30% of the grade comes from unseen evaluator `.mp4` files whose noise and
   brightness characteristics we cannot predict. Hardcoded thresholds tuned to our own simulator
   are the single most likely cause of failure.

2. **ROI-LIMITED PROCESSING.** Once the tracker is locked, the vision pipeline must operate on a
   small window (default 64×64) around the Kalman-predicted position — never on the full
   2000×2000 canvas. Full-frame processing is only permitted during initial acquisition search.
   This is what makes ≥20 FPS achievable.

3. **ONE VISION PIPELINE, TWO INPUT MODES.** Mode A (closed-loop simulation) and Mode B
   (pre-recorded `.mp4` ingestion) must feed the *identical* vision code through the
   `FrameSource` abstraction in `src/framesource.py`. No mode-specific branches inside the
   vision, filtering, or telemetry modules. If you find yourself writing `if mode == "video"`
   inside `src/vision/`, the abstraction is wrong — fix the abstraction instead.

4. **GROUND TRUTH IS SACRED.** The simulator must know the true beacon centroid to sub-pixel
   accuracy (render on a supersampled grid, downsample). All accuracy claims are validated
   against it. Never let rendering quantisation destroy sub-pixel ground truth.

5. **EVERY PUBLIC FUNCTION GETS A DOCSTRING AND TYPE HINTS.** "Complete source code with proper
   documentation. The code shall be modular and adequately commented" is a graded deliverable.
   Writing docs later never happens.

6. **NO GUI DEPENDENCIES IN CORE MODULES.** `src/sim`, `src/noise`, `src/camera`, `src/vision`,
   `src/filtering`, `src/control`, `src/telemetry` must be importable and runnable headless. The
   GUI is a thin view layer on top. This keeps the system testable and scriptable for benchmarks.

---

## Technology stack (decided — do not substitute without asking)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Numerics | NumPy; Numba `@njit(nogil=True)` for hotspots only |
| Computer vision | OpenCV (`opencv-python`) |
| GUI | PySide6 (Qt) + pyqtgraph for live plots |
| Video I/O | OpenCV `VideoCapture` (PyAV fallback) |
| Config | JSON (see `config/default.json`) |
| Testing | pytest |
| Packaging | PyInstaller (standalone executable is a mandatory deliverable) |
| Optional AI | ONNX Runtime for a lightweight CNN validator |

**Do not add heavy dependencies** (PyTorch, TensorFlow, scipy-full) without discussing it first.
Every dependency increases the risk that the PyInstaller build fails, and a broken executable
costs us the 20% Functional Verification marks regardless of how good the code is.

---

## Architecture at a glance

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
                                              acq + reacq timers, error, loss, FPS
                                                        |
                                                        v
                                              GUI DASHBOARD + CSV/JSON logs
```

Both `SimulationFrameSource` (Mode A) and `VideoFrameSource` (Mode B) implement the same
`FrameSource` protocol. In Mode B the PTZ loop is bypassed — the video *is* the scene.

---

## Key numbers you will need constantly

- **Angular resolution**: FOV 4°×3° over 640×480 px → **0.00625 °/pixel** (both axes).
- **Slew ceiling**: max pan 5 °/s = 800 px/s = **26.7 px/frame at 30 Hz**. At 10 °/s it is
  ~53 px/frame. A target moving faster than this between frames **cannot be kept centred**.
  This is a real physical limit of the system — quantify it and report it honestly, do not hide
  it by only testing slow targets.
- **Canvas**: 2000×2000 px minimum. Camera starts at centre (1000, 1000).
- **Beacon**: default 10×10 px, range 5–20 px. Gaussian profile, σ ≈ 2–3 px for a 10 px spot.
- **Search-time ceiling**: covering a 2000×2000 uncertainty region with a spiral whose arm
  spacing equals the limiting FOV dimension (480 px) needs a path of roughly
  `area / spacing = 4×10⁶ / 480 ≈ 8.3×10³ px`. At the 5 °/s slew ceiling (800 px/s) that is
  **≈10 s worst case**, and ≈5 s even at 10 °/s. **The ≤2 s acquisition budget is therefore
  unachievable whenever the beacon starts outside the initial viewport** — that is arithmetic,
  not a tuning failure. `src/config.py` computes this number for the active configuration and
  warns at startup when it exceeds the budget. Report it; do not hide it by only ever starting
  the beacon in frame.

---

## Coordinate conventions (decided — never deviate)

Sub-pixel ground truth is only meaningful if every stage agrees on where a pixel *is*. One
inconsistent convention is a silent systematic half-pixel bias that survives every test we would
normally write, and it would quietly invalidate the centroid-error-vs-SNR curve.

- **Pixel centres sit at integer indices.** Pixel `[r, c]` has its centre at exactly `(x=c, y=r)`.
  A frame of width `W` therefore spans x from `-0.5` to `W-0.5`, and its centre — the camera
  boresight — is at `((W-1)/2, (H-1)/2)`. This matches `FrameData.center` in
  `src/framesource.py`, which is the reference implementation.
- **Origin is the top-left pixel**, x increasing right, y increasing down. This is OpenCV/NumPy
  order; note that NumPy indexes `[y, x]` while all our coordinate tuples are `(x, y)`.
- **Order of coordinate tuples is always `(x, y)`.** Never `(row, col)` outside of raw NumPy
  indexing.
- **The supersampled renderer must preserve this.** When rendering at factor `S` and downsampling
  by block-averaging, supersample pixel `[R, C]` has centre `((C + 0.5)/S - 0.5, (R + 0.5)/S - 0.5)`
  in output coordinates. Getting this off by one produces a `(S-1)/(2S)` px offset — 0.375 px at
  S=4, which is 7.5× the Phase 1 accuracy target.
- **Phase 1 must assert this.** The renderer test places a beacon at a known non-integer position
  and requires the recovered centroid within 0.05 px, with a dedicated case at an exact integer
  position and one at an exact half-integer position to catch off-by-half errors specifically.
- **Mode B reports centroids in full-frame source pixel coordinates** under this same convention,
  so evaluator comparison needs no coordinate negotiation. State the convention in the log header.

---

## Metric definitions (be precise — ambiguity here costs marks)

Implement exactly these definitions and document them in the logs:

- **Acquisition time**: clock starts at first frame of the run; stops the first frame the lock
  criterion holds for `K` consecutive frames (default K=3).
  → **Log this split into two populations, never as a single pooled number:**
  - **In-FOV acquisition** — the beacon was already inside the initial viewport at frame 0. This
    is detection-limited, and it is the case the spec's ≤2 s budget is realistic for.
  - **Search-limited acquisition** — the beacon was outside the initial viewport, so the time is
    dominated by how long the SEARCH spiral takes to sweep the uncertainty region, which is
    bounded below by the slew-rate ceiling and can legitimately exceed 2 s (see the
    search-time envelope below). Report it honestly and separately, exactly as we report the
    maximum trackable velocity envelope.
  Pooling the two hides a physical limit behind an initial-condition lottery and makes the
  headline number unreproducible.
- **Lock criterion**: a valid detection whose SNR exceeds the adaptive threshold AND whose
  centroid passes the Kalman validation gate.
- **Re-acquisition time**: clock starts when an established lock is lost (`N` consecutive missed
  detections, default N=5); stops when the lock criterion is re-satisfied.
- **Centroiding error**: Euclidean distance between estimated centroid and ground-truth centroid,
  in pixels, per frame.
- **Pointing error**: Euclidean distance between the target position and the camera boresight
  (viewport centre), in pixels, per frame.
  → **Log BOTH separately.** The spec's "tracking error ≤ 10 px" is ambiguous between them, and
  Benchmark-2 explicitly scores "centroiding error".
- **SNR** — two definitions, both logged, one on the plot axis. The literature uses "SNR"
  inconsistently (peak vs. integrated vs. ROI), and the centroid-error-vs-SNR curve is a graded
  deliverable, so this is settled here rather than per-call-site:
  - **`snr_aperture` (PRIMARY — this is the x-axis of the SNR curve, and what `sigma_meas`
    derives from):**
    `sum(I - mu_bg over aperture) / (sigma_bg * sqrt(N_aperture))`.
    Aperture = disc of radius `1.5 * FWHM` centred on the estimated centroid.
    Chosen as primary because it is the SNR the accuracy law `sigma_x ~ FWHM / (2 * SNR)`
    (DESIGN §5.3) is written in terms of — using peak SNR there would make the quoted law wrong
    by a spot-shape-dependent factor.
  - **`snr_peak` (SECONDARY, logged):** `(I_peak - mu_bg) / sigma_bg`. This is what detection
    thresholding keys on, so it is the natural companion to the lock criterion.
  - **Background estimation for both is robust, not mean/std:** `mu_bg` = median and
    `sigma_bg` = `1.4826 * MAD`, measured over an annulus from `2.5 * FWHM` to `4 * FWHM`.
    Plain mean/std would be inflated by salt-and-pepper impulses — at 10% density the estimate
    of `sigma_bg` roughly doubles, which would silently halve every reported SNR.
  - Both definitions, the aperture and annulus radii, and the robust estimator go in the log
    header and on the plot axis label.
- **RMSE**: sqrt(mean(error²)) over frames where lock was held.
- **Loss rate**: frames without valid lock ÷ frames where the target was present.

---

## How this is graded (drives priority)

| Stage | Weight | What it means for us |
|---|---|---|
| Functional Verification | 20% | 10–15 min live demo. Every mandatory function must be visibly present and working, with a usable GUI. |
| Benchmark Performance-1 | 30% | Evaluators give us scenarios to run. Needs scenario loading + automatic centroiding-error logs. |
| Benchmark Performance-2 | 30% | Evaluators give us **unseen `.mp4` files**. Our PTZ camera is bypassed; the video is the input. Robustness to unknown noise is everything here. |
| Technical Evaluation | 20% | Presentation + Q&A on architecture, algorithm choice, AI/CV, innovation. |

**60% of the grade is objective benchmark performance on inputs we do not control.** Build the
measurable core first; the GUI is a view on top of it, not the product.

---

## The "AI" question — our position

The beacon is a bright, symmetric blob on a near-uniform background. Classical weighted
centroiding is near-optimal for this: localisation error scales as spot-width / sqrt(photons)
in the shot-noise limit, and roughly FWHM / (2 · SNR) in general. A heavy CNN would be slower,
less accurate, and would jeopardise the ≥20 FPS budget.

So our architecture is a **hybrid**, and this is a deliberate, defensible engineering decision
we must be ready to argue in the Q&A:

- **Primary fast path (every frame)**: classical CV — median filter, top-hat morphology,
  adaptive/Otsu threshold, intensity-weighted centroiding.
- **State estimation**: Kalman filter for prediction, jitter rejection, validation gating, and
  coasting through dropouts. This is the adaptive/"learned dynamics" layer.
- **AI fallback (only when the classical path fails)**: a lightweight CNN spot validator /
  re-acquisition detector via ONNX Runtime, analogous to the infrared small-target detection
  literature.

Do not replace the classical path with a neural network. Do not drop the AI component entirely
either — the problem is titled "AI-Based" and Technical Evaluation rewards it.

---

## Working conventions

- One module per session. Write the module, then write its test, then move on.
- Prefer pure functions with signature `(frame: np.ndarray, params: dataclass) -> np.ndarray`
  for all noise and preprocessing steps. Easy to test, easy to compose, easy to reorder.
- All tunable values live in `config/default.json` and are loaded into dataclasses. Nothing
  tunable should be literal in the source.
- Use `uint8` frames where possible; allocate the 2000×2000 canvas once and reuse the buffer.
  Do not reallocate per frame.
- Seed all RNGs from config so runs are reproducible for the report.
- When adding a module, update `docs/ROADMAP.md` to tick it off.

## Testing

- `pytest tests/` must pass before any phase is considered done.
- Noise functions are tested statistically (mean, variance, impulse density), not by eyeball.
- The vision pipeline has an **SNR sweep test** that produces the centroid-error-vs-SNR curve.
  That curve goes in the technical report and is a Q&A centrepiece — treat it as a deliverable,
  not a debug tool.
- Mode B is tested against `.mp4` files we generate ourselves with known ground truth, so the
  logging format is validated end-to-end before evaluators ever hand us a video.

## Standing rule: never attribute a forced-path measurement to the shipped system

**A diagnostic that overrides a runtime-derived value measures a hypothetical. It cannot be
attributed to the shipped system without re-running unforced.**

This has now caused two wrong conclusions, both of which survived review and had to be retracted:

1. Phase 6 measured ROI vs full-frame vision cost by calling `pipeline.process(..., roi=...)`
   directly, and reported the speed-up as the system's throughput. The run loop did not pass an
   ROI at all — the path was unwired until it was found much later. The report carried a
   throughput claim the shipped code did not realise.
2. A large-spot association failure was diagnosed by calling
   `pipeline.process(frame, fwhm_px=5.887, from_fallback=True)`, which forces the fallback spot
   scale. The runner passes the live `ScaleTracker` estimate instead. The forced run showed the
   beacon gated out 0/26 frames and produced a confident, wrong root cause; the estimator was in
   fact reporting 15.8–16.3 px against a true 14.1 and working correctly.

Both had the same shape: a measurement that was *numerically correct* about a configuration the
production loop never enters.

So, when writing any diagnostic:

- **Prefer driving `TrackingRunner`** over calling `VisionPipeline.process` directly. The runner
  is the shipped path; anything else is a model of it.
- If a parameter must be forced to isolate a variable, **say so in the output**, and confirm the
  finding against an unforced run before it leaves the session.
- Treat `from_fallback=True`, an explicit `fwhm_px=`, an explicit `roi=`, a hand-built config, or
  a monkeypatched attribute as a flag that the result is conditional.
- When a diagnostic and the shipped system disagree, **the diagnostic is wrong until proven
  otherwise** — that is the more common case, and it is the one that wastes a session.

## Things that have killed other teams — avoid these

- Hardcoded thresholds that work on own data and fail on evaluator videos.
- Processing the full canvas every frame and never hitting 20 FPS.
- Discovering on the final night that PyInstaller cannot bundle Numba's llvmlite DLL or the Qt
  platform plugins. **Build the executable at ~50% completion and test it on a clean machine.**
- A beautiful GUI wrapped around a tracker that loses lock under noise.
- Being unable to explain in Q&A why classical CV was chosen over deep learning.
