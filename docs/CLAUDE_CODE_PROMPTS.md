# Claude Code session prompts

Ready-to-paste prompts, one per session. Work them in order. After each, run `pytest tests/ -v`
and tick the corresponding box in `docs/ROADMAP.md`.

**General rule:** never ask for more than one module at a time. "Build the FSOC tracker" produces
a plausible-looking system that fails under noise. One module, one test, one commit.

---

## Session 0 — Orientation

```
Read CLAUDE.md, docs/PROBLEM_STATEMENT.md, docs/DESIGN.md and docs/ROADMAP.md.

Then, without writing any code yet, tell me:
1. Your understanding of what we're building in three sentences
2. Anything in the design you think is wrong, risky, or underspecified
3. Which Phase 0 item you'd tackle first and why

I want to check we're aligned before you start writing code.
```

---

## Session 1 — Config loader (Phase 0)

```
Implement src/config.py.

Load config/default.json into typed, frozen dataclasses mirroring the JSON structure. Requirements:
- Validate ranges against the spec (target size 5-20 px, pan/tilt speed 5-10 deg/s, noise sigma
  <= 20, salt-pepper density <= 0.10, camera update rate >= 30 Hz, control rate >= 20 Hz)
- Raise a clear, actionable error naming the offending key when validation fails
- Compute derived values: deg_per_pixel (both axes), max_px_per_frame at the configured rate
- Support overriding any value from a second partial JSON (for evaluator scenarios)
- Ignore keys beginning with underscore; they are documentation

Then write tests/test_config.py covering: valid load, each range violation, derived value
correctness (FOV 4x3 over 640x480 must give 0.00625 deg/px), and partial-override merging.
```

---

## Session 2 — Simulation engine (Phase 1)

```
Implement Phase 1 of docs/ROADMAP.md: src/sim/canvas.py, src/sim/beacon.py,
src/sim/trajectories.py.

Key requirements from docs/DESIGN.md sections 2 and 3:
- Canvas buffer allocated ONCE and reused; no per-frame allocation
- Beacon rendered as a 2D Gaussian on a 4x supersampled grid then downsampled, so the
  ground-truth centroid is exact at sub-pixel positions
- Shapes: gaussian, square, circle. Size 5-20 px.
- Trajectories: linear, circular, figure8 (Lissajous 1:2), random walk are MANDATORY.
  Spiral, sinusoidal, Ornstein-Uhlenbeck are optional - implement if straightforward.
- All trajectories are pure functions of time returning (x, y) in canvas coordinates
- Boundary behaviour configurable: bounce (default), wrap, clamp

Critical test: render a beacon at a known sub-pixel position, run a plain centroid over the
clean output, and assert recovery to within 0.05 px. If that fails, our ground truth is
worthless and every accuracy number downstream is a lie.
```

---

## Session 3 — Camera and viewport (Phase 1)

```
Implement src/camera/model.py and src/camera/viewport.py.

- model.py: FOV <-> pixel mapping, deg_per_pixel, slew-rate and acceleration limiting, pan/tilt
  state integration. Enforce limits INSIDE the model so the controller cannot exceed them.
- viewport.py: extract the 640x480 sub-viewport from the 2000x2000 canvas at the current
  boresight. Handle edge cases where the viewport overlaps the canvas boundary (pad, don't crash).

Then implement SimulationFrameSource in src/sim/source.py, conforming to the FrameSource protocol
in src/framesource.py. It should compose canvas + beacon + trajectory + camera, and return
FrameData with exact ground truth converted to FRAME-LOCAL coordinates.

Test: the slew limiter must clamp a 100 deg/s command to the configured 5 deg/s, and the
resulting displacement must be 26.7 px/frame at 30 Hz (this is the feasibility number from
docs/DESIGN.md section 7.3 - assert it explicitly so a regression is caught).
```

---

## Session 4 — Noise pipeline (Phase 2)

```
Implement Phase 2 of docs/ROADMAP.md.

All noise functions must be PURE: (frame, params) -> frame. Composable in the configurable order
from config. See docs/DESIGN.md section 4 for the models.

- src/noise/sensor.py: Gaussian (sigma <= 20), Poisson shot noise, salt & pepper (<= 10% density)
- src/noise/atmospheric.py: Koschmieder model I = J*t + A*(1-t), with clear/haze/fog/rain/
  low_light presets from config
- src/noise/turbulence.py: log-normal and Gamma-Gamma scintillation, Ornstein-Uhlenbeck beam
  wander. Cheap multiplicative time series - do NOT implement phase-screen propagation.
- src/noise/disturbance.py: camera jitter and platform motion, both bounded at 20 px/frame
- src/noise/pipeline.py: ordered composition driven by config

Tests must be STATISTICAL, not visual: measured sigma matches requested within tolerance,
salt-pepper density matches requested within tolerance, Poisson variance tracks the mean,
fog reduces measured contrast monotonically with beta.
```

---

## Session 5 — Vision pipeline (Phase 3) — the scoring core

```
Implement Phase 3 of docs/ROADMAP.md. This is the most important module in the project - 60% of
the grade depends on it working on inputs we have never seen.

Pipeline order (docs/DESIGN.md section 5.1):
median filter -> top-hat morphology -> adaptive/Otsu threshold -> connected components ->
size/shape gating -> thresholded intensity-weighted centroid

ABSOLUTE REQUIREMENT: no hardcoded intensity thresholds anywhere. Every threshold derives from
frame statistics at runtime (Otsu, adaptive, mean + k*sigma, percentile - selectable from config).
If you write a bare numeric intensity comparison, you have introduced the exact failure mode that
loses us Benchmark-2. Per-frame normalisation before thresholding.

The pipeline must accept an optional ROI window and process only that region when given one.

Centroid estimators: cog, thresholded_cog, iwcog (default, 3 iterations), gaussian_fit.

Then write tests/test_vision_snr_sweep.py: sweep SNR from 1 to 100, 200 trials each, and produce
a matplotlib plot of centroid error vs SNR saved to docs/figures/centroid_error_vs_snr.png.
Assert error < 1 px at SNR > 10. This plot is a report deliverable, not a debug tool.
```

---

## Session 6 — Mode B video ingestion (Phase 6, pulled early)

```
Implement src/video_source.py: VideoFrameSource conforming to the FrameSource protocol.

We are doing this EARLY, before the controller, because it validates the whole dual-mode
abstraction while it's still cheap to fix.

Requirements (docs/DESIGN.md section 8):
- Auto-detect resolution; never assume 640x480 or 2000x2000
- Convert colour to grayscale at the source boundary
- Normalise intensity per frame
- supports_pan_tilt returns False; apply_pan_tilt is a no-op
- ground_truth is None unless a sidecar CSV is supplied via config
- deg_per_pixel returns None (unknown calibration) so angular metrics are omitted, not faked

Then write a script scripts/make_test_video.py that renders a beacon on a full 2000x2000 scene
with configurable noise and writes an .mp4 at 30 fps plus a ground-truth CSV. Use it to produce
three test videos in scenarios/ with DIFFERENT noise characteristics from our defaults - the
whole point is to catch overfitting.

Verify the vision pipeline from Session 5 runs on these videos completely unmodified.
```

---

## Session 7 — Kalman filter and tracking (Phase 4)

```
Implement src/filtering/kalman.py and src/filtering/track.py per docs/DESIGN.md section 6.

Constant-velocity model, state [px, py, vx, vy]. Use the continuous white-noise Q matrix given in
the design doc. Include:
- Mahalanobis validation gating (chi-squared 99%, 2 DOF = 9.21) to reject salt-pepper false
  positives
- Coast mode: predict-only when detection fails, enabling <= 1 s re-acquisition
- M-of-N track confirmation and deletion

dt must always come from FrameData.timestamp deltas, never assumed constant.

Test: feed a known constant-velocity trajectory plus measurement noise and assert the filter's
position RMSE is lower than the raw measurement RMSE. Then feed a 10-frame dropout and assert
the coast prediction stays within 20 px of truth.
```

---

## Session 8 — Controller and state machine (Phase 4)

```
Implement src/control/pid.py, controller.py, statemachine.py, search.py per docs/DESIGN.md
section 7.

- PID with anti-windup and slew-rate limiting on angular error
- Kalman-velocity FEEDFORWARD (this is what cancels steady-state phase lag - do not omit it)
- State machine SEARCH -> TRACK -> COAST with hysteresis (lock_window_px 40 / unlock_window_px 80)
- Archimedean spiral acquisition search with arm spacing <= FOV width

Then write tests/test_trackable_velocity.py: sweep target angular velocity and find the maximum
rate at which steady-state pointing error stays <= 10 px. Plot it to
docs/figures/trackable_velocity.png.

I want the honest number here, including the regime where we CANNOT track. Do not tune the test
to only use velocities that succeed. This plot is a Q&A centrepiece - teams that characterise
their own limits do better than teams that hide them.
```

---

## Session 9 — Telemetry (Phase 5)

```
Implement Phase 5 of docs/ROADMAP.md.

Implement EXACTLY the metric definitions in CLAUDE.md. In particular:
- Acquisition: clock from frame 0, stops when lock criterion holds K consecutive frames
- Re-acquisition: from loss declaration (N missed) to re-lock
- Log centroiding error AND pointing error separately, both labelled
- Three separate rates: camera update Hz, control Hz, processing FPS

Per-frame CSV with the columns listed in docs/DESIGN.md section 9. Auto-generated HTML summary
with plots, config snapshot, and the metric definitions printed in the header so evaluators can
see exactly what we measured.

Test against synthetic sequences with known answers: a run that locks at frame 30 at 30 Hz must
report acquisition time 1.0 s, not 0.97 or 1.03.
```

---

## Session 10 — Packaging (do NOT leave to the end)

```
Set up PyInstaller packaging now, at roughly half completion, so we discover problems on a normal
day rather than the night before submission.

- Write the .spec file
- Bundle the Numba llvmlite binary explicitly (it is not auto-detected by freezing tools)
- Bundle Qt platform plugins and OpenCV binaries
- Add necessary hidden-imports
- Produce a build script

Then tell me exactly how to test the frozen executable on a clean machine with no Python
installed, because that is the only test that counts.
```

---

## Session 11+ — GUI (Phase 7)

```
Implement Phase 7 of docs/ROADMAP.md: the PySide6 dashboard.

Constraint from CLAUDE.md: core modules must stay importable and runnable headless. The GUI is a
thin view layer - no business logic in widgets.

- Live viewport with estimated centroid (and ground truth when available) overlaid
- pyqtgraph strip charts: centroiding error and FPS over a rolling 30 s window
- Controls exposing EVERY mandatory spec function: motion selector, each noise type toggle,
  atmospheric preset, target size/shape, pan/tilt speed, Mode A/B switch with file picker
- Current state (SEARCH/TRACK/COAST) and live metrics vs targets

Every mandatory function from docs/PROBLEM_STATEMENT.md must be demonstrable from this GUI within
a 10-15 minute demo. Walk through the spec's parameter table and confirm each one is reachable.
```

---

## Useful mid-session prompts

```
Review src/vision/ for hardcoded numeric thresholds. Anything that isn't derived from frame
statistics or read from config is a bug. List what you find before changing anything.
```

```
Profile a 60-second simulation run with cProfile. Show me the top 10 hotspots by cumulative time
and tell me whether we're meeting 20 FPS. Do not optimise anything yet - I want the measurement
first.
```

```
Run the full pipeline against scenarios/*.mp4 and report centroiding RMSE, acquisition time,
lock retention and FPS for each. These videos have different noise characteristics from our
defaults, so treat any large gap versus simulation performance as an overfitting signal and tell
me where you think it's coming from.
```

```
I have a technical Q&A coming up. Ask me five hard questions an ISRO evaluator would ask about
our algorithm choices, then critique my answers.
```
