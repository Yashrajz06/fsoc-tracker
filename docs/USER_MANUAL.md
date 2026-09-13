# User Manual

**AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals**
Smart India Hackathon — Problem Statement 4, Department of Space / ISRO

---

## 1. Installation

### 1.1 Running the standalone executable (no Python needed)

Two executables are built and shipped side by side:

| file | size | contents |
|---|---|---|
| `dist/fsoc-tracker` | ~99 MB | headless: simulation, video mode, logging, reports |
| `dist/fsoc-tracker-gui` | ~318 MB | everything above plus the Qt dashboard |

They are kept separate deliberately. Qt plugin problems surface at runtime on the target machine
with opaque messages, so a headless executable that passes its own self-test means a Qt problem
never costs the whole demonstration.

Both are self-contained — no Python installation, no `pip install`, no virtual environment.

```
./dist/fsoc-tracker --selftest
```

A healthy bundle ends with `self-test PASSED`. Run this first on any new machine.

**Linux** — the executables are built for x86_64 with glibc 2.35 or newer (Ubuntu 22.04+,
Debian 12+). On older distributions, run from source instead (§1.2).

**macOS / Windows** — the shipped binaries are Linux builds and will not run. Use §1.2, or
rebuild on the target platform with `scripts/build.sh`.

### 1.2 Running from source

Requires **Python 3.11 or 3.12** (not 3.13 — `numba` wheels lag new releases).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.main --selftest
```

`ffmpeg` is needed only to *generate* test video, not to run the tracker:

```
sudo apt install ffmpeg        # Debian / Ubuntu
brew install ffmpeg            # macOS
```

Verify the installation:

```
.venv/bin/python -m pytest tests/ -q -m "not slow"
```

Expect **663 passed**. Fewer, with skips, if `ffmpeg` is absent.

### 1.3 Rebuilding the executables

```
scripts/build.sh              # headless
scripts/build.sh --gui        # GUI build
scripts/build.sh --clean      # remove previous build artefacts first
```

See `docs/PACKAGING.md` for the clean-machine verification procedure.

---

## 2. Application operation

### 2.1 Graphical mode (recommended for demonstration)

```
./dist/fsoc-tracker-gui --gui
```

or from source:

```
.venv/bin/python -m src.main --gui
```

Set parameters in the control panel, press **Start**, and watch the viewport and live metrics.
**Stop** ends the run and writes the logs. Parameters are read when Start is pressed; changing a
control mid-run has no effect until the next run.

### 2.2 Command line

| flag | meaning |
|---|---|
| `--headless` | run with no GUI, print a summary, write logs |
| `--gui` | launch the dashboard |
| `--config PATH` | use a configuration file other than the bundled default |
| `--scenario PATH` | apply a partial override file; repeatable, later files win |
| `--mode simulation\|video` | select Mode A or Mode B |
| `--video PATH` | the `.mp4` to ingest in Mode B |
| `--duration SECONDS` | length of a Mode A run |
| `--selftest` | verify the installation and exit |
| `--check-config` | validate configuration, print derived physics, and exit |

**A ten-second simulated run:**

```
./dist/fsoc-tracker --headless --duration 10
```

**Running an evaluator scenario file:**

```
./dist/fsoc-tracker --headless --scenario scenarios/case1.json
```

**Running an evaluator video (Mode B):**

```
./dist/fsoc-tracker --headless --mode video --video /path/to/clip.mp4
```

In Mode B the pan/tilt camera is bypassed — the video *is* the scene. The vision pipeline is
byte-for-byte the same code used in Mode A. Centroids are reported in full-frame source pixel
coordinates, so no coordinate negotiation is needed to compare against evaluator ground truth.

---

## 3. Parameter configuration

All tunable values live in `config/default.json` and are validated on load. Validation runs
*after* any override files are merged, so a scenario that pushes the system outside the
specification fails immediately with a message naming the offending parameter, rather than
producing a completed run whose numbers are quietly invalid.

### 3.1 Scenario override files

A scenario file is a **partial** JSON document — only the keys you wish to change:

```json
{
  "run":    { "duration_seconds": 30 },
  "target": { "shape": "circle", "size_px": 15, "motion": { "type": "figure8" } },
  "noise":  { "gaussian": { "enabled": true, "sigma": 12.0 },
              "salt_pepper": { "enabled": true, "density": 0.08 } }
}
```

```
./dist/fsoc-tracker --headless --scenario my_case.json
```

Check a scenario before running it:

```
./dist/fsoc-tracker --check-config --scenario my_case.json
```

This prints the derived physics — degrees per pixel, slew ceiling in pixels per frame, ROI size,
worst-case search time — and exits.

### 3.2 The specification parameter table

Numbers in brackets are the specification's parameter numbers.

| # | Parameter | Config key | Default | Accepted range |
|---|---|---|---|---|
| 1 | Screen size | `scene.width` / `.height` | 2000 × 2000 | ≥ 2000 |
| 2 | Camera type | — | monochrome FPA | colour not implemented (optional in spec) |
| 3 | Camera resolution | `camera.resolution_*` | 640 × 480 | user-defined |
| 4 | Camera FOV | `camera.fov_*_deg` | 4° × 3° | user-defined |
| 5 | Camera update rate | `camera.update_rate_hz` | 30 Hz | ≥ 30 |
| 6 | Initial camera position | `camera.initial_position` | centre | centre / custom |
| 7 | Target type | — | beacon spot | fixed by spec |
| 8 | Number of targets | `target.count` | 1 | 1 (multiple optional, not implemented) |
| 9 | Target shape | `target.shape` | **square** | square / circle / gaussian |
| 10 | Target size | `target.size_px` | 10 | 5–20 |
| 11 | Initial target location | `target.initial_position` | random | random / center / custom |
| 12 | Motion | `target.motion.type` | circular | linear, circular, figure8, random, spiral, sinusoidal, ornstein_uhlenbeck |
| 13 | Max pan speed | `camera.max_pan_speed_deg_s` | 5 °/s | 5–10 |
| 14 | Max tilt speed | `camera.max_tilt_speed_deg_s` | 5 °/s | 5–10 |
| 15 | Control update interval | `control.update_rate_hz` | 30 Hz | ≥ 20 |
| 21 | Image noise | `noise.gaussian` / `.poisson` / `.salt_pepper` | off | any combination |
| 22 | Max noise std dev | `noise.gaussian.sigma` | 10 | 0–20 |
| 23 | Max camera jitter | `noise.camera_jitter.max_px_per_frame` | 5 | 0–20 |
| 24 | Atmospheric disturbance | `noise.atmospheric.preset` | clear | clear, haze, fog, rain, low_light |
| 25 | Platform motion | `noise.platform_motion` | linear, 5 px/frame | 0–20 px/frame; linear, circular, random, spiral, figure8 |

Parameters 16–20 are performance *specifications*, not inputs. They appear in the report as
measured values against their targets: acquisition ≤ 2 s, tracking error ≤ 10 px, target loss
< 5 %, re-acquisition ≤ 1 s, processing ≥ 20 FPS.

---

## 4. GUI description

The window has three regions: controls on the left, live viewport top right, charts and metrics
bottom right.

### 4.1 Control panel — four tabs

**Target** — count (8), shape (9), size in pixels (10), initial location (11), motion model (12),
and edge behaviour (bounce, wrap or clamp at the scene boundary).

**Camera** — scene width and height (1), camera type (2, read-only label: monochrome FPA),
resolution (3), horizontal and vertical FOV (4), camera rate (5), initial position (6), maximum
pan and tilt speed (13, 14), control rate (15).

**Noise** — Gaussian with standard deviation capped at 20 (22), Poisson, salt & pepper with
density, camera jitter in pixels per frame (23), atmospheric preset (24), platform motion type
and magnitude (25), and optical turbulence.

**Mode** — Mode A or Mode B, run duration, random seed for reproducibility, and file pickers for
the video and its optional ground-truth CSV in Mode B.

Every control is bounded by the specification. Out-of-range values are rejected by the
configuration layer with a message citing the parameter, **not** silently clamped by the widget —
so the GUI cannot accept something the command line would refuse.

### 4.2 Viewport

The live camera image, with the estimated centroid marked in orange and, when ground truth is
available, the true position in green. The current state — SEARCH, TRACK or COAST — is overlaid.

- **SEARCH** — no lock; the camera is sweeping a spiral pattern
- **TRACK** — locked; the ROI follows the Kalman prediction
- **COAST** — lock held but detections are being missed; running on prediction

### 4.3 Strip charts

Two rolling 30-second plots. Centroiding error on a logarithmic axis with the 10 px requirement
marked, and processing FPS with the 20 FPS requirement marked.

### 4.4 Metrics panel

Live values against their targets, coloured green when met and red when not: centroiding RMSE and
median, pointing RMSE, lock retention, loss rate, processing FPS, and association failure rate.

Centroiding and pointing error are shown separately and deliberately. The specification's
"tracking error ≤ 10 px" is ambiguous between them, and Benchmark-2 scores centroiding
specifically.

---

## 5. Output files

Written to `logs/` at the end of every run.

**`frames.csv`** — one row per frame with a commented header recording the coordinate convention,
every metric definition, and the configuration. Columns include timestamp, state, estimated and
true centroid, centroid error, detection error, pointing error, SNR (aperture and peak), spot
scale and its provenance, filter diagnostics (NIS, measurement sigma, gate outcome), and
per-frame processing time. Every summary figure can be recomputed from this file independently.

**`report.html`** — the performance report, containing simulation duration, frame/control/
processing rates, acquisition time, mean and maximum tracking error, RMSE, median and p95, lock
retention, loss rate and processing time, with every metric definition stated inline.

> **Note:** the test suite also writes into `logs/`. Run the tests *before* generating a report
> you intend to keep.

---

## 6. Troubleshooting

**`self-test FAILED`** — the bundle is incomplete or the platform is unsupported. Check the glibc
version (`ldd --version`); the binaries need 2.35 or newer. Fall back to running from source.

**GUI will not start, "could not load the Qt platform plugin"** — a display is required; check
`$DISPLAY`. Over SSH use `ssh -X`. For a headless machine, use `--headless`, or
`QT_QPA_PLATFORM=offscreen` to run the GUI without a window.

**Mode B reports zero locked frames** — the beacon is likely below the detection threshold. The
report says so explicitly, quoting the measured SNR against the SNR ≈ 10 threshold characterised
in the technical report. This is a genuine input limitation, not a crash.

**Acquisition takes far longer than 2 seconds** — expected when the beacon starts outside the
initial viewport. Sweeping a 2000 × 2000 region at the 5 °/s slew ceiling takes about 10 seconds
in the worst case. The report separates in-FOV from search-limited acquisition for this reason,
and `--check-config` prints the worst-case search time for the active configuration.

**Throughput below 20 FPS** — full-frame processing during acquisition is expensive at large
resolutions. Once locked, processing is ROI-limited and roughly independent of frame size. Check
whether the run ever achieved lock.

---

## 7. Reproducibility

Every random draw derives from `run.random_seed`, so a given configuration and seed reproduce
exactly. Generated test videos are bit-reproducible: the encoder is pinned to single-threaded
deterministic mode, because x264's thread scheduling otherwise changes the decoded pixels between
runs of identical input.

To reproduce a reported figure, run with the same scenario file and seed; `frames.csv` carries
both in its header.
