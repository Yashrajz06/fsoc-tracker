# Packaging and clean-machine verification

The standalone executable is a mandatory graded deliverable. This document covers building it and
— more importantly — verifying it on a machine with no Python installed, which is the only test
that counts.

## Building

```bash
scripts/build.sh                # headless, no Numba            (~99 MB)
scripts/build.sh --with-numba   # headless + JIT runtime        (~173 MB)
scripts/build.sh --gui          # GUI build, adds Qt            (~318 MB)
scripts/build.sh --clean        # remove build artefacts first
```

Headless lands at `dist/fsoc-tracker`, GUI at `dist/fsoc-tracker-gui`. **They are kept side by
side, not one replacing the other.** Qt plugin failures happen at *runtime* on the target machine
with an opaque message, so keeping a headless executable that passes its own clean-machine test
means a Qt problem never costs us the deliverable — and any failure that does appear is
unambiguously a Qt failure rather than a base-packaging one.

One spec file, not two: `.gitignore` ignores `*.spec` with a single `!fsoc-tracker.spec`
negation, so a second spec would be silently untracked. `BUILD_GUI=1` selects the variant.

### A real Qt packaging failure, and its fix

`opencv-python` ships **its own Qt plugins** under `cv2/qt/plugins`. In a GUI build they collide
with PySide6's, and PyInstaller fails to extract the duplicate:

```
[PYI-xxxxx:ERROR] Failed to extract entry: cv2/qt/plugins/platforms/libqxcb.so
```

The application dies at startup before any window appears. The spec filters `cv2/qt` out of both
`binaries` and `datas` after analysis — cv2's Qt plugins exist only to serve `cv2.imshow`, which
this application never calls, so dropping them costs nothing. The headless build never hit this,
which is exactly the isolation the two-build split was meant to provide.

Output: `dist/fsoc-tracker`. The spec is `fsoc-tracker.spec` — the name matters, because
`.gitignore` ignores `*.spec` with an explicit `!fsoc-tracker.spec` negation. Any other name
silently falls back into the ignore rule.

### Why headless first

Qt is deliberately excluded from this build. Freezing is where projects discover that neither
their JIT runtime nor their GUI toolkit bundles cleanly, and debugging both at once is
substantially harder than debugging either alone. Phase 7 adds Qt on top of a packaging setup
already known to work.

### Why Numba is off by default

`BUNDLE_NUMBA=0` is a measured decision, not an oversight:

- Nothing in `src/` imports `numba` or `llvmlite` today. The vision pipeline is pure NumPy and
  OpenCV; `performance.use_numba` in the config is plumbing for a future hotspot.
- `libllvmlite.so` alone is **179 MB**, with `numba` adding a further 33 MB. Bundling them now
  roughly doubles the executable for code that is never called (103 MB → 181 MB measured).

The mechanism is implemented and **tested anyway**, because the point of packaging early is to
prove the path before it is needed. `llvmlite`'s shared library is loaded through `ctypes` at
runtime, so no import graph leads to it and PyInstaller cannot detect it — the spec names it
explicitly. When a Numba hotspot lands, the flag flips and nothing else changes.

## ffmpeg is a development dependency, not a runtime one

`ffmpeg` is required **only** to *generate* test videos (`scenarios/generate.py`, used by
`tests/test_video_mode.py`). Neither ships in the executable, and the shipped application never
invokes the `ffmpeg` binary.

Mode B *playback* uses OpenCV's `VideoCapture`, whose decoder libraries PyInstaller bundles
alongside `cv2`. So an evaluator running the frozen executable on a `.mp4` needs nothing on PATH.

The failure mode is explicit rather than a crash at first use: `scenarios/generate.py` probes for
a **software** H.264 encoder up front and raises `RuntimeError` naming the problem if none is
found. It deliberately does not fall back to another codec — see below.

## Verifying on a clean machine

This is the only test that counts. A frozen build failing is almost always an *import* or
*shared-library* failure that does not reproduce from source, and it typically surfaces on first
use rather than at startup.

### Step 0 — build a Mode B fixture first

`--selftest` does not decode anything, so it cannot catch a missing `libavcodec` or
`libavformat`. Mode B is 30% of the grade, so the video path needs its own step. Generate a
small fixture on the build machine before starting:

```bash
.venv/bin/python -c "
from pathlib import Path
from scenarios.generate import VideoSpec, generate_video
generate_video(VideoSpec('clean_probe', width=640, height=480, frames=60,
                         gaussian_sigma=8.0), Path('dist'))
"
ls dist/clean_probe.mp4 dist/clean_probe_truth.csv
```

This needs `ffmpeg` on the **build** machine only — it is a development dependency, and the
container never needs it.

### Steps 1-7 — in the container

```bash
# 1. Copy only dist/ into the container. Nothing else from the project.
docker run --rm -it -v "$PWD/dist:/opt/fsoc:ro" ubuntu:24.04 bash

# 2. Confirm there is genuinely no Python.
which python3 python || echo "no python present — good"

# 3. Self-test: exercises each bundled dependency deliberately.
/opt/fsoc/fsoc-tracker --selftest

# 4. Configuration resolution, using the config bundled inside the binary.
cd /root && /opt/fsoc/fsoc-tracker --check-config | head -20

# 5. Mode A: a full headless run from a directory with no project files.
cd /root && /opt/fsoc/fsoc-tracker --headless --duration 5

# 6. Mode B: decode an H.264 file. This is the step --selftest cannot cover.
cd /root && cat > video.json <<'JSON'
{ "run": {"mode": "video"},
  "video_input": {"path": "/opt/fsoc/clean_probe.mp4",
                  "ground_truth_path": "/opt/fsoc/clean_probe_truth.csv"} }
JSON
/opt/fsoc/fsoc-tracker --scenario video.json --headless

# 7. Confirm the outputs, and that Mode B decoded the right number of frames.
ls -la /root/logs/
grep -vc '^#' /root/logs/frames.csv     # expect 61: header + 60 video frames
head -30 /root/logs/frames.csv          # header carries the metric definitions
```

A frame count of anything other than 61 in step 7 means the video was not decoded and some other
source ran instead — exactly the failure that let a 150-frame file report 1800 frames during
development.

### What each step catches

| Step | Failure it catches |
|---|---|
| `--selftest` | missing shared libraries; NumPy/OpenCV import failures; a broken llvmlite bundle |
| `--check-config` | bundled data not found, i.e. `sys._MEIPASS` resolution |
| `--headless` run | path handling for *writes* — output directories resolve against the working directory, not the bundle |
| output inspection | report and CSV generation, which changes behaviour under freezing |

### Expected self-test output

```
  [PASS] numpy + linalg               2.5.3
  [PASS] opencv + GaussianBlur        4.14.0
  [PASS] bundled config               /tmp/_MEIxxxxxx/config/default.json
  [PASS] vision pipeline round-trip   centroid error 0.0037 px
  [PASS] numba JIT (optional)         not bundled (lean build; expected)
self-test PASSED
```

`numba JIT` reporting "not bundled" is **correct** for the lean build. In a `--with-numba` build
it must instead read "llvmlite loaded and compiled"; anything else means the JIT runtime shipped
broken.

### Second base image — and what it is expected to show

`ubuntu:24.04` matches the build host, which is the *most forgiving* case. A different glibc is
what actually catches portability failures, so repeat steps 1-7 on a second image:

```bash
docker run --rm -it -v "$PWD/dist:/opt/fsoc:ro" debian:12 bash
```

**Expect this to fail, and that is the useful result.** Symbol analysis of the current build:

| component | highest glibc symbol required |
|---|---|
| PyInstaller bootloader | `GLIBC_2.14` |
| bundled wheels (`cv2`, `numpy` — manylinux) | `GLIBC_2.27` |
| **collected system libraries** | **`GLIBC_2.38`** |

The wheels are portable; the *system* libraries PyInstaller collects from the build host are not.
`libpython3.12.so.1.0`, `libstdc++.so.6`, `libcrypto.so.3` and others all carry `GLIBC_2.38`
symbols, and `libpython` is not optional. So the executable requires **glibc >= 2.38**:

| distro | glibc | expected |
|---|---|---|
| Ubuntu 24.04 | 2.39 | works |
| Debian 13 | 2.41 | works |
| Debian 12 | 2.36 | **fails** — below 2.38 |
| Ubuntu 22.04 | 2.35 | **fails** — below 2.38 |

The failure mode is a loader error at startup (`version 'GLIBC_2.38' not found`), not a subtle
misbehaviour, so it is unmistakable.

**The fix, if an older target matters:** build inside a container matching the *oldest* machine
the executable must run on. PyInstaller links against whatever the build host provides, so the
build host sets the floor:

```bash
docker run --rm -v "$PWD:/src" -w /src python:3.12-bookworm bash -c \
  "pip install -r requirements.txt && python -m PyInstaller --noconfirm fsoc-tracker.spec"
```

That lowers the floor to Debian 12's glibc 2.36. Building on the demo machine itself is the
simplest option if the demo machine is known in advance.

### Verification status

Stated honestly rather than implying general portability:

| target | status |
|---|---|
| Build host (Ubuntu 24.04, glibc 2.39), source run | **verified** |
| Build host, frozen run from an unrelated working directory | **verified** — this caught the bundled-config bug |
| Build host, frozen Mode B video decode | **verified** — 150-frame file, identical results to source |
| `ubuntu:24.04` container, no Python | **not yet run** |
| `debian:12` container, no Python | **not yet run** — predicted to fail on glibc, see above |
| Windows, macOS | **not attempted.** The spec is platform-neutral but nothing has been built or run there. |

Running from a different working directory on the build machine catches path-handling bugs but
**not** missing shared libraries, because the build host already has them. It is not a substitute
for a container.

## Frozen-versus-source output equivalence

The frozen build must produce identical results, not merely run. Verified by running the same
scenario both ways and diffing:

- All **28 non-timing columns** of the per-frame CSV are byte-identical across 180 frames.
- `processing_ms` differs, as it must — it is wall-clock time.
- The HTML report differs only in the generated-at timestamp, the config path, and the derived
  FPS figures.

`tests/test_packaging.py` re-runs this comparison whenever a build exists.
