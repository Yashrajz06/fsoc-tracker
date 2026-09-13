#!/usr/bin/env bash
# Build the standalone headless executable.
#
# Usage:
#   scripts/build.sh                # lean build, no Numba (default)
#   scripts/build.sh --with-numba   # include the Numba/llvmlite JIT runtime (+~200 MB)
#   scripts/build.sh --gui          # build the GUI executable (adds Qt) as fsoc-tracker-gui
#   scripts/build.sh --clean        # remove build artefacts first
#
# The headless and GUI executables are built separately and kept side by side. Qt plugin
# failures are a different class from base packaging failures, so a working headless binary
# remains a deliverable even if Qt packaging proves difficult.
#
# The executable lands in dist/fsoc-tracker. It is self-contained: no Python required on the
# target machine. See docs/PACKAGING.md for the clean-machine verification procedure, which is
# the only test that actually counts.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
export BUNDLE_NUMBA=0
export BUILD_GUI=0

for arg in "$@"; do
  case "$arg" in
    --with-numba) export BUNDLE_NUMBA=1 ;;
    --gui)        export BUILD_GUI=1 ;;
    --clean)      rm -rf build dist ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "==> building (BUNDLE_NUMBA=$BUNDLE_NUMBA BUILD_GUI=$BUILD_GUI)"
"$PYTHON" -m PyInstaller --noconfirm --log-level=WARN fsoc-tracker.spec

if [ "$BUILD_GUI" = "1" ]; then BIN=dist/fsoc-tracker-gui; else BIN=dist/fsoc-tracker; fi
if [ ! -x "$BIN" ]; then
  echo "build produced no executable at $BIN" >&2
  exit 1
fi

echo "==> built $BIN ($(du -h "$BIN" | cut -f1))"
echo "==> smoke test: --selftest"
"$BIN" --selftest >/dev/null
echo "==> smoke test passed"
