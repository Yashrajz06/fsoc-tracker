"""Headless entry point.

Phase 0 deliverable: this must run a scenario end-to-end without error, even while the
simulation, vision, filtering and control modules are still stubs. Its job right now is to prove
that configuration loading, the :class:`~src.framesource.FrameSource` seam and the run loop fit
together, and to put the derived configuration quantities and metric definitions on screen where
they cannot be ignored.

Usage::

    python -m src.main --config config/default.json --headless

As later phases land, :func:`build_frame_source` gains the real
``SimulationFrameSource`` (Phase 1) and ``VideoFrameSource`` (Phase 6), and :func:`run` gains the
vision, filtering, control and telemetry stages. The structure here is deliberately the final
structure so those phases are drop-in rather than rewrites.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple

import numpy as np

from src.config import AppConfig, ConfigError, load_config
from src.framesource import BaseFrameSource, FrameData, FrameSource, GroundTruth
from src.runner import TrackingRunner, build_frame_source
from src.telemetry.report import write_report


class PlaceholderFrameSource(BaseFrameSource):
    """A frame source that emits blank frames at the configured rate.

    **Temporary Phase 0 scaffolding.** It exists only so the run loop has something real to
    iterate while ``src/sim/`` is unwritten, and it is replaced by ``SimulationFrameSource`` in
    Phase 1. It renders no beacon and reports no ground truth, so no tracking metric computed
    from it is meaningful.

    Attributes:
        config: The active application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialise the placeholder source.

        Args:
            config: Active configuration, supplying resolution, rate and duration.
        """
        self.config = config
        self._index = 0
        self._width = config.camera.resolution_width
        self._height = config.camera.resolution_height
        self._rate = config.camera.update_rate_hz
        self._total = max(1, int(config.run.duration_seconds * self._rate))
        self._frame = np.full((self._height, self._width),
                              config.scene.background_level, dtype=np.uint8)
        self.pan_deg = 0.0
        self.tilt_deg = 0.0

    @property
    def supports_pan_tilt(self) -> bool:
        """Placeholder simulates a steerable camera, so pan/tilt is accepted."""
        return True

    @property
    def nominal_rate_hz(self) -> float:
        """Configured camera update rate in hertz."""
        return self._rate

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        """Angular resolution derived from the camera configuration."""
        return self.config.camera.deg_per_pixel

    def get_frame(self) -> Optional[FrameData]:
        """Return the next blank frame, or ``None`` once the run duration is reached.

        Returns:
            A :class:`FrameData` with a uniform background frame and no ground truth, or ``None``
            when the configured duration has elapsed.
        """
        if self._index >= self._total:
            return None
        data = FrameData(
            frame=self._frame,
            timestamp=self._index / self._rate,
            frame_index=self._index,
            ground_truth=None,
            camera_pan_deg=self.pan_deg,
            camera_tilt_deg=self.tilt_deg,
        )
        self._index += 1
        return data

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """Integrate a rate command, clamped to the configured slew limits.

        Args:
            pan_rate_deg_s: Requested pan rate, clamped to the configured maximum.
            tilt_rate_deg_s: Requested tilt rate, clamped to the configured maximum.
            dt: Interval over which the rate applies, in seconds.
        """
        max_pan = self.config.camera.max_pan_speed_deg_s
        max_tilt = self.config.camera.max_tilt_speed_deg_s
        self.pan_deg += max(-max_pan, min(max_pan, pan_rate_deg_s)) * dt
        self.tilt_deg += max(-max_tilt, min(max_tilt, tilt_rate_deg_s)) * dt

    def reset(self) -> None:
        """Rewind to the first frame and zero the boresight."""
        self._index = 0
        self.pan_deg = 0.0
        self.tilt_deg = 0.0


def print_header(config: AppConfig, stream: Optional[TextIO] = None) -> None:
    """Print the configuration summary, derived quantities and metric definitions.

    Everything printed here also belongs in the telemetry log header (Phase 5). Stating the
    metric definitions and the coordinate convention up front is what removes the specification's
    ambiguity for an evaluator rather than leaving it to be guessed.

    Args:
        config: Validated application configuration.
        stream: Output stream to write to. Defaults to the *current* ``sys.stdout``, resolved at
            call time rather than at import time so that redirection works.
    """
    if stream is None:
        stream = sys.stdout
    print("=" * 88, file=stream)
    print("FSOC coarse-alignment virtual camera tracking system", file=stream)
    if config.source_path:
        print(f"configuration           : {config.source_path}", file=stream)
    print("=" * 88, file=stream)
    for line in config.summary_lines():
        print(line, file=stream)

    warnings_list: List[str] = config.startup_warnings()
    if warnings_list:
        print("-" * 88, file=stream)
        for message in warnings_list:
            print(f"[WARNING] {message}", file=stream)

    print("-" * 88, file=stream)
    print("metric definitions in force for this run:", file=stream)
    for name, definition in config.metric_definitions().items():
        print(f"  {name}: {definition}", file=stream)
    print("=" * 88, file=stream)


def selftest(stream=None) -> int:
    """Verify that a frozen build's bundled components actually load and work.

    This is the check to run first on a machine with no Python installed. Freezing failures are
    almost always *import* or *shared-library* failures that do not reproduce from source, and
    they surface as a crash on first use rather than at startup -- so this exercises each
    dependency deliberately instead of waiting for a run to stumble into it.

    Args:
        stream: Output stream. Defaults to the current ``sys.stdout``.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    import platform

    if stream is None:
        stream = sys.stdout

    frozen = getattr(sys, "frozen", False)
    results: List[Tuple[str, bool, str]] = []

    print("FSOC tracker self-test", file=stream)
    print(f"  frozen build      : {frozen}", file=stream)
    print(f"  python            : {sys.version.split()[0]}", file=stream)
    print(f"  platform          : {platform.platform()}", file=stream)
    if frozen:
        print(f"  bundle directory  : {getattr(sys, '_MEIPASS', '?')}", file=stream)

    try:
        import numpy
        array = numpy.linalg.inv(numpy.eye(3) * 2.0)
        ok = bool(numpy.allclose(array, numpy.eye(3) * 0.5))
        results.append(("numpy + linalg", ok, numpy.__version__))
    except Exception as exc:  # pragma: no cover - only reachable on a broken bundle
        results.append(("numpy + linalg", False, str(exc)))

    try:
        import cv2
        import numpy as _np
        blurred = cv2.GaussianBlur(_np.zeros((16, 16), _np.float32), (5, 5), 1.0)
        results.append(("opencv + GaussianBlur", blurred.shape == (16, 16), cv2.__version__))
    except Exception as exc:  # pragma: no cover
        results.append(("opencv + GaussianBlur", False, str(exc)))

    config_path = default_config_path()
    try:
        # When frozen, resolve the AI model path through the bundle so validation succeeds.
        # The config stores model_path as a relative string; in a frozen build that must be
        # mapped to sys._MEIPASS before the AiConfig.validate() path-existence check runs.
        import json as _json
        _raw = _json.loads(open(config_path, encoding="utf-8").read())
        if (getattr(sys, "frozen", False) and
                isinstance(_raw.get("ai"), dict) and _raw["ai"].get("enabled")):
            _model_rel = _raw["ai"].get("model_path")
            if _model_rel:
                _resolved = bundled_resource(_model_rel)
                if _resolved:
                    _raw["ai"]["model_path"] = _resolved
        from src.config import AppConfig as _AppConfig
        config = _AppConfig.from_dict(_raw, source_path=config_path)
        results.append(("bundled config", True, config_path))
    except Exception as exc:
        config = None
        results.append(("bundled config", False, f"{config_path}: {exc}"))

    if config is not None:
        try:
            import numpy as _np

            from src.sim.beacon import BeaconParams, render_beacon
            from src.vision.pipeline import VisionPipeline

            frame = _np.full((160, 160), 30, dtype=_np.uint8)
            patch = render_beacon(80.4, 80.6, BeaconParams(peak_intensity=200.0))
            frame[patch.y0:patch.y0 + patch.data.shape[0],
                  patch.x0:patch.x0 + patch.data.shape[1]] += patch.data.astype(_np.uint8)
            measurement = VisionPipeline.from_config(config).process(
                frame, fwhm_px=5.887, from_fallback=False)
            error = math.hypot(measurement.x - 80.4, measurement.y - 80.6)
            results.append(("vision pipeline round-trip", measurement.found and error < 1.0,
                            f"centroid error {error:.4f} px"))
        except Exception as exc:  # pragma: no cover
            results.append(("vision pipeline round-trip", False, str(exc)))

    # Numba is optional: absent in the lean build by design, so report rather than fail.
    try:
        # Import llvmlite.binding to force the shared library to load: that is the failure mode
        # freezing actually produces, and it is invisible if only `import numba` is attempted.
        # Do NOT call binding.initialize() -- it is deprecated in current llvmlite and raises,
        # which would report a healthy bundle as broken.
        import llvmlite.binding  # noqa: F401
        from numba import njit

        @njit(cache=False)
        def _add(a, b):
            return a + b

        results.append(("numba JIT (optional)", _add(2, 3) == 5, "llvmlite loaded and compiled"))
    except ImportError:
        results.append(("numba JIT (optional)", True, "not bundled (lean build; expected)"))
    except Exception as exc:  # pragma: no cover
        results.append(("numba JIT (optional)", False, f"bundled but broken: {exc}"))

    print(file=stream)
    failures = 0
    for name, ok, detail in results:
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {detail}", file=stream)

    print(file=stream)
    print(f"self-test {'PASSED' if failures == 0 else f'FAILED ({failures})'}", file=stream)
    return 0 if failures == 0 else 1


def launch_gui(args: argparse.Namespace) -> int:
    """Launch the graphical interface.

    Imported lazily so that ``src.main`` stays importable, and the headless build stays
    runnable, on a machine with no Qt installed. ``CLAUDE.md`` requires the core modules to be
    importable headless; a module-level Qt import would break that for every CLI invocation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    try:
        from PySide6.QtWidgets import QApplication

        from src.gui.main_window import MainWindow
    except ImportError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        print("Install PySide6, or use the headless interface.", file=sys.stderr)
        return 3

    config_path = args.config or default_config_path()
    try:
        config = apply_overrides(load_config(config_path, args.scenario), args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    app = QApplication.instance() or QApplication([])
    window = MainWindow(config, config_path)
    window.resize(1360, 820)
    window.show()
    return int(app.exec())


def run(config: AppConfig) -> int:
    """Execute one run end-to-end.

    The per-frame logic lives in :class:`~src.runner.TrackingRunner`, shared with the GUI, so
    there is exactly one implementation of it and the GUI cannot drift from the CLI.

    Args:
        config: Validated application configuration.

    Returns:
        Process exit code: 0 on success.
    """
    print_header(config)

    runner = TrackingRunner(config)
    started = time.perf_counter()
    frames = runner.run()
    elapsed = time.perf_counter() - started

    logger = runner.logger
    summary = logger.summary()

    outputs = []
    if config.telemetry.enabled:
        if config.telemetry.per_frame_csv:
            outputs.append(logger.write_csv())
        if config.telemetry.per_frame_json:
            outputs.append(logger.write_json())
        if config.telemetry.summary_report:
            outputs.append(write_report(logger, config))

    print(f"frames processed        : {frames}")
    print(f"wall-clock elapsed      : {elapsed:.3f} s")
    print(f"real-time throughput    : "
          f"{frames / elapsed if elapsed > 0 else float('inf'):.1f} FPS")
    if summary.processing_ms_mean is not None:
        print(f"mean processing time    : {summary.processing_ms_mean:.2f} ms/frame")
    print(f"lock retention          : {100.0 * summary.lock_retention:.1f} %")
    if summary.centroid_rmse_px is not None:
        print(f"centroiding RMSE        : {summary.centroid_rmse_px:.4f} px")
    if summary.below_characterised_envelope:
        print("NOTE                    : input below characterised envelope; no lock achieved")
    for output in outputs:
        print(f"wrote                   : {output}")
    return 0



def bundled_resource(relative: str) -> Optional[str]:
    """Resolve a data file that travels inside the frozen executable.

    PyInstaller unpacks bundled data to a temporary directory and records it on
    ``sys._MEIPASS``. A relative default path such as ``config/default.json`` therefore resolves
    against the *working directory* rather than the bundle, so a user on a clean machine running
    ``./fsoc-tracker --headless`` from their home directory gets "configuration file not found"
    even though the file shipped with the binary.

    Args:
        relative: Path relative to the project root, e.g. ``"config/default.json"``.

    Returns:
        An absolute path inside the bundle when frozen and the file is present, otherwise
        ``None`` so the caller falls back to the ordinary relative path.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is None:
        return None
    candidate = Path(bundle) / relative
    return str(candidate) if candidate.exists() else None


def default_config_path() -> str:
    """Return the default configuration path, bundle-aware.

    Returns:
        The bundled configuration when running frozen, otherwise the source-tree relative path.
    """
    return bundled_resource("config/default.json") or "config/default.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="src.main",
        description="AI-based virtual camera tracking system for FSOC coarse alignment.",
    )
    parser.add_argument("--config", default=None,
                        help="Path to the JSON configuration file. Defaults to the copy bundled "
                             "with the executable when frozen, or config/default.json in the "
                             "source tree.")
    parser.add_argument("--scenario", action="append", default=None, metavar="PATH",
                        help="Partial JSON override applied on top of --config. May be repeated; "
                             "later files win. This is how evaluator scenarios are loaded.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the GUI. Currently the only supported mode.")
    parser.add_argument("--mode", choices=("simulation", "video"), default=None,
                        help="Override run.mode from the configuration file.")
    parser.add_argument("--video", default=None,
                        help="Path to an input .mp4. Implies --mode video.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override run.duration_seconds, in seconds.")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the graphical interface. Requires PySide6; the headless "
                             "build deliberately does not bundle Qt.")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify that bundled components load and work, then exit. Run this "
                             "first on a machine with no Python installed.")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate the configuration, print the summary, and exit without "
                             "running. Useful in CI.")
    return parser.parse_args(argv)


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides to a loaded configuration.

    Overrides are applied by rebuilding the affected frozen dataclasses and re-validating, so an
    override can never bypass a specification check.

    Args:
        config: The configuration as loaded from disk.
        args: Parsed command-line arguments.

    Returns:
        A validated configuration with the overrides applied. The original is unmodified.

    Raises:
        ConfigError: If the overridden configuration fails validation.
    """
    from dataclasses import replace

    run_cfg = config.run
    video_cfg = config.video_input

    if args.video is not None:
        video_cfg = replace(video_cfg, path=args.video)
        run_cfg = replace(run_cfg, mode="video")
    if args.mode is not None:
        run_cfg = replace(run_cfg, mode=args.mode)
    if args.duration is not None:
        run_cfg = replace(run_cfg, duration_seconds=args.duration)
    if args.headless:
        run_cfg = replace(run_cfg, headless=True)

    updated = replace(config, run=run_cfg, video_input=video_cfg,
                      gui=replace(config.gui, enabled=not run_cfg.headless))
    updated.validate()
    return updated


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 2 on a configuration error.
    """
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.gui:
        return launch_gui(args)
    try:
        config = apply_overrides(
            load_config(args.config or default_config_path(), args.scenario), args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.check_config:
        print_header(config)
        return 0
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
