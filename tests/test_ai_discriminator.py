"""Pytest tests for CandidateDiscriminator.should_run() and basic inference.

Tests 1–5 and 7 use a stub ONNX session so they run without onnxruntime installed and
without a model file on disk. Test 6 skips gracefully when either is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.ai.validator import CandidateDiscriminator
from src.vision.detect import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubSession:
    """Minimal ONNX session stand-in that satisfies CandidateDiscriminator.__init__."""

    def get_inputs(self):
        class _Input:
            name = "input"
        return [_Input()]

    def run(self, output_names, feed):
        n = next(iter(feed.values())).shape[0]
        return [np.full((n, 1), 0.5, dtype=np.float32)]


def _make_discriminator(invoke_on: str, flux_ratio_threshold: float = 1.5,
                         confidence_threshold: float = 0.5,
                         max_inference_ms: float = 20.0) -> CandidateDiscriminator:
    """Build a CandidateDiscriminator backed by the stub session."""
    return CandidateDiscriminator(
        session=_StubSession(),
        invoke_on=invoke_on,
        confidence_threshold=confidence_threshold,
        max_inference_ms=max_inference_ms,
        flux_ratio_threshold=flux_ratio_threshold,
    )


def _detection(flux: float = 100.0) -> Detection:
    """Construct a minimal Detection with the given flux."""
    return Detection(
        x=40.0, y=40.0,
        area_px=9.0, peak=flux,
        flux=flux,
        circularity=0.9,
        bbox=(37, 37, 6, 6),
    )


# ---------------------------------------------------------------------------
# Test 1: single candidate
# ---------------------------------------------------------------------------

def test_should_run_single_candidate():
    """classical_failure and multi_candidate return False for one detection; always returns True."""
    one = [_detection(100.0)]

    d_classical = _make_discriminator("classical_failure")
    assert d_classical.should_run(one) is False

    d_multi = _make_discriminator("multi_candidate")
    assert d_multi.should_run(one) is False

    d_always = _make_discriminator("always")
    assert d_always.should_run(one) is True


# ---------------------------------------------------------------------------
# Test 2: zero candidates
# ---------------------------------------------------------------------------

def test_should_run_zero_candidates():
    """All modes return False when there are no candidates."""
    for mode in ("never", "classical_failure", "always", "multi_candidate"):
        d = _make_discriminator(mode)
        assert d.should_run([]) is False, f"mode={mode!r} should return False for empty list"


# ---------------------------------------------------------------------------
# Test 3: multi_candidate skips when flux ratio is high
# ---------------------------------------------------------------------------

def test_multi_candidate_skips_on_high_flux_ratio():
    """Ratio 100/50 = 2.0 exceeds threshold 1.5, so should_run returns False."""
    d = _make_discriminator("multi_candidate", flux_ratio_threshold=1.5)
    detections = [_detection(100.0), _detection(50.0)]   # ratio = 2.0 > 1.5
    assert d.should_run(detections) is False


# ---------------------------------------------------------------------------
# Test 4: multi_candidate invokes when flux ratio is within threshold
# ---------------------------------------------------------------------------

def test_multi_candidate_invokes_on_close_flux():
    """Ratio 100/80 = 1.25 <= threshold 1.5, so should_run returns True."""
    d = _make_discriminator("multi_candidate", flux_ratio_threshold=1.5)
    detections = [_detection(100.0), _detection(80.0)]   # ratio = 1.25 <= 1.5
    assert d.should_run(detections) is True


# ---------------------------------------------------------------------------
# Test 5: multi_candidate guards against zero second flux
# ---------------------------------------------------------------------------

def test_multi_candidate_zero_second_flux():
    """Division-by-zero guard: second flux == 0 must return False, never raise."""
    d = _make_discriminator("multi_candidate", flux_ratio_threshold=1.5)
    detections = [_detection(100.0), _detection(0.0)]
    assert d.should_run(detections) is False


# ---------------------------------------------------------------------------
# Test 6: real inference smoke test (skipped when model or onnxruntime absent)
# ---------------------------------------------------------------------------

def test_inference_does_not_crash():
    """Load the real model and run rank() on a synthetic frame; assert no exception."""
    model_path = Path("models/discriminator.onnx")
    if not model_path.exists():
        pytest.skip("models/discriminator.onnx not present")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime is not installed")

    from src.ai.validator import load_discriminator
    from src.config import AiConfig

    ai_config = AiConfig(
        enabled=True,
        model_path=str(model_path),
        invoke_on="always",
        confidence_threshold=0.5,
        flux_ratio_threshold=1.5,
        max_inference_ms=20.0,
    )
    discriminator = load_discriminator(ai_config)
    if discriminator is None:
        pytest.skip("load_discriminator returned None (onnxruntime unavailable or model bad)")

    frame = np.zeros((480, 640), dtype=np.uint8)
    detection = _detection(100.0)

    result = discriminator.rank(frame, [detection])

    assert isinstance(result, tuple), "rank() must return a tuple"
    assert len(result) == 2, "rank() must return a 2-tuple"


# ---------------------------------------------------------------------------
# Test 7: never mode never runs
# ---------------------------------------------------------------------------

def test_never_mode_never_runs():
    """invoke_on='never' must return False regardless of the candidate count."""
    d = _make_discriminator("never")

    assert d.should_run([]) is False
    assert d.should_run([_detection()]) is False
    assert d.should_run([_detection()] * 5) is False
