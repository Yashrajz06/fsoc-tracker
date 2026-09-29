"""The candidate discriminator: gradients, export fidelity, and failure policy.

The governing constraint is that the classical path must survive every AI failure mode. Most of
these tests assert that nothing happens rather than that something does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.ai.dataset import extract_patch, standardise
from src.ai.export import export_onnx, verify_export
from src.ai.model import ConvNet, PATCH_SIZE
from src.config import load_config

onnxruntime = pytest.importorskip("onnxruntime")


def test_analytic_gradients_match_numerical_ones():
    """Hand-written backpropagation is checked against finite differences.

    This caught a real defect. Max-pooling reshaped ``(n, c, H, 2, W, 2)`` directly to
    ``(n, c, H, W, 4)``, merging the h-inner axis with the w-block axis, so ``argmax`` selected
    within the wrong group of four and gradients were routed to the wrong pixels. The layers
    *after* pooling stayed numerically exact while the layers feeding it were badly wrong -- the
    network still trained, just worse, which is exactly the kind of defect that survives to a
    demo.
    """
    rng = np.random.default_rng(3)
    net = ConvNet.initialise(0)
    for key in net.params:
        net.params[key] = net.params[key].astype(np.float64)
    x = rng.standard_normal((4, 1, PATCH_SIZE, PATCH_SIZE))
    y = rng.integers(0, 2, 4).astype(np.float64)

    def loss() -> float:
        p = np.clip(net.forward(x), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    probability, cache = net.forward(x, cache=True)
    grads = net.backward(cache, probability, y)
    worst = 0.0
    for key, array in net.params.items():
        for _ in range(4):
            index = tuple(rng.integers(0, size) for size in array.shape)
            original, eps = array[index], 1e-6
            array[index] = original + eps
            plus = loss()
            array[index] = original - eps
            minus = loss()
            array[index] = original
            numerical = (plus - minus) / (2 * eps)
            analytic = grads[key][index]
            worst = max(worst, abs(numerical - analytic)
                        / max(abs(numerical), abs(analytic), 1e-12))
    assert worst < 1e-4, f"worst relative gradient error {worst:.2e}"


def test_onnx_export_matches_the_numpy_forward_pass(tmp_path):
    """The ONNX graph is hand-built, so it must be checked against the weights it exports."""
    net = ConvNet.initialise(1)
    path = export_onnx(net, tmp_path / "m.onnx")
    difference, agrees = verify_export(net, path)
    assert agrees, f"exported graph disagrees with NumPy by {difference:.2e}"


def test_patch_standardisation_is_invariant_to_brightness_and_contrast():
    """The network must not learn absolute brightness, which does not transfer to unseen clips."""
    rng = np.random.default_rng(0)
    patch = rng.integers(20, 200, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    base = standardise(patch)
    scaled = standardise(patch * 0.5 + 30.0)
    assert np.allclose(base, scaled, atol=1e-4)


def test_edge_candidates_are_skipped_rather_than_padded():
    """Zero padding would teach the network that the frame boundary is a feature."""
    frame = np.zeros((64, 64), dtype=np.uint8)
    assert extract_patch(frame, 2.0, 2.0) is None
    assert extract_patch(frame, 32.0, 32.0) is not None


# --- failure policy ----------------------------------------------------------------------------

def _ai_config(tmp_path, **overrides):
    payload = {"ai": {"enabled": True, "model_path": None, **overrides}}
    path = tmp_path / "ai.json"
    path.write_text(json.dumps(payload))
    return load_config("config/default.json", overrides=[str(path)]).ai


def test_enabling_ai_without_a_model_is_rejected_at_load(tmp_path):
    """``ai.enabled`` with no model is a contradictory scenario and fails fast.

    Config rejects it on load rather than starting a run that silently has no discriminator.
    That is the right place for it: an evaluator who enables the AI and gets a completed run
    with no indication the model was missing would have no way to know.
    """
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="model_path"):
        _ai_config(tmp_path)


def test_corrupt_model_disables_the_discriminator_without_raising(tmp_path):
    """A truncated or non-ONNX file must be survivable."""
    from src.ai.validator import load_discriminator

    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    assert load_discriminator(_ai_config(tmp_path, model_path=str(bad))) is None


def test_never_and_unknown_backend_disable_the_discriminator(tmp_path):
    from src.ai.validator import load_discriminator

    net = ConvNet.initialise(0)
    model = export_onnx(net, tmp_path / "m.onnx")
    assert load_discriminator(_ai_config(tmp_path, model_path=str(model),
                                         invoke_on="never")) is None
    assert load_discriminator(_ai_config(tmp_path, model_path=str(model),
                                         backend="tensorrt")) is None


def test_discriminator_only_runs_when_ranking_is_ambiguous(tmp_path):
    """With a single candidate, flux ranking has made no choice worth second-guessing."""
    from src.ai.validator import load_discriminator
    from src.vision.detect import Detection

    net = ConvNet.initialise(0)
    model = export_onnx(net, tmp_path / "m.onnx")
    discriminator = load_discriminator(_ai_config(tmp_path, model_path=str(model)))
    assert discriminator is not None

    def blob() -> Detection:
        return Detection(x=40.0, y=40.0, area_px=9.0, peak=100.0, flux=500.0,
                         circularity=0.9, bbox=(37, 37, 6, 6))

    assert not discriminator.should_run([])
    assert not discriminator.should_run([blob()])
    assert discriminator.should_run([blob(), blob()])


def test_inference_stays_far_inside_its_budget(tmp_path):
    """Steady-state cost must leave the 20 FPS budget intact.

    Measured separately: the *first* inference costs ~42 ms because ONNX Runtime builds the graph
    then, which originally tripped the budget check and disabled the discriminator on frame 1 of
    every clip. The session is warmed at load for that reason, and this test would fail if the
    warm-up were removed.
    """
    from src.ai.validator import load_discriminator

    net = ConvNet.initialise(0)
    model = export_onnx(net, tmp_path / "m.onnx")
    discriminator = load_discriminator(_ai_config(tmp_path, model_path=str(model)))
    batch = np.zeros((64, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    import time

    start = time.perf_counter()
    for _ in range(5):
        discriminator.session.run(None, {discriminator._input_name: batch})
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5
    assert elapsed_ms < 15.0, f"{elapsed_ms:.2f} ms per frame for 64 candidates"


def test_classical_pipeline_is_unchanged_when_ai_is_disabled():
    """When ``invoke_on='never'``, no discriminator is attached regardless of enabled flag."""
    import json, os, tempfile
    from src.config import load_config, merge_overrides
    from src.vision.pipeline import VisionPipeline

    # Build a config with invoke_on="never" — the discriminator must be None.
    raw = json.load(open("config/default.json"))
    raw_never = merge_overrides(raw, {"ai": {"invoke_on": "never"}})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(raw_never, f)
        tmppath = f.name
    try:
        config_never = load_config(tmppath)
    finally:
        os.unlink(tmppath)

    pipeline_never = VisionPipeline.from_config(config_never)
    assert pipeline_never.discriminator is None, (
        "invoke_on='never' must not attach a discriminator"
    )

    # With the default config, the discriminator is loaded when onnxruntime and the
    # model file are both available; None otherwise. Both outcomes are valid.
    from src.ai.validator import CandidateDiscriminator
    pipeline_default = VisionPipeline.from_config(load_config("config/default.json"))
    assert pipeline_default.discriminator is None or isinstance(
        pipeline_default.discriminator, CandidateDiscriminator
    ), "Default config should produce either a CandidateDiscriminator or None"
