"""Export the NumPy network to ONNX.

The graph is built explicitly rather than traced, because there is no framework to trace. Every
operator here has a direct counterpart in :meth:`src.ai.model.ConvNet.forward`, and
:func:`verify_export` checks the two agree numerically -- a hand-built graph that silently
disagrees with the trained weights is the obvious failure mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.ai.model import ConvNet, PATCH_SIZE

__all__ = ["export_onnx", "verify_export"]


def export_onnx(net: ConvNet, path: Path) -> Path:
    """Write the network to an ONNX file.

    Args:
        net: Trained network.
        path: Destination ``.onnx`` path.

    Returns:
        The written path.
    """
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    initialisers = []
    current = "input"
    for index in range(1, len(net.LAYERS) + 1):
        w, b = net.params[f"w{index}"], net.params[f"b{index}"]
        initialisers += [numpy_helper.from_array(w.astype(np.float32), f"W{index}"),
                         numpy_helper.from_array(b.astype(np.float32), f"B{index}")]
        nodes.append(helper.make_node("Conv", [current, f"W{index}", f"B{index}"], [f"c{index}"],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
        nodes.append(helper.make_node("Relu", [f"c{index}"], [f"r{index}"]))
        current = f"r{index}"
        if index < len(net.LAYERS):
            nodes.append(helper.make_node("MaxPool", [current], [f"p{index}"],
                                          kernel_shape=[2, 2], strides=[2, 2]))
            current = f"p{index}"
    nodes.append(helper.make_node("GlobalAveragePool", [current], ["gap"]))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], axis=1))
    initialisers += [
        numpy_helper.from_array(net.params["wfc"].astype(np.float32), "Wfc"),
        numpy_helper.from_array(net.params["bfc"].astype(np.float32), "Bfc"),
    ]
    nodes.append(helper.make_node("Gemm", ["flat", "Wfc", "Bfc"], ["logit"], transB=1))
    nodes.append(helper.make_node("Sigmoid", ["logit"], ["output"]))

    graph = helper.make_graph(
        nodes, "fsoc_candidate_discriminator",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       ["N", 1, PATCH_SIZE, PATCH_SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 1])],
        initialisers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(model.SerializeToString())
    return path


def verify_export(net: ConvNet, path: Path, seed: int = 0) -> Tuple[float, bool]:
    """Check the exported graph reproduces the NumPy forward pass.

    Args:
        net: The network that was exported.
        path: The written ``.onnx`` file.
        seed: RNG seed for the probe batch.

    Returns:
        ``(max_absolute_difference, agrees)``.
    """
    import onnxruntime

    rng = np.random.default_rng(seed)
    batch = rng.standard_normal((16, 1, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    expected = net.forward(batch)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: batch})[0].reshape(-1)
    difference = float(np.max(np.abs(expected - actual)))
    return difference, difference < 1e-5
