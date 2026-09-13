"""A small convolutional discriminator, implemented in NumPy.

Why NumPy and not a framework
-----------------------------
``CLAUDE.md`` forbids adding heavy dependencies without discussion, because every one raises the
chance the PyInstaller build fails -- and a broken executable costs the Functional Verification
marks regardless of how good the model is. PyTorch would be a ~2 GB dependency used for a network
of ~3.6k parameters.

So training is written here directly against NumPy. Only two things reach the runtime: the
exported ``.onnx`` file and ``onnxruntime``, which ``requirements.txt`` already sanctioned for
this phase. The training code is developer-only, in the same way the fixture generator's external video
encoder is developer-only and never invoked by shipped code.

Architecture
------------
Input is a single-channel 32x32 patch, per-patch standardised (see :mod:`src.ai.dataset`)::

    conv 1->8   3x3  -> ReLU -> maxpool 2   (32 -> 16)
    conv 8->16  3x3  -> ReLU -> maxpool 2   (16 -> 8)
    conv 16->16 3x3  -> ReLU -> global average pool
    linear 16->1     -> sigmoid

Global average pooling rather than a flatten-and-dense head: it holds the parameter count at
~3.6k, and it makes the score depend on *how much of the patch looks beacon-like* rather than on
where in the patch the evidence sits, which is the property wanted when the candidate's coarse
position is itself uncertain by a pixel or two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["ConvNet", "PATCH_SIZE"]

#: Side length of the patch the network scores. Chosen to comfortably contain the largest
#: specified beacon (20 px) plus surrounding background for context.
PATCH_SIZE = 32


def _im2col(x: np.ndarray, k: int, pad: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Expand a batch of images into column form for convolution as a matrix multiply.

    Args:
        x: Input of shape ``(N, C, H, W)``.
        k: Square kernel side length.
        pad: Zero padding applied to each spatial edge.

    Returns:
        ``(columns, (out_h, out_w))`` where columns has shape ``(N, C*k*k, out_h*out_w)``.
    """
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_h, out_w = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[3],
               x.strides[2], x.strides[3])
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, out_h, out_w, k, k), strides=strides, writeable=False)
    cols = windows.transpose(0, 1, 4, 5, 2, 3).reshape(n, c * k * k, out_h * out_w)
    return cols, (out_h, out_w)


def _col2im(cols: np.ndarray, shape: Tuple[int, int, int, int], k: int,
            pad: int) -> np.ndarray:
    """Accumulate column-form gradients back into image form (the transpose of :func:`_im2col`)."""
    n, c, h, w = shape
    out_h, out_w = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    padded = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    reshaped = cols.reshape(n, c, k, k, out_h, out_w)
    for i in range(k):
        for j in range(k):
            padded[:, :, i:i + out_h, j:j + out_w] += reshaped[:, :, i, j]
    return padded[:, :, pad:pad + h, pad:pad + w] if pad else padded


@dataclass
class ConvNet:
    """The discriminator: parameters, forward pass, and gradients.

    Attributes:
        params: Weight and bias arrays keyed by name.
    """

    params: Dict[str, np.ndarray]

    # Channel widths per convolution, input to output.
    LAYERS: Tuple[Tuple[int, int], ...] = ((1, 8), (8, 16), (16, 16))

    @classmethod
    def initialise(cls, seed: int = 0) -> "ConvNet":
        """Create a network with He-initialised weights.

        Args:
            seed: RNG seed, so training is reproducible for the report.

        Returns:
            An initialised network.
        """
        rng = np.random.default_rng(seed)
        params: Dict[str, np.ndarray] = {}
        for index, (cin, cout) in enumerate(cls.LAYERS, start=1):
            scale = np.sqrt(2.0 / (cin * 9))
            params[f"w{index}"] = (rng.standard_normal((cout, cin, 3, 3)) * scale).astype(
                np.float32)
            params[f"b{index}"] = np.zeros(cout, dtype=np.float32)
        last = cls.LAYERS[-1][1]
        params["wfc"] = (rng.standard_normal((1, last)) * np.sqrt(1.0 / last)).astype(np.float32)
        params["bfc"] = np.zeros(1, dtype=np.float32)
        return cls(params=params)

    def forward(self, x: np.ndarray, cache: bool = False):
        """Score a batch of patches.

        Args:
            x: Patches of shape ``(N, 1, 32, 32)``, already standardised.
            cache: Retain intermediates needed by :meth:`backward`.

        Returns:
            Probabilities of shape ``(N,)``, or ``(probabilities, cache)`` when ``cache``.
        """
        store: List = []
        h = x.astype(np.float32)
        for index in range(1, len(self.LAYERS) + 1):
            w, b = self.params[f"w{index}"], self.params[f"b{index}"]
            cols, (oh, ow) = _im2col(h, 3, 1)
            flat = w.reshape(w.shape[0], -1)
            out = np.einsum("ok,nkp->nop", flat, cols) + b[None, :, None]
            out = out.reshape(h.shape[0], w.shape[0], oh, ow)
            relu_mask = out > 0
            out = out * relu_mask
            if cache:
                store.append((h.shape, cols, relu_mask))
            h = out
            if index < len(self.LAYERS):          # pool after all but the last convolution
                n, c, hh, ww = h.shape
                # The transpose is load-bearing. Reshaping (n, c, H, 2, W, 2) straight to
                # (n, c, H, W, 4) merges the h-inner axis with the w-block axis, so argmax
                # selects within the wrong group of four and the backward pass routes each
                # gradient to the wrong pixel. Layers followed by pooling then get wrong
                # gradients while the unpooled layers stay exact -- which is precisely how the
                # numerical gradient check surfaced it.
                blocks = h.reshape(n, c, hh // 2, 2, ww // 2, 2).transpose(0, 1, 2, 4, 3, 5)
                flat = blocks.reshape(n, c, hh // 2, ww // 2, 4)
                idx = flat.argmax(axis=-1)
                h = flat.max(axis=-1)
                if cache:
                    store.append(("pool", idx, (n, c, hh, ww)))
        gap = h.mean(axis=(2, 3))
        logit = gap @ self.params["wfc"].T + self.params["bfc"]
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit[:, 0], -30, 30)))
        if cache:
            store.append(("head", gap, h.shape))
            return prob, store
        return prob

    def backward(self, store: List, prob: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Gradients of mean binary cross-entropy with respect to every parameter.

        Args:
            store: Cache from :meth:`forward`.
            prob: Predicted probabilities.
            labels: Binary targets.

        Returns:
            Gradient arrays keyed as in :attr:`params`.
        """
        n = labels.shape[0]
        grads: Dict[str, np.ndarray] = {}
        _, gap, h_shape = store.pop()
        dlogit = ((prob - labels) / n).astype(np.float32)[:, None]
        grads["wfc"] = dlogit.T @ gap
        grads["bfc"] = dlogit.sum(axis=0)
        dgap = dlogit @ self.params["wfc"]
        dh = np.broadcast_to(dgap[:, :, None, None] / (h_shape[2] * h_shape[3]),
                             h_shape).astype(np.float32).copy()
        for index in range(len(self.LAYERS), 0, -1):
            if index < len(self.LAYERS):
                _, idx, full = store.pop()
                nn, cc, hh, ww = full
                expanded = np.zeros((nn, cc, hh // 2, ww // 2, 4), dtype=np.float32)
                np.put_along_axis(expanded, idx[..., None], dh[..., None], axis=-1)
                dh = expanded.reshape(nn, cc, hh // 2, ww // 2, 2, 2)
                dh = dh.transpose(0, 1, 2, 4, 3, 5).reshape(nn, cc, hh, ww)
            in_shape, cols, relu_mask = store.pop()
            dh = dh * relu_mask
            w = self.params[f"w{index}"]
            dout = dh.reshape(dh.shape[0], dh.shape[1], -1)
            grads[f"w{index}"] = np.einsum("nop,nkp->ok", dout, cols).reshape(w.shape)
            grads[f"b{index}"] = dout.sum(axis=(0, 2))
            dcols = np.einsum("ok,nop->nkp", w.reshape(w.shape[0], -1), dout)
            dh = _col2im(dcols, in_shape, 3, 1)
        return grads
