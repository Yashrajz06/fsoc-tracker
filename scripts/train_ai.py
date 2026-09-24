#!/usr/bin/env python3
"""Train and export the candidate discriminator. Developer tooling, not runtime.

Usage::

    PYTHONPATH=. python scripts/train_ai.py --out models/discriminator.onnx

The training clips are deliberately **disjoint from the six evaluation fixtures**. Training on
the clips the model is then scored against would make the headline result self-confirming, which
is the single easiest way to produce an AI component that looks excellent and fails on an
evaluator's video. Training conditions are chosen to bracket the failure regime -- low peak
intensity, heavy impulse noise, low bitrate -- and to include clean clips so the network is not
trained only on pathology.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenarios.generate import AtmosphericParams, VideoSpec, generate_video  # noqa: E402
from src.ai.collect import collect_from_videos  # noqa: E402
from src.ai.export import export_onnx, verify_export  # noqa: E402
from src.ai.model import ConvNet  # noqa: E402
from src.config import load_config  # noqa: E402

#: Training clips. Distinct in name, geometry, noise and seed from ``DEFAULT_SPECS``.
#: Specs now cover the full spec-mandated beacon size range (5–20 px) to prevent
#: the discriminator from being confidently wrong on large-spot clips.
TRAIN_SPECS = (
    VideoSpec("tr_impulse_dim", width=640, height=480, frames=90, peak=65.0, background=10.0,
              gaussian_sigma=14.0, sp_density=0.09, bitrate_kbps=1100),
    VideoSpec("tr_impulse_mid", width=800, height=600, frames=90, peak=95.0, background=12.0,
              gaussian_sigma=12.0, sp_density=0.06, bitrate_kbps=1500, motion="circular"),
    VideoSpec("tr_impulse_heavy", width=720, height=576, frames=90, peak=80.0, background=6.0,
              gaussian_sigma=16.0, sp_density=0.12, bitrate_kbps=900, motion="linear"),
    VideoSpec("tr_clean_small", width=640, height=480, frames=60, size_px=6.0, sigma_px=1.6,
              peak=180.0, gaussian_sigma=6.0, bitrate_kbps=3000),
    VideoSpec("tr_clean_big", width=960, height=720, frames=60, size_px=18.0, sigma_px=5.0,
              peak=220.0, background=50.0, gaussian_sigma=7.0, motion="figure8"),
    VideoSpec("tr_fog", width=800, height=600, frames=60, size_px=12.0, sigma_px=3.5,
              peak=200.0, background=20.0, gaussian_sigma=9.0, motion="circular",
              atmosphere=AtmosphericParams(beta=0.6, airlight=150.0, blur_sigma=1.2)),
    VideoSpec("tr_large_spot", width=960, height=720, frames=90, size_px=20.0, sigma_px=5.5,
              peak=200.0, background=15.0, gaussian_sigma=8.0, sp_density=0.05, bitrate_kbps=2000,
              motion="circular"),
    VideoSpec("tr_medium_spot", width=800, height=600, frames=90, size_px=14.0, sigma_px=3.8,
              peak=180.0, background=12.0, gaussian_sigma=10.0, sp_density=0.07, bitrate_kbps=1600,
              motion="figure8"),
    VideoSpec("tr_small_spot", width=640, height=480, frames=90, size_px=5.0, sigma_px=1.2,
              peak=160.0, gaussian_sigma=8.0, sp_density=0.08, bitrate_kbps=1200),
)


def main() -> int:
    """Generate clips, collect patches, train, export and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/discriminator.onnx")
    parser.add_argument("--clips", default="build/train_clips")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clip_dir = Path(args.clips)
    clip_dir.mkdir(parents=True, exist_ok=True)
    print(f"generating {len(TRAIN_SPECS)} training clips into {clip_dir} ...")
    pairs = []
    for spec in TRAIN_SPECS:
        video, sidecar = generate_video(spec, clip_dir)
        pairs.append((video, sidecar))
        print(f"   {spec.name}")

    config = load_config("config/default.json")
    print("collecting labelled candidate patches ...")
    corpus = collect_from_videos(pairs, config)
    positives = int((corpus.labels > 0.5).sum())
    print(f"   {len(corpus)} patches: {positives} beacon, {len(corpus) - positives} artifact")
    if positives == 0:
        print("ERROR: no positive examples; the detector never found the target in training "
              "clips. Training would be meaningless.")
        return 1

    rng = np.random.default_rng(args.seed)
    corpus = corpus.balanced(rng)
    order = rng.permutation(len(corpus))
    split = int(0.8 * len(order))
    train_idx, val_idx = order[:split], order[split:]
    xtr, ytr = corpus.patches[train_idx], corpus.labels[train_idx]
    xva, yva = corpus.patches[val_idx], corpus.labels[val_idx]
    print(f"   balanced to {len(corpus)}; train {len(train_idx)}, validation {len(val_idx)}")

    net = ConvNet.initialise(args.seed)
    velocity = {k: np.zeros_like(v) for k, v in net.params.items()}
    best_accuracy, best_params = 0.0, None

    for epoch in range(args.epochs):
        shuffled = rng.permutation(len(xtr))
        total = 0.0
        for start in range(0, len(shuffled), args.batch):
            batch = shuffled[start:start + args.batch]
            if batch.size < 2:
                continue
            probability, cache = net.forward(xtr[batch], cache=True)
            clipped = np.clip(probability, 1e-7, 1 - 1e-7)
            total += float(-(ytr[batch] * np.log(clipped)
                             + (1 - ytr[batch]) * np.log(1 - clipped)).mean()) * batch.size
            grads = net.backward(cache, probability, ytr[batch])
            for key in net.params:                      # SGD with momentum
                velocity[key] = 0.9 * velocity[key] - args.lr * grads[key]
                net.params[key] = net.params[key] + velocity[key]
        accuracy = float(((net.forward(xva) > 0.5) == (yva > 0.5)).mean()) if len(xva) else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {k: v.copy() for k, v in net.params.items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"   epoch {epoch:>3}  loss {total / max(len(xtr), 1):.4f}  "
                  f"val accuracy {accuracy:.4f}")

    if best_params is not None:
        net.params = best_params
    print(f"best validation accuracy: {best_accuracy:.4f}")

    out = Path(args.out)
    export_onnx(net, out)
    difference, agrees = verify_export(net, out)
    print(f"exported {out} ({out.stat().st_size} bytes); "
          f"ONNX vs NumPy max difference {difference:.2e} -> {'agrees' if agrees else 'MISMATCH'}")
    return 0 if agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
