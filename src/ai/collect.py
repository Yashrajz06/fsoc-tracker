"""Build a labelled patch corpus by running the real pipeline over generated video.

Developer tooling, not runtime. Invoked by ``scripts/train_ai.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from src.ai.dataset import PatchSet, extract_patch, label_for
from src.config import AppConfig
from src.video_source import VideoFrameSource, load_ground_truth_sidecar
from src.vision.pipeline import VisionPipeline

__all__ = ["collect_from_video", "collect_from_videos"]


def collect_from_video(video: Path, sidecar: Path, config: AppConfig,
                       max_candidates: int = 12) -> Tuple[List[np.ndarray], List[int]]:
    """Run the pipeline over one clip and label every candidate it produces.

    Args:
        video: Clip to read.
        sidecar: Ground-truth CSV for that clip.
        config: Validated configuration supplying the vision parameters.
        max_candidates: Cap on candidates kept per frame, strongest first by flux. Bounds the
            corpus so a heavily-impulsive clip cannot dominate it.

    Returns:
        ``(patches, labels)`` accumulated across the clip.
    """
    pipeline = VisionPipeline.from_config(config)
    truth = load_ground_truth_sidecar(sidecar)
    patches: List[np.ndarray] = []
    labels: List[int] = []
    with VideoFrameSource(video) as source:
        for frame_data in source:
            if frame_data.frame_index not in truth:
                continue
            target = truth[frame_data.frame_index]
            measurement = pipeline.process(frame_data.frame, fwhm_px=5.887, from_fallback=True)
            for detection in measurement.detections[:max_candidates]:
                label = label_for(detection, target)
                if label is None:
                    continue
                patch = extract_patch(frame_data.frame, detection.x, detection.y)
                if patch is None:
                    continue
                patches.append(patch)
                labels.append(label)
    return patches, labels


def collect_from_videos(pairs: Sequence[Tuple[Path, Path]], config: AppConfig) -> PatchSet:
    """Accumulate a corpus across several clips.

    Args:
        pairs: ``(video, sidecar)`` pairs.
        config: Validated configuration.

    Returns:
        The combined labelled set, unbalanced.
    """
    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    for video, sidecar in pairs:
        patches, labels = collect_from_video(video, sidecar, config)
        all_patches.extend(patches)
        all_labels.extend(labels)
    return PatchSet.from_lists(all_patches, all_labels)
