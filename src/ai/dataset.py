"""Labelled patch extraction for the candidate discriminator.

The training set is built from *the classical detector's own candidates*, not from synthetic
positives and negatives. Each frame is passed through the real pipeline, every surviving
candidate blob is cropped, and it is labelled by distance to the known ground-truth centroid.

That choice matters more than the architecture. A network trained on hand-made "beacon" and
"noise" images learns to separate a distribution it will never see; a network trained on detector
candidates learns to separate the two things that are *actually confusable at inference* -- a dim
beacon and a compression artifact that outranks it by flux. The failure being targeted is
``lowlight_impulse``, where 149 of 150 frames pick an artifact over the target.

Patches are standardised per-patch, never globally: subtract the median and divide by
``1.4826 * MAD`` (floored at the quantisation limit, as everywhere else in this codebase). This
is what keeps the network from learning absolute brightness, which is exactly the cue that does
not transfer to an evaluator's unseen clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.ai.model import PATCH_SIZE
from src.vision.detect import Detection

__all__ = ["extract_patch", "standardise", "PatchSet", "label_for"]

#: A candidate is a positive example when its coarse centroid lies within this many pixels of
#: truth. Sized from the detector's own coarse-centroid spread, which is a pixel or two; a tighter
#: radius would label genuine beacon detections as negatives and teach the network to reject the
#: thing it exists to find.
POSITIVE_RADIUS_PX = 3.0

#: Candidates between the positive radius and this are neither clearly beacon nor clearly
#: artifact, and are discarded rather than forced into a class.
AMBIGUOUS_RADIUS_PX = 8.0


def standardise(patch: np.ndarray) -> np.ndarray:
    """Normalise a patch to zero median and unit robust scale.

    Robust statistics rather than mean and standard deviation: at the impulse densities this
    model exists to survive, a plain standard deviation is inflated by the impulses themselves,
    which would compress the beacon's contrast towards zero in exactly the regime that matters.

    Args:
        patch: Raw patch, any dtype.

    Returns:
        ``float32`` patch, median-centred and MAD-scaled.
    """
    values = patch.astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # Floored at the 8-bit quantisation limit 1/sqrt(12), matching src.vision robust_sigma.
    scale = max(1.4826 * mad, 1.0 / np.sqrt(12.0))
    return (values - median) / scale


def extract_patch(frame: np.ndarray, x: float, y: float,
                  size: int = PATCH_SIZE) -> Optional[np.ndarray]:
    """Crop a standardised patch centred on a candidate.

    Args:
        frame: Single-channel source frame.
        x: Candidate centre x, in frame coordinates.
        y: Candidate centre y, in frame coordinates.
        size: Patch side length.

    Returns:
        A ``(size, size)`` standardised patch, or ``None`` when the crop would leave the frame.
        Edge candidates are skipped rather than zero-padded: padding introduces a hard artificial
        edge that the network would learn to associate with the frame boundary.
    """
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    if left < 0 or top < 0 or left + size > frame.shape[1] or top + size > frame.shape[0]:
        return None
    return standardise(frame[top:top + size, left:left + size])


def label_for(detection: Detection, truth: Tuple[float, float]) -> Optional[int]:
    """Label a candidate against ground truth.

    Args:
        detection: The candidate blob.
        truth: True beacon centroid as ``(x, y)`` in the same frame coordinates.

    Returns:
        ``1`` for beacon, ``0`` for artifact, or ``None`` when the candidate is ambiguous and
        should be discarded.
    """
    distance = float(np.hypot(detection.x - truth[0], detection.y - truth[1]))
    if distance <= POSITIVE_RADIUS_PX:
        return 1
    if distance >= AMBIGUOUS_RADIUS_PX:
        return 0
    return None


@dataclass
class PatchSet:
    """A labelled patch collection.

    Attributes:
        patches: Array of shape ``(N, 1, 32, 32)``.
        labels: Array of shape ``(N,)``, 1 for beacon and 0 for artifact.
    """

    patches: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @classmethod
    def from_lists(cls, patches: List[np.ndarray], labels: List[int]) -> "PatchSet":
        """Stack accumulated patches into arrays.

        Args:
            patches: Standardised patches.
            labels: Matching binary labels.

        Returns:
            The stacked set, empty-safe.
        """
        if not patches:
            return cls(patches=np.zeros((0, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32),
                       labels=np.zeros((0,), dtype=np.float32))
        return cls(patches=np.stack(patches)[:, None, :, :].astype(np.float32),
                   labels=np.asarray(labels, dtype=np.float32))

    def balanced(self, rng: np.random.Generator) -> "PatchSet":
        """Down-sample the majority class to equal counts.

        Artifacts vastly outnumber beacons -- one beacon per frame against many impulse blobs --
        and an unbalanced set trains a network that scores everything as artifact and still
        reports high accuracy.

        Args:
            rng: Seeded generator.

        Returns:
            A class-balanced set.
        """
        positive = np.flatnonzero(self.labels > 0.5)
        negative = np.flatnonzero(self.labels < 0.5)
        if positive.size == 0 or negative.size == 0:
            return self
        keep = min(positive.size, negative.size)
        chosen = np.concatenate([rng.choice(positive, keep, replace=False),
                                 rng.choice(negative, keep, replace=False)])
        rng.shuffle(chosen)
        return PatchSet(patches=self.patches[chosen], labels=self.labels[chosen])
