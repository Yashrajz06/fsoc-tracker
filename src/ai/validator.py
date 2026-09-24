"""ONNX candidate discriminator: scores blobs, never localises.

Scope, deliberately narrow
--------------------------
This is not a detector. The classical pipeline finds candidate blobs and computes the centroid;
the network's only job is to say *which candidate is the beacon* when flux ranking cannot. The
centroid remains intensity-weighted centre of gravity in every configuration, because classical
centroiding is near-optimal for a symmetric spot and a CNN would be both slower and less accurate
at that task.

Why a discriminator is the right place for a network
----------------------------------------------------
Flux ranking has no way to distinguish a bright compression artifact from a dim beacon -- they
differ in *shape*, not in integrated intensity. On ``lowlight_impulse`` (peak 70 over background
8, 8% salt-and-pepper at 1200 kbps) the classical detector picks an artifact on 149 of 150
frames, and no threshold tuning fixes that, because the artifact genuinely has more flux. Shape
discrimination is precisely what a small convolutional network does well.

Failure policy
--------------
Every failure path falls back to the classical answer. A missing ``onnxruntime``, an absent or
corrupt model, or an inference that exceeds ``max_inference_ms`` all leave the classical ranking
in place rather than raising. The AI is an enhancement to a working system, never a dependency of
one.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.ai.dataset import extract_patch
from src.ai.model import PATCH_SIZE
from src.vision.detect import Detection

__all__ = ["CandidateDiscriminator", "load_discriminator"]

LOGGER = logging.getLogger(__name__)

#: Most candidates scored in one frame, strongest first by flux.
#:
#: Not a tuning constant but a cost bound. On a heavily-impulsive clip the detector surfaces ~700
#: candidates per frame (peak 1088); batching all of them cost more than the whole inference
#: budget, so the discriminator disabled itself on frame 1 and appeared to do nothing. 64 patches
#: cost well under a millisecond, and measured on ``lowlight_impulse`` the true beacon -- when it
#: is surfaced at all -- lies within the top 64 by flux on 74% of those frames, against 29% for
#: a top-16 cap. Scoring beyond 64 buys little and costs linearly.
MAX_SCORED_CANDIDATES = 64


class CandidateDiscriminator:
    """Scores detector candidates with an ONNX model.

    Attributes:
        session: The ``onnxruntime`` inference session.
        invoke_on: ``"never"``, ``"classical_failure"`` or ``"always"``.
        confidence_threshold: Minimum score for the network's pick to displace the classical one.
        max_inference_ms: Per-frame budget; exceeding it disables the discriminator for the rest
            of the run rather than silently eroding the FPS requirement.
        flux_ratio_threshold: Maximum flux ratio between the top two candidates below which the
            discriminator is invoked. Only used when ``invoke_on="multi_candidate"``.
    """

    def __init__(self, session, invoke_on: str, confidence_threshold: float,
                 max_inference_ms: float, flux_ratio_threshold: float = 1.5) -> None:
        """Initialise the discriminator.

        Args:
            session: An ``onnxruntime.InferenceSession``.
            invoke_on: When to run. See :attr:`invoke_on`.
            confidence_threshold: Minimum accepted confidence.
            max_inference_ms: Per-frame inference budget in milliseconds.
            flux_ratio_threshold: See :attr:`flux_ratio_threshold`.
        """
        self.session = session
        self.invoke_on = invoke_on
        self.confidence_threshold = float(confidence_threshold)
        self.max_inference_ms = float(max_inference_ms)
        self.flux_ratio_threshold = float(flux_ratio_threshold)
        self._input_name = session.get_inputs()[0].name
        self._disabled = False
        self.last_inference_ms = 0.0
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one throwaway inference so the budget check measures steady-state cost.

        ONNX Runtime performs graph optimisation, memory-arena setup and kernel selection on the
        *first* ``run`` call. Measured here that first call cost 41.8 ms against a 20 ms budget,
        so the discriminator disabled itself on frame 1 of every clip and the measured result was
        "the AI changes nothing" -- a conclusion about initialisation cost masquerading as a
        conclusion about the model. Steady-state inference is roughly two orders faster.
        """
        try:
            probe = np.zeros((1, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            self.session.run(None, {self._input_name: probe})
        except Exception as error:                        # pragma: no cover - backend failure
            LOGGER.warning("discriminator warm-up failed: %s", error)
            self._disabled = True

    def should_run(self, detections: Sequence[Detection]) -> bool:
        """Decide whether to score this frame's candidates.

        Under ``classical_failure`` the discriminator runs only when there are at least two
        surviving candidates. With a single candidate, flux ranking has made no choice and there
        is nothing to second-guess; with several, the ranking is a guess that can be wrong. This
        keeps the cost off the common case without introducing a tuned ambiguity margin.

        Under ``multi_candidate`` the discriminator runs only when there are at least two
        candidates **and** the flux ratio of the top two candidates is at or below
        ``flux_ratio_threshold``. When the top candidate massively outranks all others by flux,
        classical ranking is already reliable; the discriminator is only needed when the choice
        is genuinely ambiguous. This prevents the network from overriding a correct classical
        ranking, which was the failure mode on ``fhd_1920_bigspot`` (96.6% association failure
        when the discriminator overrode a reliable flux-ranked answer).

        Args:
            detections: Surviving candidates, flux-ranked (strongest first).

        Returns:
            Whether :meth:`rank` should be called.
        """
        if self._disabled or self.invoke_on == "never":
            return False
        if self.invoke_on == "always":
            return bool(detections)
        if self.invoke_on == "multi_candidate":
            if len(detections) < 2:
                return False
            second_flux = detections[1].flux
            if second_flux <= 0:
                return False
            ratio = detections[0].flux / second_flux
            return ratio <= self.flux_ratio_threshold
        # "classical_failure" (default)
        return len(detections) >= 2

    def rank(self, frame: np.ndarray,
             detections: Sequence[Detection]
             ) -> Tuple[Optional[Tuple[float, ...]], Optional[Detection]]:
        """Score every candidate and return the network's pick.

        Args:
            frame: The frame the candidates were found in.
            detections: Surviving candidates, flux-ranked.

        Returns:
            ``(scores, chosen)``. ``chosen`` is ``None`` when the discriminator declined or
            failed, in which case the caller keeps the classical selection. ``scores`` aligns
            with ``detections``; candidates too close to the edge to crop score ``nan`` and are
            never selected.
        """
        patches: List[np.ndarray] = []
        index_map: List[int] = []
        for index, detection in enumerate(detections[:MAX_SCORED_CANDIDATES]):
            patch = extract_patch(frame, detection.x, detection.y)
            if patch is not None:
                patches.append(patch)
                index_map.append(index)
        if not patches:
            return None, None

        batch = np.stack(patches)[:, None, :, :].astype(np.float32)
        start = time.perf_counter()
        try:
            raw = self.session.run(None, {self._input_name: batch})[0]
        except Exception as error:                        # pragma: no cover - backend failure
            LOGGER.warning("discriminator inference failed, keeping classical ranking: %s", error)
            self._disabled = True
            return None, None
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        if self.last_inference_ms > self.max_inference_ms:
            LOGGER.warning("discriminator exceeded its %.1f ms budget (%.1f ms); disabling it "
                           "for the rest of the run and keeping the classical path",
                           self.max_inference_ms, self.last_inference_ms)
            self._disabled = True
            return None, None

        scores = np.full(len(detections), np.nan, dtype=np.float64)
        scores[index_map] = np.asarray(raw, dtype=np.float64).reshape(-1)
        best_index = int(np.nanargmax(scores)) if np.any(np.isfinite(scores)) else None
        if best_index is None or not np.isfinite(scores[best_index]):
            return tuple(scores), None
        if scores[best_index] < self.confidence_threshold:
            # The network is not confident about any candidate. Deferring to flux ranking is the
            # conservative choice: an unconfident override is how a discriminator turns a working
            # clip into a broken one.
            return tuple(scores), None
        return tuple(scores), detections[best_index]


def load_discriminator(ai_config) -> Optional[CandidateDiscriminator]:
    """Build a discriminator from the ``ai`` configuration block.

    Args:
        ai_config: The validated ``ai`` configuration.

    Returns:
        A discriminator, or ``None`` when it is disabled, unavailable or fails to load. Never
        raises: the classical path must survive every AI failure mode.
    """
    if not ai_config.enabled or ai_config.invoke_on == "never":
        return None
    if ai_config.backend != "onnxruntime":
        LOGGER.warning("unsupported AI backend %r; classical path only", ai_config.backend)
        return None
    if not ai_config.model_path:
        LOGGER.warning("ai.enabled is true but ai.model_path is unset; classical path only")
        return None
    try:
        import onnxruntime
    except ImportError:
        LOGGER.warning("onnxruntime is not available; classical path only")
        return None
    try:
        options = onnxruntime.SessionOptions()
        # Single-threaded on purpose. The model is ~3.6k parameters on 32x32 patches, so thread
        # pool setup costs more than the arithmetic, and the run loop already owns the CPU budget.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(str(ai_config.model_path), options,
                                               providers=["CPUExecutionProvider"])
    except Exception as error:
        LOGGER.warning("could not load ONNX model %s: %s; classical path only",
                       ai_config.model_path, error)
        return None
    return CandidateDiscriminator(session, ai_config.invoke_on,
                                  ai_config.confidence_threshold, ai_config.max_inference_ms,
                                  ai_config.flux_ratio_threshold)
