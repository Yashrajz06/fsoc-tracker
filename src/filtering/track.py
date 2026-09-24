"""Track lifecycle: initiation, confirmation, coasting and deletion.

The Kalman filter in :mod:`src.filtering.kalman` estimates state. This module decides *when to
believe it* -- when a track begins, when it is trusted, when it is coasting, and when it is gone.

Why this module exists at all: the gate can lock itself out
-----------------------------------------------------------
Measured on a dim moving target, the filter lost the track on 3 runs in 12, and the mechanism was
not what it looked like. The measurements were **good** throughout -- 0.1 to 1.0 px error -- and
every one of them was rejected from the fourth frame onward:

* The filter initialised from a single detection, so its velocity started at **zero**.
* The target moved ~100 px/s, i.e. 3.35 px per frame, so the prediction fell behind immediately.
* The position covariance had already collapsed after two or three accepted updates, so the gate
  was tight while the *velocity* estimate was still wrong.
* A 3.35 px innovation against a ~0.2 px gate is rejected -- and a rejected measurement produces
  no update, so the filter never learns the velocity, so the lag grows without bound.

Track error grew linearly to 27 px while the detector was working perfectly. The failure is
silent: nothing looks wrong except the output.

Three mechanisms here prevent it, and each addresses a different part of that chain:

1. **Two-point initiation** -- velocity comes from the first two detections rather than being
   assumed zero, so the prediction is never systematically behind to begin with.
2. **Ungated initiation** -- during confirmation the gate is not applied, because gating against
   a state that has not yet been established is what causes the lockout.
3. **Lockout detection** -- consecutive rejections are counted, and a track that rejects
   everything is declared unhealthy and re-initiated rather than left to diverge silently. A gate
   that rejects every measurement is evidence about the *track*, not about the measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.filtering.kalman import GateResult, KalmanFilterCV, KalmanParams

__all__ = ["TrackStatus", "TrackParams", "TrackUpdate", "Track", "ContestState"]


class TrackStatus(str, Enum):
    """Lifecycle state of a track."""

    #: No track; awaiting detections to initiate one.
    EMPTY = "empty"
    #: Detections seen but M-of-N confirmation not yet satisfied.
    INITIATING = "initiating"
    #: Confirmed and being updated from gated measurements.
    CONFIRMED = "confirmed"
    #: Confirmed but currently missing detections; running on prediction alone.
    COASTING = "coasting"


@dataclass
class ContestState:
    """A competing-hypothesis contest between the incumbent track and a challenger.

    Opened when the incumbent has rejected several consecutive detections. Both hypotheses are
    then scored on **the same detections** over a fixed window, and the incumbent is replaced only
    if the challenger explains them materially better.

    Attributes:
        opened_at_frame: Timestamp at which the contest opened.
        frames_evaluated: Detections scored so far.
        incumbent_nis: Accumulated normalised innovation squared for the incumbent.
        challenger_nis: Accumulated NIS for the challenger.
        challenger_detections: Detections fed to the challenger, for its two-point initiation.
    """

    opened_at_s: float
    frames_evaluated: int = 0
    incumbent_nis: float = 0.0
    challenger_nis: float = 0.0
    challenger_detections: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def incumbent_mean_nis(self) -> float:
        """Mean NIS for the incumbent over the contest window."""
        return self.incumbent_nis / max(self.frames_evaluated, 1)

    @property
    def challenger_mean_nis(self) -> float:
        """Mean NIS for the challenger over the contest window."""
        return self.challenger_nis / max(self.frames_evaluated, 1)


@dataclass(frozen=True)
class TrackParams:
    """Track-management configuration.

    Attributes:
        confirm_m_of_n: ``(M, N)`` detections required to confirm a track.
        delete_after_missed: Consecutive missed detections before a track is deleted. This is the
            ``N`` in the re-acquisition metric definition.
        max_consecutive_rejections: Consecutive gate rejections after which a **challenger
            hypothesis is opened**. It is no longer a signal to discard the incumbent outright:
            see :attr:`contest_window_frames`.
        contest_window_frames: Frames over which incumbent and challenger are scored on the same
            detections before a winner is declared.

            **Duration is the discriminator, and it has to be.** Two situations produce identical
            symptoms over a few frames -- an incumbent with a genuinely wrong state (the Phase 4
            zero-velocity lockout) and a correct incumbent whose *detector* has latched onto a
            spurious object. In both, the incumbent rejects everything and a challenger fitted to
            the rejected detections scores well, because a spurious object that persists for a
            few frames is just as self-consistent as a real one. What separates them is
            persistence: a detector excursion ends and the true detections return, at which point
            the incumbent starts explaining them again, while a genuinely wrong incumbent never
            recovers. A window longer than a typical excursion therefore resolves the two
            oppositely. The cost is slower recovery from a real lockout, which is the right trade:
            discarding a correct track is worse than taking longer to replace a wrong one.
        contest_nis_margin: Factor by which the challenger's mean NIS must beat the incumbent's to
            win. A margin above 1 means ties go to the incumbent, which is the conservative
            direction -- an established track is evidence in itself.
        incumbent_max_mean_nis: Upper bound on the incumbent's mean NIS over the contest window.
            An incumbent that exceeds this threshold **never wins**, regardless of the challenger's
            score. A consistent 2-DOF filter has a mean NIS of about 2; any value in the
            hundreds means the incumbent's state is explaining nothing.

            This closes the case found in Phase A where an incumbent with NIS 1147 held because
            the challenger had not yet accumulated enough evidence, and the challenger could not
            win a contest the incumbent should never survive. The bound is intentionally generous
            (default 100) -- far above the chi-squared 99% consistency boundary, but far below
            the thousands seen in a phantom-velocity track.
        gate_during_initiation: Apply the validation gate while initiating. Off by default:
            gating against a state that is not yet established is precisely the lockout mechanism.
        max_association_px: Sanity bound on how far a detection may be from the prediction during
            initiation, when the gate is not in use.
        max_initiation_velocity_px_s: Upper bound on the speed implied by a two-point initiation
            pair. A pair that implies a speed above this means the **first** detection was spurious
            -- one bad measurement is far more likely than a genuine target moving at an absurd
            fraction above the slew ceiling. The response is to **discard the oldest buffered
            detection and wait for a fresh pair** rather than confirming a phantom track.

            The root cause (Phase A §P3): the first detection on a large-spot clip was 499.9 px
            from the true position, and the second was correct. Two-point initiation derived
            7550 px/s from that pair, far above the measured 279 px/s trackable ceiling and the
            800 px/s slew ceiling. The track then flew off in a straight line at 251.7 px/frame
            and gated out every subsequent correct detection, producing the observed 39.8% / 13.6%
            association failure on 15 px / 20 px spots.

            The default is **5 × the slew ceiling = 4000 px/s**. Rationale:

            * A real target passing through the FOV at above the slew ceiling is physically
              possible even if untrackable -- the initiation should complete so the filter can
              at least coast.
            * Initiation measurement noise on two closely-spaced frames can magnify a moderate
              actual speed into a number somewhat above the slew ceiling (800 px/s) without
              any detection being spurious. A margin of 5× absorbs that.
            * 7550 px/s >> 4000 px/s, so the phantom case is still caught decisively.
            * Setting the bound at exactly the slew ceiling (800 px/s) was tried and broke the
              existing re-acquisition test at 900 px/s, which is a legitimate above-ceiling speed
              that the tracker should still survive.
    """

    confirm_m_of_n: Tuple[int, int] = (3, 5)
    delete_after_missed: int = 5
    max_consecutive_rejections: int = 5
    contest_window_frames: int = 12
    contest_nis_margin: float = 4.0
    incumbent_max_mean_nis: float = 100.0
    gate_during_initiation: bool = False
    max_association_px: float = 100.0
    max_initiation_velocity_px_s: float = 4000.0


@dataclass(frozen=True)
class TrackUpdate:
    """Outcome of feeding one frame to a track.

    Attributes:
        status: Lifecycle state after the update.
        position: Best position estimate as ``(x, y)``, or ``None`` when there is no track.
        velocity: Velocity estimate as ``(vx, vy)`` in pixels per second.
        accepted: Whether a measurement was folded in this frame.
        gated_out: Whether a measurement was rejected by the validation gate.
        coasting: Whether the estimate came from prediction alone.
        nis: Normalised innovation squared, when a measurement was tested.
        gain: Scalar Kalman gain for position, ``P / (P + R)``, when a measurement was folded in.
            **A measurement accepted with near-zero gain contributed nothing**, so reporting the
            frame as "measured" would be technically true and materially misleading -- at least a
            coasting frame is honest about running on the model. Callers use this to classify the
            estimate's real provenance.
        sigma_meas_px: Measurement sigma used, when a measurement was tested.
        consecutive_missed: Consecutive frames without a usable detection.
        consecutive_rejected: Consecutive frames whose detection failed the gate.
        contest_open: Whether a challenger hypothesis is currently being evaluated.
        contest_frames: Detections scored in the current or just-concluded contest.
        contest_incumbent_nis: Incumbent's mean NIS over the contest window.
        contest_challenger_nis: Challenger's mean NIS over the same frames.
        reason: Machine-readable code for the per-frame trace. Contest transitions appear as
            ``contest_opened``, ``contest_incumbent_held`` and ``contest_challenger_won`` so a
            track replacement is never invisible in the log.
    """

    status: TrackStatus
    position: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    accepted: bool = False
    gated_out: bool = False
    coasting: bool = False
    nis: Optional[float] = None
    sigma_meas_px: Optional[float] = None
    gain: Optional[float] = None
    consecutive_missed: int = 0
    consecutive_rejected: int = 0
    contest_open: bool = False
    contest_frames: int = 0
    contest_incumbent_nis: Optional[float] = None
    contest_challenger_nis: Optional[float] = None
    reason: str = ""

    @property
    def has_estimate(self) -> bool:
        """Whether a usable position estimate is available this frame."""
        return self.position is not None


class Track:
    """A single target track with M-of-N initiation, coasting and lockout recovery.

    Attributes:
        params: Track-management configuration.
        filter: The underlying Kalman filter.
    """

    def __init__(self, params: Optional[TrackParams] = None,
                 kalman_params: Optional[KalmanParams] = None) -> None:
        """Create an empty track.

        Args:
            params: Track-management configuration.
            kalman_params: Filter configuration.
        """
        self.params = params or TrackParams()
        self.filter = KalmanFilterCV(kalman_params)
        self._status = TrackStatus.EMPTY
        self._pending: List[Tuple[float, float, float]] = []
        self._missed = 0
        self._rejected = 0
        self._challenger: Optional[KalmanFilterCV] = None
        self._contest: Optional[ContestState] = None
        self._last_contest: Optional[ContestState] = None

    @property
    def status(self) -> TrackStatus:
        """Current lifecycle state."""
        return self._status

    @property
    def is_established(self) -> bool:
        """Whether the track is confirmed or coasting, i.e. has a trustworthy estimate."""
        return self._status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)

    def _initiate(self, x: float, y: float, t: float) -> None:
        """Add a detection to the initiation buffer and confirm when M-of-N is met.

        Velocity is estimated from the first and last buffered detections rather than assumed
        zero. That single change removes the systematic prediction lag that otherwise causes the
        gate to lock out every subsequent measurement.

        **Velocity sanity check (Phase A §P3 fix).** If the implied speed exceeds
        :attr:`TrackParams.max_initiation_velocity_px_s`, the *oldest* buffered detection is
        discarded and initiation waits for a fresh pair. One bad measurement is far more likely
        than a genuine target moving at 10x the trackable envelope. Without this check, a single
        spurious detection at frame 0 produced a 7550 px/s phantom track that gated out every
        subsequent correct detection for the entire clip.

        Args:
            x: Detection x.
            y: Detection y.
            t: Timestamp in seconds.
        """
        self._pending.append((x, y, t))
        m, n = self.params.confirm_m_of_n
        if len(self._pending) > n:
            self._pending.pop(0)

        if len(self._pending) < m:
            self._status = TrackStatus.INITIATING
            return

        first_x, first_y, first_t = self._pending[0]
        last_x, last_y, last_t = self._pending[-1]
        span = last_t - first_t
        if span > 0:
            vx = (last_x - first_x) / span
            vy = (last_y - first_y) / span
        else:
            vx = vy = 0.0

        speed = math.hypot(vx, vy)
        if speed > self.params.max_initiation_velocity_px_s:
            # The oldest detection is the likely culprit: the most recent measurement caused the
            # speed to look implausible, so the candidate pair that spans from a bad old point to
            # a good new one is the problem. Discard the oldest and stay in INITIATING so a fresh
            # pair that was never poisoned by the bad first detection can complete initiation.
            self._pending.pop(0)
            self._status = TrackStatus.INITIATING
            return

        self.filter.initialise(last_x, last_y, vx, vy)
        self._status = TrackStatus.CONFIRMED
        self._pending.clear()
        self._rejected = 0
        self._missed = 0

    def update(self, detection: Optional[Tuple[float, float]], dt: float, t: float,
               snr_aperture: Optional[float] = None, fwhm_px: float = 5.887,
               clipped: bool = False, saturated: bool = False) -> TrackUpdate:
        """Feed one frame to the track.

        Args:
            detection: Measured position as ``(x, y)``, or ``None`` when nothing was detected.
            dt: Time since the previous frame, in seconds.
            t: Absolute timestamp in seconds.
            snr_aperture: Measured aperture SNR, driving the adaptive measurement noise.
            fwhm_px: Spot scale.
            clipped: Detection touches the frame edge.
            saturated: Detection core is saturated.

        Returns:
            A :class:`TrackUpdate` carrying the estimate and the per-frame trace.
        """
        if not self.is_established:
            if detection is None:
                self._missed += 1
                if self._missed > self.params.delete_after_missed:
                    self._pending.clear()
                    self._status = TrackStatus.EMPTY
                return TrackUpdate(status=self._status, consecutive_missed=self._missed,
                                   reason="awaiting_detection")
            self._missed = 0
            self._initiate(detection[0], detection[1], t)
            if self._status is TrackStatus.CONFIRMED:
                return TrackUpdate(status=self._status, position=self.filter.position,
                                   velocity=self.filter.velocity, accepted=True,
                                   reason="confirmed")
            return TrackUpdate(status=self._status, reason="initiating")

        self.filter.predict(dt)

        if detection is None:
            self._missed += 1
            if self._missed > self.params.delete_after_missed:
                self.reset()
                return TrackUpdate(status=self._status, consecutive_missed=self._missed,
                                   reason="deleted")
            self._status = TrackStatus.COASTING
            return TrackUpdate(status=self._status, position=self.filter.position,
                               velocity=self.filter.velocity, coasting=True,
                               consecutive_missed=self._missed, reason="coasting")

        self._missed = 0
        sigma = self.filter.sigma_for(snr_aperture, fwhm_px, clipped, saturated)
        gate = self.filter.gate(detection, sigma)

        if gate.accepted:
            resolved = None
            if self._contest is not None:
                # The incumbent is explaining detections again, so the contest is already
                # decided. Without this the contest only advanced on *rejected* frames and would
                # stay open indefinitely once the detector excursion ended -- the exact situation
                # it exists to survive.
                resolved = self._close_contest(winner_is_challenger=False)

            # Scalar position gain before the update: how much this measurement will actually
            # move the state.
            position_variance = float(self.filter.covariance[0, 0])
            effective = gate.sigma_meas_px
            gain = position_variance / (position_variance + effective ** 2)
            self.filter.update(detection, sigma)
            self._rejected = 0
            self._status = TrackStatus.CONFIRMED
            return TrackUpdate(status=self._status, position=self.filter.position,
                               velocity=self.filter.velocity, accepted=True,
                               nis=gate.nis, sigma_meas_px=effective, gain=gain,
                               reason=resolved or "updated")

        self._rejected += 1
        reason = "gated_out"

        if self._contest is None and self._rejected >= self.params.max_consecutive_rejections:
            self._contest = ContestState(opened_at_s=t)
            reason = "contest_opened"

        # The incumbent keeps running throughout. It is never suspended while the challenger is
        # evaluated -- in the failure that motivated this, the incumbent was the only correct
        # thing in the system during the contest, and its output is what the controller consumed.
        if self._contest is not None:
            resolution = self._score_contest(detection, sigma, gate, t, dt)
            if resolution is not None:
                reason = resolution

        self._status = TrackStatus.COASTING
        return TrackUpdate(status=self._status, position=self.filter.position,
                           velocity=self.filter.velocity, gated_out=True, coasting=True,
                           nis=gate.nis, sigma_meas_px=sigma,
                           consecutive_rejected=self._rejected,
                           contest_open=self._contest is not None,
                           contest_frames=self._reported_contest().frames_evaluated
                           if self._reported_contest() else 0,
                           contest_incumbent_nis=(self._reported_contest().incumbent_mean_nis
                                                  if self._reported_contest() else None),
                           contest_challenger_nis=(self._reported_contest().challenger_mean_nis
                                                   if self._reported_contest() else None),
                           reason=reason)

    def _reported_contest(self) -> Optional[ContestState]:
        """Return the contest to report on this frame: the live one, or the one just closed."""
        return self._contest if self._contest is not None else self._last_contest

    def _score_contest(self, detection: Tuple[float, float], sigma: float,
                       gate: GateResult, t: float, dt: float) -> Optional[str]:
        """Score one detection against both hypotheses and resolve the contest when due.

        Both hypotheses are scored on the *same* detections, so the comparison is like-for-like.
        The score is accumulated normalised innovation squared: the hypothesis that better
        explains the observed detections accumulates less.

        Args:
            detection: The measured position the incumbent rejected.
            sigma: Measurement sigma for this frame.
            gate: The incumbent's gate result, whose NIS is reused rather than recomputed.
            t: Timestamp in seconds.
            dt: Frame interval. The challenger must advance by the **same** step as the
                incumbent, or the comparison is not like-for-like. Deriving it from elapsed
                contest time divided by frames scored gave half the true step on the first
                comparison, so the challenger lagged, accumulated innovation it had not earned,
                and lost contests it should have won.

        Returns:
            A reason code when the contest resolves, otherwise ``None``.
        """
        contest = self._contest
        assert contest is not None
        self._last_contest = None
        contest.incumbent_nis += gate.nis
        contest.frames_evaluated += 1

        if self._challenger is None:
            # Build the challenger by two-point initiation from the rejected detections, exactly
            # as a fresh track starts -- never from the incumbent's state, which is the thing
            # under challenge.
            contest.challenger_detections.append((detection[0], detection[1], t))
            if len(contest.challenger_detections) >= 2:
                first_x, first_y, first_t = contest.challenger_detections[0]
                last_x, last_y, last_t = contest.challenger_detections[-1]
                span = last_t - first_t
                vx = (last_x - first_x) / span if span > 0 else 0.0
                vy = (last_y - first_y) / span if span > 0 else 0.0
                self._challenger = KalmanFilterCV(self.filter.params)
                self._challenger.initialise(last_x, last_y, vx, vy)
        else:
            self._challenger.predict(dt)
            challenger_gate = self._challenger.gate(detection, sigma)
            contest.challenger_nis += challenger_gate.nis
            if challenger_gate.accepted:
                self._challenger.update(detection, sigma)

        if contest.frames_evaluated < self.params.contest_window_frames:
            return None

        # Resolve. Ties go to the incumbent: an established track is evidence in itself.
        # Exception: an incumbent whose mean NIS exceeds the hard ceiling is explaining nothing
        # and must not hold regardless of the challenger's score (Phase A §P5 fix 2). A
        # consistent 2-DOF filter has mean NIS ~2; any value in the hundreds or thousands means
        # the state is physically impossible and the incumbent should never survive the contest.
        incumbent_failed_nis_gate = (
            contest.incumbent_mean_nis > self.params.incumbent_max_mean_nis)
        challenger_wins = (
            self._challenger is not None
            and (incumbent_failed_nis_gate
                 or contest.challenger_mean_nis * self.params.contest_nis_margin
                 < contest.incumbent_mean_nis))
        return self._close_contest(winner_is_challenger=challenger_wins)

    def _close_contest(self, winner_is_challenger: bool) -> str:
        """Resolve the contest and record its final statistics for the trace.

        The final scores are captured *before* the contest state is cleared, so the frame that
        reports the outcome also carries the evidence behind it. Clearing first would leave the
        decisive frame showing zeroes, which is precisely the kind of invisible state change the
        trace exists to prevent.

        Args:
            winner_is_challenger: Whether the challenger won.

        Returns:
            The reason code for the resolving frame.
        """
        contest = self._contest
        assert contest is not None
        self._last_contest = contest

        if winner_is_challenger and self._challenger is not None:
            self.filter = self._challenger
            self._status = TrackStatus.CONFIRMED
            outcome = "contest_challenger_won"
        else:
            outcome = "contest_incumbent_held"

        self._rejected = 0
        self._challenger = None
        self._contest = None
        return outcome

    def reset(self) -> None:
        """Discard the track entirely."""
        self.filter.reset()
        self._status = TrackStatus.EMPTY
        self._pending.clear()
        self._missed = 0
        self._rejected = 0
        self._challenger = None
        self._contest = None
