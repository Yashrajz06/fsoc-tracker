"""Acquisition state machine: SEARCH / TRACK / COAST, with hysteresis and split timing.

::

    SEARCH  --(lock criterion held K frames)-->  TRACK
    TRACK   --(N consecutive missed detections)-->  COAST
    COAST   --(re-lock)-->  TRACK
    COAST   --(coast timeout)-->  SEARCH

**Hysteresis** uses different enter and exit windows -- 40 px to lock, 80 px to unlock -- so a
target hovering near the boundary cannot chatter the mode at frame rate. Mode chatter is worse
than either state: it resets timers and restarts the integral.

**Acquisition is timed as two populations, never pooled** (``docs/DESIGN.md`` section 7.5):

* **in-FOV** -- the beacon was already inside the initial viewport. Detection-limited, and the
  case the spec's <=2 s budget is realistic for.
* **search-limited** -- the beacon started outside, so the time is dominated by the spiral sweep
  and is bounded below by the slew ceiling. For the default configuration that bound is 11.6 s,
  which exceeds the budget by arithmetic rather than by any deficiency of the algorithm.

Pooling the two hides a physical limit behind an initial-condition lottery and makes the headline
number depend on where the beacon happened to start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

__all__ = ["TrackState", "AcquisitionClass", "StateMachineParams", "AcquisitionEvent",
           "StateUpdate", "TrackingStateMachine"]


class TrackState(str, Enum):
    """Operating mode."""

    SEARCH = "search"
    TRACK = "track"
    COAST = "coast"


class AcquisitionClass(str, Enum):
    """Which population an acquisition event belongs to."""

    IN_FOV = "in_fov"
    SEARCH_LIMITED = "search_limited"


@dataclass(frozen=True)
class StateMachineParams:
    """State machine configuration.

    Attributes:
        lock_confirm_frames: ``K`` consecutive qualifying frames to declare lock.
        loss_declare_frames: ``N`` consecutive missed detections to declare loss.
        coast_timeout_seconds: Time in COAST before falling back to SEARCH.
        lock_window_px: Enter-lock distance from boresight.
        unlock_window_px: Exit-lock distance. Strictly greater than ``lock_window_px``; the gap
            is the hysteresis.
        require_pointing_window: Whether the pointing-error window forms part of the lock
            criterion.

            **Set this from the frame source's ``supports_pan_tilt``, never from a mode string.**
            The criterion is "include pointing error in the lock gate if the source has a
            steerable camera", which keeps mode out of the metrics logic exactly as it stays out
            of the vision pipeline.

            With a steerable camera, holding the target near boresight is the *objective*, so
            failing to do so is a genuine loss of lock. With pre-recorded video there is no
            camera to steer: the target's distance from frame centre is a property of the file,
            not of our tracking, and gating on it would report a perfectly tracked beacon as
            never locked. That is not hypothetical -- on a Mode B clip whose target sat a median
            202 px from frame centre, lock retention read 0% and **the summary therefore omitted
            centroiding error entirely**, which is the one quantity Benchmark-2 scores.
    """

    lock_confirm_frames: int = 3
    loss_declare_frames: int = 5
    coast_timeout_seconds: float = 1.0
    lock_window_px: float = 40.0
    unlock_window_px: float = 80.0
    require_pointing_window: bool = True

    def validate(self) -> None:
        """Check the configuration.

        Raises:
            ValueError: If the hysteresis windows are not strictly ordered.
        """
        if not 0 < self.lock_window_px < self.unlock_window_px:
            raise ValueError(
                f"Hysteresis requires 0 < lock_window_px < unlock_window_px, got "
                f"{self.lock_window_px} and {self.unlock_window_px}")


@dataclass(frozen=True)
class AcquisitionEvent:
    """A completed acquisition, tagged with its population.

    Attributes:
        duration_s: Time from clock start to lock.
        population: Whether the beacon started inside the viewport.
        reacquisition: Whether this followed a loss rather than being the initial acquisition.
    """

    duration_s: float
    population: AcquisitionClass
    reacquisition: bool = False


@dataclass(frozen=True)
class StateUpdate:
    """State machine output for one frame.

    Attributes:
        state: Mode after this frame.
        locked: Whether the lock criterion currently holds.
        acquired: An acquisition completed on this frame, if any.
        consecutive_hits: Consecutive qualifying frames.
        consecutive_misses: Consecutive missed detections.
        pointing_error_px: Distance from estimate to boresight this frame.
        reason: Machine-readable code for the trace.
    """

    state: TrackState
    locked: bool = False
    acquired: Optional[AcquisitionEvent] = None
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    pointing_error_px: float = math.inf
    reason: str = ""


class TrackingStateMachine:
    """SEARCH / TRACK / COAST with hysteresis and split acquisition timing.

    Attributes:
        params: Configuration.
    """

    def __init__(self, params: Optional[StateMachineParams] = None,
                 target_initially_in_fov: bool = False) -> None:
        """Initialise in SEARCH.

        Args:
            params: Configuration.
            target_initially_in_fov: Whether the beacon lay inside the initial viewport. This is
                what classifies the first acquisition, and it must be recorded at frame 0 rather
                than inferred later.
        """
        self.params = params or StateMachineParams()
        self.params.validate()
        self.target_initially_in_fov = bool(target_initially_in_fov)

        self._state = TrackState.SEARCH
        self._hits = 0
        self._misses = 0
        self._coast_elapsed = 0.0
        self._clock_start: Optional[float] = 0.0
        self._has_acquired = False
        self._events: List[AcquisitionEvent] = []

    @property
    def state(self) -> TrackState:
        """Current mode."""
        return self._state

    @property
    def events(self) -> Tuple[AcquisitionEvent, ...]:
        """All acquisition events recorded so far."""
        return tuple(self._events)

    def acquisitions(self, population: AcquisitionClass) -> Tuple[AcquisitionEvent, ...]:
        """Return acquisition events belonging to one population.

        Args:
            population: Which population to filter for.

        Returns:
            The matching events. Reported separately, never pooled.
        """
        return tuple(e for e in self._events if e.population is population)

    def update(self, detected: bool, pointing_error_px: float, now_s: float,
               dt: float) -> StateUpdate:
        """Advance the state machine by one frame.

        Args:
            detected: Whether a validated detection was available this frame.
            pointing_error_px: Distance from the estimate to the boresight.
            now_s: Absolute run time in seconds.
            dt: Interval since the previous frame.

        Returns:
            A :class:`StateUpdate`.
        """
        params = self.params
        # Hysteresis: the window depends on whether we are already locked.
        window = (params.unlock_window_px if self._state is TrackState.TRACK
                  else params.lock_window_px)
        # With no steerable camera the pointing window is inapplicable: the lock criterion
        # reduces to a validated detection that passed the Kalman gate.
        qualifies = detected and (pointing_error_px <= window
                                  if params.require_pointing_window else True)

        if detected:
            self._misses = 0
        else:
            self._misses += 1

        self._hits = self._hits + 1 if qualifies else 0

        acquired: Optional[AcquisitionEvent] = None
        reason = ""

        if self._state is TrackState.SEARCH:
            self._coast_elapsed = 0.0
            if self._hits >= params.lock_confirm_frames:
                self._state = TrackState.TRACK
                acquired = self._record_acquisition(now_s)
                reason = "acquired"
            else:
                reason = "searching"

        elif self._state is TrackState.TRACK:
            if self._misses >= params.loss_declare_frames:
                self._state = TrackState.COAST
                self._coast_elapsed = 0.0
                self._clock_start = now_s
                reason = "lost"
            elif (params.require_pointing_window and detected
                  and pointing_error_px > params.unlock_window_px):
                self._state = TrackState.COAST
                self._coast_elapsed = 0.0
                self._clock_start = now_s
                reason = "unlocked"
            else:
                reason = "tracking"

        else:  # COAST
            self._coast_elapsed += dt
            if self._hits >= params.lock_confirm_frames:
                self._state = TrackState.TRACK
                acquired = self._record_acquisition(now_s, reacquisition=True)
                reason = "reacquired"
            elif self._coast_elapsed >= params.coast_timeout_seconds:
                self._state = TrackState.SEARCH
                reason = "coast_timeout"
            else:
                reason = "coasting"

        return StateUpdate(state=self._state, locked=self._state is TrackState.TRACK,
                           acquired=acquired, consecutive_hits=self._hits,
                           consecutive_misses=self._misses,
                           pointing_error_px=pointing_error_px, reason=reason)

    def _record_acquisition(self, now_s: float,
                            reacquisition: bool = False) -> AcquisitionEvent:
        """Close the acquisition clock and classify the event.

        Args:
            now_s: Time of lock.
            reacquisition: Whether this followed a loss.

        Returns:
            The recorded :class:`AcquisitionEvent`.
        """
        start = self._clock_start if self._clock_start is not None else now_s
        # A re-acquisition is always local -- the target's last position is known -- so it is
        # never charged to the search-limited population.
        population = (AcquisitionClass.IN_FOV
                      if (reacquisition or self.target_initially_in_fov)
                      else AcquisitionClass.SEARCH_LIMITED)
        event = AcquisitionEvent(duration_s=max(0.0, now_s - start), population=population,
                                 reacquisition=reacquisition)
        self._events.append(event)
        self._has_acquired = True
        self._clock_start = None
        return event

    def reset(self) -> None:
        """Return to SEARCH and clear all timers and events."""
        self._state = TrackState.SEARCH
        self._hits = 0
        self._misses = 0
        self._coast_elapsed = 0.0
        self._clock_start = 0.0
        self._has_acquired = False
        self._events.clear()
