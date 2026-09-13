"""Scene composition: canvas, beacon and trajectory driven together on one clock.

This is the piece that satisfies the Phase 1 exit criterion -- a clean beacon moving in any of
the mandatory patterns, with exact sub-pixel ground truth available for every frame. It is
deliberately noise-free: sensor noise, atmospheric degradation and turbulence arrive in Phase 2
and compose *on top* of what this produces, so the pristine scene stays available as the
reference against which every degradation is measured.

The scene knows nothing about the camera. It renders the world; extracting a viewport from it is
:mod:`src.camera.viewport`'s job. Keeping that split means ground truth is always expressed in
canvas coordinates and converted to frame coordinates at exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config import AppConfig
from src.sim.beacon import BeaconParams, render_beacon
from src.sim.canvas import Canvas
from src.sim.trajectories import Trajectory, build_trajectory

__all__ = ["Scene", "SceneState"]


@dataclass(frozen=True)
class SceneState:
    """The true state of the scene at one instant.

    Attributes:
        time: Elapsed scene time in seconds.
        x: True beacon centre x in canvas coordinates, sub-pixel exact.
        y: True beacon centre y in canvas coordinates, sub-pixel exact.
        intensity_scale: Multiplicative brightness factor applied this frame.
        visible: Whether the beacon was drawn onto the canvas at all.
    """

    time: float
    x: float
    y: float
    intensity_scale: float = 1.0
    visible: bool = True

    @property
    def position(self) -> Tuple[float, float]:
        """True beacon position as ``(x, y)`` in canvas coordinates."""
        return self.x, self.y


class Scene:
    """A world canvas with a single moving beacon.

    Attributes:
        canvas: The world canvas, allocated once and reused.
        trajectory: The beacon's motion model.
        beacon: Beacon rendering parameters.
    """

    def __init__(self, canvas: Canvas, trajectory: Trajectory, beacon: BeaconParams,
                 composite_mode: str = "add") -> None:
        """Initialise the scene.

        Args:
            canvas: World canvas to render into.
            trajectory: Motion model driving the beacon.
            beacon: Beacon rendering parameters.
            composite_mode: ``"add"`` or ``"max"``, forwarded to :meth:`Canvas.composite`.
        """
        self.canvas = canvas
        self.trajectory = trajectory
        self.beacon = beacon
        self.composite_mode = composite_mode
        self._state = SceneState(time=0.0, x=0.0, y=0.0)
        self._initialised = False

    @classmethod
    def from_config(cls, config: AppConfig,
                    rng: Optional[np.random.Generator] = None) -> "Scene":
        """Build a scene from an application configuration.

        Args:
            config: Validated application configuration.
            rng: Seeded generator for stochastic motion. Derived from
                ``config.run.random_seed`` when omitted, so runs stay reproducible.

        Returns:
            A ready-to-step :class:`Scene`.
        """
        if rng is None:
            rng = np.random.default_rng(config.run.random_seed)
        return cls(
            canvas=Canvas.from_config(config),
            trajectory=build_trajectory(config, rng),
            beacon=BeaconParams.from_config(config),
        )

    @property
    def state(self) -> SceneState:
        """The most recent scene state, i.e. the current ground truth."""
        return self._state

    def step(self, dt: float, intensity_scale: float = 1.0) -> SceneState:
        """Advance the scene by ``dt`` seconds and redraw the canvas.

        The canvas is cleared and redrawn in place; no buffer is reallocated.

        Args:
            dt: Time increment in seconds. Must be non-negative. Pass 0.0 on the first call to
                render the initial position without advancing the trajectory.
            intensity_scale: Brightness multiplier for this frame, used by the Phase 2
                scintillation model.

        Returns:
            The :class:`SceneState` holding exact ground truth for the frame just rendered.
        """
        if dt > 0 or self._initialised:
            x, y = self.trajectory.step(dt)
        else:
            # First frame: render where the trajectory currently is, without advancing it.
            x, y = self.trajectory.step(0.0)
        self._initialised = True

        self.canvas.clear()
        patch = render_beacon(x, y, self.beacon, intensity_scale=intensity_scale)
        self.canvas.composite(patch, mode=self.composite_mode)

        self._state = SceneState(
            time=self.trajectory.elapsed,
            x=x,
            y=y,
            intensity_scale=intensity_scale,
            visible=self.canvas.contains(x, y),
        )
        return self._state

    def reset(self) -> None:
        """Return the scene to its initial state so a scenario can be re-run."""
        self.trajectory.reset()
        self.canvas.clear()
        self._initialised = False
        self._state = SceneState(time=0.0, x=0.0, y=0.0)
