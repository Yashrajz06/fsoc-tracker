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
from typing import List, Optional, Tuple

import numpy as np

from src.config import AppConfig
from src.sim.beacon import BeaconParams, render_beacon
from src.sim.canvas import Canvas
from src.sim.trajectories import Trajectory, build_trajectory

__all__ = ["Scene", "SceneState", "MultiScene"]


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


class MultiScene:
    """Multiple independent beacons sharing one world canvas.

    Each beacon has its own :class:`Scene` (and therefore its own :class:`Trajectory` and
    :class:`~src.sim.beacon.BeaconParams`), but all scenes render onto the single shared
    :class:`Canvas` allocated at construction. The canvas is cleared once per step and all
    beacons are composited in order, so the output frame is physically correct for any number
    of simultaneous targets.

    This is the ``target.count > 1`` variant of :class:`Scene`. The primary sub-scene (index 0)
    drives the shared canvas and its state is the "primary" ground truth; additional sub-scenes
    add their beacons to the same canvas and contribute ``extra_ground_truths``.

    Attributes:
        scenes: Ordered list of :class:`Scene` objects. ``scenes[0]`` is the primary target.
    """

    def __init__(self, scenes: List[Scene]) -> None:
        """Initialise with a list of pre-built scenes that share one canvas.

        The first scene's canvas is the shared canvas. All subsequent scenes must reference
        the same :class:`Canvas` object. Use :meth:`from_config` rather than this constructor
        directly -- it handles the canvas sharing correctly.

        Args:
            scenes: One or more :class:`Scene` objects. Must not be empty. All must share
                the same :class:`Canvas` instance.

        Raises:
            ValueError: If ``scenes`` is empty.
        """
        if not scenes:
            raise ValueError("MultiScene requires at least one Scene")
        self.scenes = scenes

    @property
    def canvas(self) -> Canvas:
        """The shared world canvas."""
        return self.scenes[0].canvas

    @property
    def state(self) -> SceneState:
        """The primary target's most recent state (ground truth for the tracked beacon)."""
        return self.scenes[0].state

    @classmethod
    def from_config(cls, config: AppConfig,
                    rng: Optional[np.random.Generator] = None) -> "MultiScene":
        """Build N scenes from configuration, sharing one canvas.

        Each sub-scene gets a distinct RNG derived from the master seed so initial positions
        and trajectory phases differ between targets. All sub-scenes share the same
        :class:`Canvas` object (allocated once here).

        Args:
            config: Validated application configuration. ``config.target.count`` determines N.
            rng: Master seeded generator. Derived from ``config.run.random_seed`` when omitted.

        Returns:
            A :class:`MultiScene` ready for stepping.
        """
        if rng is None:
            rng = np.random.default_rng(config.run.random_seed)

        n = max(1, config.target.count)
        shared_canvas = Canvas.from_config(config)
        scenes: List[Scene] = []
        import dataclasses as _dc
        import math as _math

        for i in range(n):
            sub_seed = int(rng.integers(0, 2**31))
            sub_rng = np.random.default_rng(sub_seed)

            # Distribute sub-scenes evenly around the trajectory period so they start at
            # visually distinct positions rather than all at phase 0.  Only analytic periodic
            # trajectories need this; stochastic ones (random, OU) already differ by RNG seed,
            # and linear trajectories separate by initial position (also RNG-seeded).
            traj_config = config
            if i > 0:
                motion_type = config.target.motion_type
                phase_step = (2.0 * _math.pi * i) / n
                motion_raw = dict(config.target.motion)

                if motion_type == "circular":
                    params = dict(motion_raw.get("circular", {}))
                    params["phase_offset_rad"] = (
                        float(params.get("phase_offset_rad", 0.0)) + phase_step)
                    motion_raw = dict(motion_raw, circular=params)
                    traj_config = _dc.replace(
                        config, target=_dc.replace(config.target, motion=motion_raw))

                elif motion_type == "figure8":
                    params = dict(motion_raw.get("figure8", {}))
                    params["phase_offset_rad"] = (
                        float(params.get("phase_offset_rad", 0.0)) + phase_step)
                    motion_raw = dict(motion_raw, figure8=params)
                    traj_config = _dc.replace(
                        config, target=_dc.replace(config.target, motion=motion_raw))

                elif motion_type == "sinusoidal":
                    # Offset x0 so beacons are spread along the horizontal axis.
                    params = dict(motion_raw.get("sinusoidal", {}))
                    shift = float(config.scene.width) * i / n
                    params["x0"] = float(params.get("x0", config.scene.width / 2.0)) + shift
                    motion_raw = dict(motion_raw, sinusoidal=params)
                    traj_config = _dc.replace(
                        config, target=_dc.replace(config.target, motion=motion_raw))

                elif motion_type == "spiral":
                    # Stagger angular start so spirals don't overlap at t=0.
                    params = dict(motion_raw.get("spiral", {}))
                    # Not a phase param — use a different growth rate per target.
                    growth = float(params.get("growth_rate_b", 12.0))
                    params["growth_rate_b"] = growth * (1.0 + 0.4 * i)
                    motion_raw = dict(motion_raw, spiral=params)
                    traj_config = _dc.replace(
                        config, target=_dc.replace(config.target, motion=motion_raw))
                # linear, random, OU, custom — variety comes from the sub_rng seed alone.

            scene = Scene(
                canvas=shared_canvas,
                trajectory=build_trajectory(traj_config, sub_rng),
                beacon=BeaconParams.from_config(config),
            )
            scenes.append(scene)
        return cls(scenes)

    def step(self, dt: float, intensity_scale: float = 1.0) -> List[SceneState]:
        """Advance all scenes by ``dt`` seconds and redraw all beacons onto the shared canvas.

        The canvas is cleared exactly once, then every beacon is composited in order. The
        returned list has one :class:`SceneState` per target; index 0 is the primary target.

        Args:
            dt: Time increment in seconds. Must be non-negative.
            intensity_scale: Brightness multiplier for this frame (scintillation).

        Returns:
            A list of :class:`SceneState`, one per target, in the same order as :attr:`scenes`.
        """
        self.canvas.clear()
        states: List[SceneState] = []
        for scene in self.scenes:
            if dt > 0 or scene._initialised:
                x, y = scene.trajectory.step(dt)
            else:
                x, y = scene.trajectory.step(0.0)
            scene._initialised = True

            patch = render_beacon(x, y, scene.beacon, intensity_scale=intensity_scale)
            self.canvas.composite(patch, mode=scene.composite_mode)

            state = SceneState(
                time=scene.trajectory.elapsed,
                x=x,
                y=y,
                intensity_scale=intensity_scale,
                visible=self.canvas.contains(x, y),
            )
            scene._state = state
            states.append(state)
        return states

    def reset(self) -> None:
        """Return all scenes to their initial states."""
        self.canvas.clear()
        for scene in self.scenes:
            scene.trajectory.reset()
            scene._initialised = False
            scene._state = SceneState(time=0.0, x=0.0, y=0.0)
