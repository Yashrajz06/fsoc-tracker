"""Mode A frame source: the closed-loop simulation behind the :class:`FrameSource` protocol.

Assembles the pieces built in earlier phases -- scene, camera, viewport, noise pipeline -- into
something the run loop can iterate without knowing it is a simulation. The vision, filtering,
control and telemetry modules receive a :class:`~src.framesource.FrameSource` and never learn
which concrete class they were given, which is the structural guarantee that keeps Mode A and
Mode B running identical vision code.

Ground truth is reported in **frame-local** coordinates, converted at this single boundary, so no
consumer has to know the canvas coordinate system. Beam wander is applied to the *true* position
before rendering, so ground truth reflects where the beacon actually appeared rather than where
it nominally was -- wander is a real displacement of the received beam, not an imaging artefact.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from src.camera.model import CameraModel
from src.camera.viewport import Viewport
from src.config import AppConfig
from src.framesource import BaseFrameSource, FrameData, GroundTruth
from src.noise.pipeline import NoisePipeline
from src.sim.scene import MultiScene, Scene

__all__ = ["SimulationFrameSource"]


class SimulationFrameSource(BaseFrameSource):
    """Renders the virtual scene and returns degraded camera frames with exact ground truth.

    Attributes:
        config: Validated application configuration.
        camera: The steerable camera.
        viewport: Viewport extraction, including camera jitter.
        noise: The degradation pipeline.
    """

    def __init__(self, config: AppConfig, rng: Optional[np.random.Generator] = None) -> None:
        """Build the simulation.

        Args:
            config: Validated application configuration.
            rng: Seeded generator shared by every stochastic component, so a whole run is
                reproducible from ``config.run.random_seed``.
        """
        self.config = config
        self._rng = rng if rng is not None else np.random.default_rng(config.run.random_seed)

        if config.target.count > 1:
            self._scene_impl: Union[Scene, MultiScene] = MultiScene.from_config(config, self._rng)
        else:
            self._scene_impl = Scene.from_config(config, self._rng)

        self.camera = CameraModel.from_config(config)
        jitter = config.noise.camera_jitter
        self.viewport = Viewport(
            self.camera,
            jitter_px=float(jitter.get("max_px_per_frame", 0.0)) if jitter.get("enabled") else 0.0,
            rng=self._rng,
            jitter_distribution=str(jitter.get("distribution", "gaussian")))
        self.noise = NoisePipeline.from_config(config, self._rng)

        self._index = 0
        self._rate = config.camera.update_rate_hz
        self._total = max(1, int(config.run.duration_seconds * self._rate))

    @property
    def scene(self) -> Scene:
        """The primary scene (index 0 for multi-target, the only scene for single-target)."""
        if isinstance(self._scene_impl, MultiScene):
            return self._scene_impl.scenes[0]
        return self._scene_impl

    # -- FrameSource protocol ----------------------------------------------------------------

    @property
    def supports_pan_tilt(self) -> bool:
        """Simulation drives a real pan/tilt loop."""
        return True

    @property
    def nominal_rate_hz(self) -> float:
        """Configured frame generation rate."""
        return self._rate

    @property
    def deg_per_pixel(self) -> Optional[Tuple[float, float]]:
        """Angular resolution as ``(horizontal, vertical)`` degrees per pixel."""
        return self.camera.deg_per_pixel

    @property
    def target_initially_in_fov(self) -> bool:
        """Whether the beacon lay inside the initial viewport.

        Recorded so acquisition can be classified as in-FOV or search-limited. It must be
        captured at the start rather than inferred later, which is why it is a property of the
        source rather than something the state machine works out.
        """
        return self.camera.is_visible(self.scene.state.x, self.scene.state.y)

    def get_frame(self) -> Optional[FrameData]:
        """Render, extract and degrade the next frame.

        Returns:
            A :class:`FrameData` with frame-local ground truth, or ``None`` when the configured
            duration has elapsed.
        """
        if self._index >= self._total:
            return None

        dt = 0.0 if self._index == 0 else 1.0 / self._rate

        # Scintillation modulates emitted intensity, and beam wander displaces the apparent
        # position -- both applied before rendering so ground truth includes them.
        intensity_scale = self.noise.next_intensity_scale()
        wander_x, wander_y = self.noise.next_beam_wander(dt)

        if isinstance(self._scene_impl, MultiScene):
            states = self._scene_impl.step(dt, intensity_scale=intensity_scale)
            state = states[0]
            true_x = state.x + wander_x
            true_y = state.y + wander_y
            extra_gts = []
        else:
            states = None
            state = self._scene_impl.step(dt, intensity_scale=intensity_scale)
            true_x = state.x + wander_x
            true_y = state.y + wander_y
            extra_gts = []

        if wander_x or wander_y:
            # Re-render at the wandered position so the image and the truth agree.
            self.scene.canvas.clear()
            from src.sim.beacon import render_beacon
            self.scene.canvas.composite(
                render_beacon(true_x, true_y, self.scene.beacon,
                              intensity_scale=intensity_scale))

        view = self.viewport.extract(self.scene.canvas)
        degraded = self.noise.apply(view.frame, intensity_scale)

        local_x, local_y = view.canvas_to_frame(true_x, true_y)

        # Re-compute extra ground truths using the final view (after wander re-render)
        if isinstance(self._scene_impl, MultiScene):
            extra_gts = []
            for extra_state in states[1:]:
                ex, ey = view.canvas_to_frame(extra_state.x, extra_state.y)
                extra_gts.append(GroundTruth(
                    x=ex, y=ey,
                    visible=view.contains(extra_state.x, extra_state.y),
                    world_x=extra_state.x,
                    world_y=extra_state.y,
                ))

        data = FrameData(
            frame=degraded.frame,
            timestamp=self._index / self._rate,
            frame_index=self._index,
            ground_truth=GroundTruth(x=local_x, y=local_y,
                                     visible=view.contains(true_x, true_y),
                                     world_x=true_x, world_y=true_y),
            camera_pan_deg=self.camera.pan_deg,
            camera_tilt_deg=self.camera.tilt_deg,
            extra_ground_truths=extra_gts if extra_gts else None,
        )
        self._index += 1
        return data

    def apply_pan_tilt(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> None:
        """Slew the camera, with limits enforced inside the camera model.

        Args:
            pan_rate_deg_s: Requested pan rate.
            tilt_rate_deg_s: Requested tilt rate.
            dt: Interval over which the rates apply.
        """
        self.camera.apply_rates(pan_rate_deg_s, tilt_rate_deg_s, dt)

    def reset(self) -> None:
        """Return the simulation to its initial state for a reproducible re-run."""
        self._scene_impl.reset()
        self.camera.reset()
        self.noise.reset()
        self._index = 0
