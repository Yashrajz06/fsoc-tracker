"""Archimedean spiral acquisition search.

When the beacon is not in the initial viewport, the camera must sweep the uncertainty region.
The spiral ``r = b*theta`` is the standard pattern: it starts at the most likely position (the
last known or nominal location) and expands outward, so the expected time to detection is
minimised when the prior is centre-weighted.

**Arm spacing is FOV-derived, never a stored constant.** The spacing uses the *limiting* -- that
is, smaller -- FOV dimension, so the sweep is gap-free on both axes, with a coverage fraction
supplying overlap margin. A hardcoded spacing (the config previously carried 300 px) either
wastes scan time or silently opens uncovered strips the moment the FOV or resolution changes.

The time this takes is bounded below by the slew ceiling and is the origin of the
search-limited acquisition population (``docs/DESIGN.md`` section 7.5): covering 2000x2000 at
432 px spacing needs ~9.3e3 px of path, which at 800 px/s is **11.6 s** against a 2 s budget.
That is arithmetic, not a tuning failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["SearchParams", "SpiralSearch"]


@dataclass(frozen=True)
class SearchParams:
    """Spiral search configuration.

    Attributes:
        arm_spacing_fov_fraction: Arm spacing as a fraction of the limiting FOV dimension.
        scan_speed_deg_s: Angular scan rate, clamped by the caller to the slew ceiling.
        max_radius_px: Radius at which the spiral restarts, normally the uncertainty region.
    """

    arm_spacing_fov_fraction: float = 0.9
    scan_speed_deg_s: float = 5.0
    max_radius_px: float = 1500.0


class SpiralSearch:
    """Generates boresight waypoints along an Archimedean spiral.

    Attributes:
        params: Search configuration.
    """

    def __init__(self, center_xy: Tuple[float, float], fov_px: Tuple[int, int],
                 scan_speed_px_s: float, params: Optional[SearchParams] = None) -> None:
        """Initialise a spiral about a centre.

        Args:
            center_xy: Spiral centre as ``(x, y)`` in canvas coordinates, normally the last known
                target position or the canvas centre.
            fov_px: Viewport size as ``(width, height)``, setting the arm spacing.
            scan_speed_px_s: Scan speed in pixels per second along the path.
            params: Search configuration.

        Raises:
            ValueError: If the FOV or scan speed is not positive.
        """
        params = params or SearchParams()
        width, height = fov_px
        if width <= 0 or height <= 0:
            raise ValueError(f"FOV must be positive, got {fov_px}")
        if scan_speed_px_s <= 0:
            raise ValueError(f"Scan speed must be positive, got {scan_speed_px_s}")

        self.params = params
        self.center = (float(center_xy[0]), float(center_xy[1]))
        self.arm_spacing_px = params.arm_spacing_fov_fraction * float(min(width, height))
        self.scan_speed_px_s = float(scan_speed_px_s)
        self._b = self.arm_spacing_px / (2.0 * math.pi)
        self._path_px = 0.0

    @property
    def path_length_px(self) -> float:
        """Distance travelled along the spiral so far."""
        return self._path_px

    @property
    def radius_px(self) -> float:
        """Current spiral radius.

        Arc length of ``r = b*theta`` is approximately ``b*theta^2/2`` for large theta, so
        ``theta = sqrt(2*s/b)`` and ``r = b*theta``.
        """
        if self._b <= 0:
            return 0.0
        theta = math.sqrt(max(0.0, 2.0 * self._path_px / self._b))
        return self._b * theta

    def coverage_time_s(self, area_px: float) -> float:
        """Estimate the time to sweep an area of ``area_px`` square pixels.

        Args:
            area_px: Uncertainty region area.

        Returns:
            Time in seconds. This is the number behind the search-limited acquisition population.
        """
        if self.arm_spacing_px <= 0 or self.scan_speed_px_s <= 0:
            return math.inf
        return (area_px / self.arm_spacing_px) / self.scan_speed_px_s

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance along the spiral and return the next boresight waypoint.

        Args:
            dt: Interval in seconds. Must be non-negative.

        Returns:
            The waypoint as ``(x, y)`` in canvas coordinates.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"Time step must be non-negative, got {dt}")
        self._path_px += self.scan_speed_px_s * dt

        radius = self.radius_px
        if radius > self.params.max_radius_px:
            self._path_px = 0.0
            radius = 0.0

        theta = math.sqrt(max(0.0, 2.0 * self._path_px / self._b)) if self._b > 0 else 0.0
        
        # Apply hexagonal scaling for "cut hexagonal spiral scan"
        angle_in_segment = (theta % (math.pi / 3.0)) - (math.pi / 6.0)
        hex_scale = (math.sqrt(3.0) / 2.0) / math.cos(angle_in_segment)
        hex_radius = radius * hex_scale
        
        return (self.center[0] + hex_radius * math.cos(theta),
                self.center[1] + hex_radius * math.sin(theta))

    def recenter(self, center_xy: Tuple[float, float]) -> None:
        """Restart the spiral about a new centre.

        Args:
            center_xy: New centre as ``(x, y)``.
        """
        self.center = (float(center_xy[0]), float(center_xy[1]))
        self._path_px = 0.0

    def reset(self) -> None:
        """Restart the spiral at its current centre."""
        self._path_px = 0.0
