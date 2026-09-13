"""Tests for the world canvas.

Two properties matter here: the buffer is genuinely reused rather than reallocated per frame,
and compositing a beacon onto the canvas does not disturb its sub-pixel centroid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.sim.beacon import BeaconParams, centroid_of, render_beacon
from src.sim.canvas import Canvas


@pytest.fixture()
def canvas() -> Canvas:
    """A default 2000x2000 canvas at background level 10."""
    return Canvas(2000, 2000, background_level=10)


def _buffer_address(array: np.ndarray) -> int:
    """Return the address of an array's underlying data buffer."""
    return array.__array_interface__["data"][0]


def test_canvas_matches_configuration() -> None:
    """A canvas built from config must match the configured scene."""
    config = load_config("config/default.json")
    built = Canvas.from_config(config)
    assert built.shape == (config.scene.height, config.scene.width)
    assert built.data.dtype == np.uint8
    assert int(built.data[0, 0]) == config.scene.background_level


def test_buffer_is_allocated_once_and_reused(canvas: Canvas) -> None:
    """Clearing must write in place, never reallocate.

    At 30 Hz a reallocated 2000x2000 uint8 canvas churns 120 MB/s through the allocator, and the
    resulting collection pauses show up directly as frame-rate jitter we would then have to
    explain in the FPS log.
    """
    address = _buffer_address(canvas.data)
    for _ in range(10):
        canvas.clear()
        canvas.composite(render_beacon(1000.5, 1000.5, BeaconParams()))
        assert _buffer_address(canvas.data) == address


def test_clear_restores_the_background(canvas: Canvas) -> None:
    """After compositing and clearing, no trace of the beacon may remain."""
    canvas.clear()
    canvas.composite(render_beacon(1000.0, 1000.0, BeaconParams()))
    assert canvas.data.max() > 10
    canvas.clear()
    assert canvas.data.min() == canvas.data.max() == 10


def test_gradient_background_has_the_requested_mean() -> None:
    """A gradient background must vary spatially while preserving its mean level."""
    plain = Canvas(200, 200, background_level=40, background_gradient=False)
    graded = Canvas(200, 200, background_level=40, background_gradient=True)
    assert plain.data.std() == 0
    assert graded.data.std() > 0
    assert graded.data.mean() == pytest.approx(40, abs=1.0)


@pytest.mark.parametrize("position", [1000.0, 1000.5, 1000.37])
def test_composite_preserves_the_sub_pixel_centroid(canvas: Canvas, position: float) -> None:
    """Compositing through uint8 quantisation must not move the spot.

    Quantisation to 8 bits is symmetric about the peak, so it costs precision but must not
    introduce bias. Ground truth is sacred: a shift introduced here would be inherited by every
    accuracy number the project reports.
    """
    canvas.clear()
    canvas.composite(render_beacon(position, position, BeaconParams()))
    window = canvas.extract(int(position) - 20, int(position) - 20, 41, 41).astype(np.float64)
    cx, cy = centroid_of(window - 10, int(position) - 20, int(position) - 20)
    assert abs(cx - position) < 0.05
    assert abs(cy - position) < 0.05


def test_additive_composite_saturates_rather_than_wrapping(canvas: Canvas) -> None:
    """Overflow must clip at 255, never wrap around to zero.

    A uint8 wrap would turn the brightest part of the beacon black -- catastrophic for a
    centroid, and exactly the kind of bug that only appears at high intensity.
    """
    canvas.clear()
    canvas.composite(render_beacon(1000.0, 1000.0, BeaconParams(peak_intensity=255.0)),
                     mode="add")
    assert canvas.data.max() == 255
    assert canvas.data[1000, 1000] == 255


def test_max_composite_does_not_accumulate_the_background(canvas: Canvas) -> None:
    """``max`` mode must take the per-pixel maximum rather than summing.

    The centre pixel lands slightly *below* the configured peak (about 197.5 for a peak of 200
    at sigma=2.5) because each pixel reports the profile integrated over its own area, not the
    analytic peak value. That is correct detector behaviour. What matters here is that the
    background is not added on top: under ``add`` the same pixel would read about 207.
    """
    canvas.clear()
    canvas.composite(render_beacon(1000.0, 1000.0, BeaconParams(peak_intensity=200.0)),
                     mode="max")
    assert canvas.data[1000, 1000] == pytest.approx(200, rel=0.02)
    assert canvas.data[1000, 1000] < 200  # area-integrated, so below the analytic peak

    canvas.clear()
    canvas.composite(render_beacon(1000.0, 1000.0, BeaconParams(peak_intensity=200.0)),
                     mode="add")
    assert canvas.data[1000, 1000] > 200  # background accumulated


def test_unknown_composite_mode_is_rejected(canvas: Canvas) -> None:
    """An unrecognised mode is a caller error."""
    with pytest.raises(ValueError, match="Composite mode"):
        canvas.composite(render_beacon(100.0, 100.0, BeaconParams()), mode="blend")


def test_composite_clips_at_canvas_edges(canvas: Canvas) -> None:
    """A beacon partly off-canvas must draw its visible part without erroring."""
    canvas.clear()
    canvas.composite(render_beacon(2.0, 2.0, BeaconParams()))
    assert canvas.data[2, 2] > 10


def test_composite_entirely_off_canvas_is_a_no_op(canvas: Canvas) -> None:
    """A fully off-canvas beacon must leave the canvas untouched."""
    canvas.clear()
    canvas.composite(render_beacon(-500.0, -500.0, BeaconParams()))
    assert canvas.data.min() == canvas.data.max() == 10


def test_extract_pads_out_of_bounds_with_the_background(canvas: Canvas) -> None:
    """Off-edge extraction must pad with background, not black.

    A hard black band would be the highest-contrast feature in the frame and adaptive
    thresholding would key on it instead of the beacon.
    """
    canvas.clear()
    window = canvas.extract(-10, -10, 32, 32)
    assert window.shape == (32, 32)
    assert window.min() == window.max() == 10


def test_extract_returns_the_requested_region(canvas: Canvas) -> None:
    """Extraction must be aligned to the coordinates asked for."""
    canvas.clear()
    canvas.data[500, 700] = 200
    window = canvas.extract(690, 495, 21, 11)
    assert window[5, 10] == 200


def test_extract_rejects_a_non_positive_size(canvas: Canvas) -> None:
    """A zero or negative window is a caller error."""
    with pytest.raises(ValueError, match="must be positive"):
        canvas.extract(0, 0, 0, 10)


def test_contains_uses_the_pixel_centre_convention(canvas: Canvas) -> None:
    """A canvas of width W spans x from -0.5 to W-0.5."""
    assert canvas.contains(-0.5, -0.5)
    assert canvas.contains(1999.5, 1999.5)
    assert not canvas.contains(-0.51, 0.0)
    assert not canvas.contains(1999.51, 0.0)


def test_invalid_canvas_dimensions_are_rejected() -> None:
    """Non-positive dimensions and out-of-range background levels must fail loudly."""
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Canvas(0, 100)
    with pytest.raises(ValueError, match="Background level"):
        Canvas(100, 100, background_level=300)
