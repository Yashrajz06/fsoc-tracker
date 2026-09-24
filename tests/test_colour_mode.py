"""Tests for the colour display mode in VideoFrameSource."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


def _make_capture_mock(frame_bgr: np.ndarray) -> MagicMock:
    """Return a mock cv2.VideoCapture that yields one BGR frame then ends."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: float(frame_bgr.shape[1]),
        cv2.CAP_PROP_FRAME_HEIGHT: float(frame_bgr.shape[0]),
        cv2.CAP_PROP_FRAME_COUNT: 1.0,
        cv2.CAP_PROP_FPS: 30.0,
    }.get(int(prop), 0.0)
    # First call returns the frame; second call (end of stream) returns (False, None)
    cap.read.side_effect = [(True, frame_bgr.copy()), (False, None)]
    return cap


def test_grayscale_mode_sets_display_frame_none():
    """With force_grayscale=True (colour_display=False), display_frame must be None."""
    from src.video_source import VideoFrameSource

    frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cap_mock = _make_capture_mock(frame_bgr)

    with patch("src.video_source.cv2.VideoCapture", return_value=cap_mock), \
         patch("pathlib.Path.exists", return_value=True):
        source = VideoFrameSource("fake.mp4", force_grayscale=True, colour_display=False)
        fd = source.get_frame()

    assert fd is not None
    assert fd.frame.ndim == 2, "Pipeline frame must be single-channel"
    assert fd.display_frame is None, "display_frame must be None when colour_display=False"


def test_colour_mode_sets_display_frame():
    """With colour_display=True, display_frame must be a 3-channel BGR array."""
    from src.video_source import VideoFrameSource

    frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cap_mock = _make_capture_mock(frame_bgr)

    with patch("src.video_source.cv2.VideoCapture", return_value=cap_mock), \
         patch("pathlib.Path.exists", return_value=True):
        source = VideoFrameSource("fake.mp4", force_grayscale=False, colour_display=True)
        fd = source.get_frame()

    assert fd is not None
    assert fd.frame.ndim == 2, "Pipeline frame must always be single-channel"
    assert fd.display_frame is not None, "display_frame must not be None when colour_display=True"
    assert fd.display_frame.ndim == 3, "display_frame must be 3-channel BGR"
    assert fd.display_frame.shape == frame_bgr.shape, "display_frame shape must match source"


def test_pipeline_frame_is_always_grayscale_regardless_of_colour_mode():
    """Whether colour or grayscale mode, frame.ndim must always be 2."""
    from src.video_source import VideoFrameSource

    frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for colour in (True, False):
        cap_mock = _make_capture_mock(frame_bgr)
        with patch("src.video_source.cv2.VideoCapture", return_value=cap_mock), \
             patch("pathlib.Path.exists", return_value=True):
            source = VideoFrameSource("fake.mp4", force_grayscale=not colour,
                                      colour_display=colour)
            fd = source.get_frame()
        assert fd is not None
        assert fd.frame.ndim == 2, \
            f"frame.ndim must be 2 with colour_display={colour}, got {fd.frame.ndim}"


def test_colour_display_false_via_config():
    """from_config with colour_display=False must set force_grayscale=True and display_frame=None."""
    from src.config import AppConfig
    from src.video_source import VideoFrameSource

    # Build a minimal video-mode config pointing at a fake path.
    # Use /dev/null for video path since VideoInputConfig.validate() checks existence and
    # /dev/null exists on Linux. We patch Path.exists to True for VideoFrameSource.__init__.
    raw = json.load(open("config/default.json"))
    raw["run"]["mode"] = "video"
    raw["video_input"] = {
        "path": "/dev/null",
        "force_grayscale": True,
        "normalise_intensity": False,
        "ground_truth_path": None,
        "colour_display": False,
        "auto_detect_resolution": True,
        "playback_realtime": False,
    }
    # Disable AI so we don't need the ONNX model path to exist during this test
    raw["ai"] = {"enabled": False}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        tmppath = f.name
    try:
        config = AppConfig.from_dict(json.load(open(tmppath)))
    finally:
        os.unlink(tmppath)

    frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cap_mock = _make_capture_mock(frame_bgr)

    with patch("src.video_source.cv2.VideoCapture", return_value=cap_mock), \
         patch("pathlib.Path.exists", return_value=True):
        source = VideoFrameSource.from_config(config)
        fd = source.get_frame()

    assert fd is not None
    assert fd.frame.ndim == 2
    assert fd.display_frame is None
