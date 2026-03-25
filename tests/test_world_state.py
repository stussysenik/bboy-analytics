"""Regression tests for world-state helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from world_state import compute_world_state


def test_compute_world_state_handles_short_clips_with_beats():
    frames = 23
    joints = np.zeros((frames, 24, 3), dtype=np.float32)
    joints[:, :, 2] = 4.0
    joints[:, :, 1] = 1.0
    joints[:, 15, 1] = 1.5
    joints[:, 0, 0] = np.linspace(0.0, 0.2, frames, dtype=np.float32)

    ws = compute_world_state(joints, fps=29.97, beat_times=np.array([0.1, 0.5], dtype=np.float32))

    assert ws.frames == frames
    assert ws.local_mu is not None
    assert ws.local_mu.shape == (frames,)
    assert np.isfinite(ws.local_mu).all()
