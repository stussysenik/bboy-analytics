"""Tests for dense JOSH 2D projection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.brace_assets import _member_matches_video
from pipeline.josh_projection import project_dense_josh_to_coco17


def test_project_dense_josh_to_coco17_outputs_visible_points():
    joints = np.zeros((2, 24, 3), dtype=np.float32)
    joints[:, :, 2] = 5.0
    joints[:, :, 1] = 1.0
    joints[:, 16, 0] = -0.5  # left shoulder
    joints[:, 17, 0] = 0.5   # right shoulder

    projected = project_dense_josh_to_coco17(joints, focal=2000.0, cx=960.0, cy=540.0)

    assert projected.shape == (2, 17, 3)
    assert np.all(projected[:, :, 2] == 1.0)
    assert projected[0, 5, 0] < projected[0, 6, 0]
    assert np.isfinite(projected[:, :, :2]).all()


def test_member_matches_video_handles_rooted_and_prefixed_paths():
    assert _member_matches_video("2011/RS0mFARO1x4/img-004582.npz", year=2011, video_id="RS0mFARO1x4")
    assert _member_matches_video(
        "dataset/2011/RS0mFARO1x4/RS0mFARO1x4_4445-4801_footwork.json",
        year=2011,
        video_id="RS0mFARO1x4",
    )
    assert not _member_matches_video("2012/OTHER/img-000001.npz", year=2011, video_id="RS0mFARO1x4")
