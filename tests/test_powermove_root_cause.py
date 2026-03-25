"""Tests for numerical powermove root-cause analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.powermove_root_cause import compute_window_projection_diagnostics, similarity_align_2d


def test_similarity_align_2d_recovers_similarity_transform():
    gt = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    pred = gt @ np.array([[0.0, -2.0], [2.0, 0.0]], dtype=np.float32).T + np.array([10.0, -5.0], dtype=np.float32)

    aligned = similarity_align_2d(pred, gt)

    assert np.allclose(aligned, gt, atol=1e-5)


def test_compute_window_projection_diagnostics_separates_raw_and_aligned_error():
    pred = np.full((5, 17, 2), np.nan, dtype=np.float32)
    gt_frames = {}
    base = np.stack([np.array([float(i), float(i * 2)], dtype=np.float32) for i in range(17)], axis=0)
    for frame in range(2, 5):
        gt_frames[frame] = base
        pred[frame] = base + np.array([100.0, -50.0], dtype=np.float32)

    metrics = compute_window_projection_diagnostics(
        pred,
        gt_frames,
        start=0,
        end_exclusive=5,
        image_width=1920,
        image_height=1080,
    )

    assert metrics["frames_compared"] == 3
    assert metrics["mean_error_px"] > 0.0
    assert metrics["translation_aligned_error_px"] < 1e-4
    assert metrics["translation_aligned_error_bbox_diag_frac"] < 1e-6
    assert metrics["similarity_aligned_error_px"] < 1e-4
    assert metrics["similarity_aligned_error_bbox_diag_frac"] < 1e-6
    assert metrics["mean_center_offset_bbox_diag_frac"] > 0.0
