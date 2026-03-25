"""Compatibility tests for the powermove debug wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.brace_benchmark import BraceSegment, BraceSequence
from pipeline.powermove_debug import (
    build_powermove_debug_report,
    build_window_ladder,
    select_target_segment,
)


def _stable_joints(n_frames: int, offset: float = 0.0) -> np.ndarray:
    joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
    t = np.linspace(0.0, 0.03, n_frames, dtype=np.float32)
    joints[:, 0, 0] = offset + t
    joints[:, :, 1] = 1.0
    joints[:, 15, 1] = 1.5
    return joints


def test_select_target_segment_prefers_powermove_when_uid_not_given():
    segments = [
        BraceSegment("seg-top", "toprock", "tester", 2011, 100, 140, 0, 40),
        BraceSegment("seg-pm", "powermove", "tester", 2011, 140, 220, 40, 120),
    ]

    selected = select_target_segment(segments)

    assert selected.uid == "seg-pm"


def test_build_window_ladder_counts_short_windows_before_benchmark_gate():
    segment = BraceSegment("seg-pm", "powermove", "tester", 2011, 100, 160, 0, 60)
    josh_windows = [
        {"start_frame": 10, "end_frame_inclusive": 32, "end_frame_exclusive": 33, "n_frames": 23},
        {"start_frame": 40, "end_frame_inclusive": 47, "end_frame_exclusive": 48, "n_frames": 8},
    ]

    ladder = build_window_ladder(segment=segment, josh_windows=josh_windows, shot_boundaries=[], thresholds=(8, 12, 24, 45))

    assert ladder == [
        {"min_frames": 8, "candidate_count": 2, "best_n_frames": 23},
        {"min_frames": 12, "candidate_count": 1, "best_n_frames": 23},
        {"min_frames": 24, "candidate_count": 0, "best_n_frames": 0},
        {"min_frames": 45, "candidate_count": 0, "best_n_frames": 0},
    ]


def test_build_powermove_debug_report_identifies_short_window_quality_present():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=4,
        start_frame=1000,
        end_frame_exclusive=1060,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.4",
    )
    segments = [
        BraceSegment("seg-pm", "powermove", "tester", 2011, 1010, 1070, 10, 60),
    ]
    josh_joints = _stable_joints(60, offset=0.0)
    gvhmr_joints = _stable_joints(60, offset=0.01)
    valid_mask = np.zeros(60, dtype=bool)
    valid_mask[15:38] = True
    josh_joints[~valid_mask] = np.nan
    meta = {
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [],
            "windows": [{"start_frame": 15, "end_frame": 37, "n_frames": 23}],
        }
    }
    gt_frames = {}
    josh_2d = np.zeros((60, 17, 2), dtype=np.float32)
    gvhmr_2d = np.zeros((60, 17, 2), dtype=np.float32)
    for frame in range(15, 38):
        gt = np.stack(
            [np.array([float(i * 10), float(i * 3)], dtype=np.float32) for i in range(17)],
            axis=0,
        )
        gt_frames[frame] = gt
        josh_2d[frame] = gt + 1.0
        gvhmr_2d[frame] = gt + 15.0

    report = build_powermove_debug_report(
        josh_joints=josh_joints,
        josh_meta=meta,
        josh_valid_mask=valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segments=segments,
        shot_boundaries=[],
        fps=30.0,
        candidate_min_frames=8,
        benchmark_min_window_frames=45,
        josh_2d=josh_2d,
        gvhmr_2d=gvhmr_2d,
        gt_status="manual",
        gt_frames=gt_frames,
    )

    assert report["summary"]["benchmark_window_count"] == 0
    assert report["summary"]["candidate_window_count"] == 1
    assert report["summary"]["dominant_issue"] == "short_window_quality_present"
    assert report["candidate_windows"][0]["recommendation"] == "keep_josh"
