"""Tests for powermove diagnostics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.brace_benchmark import BraceSegment, BraceSequence
from pipeline.powermove_diagnostics import (
    build_segment_diagnostics_report,
    render_markdown_report,
    select_target_segment,
)


def _stable_joints(n_frames: int, offset: float = 0.0) -> np.ndarray:
    joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
    t = np.linspace(0.0, 0.03, n_frames, dtype=np.float32)
    joints[:, 0, 0] = offset + t
    joints[:, :, 1] = 1.0
    joints[:, 15, 1] = 1.5
    joints[:, :, 2] = 4.0
    return joints


def test_select_target_segment_prefers_powermove():
    segments = [
        BraceSegment(
            uid="toprock",
            dance_type="toprock",
            dancer="tester",
            year=2011,
            global_start_frame=100,
            global_end_frame_exclusive=150,
            local_start_frame=0,
            local_end_frame_exclusive=50,
        ),
        BraceSegment(
            uid="powermove",
            dance_type="powermove",
            dancer="tester",
            year=2011,
            global_start_frame=160,
            global_end_frame_exclusive=220,
            local_start_frame=60,
            local_end_frame_exclusive=120,
        ),
    ]

    assert select_target_segment(segments).uid == "powermove"


def test_build_segment_diagnostics_report_flags_frames_short_of_gate():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=1,
        start_frame=100,
        end_frame_exclusive=180,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.1",
    )
    segment = BraceSegment(
        uid="vid.130.170",
        dance_type="powermove",
        dancer="tester",
        year=2011,
        global_start_frame=130,
        global_end_frame_exclusive=170,
        local_start_frame=30,
        local_end_frame_exclusive=70,
    )
    josh_joints = _stable_joints(80)
    gvhmr_joints = _stable_joints(80, offset=0.01)
    valid_mask = np.zeros(80, dtype=bool)
    valid_mask[35:50] = True
    josh_joints[~valid_mask] = np.nan
    source_track_ids = np.full(80, -1, dtype=np.int32)
    source_track_ids[35:50] = 7
    meta = {
        "fps": 30.0,
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [],
            "windows": [{"start_frame": 35, "end_frame": 49, "n_frames": 15}],
        },
    }
    gt_frames = {frame: np.zeros((17, 2), dtype=np.float32) for frame in range(35, 50)}

    report = build_segment_diagnostics_report(
        josh_joints=josh_joints,
        josh_meta=meta,
        josh_valid_mask=valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segment=segment,
        shot_boundaries=[],
        fps=30.0,
        min_window_frames=20,
        candidate_min_frames=8,
        source_track_ids=source_track_ids,
        gt_status="manual",
        gt_frames=gt_frames,
        manual_gt_frames=gt_frames,
        interpolated_gt_frames={},
    )

    assert report["segment_summary"]["max_raw_overlap_frames"] == 15
    assert report["segment_summary"]["frames_short_of_benchmark_gate"] == 5
    assert report["segment_summary"]["source_track_ids"] == [7]
    assert report["ground_truth_2d"]["segment_frames_available"] == 15
    assert report["diagnosis"]["primary_bottleneck"] == "coverage_continuity"
    assert len(report["candidate_windows"]) == 1
    assert report["candidate_windows"][0]["frames_short_of_benchmark_gate"] == 5
    assert report["candidate_windows"][0]["source_track_ids"] == [7]
    assert not report["candidate_windows"][0]["is_benchmarkable"]


def test_render_markdown_report_mentions_primary_bottleneck():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=1,
        start_frame=100,
        end_frame_exclusive=180,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.1",
    )
    segment = BraceSegment(
        uid="vid.130.170",
        dance_type="powermove",
        dancer="tester",
        year=2011,
        global_start_frame=130,
        global_end_frame_exclusive=170,
        local_start_frame=30,
        local_end_frame_exclusive=70,
    )
    josh_joints = _stable_joints(80)
    gvhmr_joints = _stable_joints(80, offset=0.01)
    valid_mask = np.zeros(80, dtype=bool)
    valid_mask[35:50] = True
    josh_joints[~valid_mask] = np.nan
    meta = {
        "fps": 30.0,
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [],
            "windows": [{"start_frame": 35, "end_frame": 49, "n_frames": 15}],
        },
    }

    report = build_segment_diagnostics_report(
        josh_joints=josh_joints,
        josh_meta=meta,
        josh_valid_mask=valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segment=segment,
        shot_boundaries=[],
        fps=30.0,
        min_window_frames=20,
        candidate_min_frames=8,
        gt_status="unavailable",
    )
    markdown = render_markdown_report(report)

    assert "Primary bottleneck:" in markdown
    assert "Candidate Windows" in markdown
