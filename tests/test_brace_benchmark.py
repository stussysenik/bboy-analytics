"""Tests for the BRACE structural benchmark helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.brace_benchmark import (
    BraceSegment,
    BraceSequence,
    build_benchmark_report,
    classify_segment_without_window,
    compute_2d_metrics,
    intersect_segment_windows,
    recommend_action,
)


def _stable_joints(n_frames: int, offset: float = 0.0) -> np.ndarray:
    joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
    t = np.linspace(0.0, 0.06, n_frames, dtype=np.float32)
    joints[:, 0, 0] = offset + t
    joints[:, :, 1] = 1.0
    joints[:, 15, 1] = 1.5  # head above pelvis in Y-up
    return joints


def test_intersect_segment_windows_uses_exclusive_segment_bounds():
    segment = BraceSegment(
        uid="seg",
        dance_type="footwork",
        dancer="tester",
        year=2011,
        global_start_frame=100,
        global_end_frame_exclusive=150,
        local_start_frame=10,
        local_end_frame_exclusive=50,
    )
    windows = [
        {"start_frame": 0, "end_frame_inclusive": 30, "end_frame_exclusive": 31, "n_frames": 31},
        {"start_frame": 40, "end_frame_inclusive": 80, "end_frame_exclusive": 81, "n_frames": 41},
    ]

    result = intersect_segment_windows(segment, windows, shot_boundaries=[12, 44], min_window_frames=10)

    assert len(result) == 2
    assert result[0].local_start_frame == 10
    assert result[0].local_end_frame_exclusive == 31
    assert result[1].local_start_frame == 40
    assert result[1].local_end_frame_exclusive == 50


def test_compute_2d_metrics_uses_gt_overlap_and_bbox_normalization():
    pred = np.zeros((20, 17, 2), dtype=np.float32)
    gt_frames = {}
    for frame in (3, 4, 5):
        gt = np.stack([np.array([float(i * 10), float(i * 5)], dtype=np.float32) for i in range(17)], axis=0)
        pred[frame] = gt + 2.0
        gt_frames[frame] = gt

    metrics = compute_2d_metrics(pred, gt_frames, start=0, end_exclusive=10)

    assert metrics is not None
    assert metrics["frames_compared"] == 3
    assert metrics["mean_error_px"] > 0.0
    assert 0.0 <= metrics["pck_0.2"] <= 1.0


def test_build_benchmark_report_produces_structural_summary():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=1,
        start_frame=100,
        end_frame_exclusive=160,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.1",
    )
    segment = BraceSegment(
        uid="vid.100.160",
        dance_type="footwork",
        dancer="tester",
        year=2011,
        global_start_frame=100,
        global_end_frame_exclusive=160,
        local_start_frame=0,
        local_end_frame_exclusive=60,
    )
    josh_joints = _stable_joints(60, offset=0.0)
    gvhmr_joints = _stable_joints(60, offset=0.01)
    valid_mask = np.ones(60, dtype=bool)
    meta = {
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
            "windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
        }
    }

    report = build_benchmark_report(
        josh_joints=josh_joints,
        josh_meta=meta,
        josh_valid_mask=valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segments=[segment],
        shot_boundaries=[],
        fps=30.0,
        min_window_frames=30,
        gt_status="unavailable",
        gt_frames={},
    )

    assert report["summary"]["segments_total"] == 1
    assert report["summary"]["benchmarkable_segments"] == 1
    assert report["summary"]["evaluated_windows_total"] == 1
    assert report["segments"][0]["best_recommendation"] in {
        "keep_josh",
        "needs_josh_tuning",
        "keep_gvhmr_baseline",
    }


def test_recommend_action_is_conservative_without_ground_truth():
    summary = {
        "renderability": "full_clip_ready",
        "coverage_pct": 100.0,
        "identity": {"pass": True},
        "bounds": {"pass": True},
    }
    assert recommend_action(summary, ["insufficient_ground_truth"]) == "keep_gvhmr_baseline"


def test_recommend_action_prefers_josh_when_2d_metrics_beat_gvhmr():
    summary = {
        "renderability": "full_clip_ready",
        "coverage_pct": 100.0,
        "identity": {"pass": True},
        "bounds": {"pass": True},
    }
    josh_2d = {"mean_error_bbox_diag_frac": 0.08, "pck_0.2": 0.95}
    gvhmr_2d = {"mean_error_bbox_diag_frac": 0.25, "pck_0.2": 0.40}
    assert recommend_action(summary, [], josh_2d_metrics=josh_2d, gvhmr_2d_metrics=gvhmr_2d) == "keep_josh"


def test_classify_segment_without_window_uses_information_limited_for_fragmented_segments():
    tags = classify_segment_without_window(
        dance_type="toprock",
        josh_coverage_pct=52.0,
        max_intersection_frames=28,
        min_window_frames=45,
    )
    assert "information_limited" in tags
    assert "tracking_failure" not in tags


def test_build_benchmark_report_rejects_length_mismatches():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=1,
        start_frame=100,
        end_frame_exclusive=160,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.1",
    )
    segment = BraceSegment(
        uid="vid.100.160",
        dance_type="footwork",
        dancer="tester",
        year=2011,
        global_start_frame=100,
        global_end_frame_exclusive=160,
        local_start_frame=0,
        local_end_frame_exclusive=60,
    )
    josh_joints = _stable_joints(60)
    gvhmr_joints = _stable_joints(59)
    valid_mask = np.ones(60, dtype=bool)
    meta = {
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
            "windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
        }
    }

    try:
        build_benchmark_report(
            josh_joints=josh_joints,
            josh_meta=meta,
            josh_valid_mask=valid_mask,
            gvhmr_joints=gvhmr_joints,
            sequence=sequence,
            segments=[segment],
            shot_boundaries=[],
            fps=30.0,
            min_window_frames=30,
            gt_status="unavailable",
            gt_frames={},
        )
    except ValueError as exc:
        assert "same frame count" in str(exc)
    else:
        raise AssertionError("Expected build_benchmark_report() to reject mismatched frame counts")


def test_build_benchmark_report_marks_window_without_gt_overlap_as_insufficient_ground_truth():
    sequence = BraceSequence(
        video_id="vid",
        seq_idx=1,
        start_frame=100,
        end_frame_exclusive=160,
        dancer="tester",
        dancer_id=1,
        year=2011,
        uid="vid.1",
    )
    segment = BraceSegment(
        uid="vid.100.160",
        dance_type="footwork",
        dancer="tester",
        year=2011,
        global_start_frame=100,
        global_end_frame_exclusive=160,
        local_start_frame=0,
        local_end_frame_exclusive=60,
    )
    josh_joints = _stable_joints(60, offset=0.0)
    gvhmr_joints = _stable_joints(60, offset=0.01)
    valid_mask = np.ones(60, dtype=bool)
    meta = {
        "stats": {
            "renderability": "window_ready",
            "recommended_windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
            "windows": [{"start_frame": 0, "end_frame": 59, "n_frames": 60}],
        }
    }
    gt_frames = {1000: np.zeros((17, 2), dtype=np.float32)}

    report = build_benchmark_report(
        josh_joints=josh_joints,
        josh_meta=meta,
        josh_valid_mask=valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segments=[segment],
        shot_boundaries=[],
        fps=30.0,
        min_window_frames=30,
        gt_status="manual",
        gt_frames=gt_frames,
    )

    window = report["segments"][0]["evaluated_windows"][0]
    assert window["gt_status"] == "unavailable"
    assert "insufficient_ground_truth" in window["failure_tags"]
    assert window["recommendation"] == "keep_gvhmr_baseline"
