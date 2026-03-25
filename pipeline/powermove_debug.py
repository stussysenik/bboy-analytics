"""Compatibility wrapper for the focused powermove diagnostics flow."""

from __future__ import annotations

from typing import Any

import numpy as np

from .brace_benchmark import BraceSegment, BraceSequence, intersect_segment_windows, load_josh_windows_from_meta
from .powermove_diagnostics import build_segment_diagnostics_report, select_target_segment


DEFAULT_WINDOW_LADDER = (8, 12, 16, 24, 30, 45)


def build_window_ladder(
    *,
    segment: BraceSegment,
    josh_windows: list[dict[str, int]],
    shot_boundaries: list[int],
    thresholds: tuple[int, ...] = DEFAULT_WINDOW_LADDER,
) -> list[dict[str, int]]:
    """Summarize surviving JOSH windows at multiple frame-length gates."""
    ladder = []
    for min_frames in thresholds:
        windows = intersect_segment_windows(
            segment,
            josh_windows,
            shot_boundaries,
            min_window_frames=min_frames,
        )
        ladder.append(
            {
                "min_frames": int(min_frames),
                "candidate_count": len(windows),
                "best_n_frames": max((window.n_frames for window in windows), default=0),
            }
        )
    return ladder


def _compat_dominant_issue(
    *,
    candidate_windows: list[dict[str, Any]],
    benchmark_window_count: int,
    coverage_pct: float,
    gt_status: str,
) -> tuple[str, str]:
    if gt_status == "unavailable":
        return (
            "ground_truth_gap",
            "No local BRACE 2D overlap is available for this segment window, so attribution remains structural only.",
        )
    if not candidate_windows:
        if coverage_pct < 35.0:
            return (
                "tracking_fragmentation",
                "JOSH validity inside the segment is too sparse to recover even a short contiguous diagnostic window.",
            )
        return (
            "no_contiguous_signal",
            "JOSH has some valid frames in the segment, but no contiguous run survives the current short-window diagnostic gate.",
        )

    best = candidate_windows[0]
    if benchmark_window_count == 0 and best["recommendation"] == "keep_josh":
        return (
            "short_window_quality_present",
            "JOSH looks locally usable on a short run, but continuity fails before the 45-frame benchmark/render gate.",
        )
    if "pose_prior_failure" in best["failure_tags"]:
        return (
            "pose_quality_gap",
            "The best available local JOSH run is contiguous enough to inspect, but the pose itself still loses against the current gate.",
        )
    if "tracking_failure" in best["failure_tags"]:
        return (
            "tracking_fragmentation",
            "The best candidate window still carries continuity or identity issues, so tracking/assembly remains the blocker.",
        )
    return (
        "mixed_failure",
        "The segment contains a short inspectable run, but the current signals do not yet isolate one single failure mode cleanly.",
    )


def build_powermove_debug_report(
    *,
    josh_joints: np.ndarray,
    josh_meta: dict[str, Any],
    josh_valid_mask: np.ndarray,
    gvhmr_joints: np.ndarray,
    sequence: BraceSequence,
    segments: list[BraceSegment],
    shot_boundaries: list[int],
    fps: float = 29.97,
    dance_type: str = "powermove",
    segment_uid: str | None = None,
    candidate_min_frames: int = 8,
    benchmark_min_window_frames: int = 45,
    source_track_ids: np.ndarray | None = None,
    josh_2d: np.ndarray | None = None,
    gvhmr_2d: np.ndarray | None = None,
    gt_status: str = "unavailable",
    gt_frames: dict[int, np.ndarray] | None = None,
    manual_gt_frames: dict[int, np.ndarray] | None = None,
    interpolated_gt_frames: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Provide the historical `powermove_debug` report shape on top of the new helpers."""
    segment = select_target_segment(segments, segment_uid=segment_uid, dance_type=dance_type)
    report = build_segment_diagnostics_report(
        josh_joints=josh_joints,
        josh_meta=josh_meta,
        josh_valid_mask=josh_valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segment=segment,
        shot_boundaries=shot_boundaries,
        fps=fps,
        min_window_frames=benchmark_min_window_frames,
        candidate_min_frames=candidate_min_frames,
        source_track_ids=source_track_ids,
        gt_status=gt_status,
        gt_frames=gt_frames,
        manual_gt_frames=manual_gt_frames,
        interpolated_gt_frames=interpolated_gt_frames,
        josh_2d=josh_2d,
        gvhmr_2d=gvhmr_2d,
    )
    _, josh_windows = load_josh_windows_from_meta(josh_meta)
    ladder = build_window_ladder(
        segment=segment,
        josh_windows=josh_windows,
        shot_boundaries=shot_boundaries,
        thresholds=DEFAULT_WINDOW_LADDER,
    )
    dominant_issue, dominant_summary = _compat_dominant_issue(
        candidate_windows=report["candidate_windows"],
        benchmark_window_count=report["segment_summary"]["benchmarkable_candidate_count"],
        coverage_pct=report["segment_summary"]["josh_valid_coverage_pct"],
        gt_status=gt_status,
    )
    report["window_ladder"] = ladder
    report["summary"] = {
        "segment_uid": segment.uid,
        "candidate_window_count": len(report["candidate_windows"]),
        "benchmark_window_count": report["segment_summary"]["benchmarkable_candidate_count"],
        "coverage_pct": report["segment_summary"]["josh_valid_coverage_pct"],
        "dominant_issue": dominant_issue,
        "dominant_issue_summary": dominant_summary,
    }
    return report
