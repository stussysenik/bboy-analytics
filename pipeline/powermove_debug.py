"""Powermove-focused diagnostics for BRACE-aligned JOSH failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .brace_benchmark import (
    BraceSegment,
    BraceSequence,
    classify_failure,
    compute_2d_metrics,
    infer_josh_valid_mask,
    intersect_segment_windows,
    load_brace_ground_truth_2d,
    load_brace_segments,
    load_brace_sequence,
    load_brace_shot_boundaries,
    load_josh_windows_from_meta,
    load_projected_2d,
    recommend_action,
    segment_coverage_pct,
    summarize_model_window,
)
from .compare import run_comparison


DEFAULT_WINDOW_LADDER = (8, 12, 16, 24, 30, 45)


def select_target_segment(
    segments: list[BraceSegment],
    *,
    segment_uid: str | None = None,
    dance_type: str = "powermove",
) -> BraceSegment:
    """Choose the target BRACE segment for diagnostics."""
    if segment_uid is not None:
        for segment in segments:
            if segment.uid == segment_uid:
                return segment
        raise ValueError(f"Segment not found: {segment_uid}")

    matching = [segment for segment in segments if segment.dance_type == dance_type]
    if not matching:
        raise ValueError(f"No BRACE segments with dance_type={dance_type!r}")
    matching.sort(
        key=lambda item: (
            -(item.local_end_frame_exclusive - item.local_start_frame),
            item.local_start_frame,
        )
    )
    return matching[0]


def build_frame_diagnostics(
    *,
    sequence: BraceSequence,
    segment: BraceSegment,
    valid_mask: np.ndarray,
    gt_frames: dict[int, np.ndarray],
    shot_boundaries: list[int],
) -> list[dict[str, Any]]:
    """Emit per-frame availability rows for one BRACE segment."""
    rows = []
    boundary_set = set(int(boundary) for boundary in shot_boundaries)
    for local_frame in range(segment.local_start_frame, segment.local_end_frame_exclusive):
        rows.append(
            {
                "segment_uid": segment.uid,
                "local_frame": local_frame,
                "segment_frame": local_frame - segment.local_start_frame,
                "global_frame": sequence.start_frame + local_frame,
                "josh_valid": bool(valid_mask[local_frame]),
                "brace_gt_available": bool(local_frame in gt_frames),
                "shot_boundary": bool(local_frame in boundary_set),
            }
        )
    return rows


def build_window_ladder(
    *,
    segment: BraceSegment,
    josh_windows: list[dict[str, int]],
    shot_boundaries: list[int],
    thresholds: tuple[int, ...] = DEFAULT_WINDOW_LADDER,
) -> list[dict[str, int]]:
    """Summarize how many candidate windows survive each frame-length gate."""
    ladder = []
    for min_frames in thresholds:
        windows = intersect_segment_windows(
            segment,
            josh_windows,
            shot_boundaries,
            min_window_frames=min_frames,
        )
        best_n = max((window.n_frames for window in windows), default=0)
        ladder.append(
            {
                "min_frames": int(min_frames),
                "candidate_count": len(windows),
                "best_n_frames": int(best_n),
            }
        )
    return ladder


def _window_gt_status(
    gt_status: str,
    gt_frames: dict[int, np.ndarray],
    start: int,
    end_exclusive: int,
) -> str:
    if gt_status == "unavailable":
        return "unavailable"
    has_overlap = any(start <= frame < end_exclusive for frame in gt_frames)
    return gt_status if has_overlap else "unavailable"


def _dominant_issue(
    *,
    candidate_windows: list[dict[str, Any]],
    benchmark_window_count: int,
    coverage_pct: float,
    gt_status: str,
) -> tuple[str, str]:
    """Condense the current segment state into one decision-oriented label."""
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


def _next_actions(
    *,
    dominant_issue: str,
    best_candidate: dict[str, Any] | None,
    benchmark_min_window_frames: int,
) -> list[str]:
    actions = []
    if dominant_issue == "short_window_quality_present":
        shortfall = benchmark_min_window_frames - int(best_candidate["n_frames"])
        actions.append(
            f"Treat continuity as the blocker: extend the best JOSH run by at least {shortfall} frames before considering a backbone pivot."
        )
        actions.append("Audit track handoffs and gap boundaries around the surviving local run before scheduling another JOSH rerun.")
        actions.append("Use the short validated powermove slice as the acceptance target for any assembly or track-selection change.")
        return actions
    if dominant_issue == "tracking_fragmentation":
        actions.append("Inspect track fragmentation and source-track switches before touching the human prior.")
        actions.append("Target denser contiguous windows first; current evidence is too sparse to justify a model pivot.")
        return actions
    if dominant_issue == "pose_quality_gap":
        actions.append("The next experiment should test a stronger pose prior or contact-aware reconstruction on the same segment window.")
        actions.append("Keep the current report as the control so any HSMR/SKEL-style prior is evaluated on identical frames.")
        return actions
    if dominant_issue == "ground_truth_gap":
        actions.append("Fetch or align BRACE 2D for the target segment before drawing model-level conclusions.")
        return actions
    actions.append("Keep the segment under diagnostics and avoid a broad rerun until one failure mode dominates.")
    return actions


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
    josh_2d: np.ndarray | None = None,
    gvhmr_2d: np.ndarray | None = None,
    gt_status: str = "unavailable",
    gt_frames: dict[int, np.ndarray] | None = None,
    shot_boundaries_status: str = "available",
) -> dict[str, Any]:
    """Build a detailed diagnostic report for one BRACE powermove segment."""
    gt_frames = gt_frames or {}
    target = select_target_segment(segments, segment_uid=segment_uid, dance_type=dance_type)
    _, josh_windows = load_josh_windows_from_meta(josh_meta)
    coverage_pct = segment_coverage_pct(
        josh_valid_mask,
        target.local_start_frame,
        target.local_end_frame_exclusive,
    )

    ladder = build_window_ladder(
        segment=target,
        josh_windows=josh_windows,
        shot_boundaries=shot_boundaries,
    )
    candidate_windows = intersect_segment_windows(
        target,
        josh_windows,
        shot_boundaries,
        min_window_frames=candidate_min_frames,
    )
    benchmark_windows = intersect_segment_windows(
        target,
        josh_windows,
        shot_boundaries,
        min_window_frames=benchmark_min_window_frames,
    )

    candidate_reports = []
    for rank, window in enumerate(candidate_windows, start=1):
        start = window.local_start_frame
        end = window.local_end_frame_exclusive
        josh_slice = josh_joints[start:end]
        gvhmr_slice = gvhmr_joints[start:end]
        josh_summary = summarize_model_window(josh_slice, fps=fps)
        gvhmr_summary = summarize_model_window(gvhmr_slice, fps=fps)
        try:
            comparison = run_comparison(josh_slice, gvhmr_slice, fps=fps)
        except ValueError:
            comparison = None
        josh_2d_metrics = compute_2d_metrics(josh_2d, gt_frames, start, end)
        gvhmr_2d_metrics = compute_2d_metrics(gvhmr_2d, gt_frames, start, end)
        window_gt_status = _window_gt_status(gt_status, gt_frames, start, end)
        failure_tags = classify_failure(
            target.dance_type,
            josh_summary,
            comparison,
            window_gt_status,
            josh_2d_metrics=josh_2d_metrics,
            gvhmr_2d_metrics=gvhmr_2d_metrics,
        )
        recommendation = recommend_action(
            josh_summary,
            failure_tags,
            comparison,
            josh_2d_metrics=josh_2d_metrics,
            gvhmr_2d_metrics=gvhmr_2d_metrics,
        )
        candidate_reports.append(
            {
                "rank": rank,
                "local_start_frame": start,
                "local_end_frame_exclusive": end,
                "global_start_frame": sequence.start_frame + start,
                "global_end_frame_exclusive": sequence.start_frame + end,
                "n_frames": end - start,
                "duration_s": round((end - start) / fps, 3),
                "source_window_start_frame": window.source_window_start_frame,
                "source_window_end_frame_inclusive": window.source_window_end_frame_inclusive,
                "shot_boundary_count": window.shot_boundary_count,
                "gt_status": window_gt_status,
                "josh": josh_summary,
                "gvhmr": gvhmr_summary,
                "comparison": comparison,
                "josh_2d": josh_2d_metrics,
                "gvhmr_2d": gvhmr_2d_metrics,
                "failure_tags": failure_tags,
                "recommendation": recommendation,
            }
        )

    dominant_issue, dominant_issue_summary = _dominant_issue(
        candidate_windows=candidate_reports,
        benchmark_window_count=len(benchmark_windows),
        coverage_pct=coverage_pct,
        gt_status=gt_status,
    )
    best_candidate = candidate_reports[0] if candidate_reports else None
    next_actions = _next_actions(
        dominant_issue=dominant_issue,
        best_candidate=best_candidate,
        benchmark_min_window_frames=benchmark_min_window_frames,
    )

    report = {
        "sequence": {
            **sequence.__dict__,
        },
        "segment": {
            **target.__dict__,
            "duration_frames": target.local_end_frame_exclusive - target.local_start_frame,
            "duration_s": round((target.local_end_frame_exclusive - target.local_start_frame) / fps, 3),
            "coverage_pct": coverage_pct,
        },
        "inputs": {
            "fps": fps,
            "dance_type": dance_type,
            "candidate_min_frames": candidate_min_frames,
            "benchmark_min_window_frames": benchmark_min_window_frames,
            "josh_renderability": josh_meta.get("stats", {}).get("renderability"),
            "gt_status": gt_status,
            "shot_boundaries_status": shot_boundaries_status,
        },
        "summary": {
            "benchmark_window_count": len(benchmark_windows),
            "candidate_window_count": len(candidate_reports),
            "best_candidate_n_frames": int(best_candidate["n_frames"]) if best_candidate else 0,
            "best_candidate_shortfall_frames": (
                max(0, benchmark_min_window_frames - int(best_candidate["n_frames"]))
                if best_candidate
                else benchmark_min_window_frames
            ),
            "dominant_issue": dominant_issue,
            "dominant_issue_summary": dominant_issue_summary,
        },
        "coverage_ladder": ladder,
        "candidate_windows": candidate_reports,
        "frame_diagnostics": build_frame_diagnostics(
            sequence=sequence,
            segment=target,
            valid_mask=josh_valid_mask,
            gt_frames=gt_frames,
            shot_boundaries=shot_boundaries,
        ),
        "next_actions": next_actions,
        "review_renders": [],
    }
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a human-readable markdown summary."""
    seq = report["sequence"]
    segment = report["segment"]
    summary = report["summary"]
    inputs = report["inputs"]
    lines = [
        f"# Powermove Debug Report — {segment['uid']}",
        "",
        "## Summary",
        "",
        f"- Sequence: `{seq['uid']}` ({seq['dancer']})",
        f"- Segment: `{segment['local_start_frame']}–{segment['local_end_frame_exclusive']}` local / "
        f"`{segment['global_start_frame']}–{segment['global_end_frame_exclusive']}` global",
        f"- Type: `{segment['dance_type']}`",
        f"- JOSH segment coverage: `{segment['coverage_pct']}%`",
        f"- Candidate windows at `{inputs['candidate_min_frames']}`-frame gate: `{summary['candidate_window_count']}`",
        f"- Benchmarkable windows at `{inputs['benchmark_min_window_frames']}`-frame gate: `{summary['benchmark_window_count']}`",
        f"- Dominant issue: `{summary['dominant_issue']}`",
        f"- Interpretation: {summary['dominant_issue_summary']}",
        "",
        "## Window Ladder",
        "",
        "| Min Frames | Candidate Windows | Best Window |",
        "|------------|-------------------|-------------|",
    ]
    for row in report["coverage_ladder"]:
        lines.append(
            f"| `{row['min_frames']}` | `{row['candidate_count']}` | `{row['best_n_frames']}` |"
        )

    lines.extend(
        [
            "",
            "## Candidate Windows",
            "",
            "| Rank | Local Frames | Global Frames | n | Root Max (m) | MPJPE (mm) | JOSH 2D frac | GVHMR 2D frac | Recommendation | Failure Tags |",
            "|------|--------------|---------------|---|--------------|------------|--------------|---------------|----------------|--------------|",
        ]
    )
    for candidate in report["candidate_windows"]:
        comparison = candidate.get("comparison") or {}
        josh_2d = candidate.get("josh_2d") or {}
        gvhmr_2d = candidate.get("gvhmr_2d") or {}
        lines.append(
            f"| `{candidate['rank']}` | "
            f"`{candidate['local_start_frame']}–{candidate['local_end_frame_exclusive']}` | "
            f"`{candidate['global_start_frame']}–{candidate['global_end_frame_exclusive']}` | "
            f"`{candidate['n_frames']}` | "
            f"`{candidate['josh']['max_root_displacement_m']}` | "
            f"`{comparison.get('mpjpe_mean_mm', '—')}` | "
            f"`{josh_2d.get('mean_error_bbox_diag_frac', '—')}` | "
            f"`{gvhmr_2d.get('mean_error_bbox_diag_frac', '—')}` | "
            f"{candidate['recommendation']} | "
            f"{', '.join(candidate['failure_tags']) or '—'} |"
        )
    if not report["candidate_windows"]:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in report["next_actions"]:
        lines.append(f"- {action}")

    if report.get("review_renders"):
        lines.extend(
            [
                "",
                "## Review Renders",
                "",
            ]
        )
        for render in report["review_renders"]:
            lines.append(
                f"- Candidate `{render['candidate_rank']}`: `{render['local_start_frame']}–{render['local_end_frame_exclusive']}` "
                f"comparison `{render['comparison_render']}`"
            )
    return "\n".join(lines) + "\n"


def write_powermove_debug_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and CSV outputs for the diagnostics report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    candidate_csv_path = output_dir / "candidate_windows.csv"
    frame_csv_path = output_dir / "frame_diagnostics.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write(render_markdown_report(report))

    candidate_fields = [
        "rank",
        "local_start_frame",
        "local_end_frame_exclusive",
        "global_start_frame",
        "global_end_frame_exclusive",
        "n_frames",
        "duration_s",
        "shot_boundary_count",
        "gt_status",
        "recommendation",
        "failure_tags",
        "mpjpe_mean_mm",
        "josh_2d_bbox_frac",
        "gvhmr_2d_bbox_frac",
        "josh_max_root_displacement_m",
    ]
    with open(candidate_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for candidate in report["candidate_windows"]:
            writer.writerow(
                {
                    "rank": candidate["rank"],
                    "local_start_frame": candidate["local_start_frame"],
                    "local_end_frame_exclusive": candidate["local_end_frame_exclusive"],
                    "global_start_frame": candidate["global_start_frame"],
                    "global_end_frame_exclusive": candidate["global_end_frame_exclusive"],
                    "n_frames": candidate["n_frames"],
                    "duration_s": candidate["duration_s"],
                    "shot_boundary_count": candidate["shot_boundary_count"],
                    "gt_status": candidate["gt_status"],
                    "recommendation": candidate["recommendation"],
                    "failure_tags": "|".join(candidate["failure_tags"]),
                    "mpjpe_mean_mm": (
                        candidate["comparison"]["mpjpe_mean_mm"]
                        if candidate["comparison"] is not None
                        else None
                    ),
                    "josh_2d_bbox_frac": (
                        candidate["josh_2d"]["mean_error_bbox_diag_frac"]
                        if candidate["josh_2d"] is not None
                        else None
                    ),
                    "gvhmr_2d_bbox_frac": (
                        candidate["gvhmr_2d"]["mean_error_bbox_diag_frac"]
                        if candidate["gvhmr_2d"] is not None
                        else None
                    ),
                    "josh_max_root_displacement_m": candidate["josh"]["max_root_displacement_m"],
                }
            )

    frame_fields = [
        "segment_uid",
        "local_frame",
        "segment_frame",
        "global_frame",
        "josh_valid",
        "brace_gt_available",
        "shot_boundary",
    ]
    with open(frame_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()
        for row in report["frame_diagnostics"]:
            writer.writerow(row)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "candidate_csv": str(candidate_csv_path),
        "frame_csv": str(frame_csv_path),
    }


def load_powermove_debug_inputs(
    *,
    josh_joints_path: str | Path,
    josh_meta_path: str | Path,
    gvhmr_joints_path: str | Path,
    brace_dir: str | Path,
    video_id: str,
    seq_idx: int,
    josh_valid_mask_path: str | Path | None = None,
    josh_2d_path: str | Path | None = None,
    gvhmr_2d_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the shared inputs needed by the powermove diagnostics CLI."""
    josh_joints = np.load(josh_joints_path)
    gvhmr_joints = np.load(gvhmr_joints_path)
    with open(josh_meta_path) as f:
        josh_meta = json.load(f)
    sequence = load_brace_sequence(brace_dir, video_id, seq_idx)
    segments = load_brace_segments(brace_dir, sequence)
    shot_boundaries, shot_boundaries_status = load_brace_shot_boundaries(brace_dir, sequence)
    gt_status, gt_frames = load_brace_ground_truth_2d(brace_dir, sequence, segments)
    return {
        "josh_joints": josh_joints,
        "josh_meta": josh_meta,
        "josh_valid_mask": infer_josh_valid_mask(josh_joints, josh_valid_mask_path),
        "gvhmr_joints": gvhmr_joints,
        "sequence": sequence,
        "segments": segments,
        "shot_boundaries": shot_boundaries,
        "shot_boundaries_status": shot_boundaries_status,
        "gt_status": gt_status,
        "gt_frames": gt_frames,
        "josh_2d": load_projected_2d(josh_2d_path),
        "gvhmr_2d": load_projected_2d(gvhmr_2d_path),
    }
