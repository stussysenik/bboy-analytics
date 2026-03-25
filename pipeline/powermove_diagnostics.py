"""Powermove-specific JOSH failure attribution helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .brace_benchmark import (
    BraceSegment,
    BraceSequence,
    classify_failure,
    compute_2d_metrics,
    intersect_segment_windows,
    load_josh_windows_from_meta,
    recommend_action,
    segment_coverage_pct,
    summarize_model_window,
)
from .compare import run_comparison

DEFAULT_WINDOW_LADDER = (8, 12, 16, 24, 30, 45)


def infer_josh_sidecar_paths(joints_path: str | Path) -> dict[str, Path]:
    """Return the expected dense JOSH sibling artifact paths."""
    joints_path = Path(joints_path)
    return {
        "valid_mask": joints_path.with_name("joints_3d_josh_valid_mask.npy"),
        "source_track_ids": joints_path.with_name("joints_3d_josh_source_track_ids.npy"),
        "metadata": joints_path.with_name("joints_3d_josh_metadata.json"),
        "projected_2d": joints_path.with_name("joints_2d_josh_coco.npy"),
    }


def select_target_segment(
    segments: list[BraceSegment],
    *,
    segment_uid: str | None = None,
    dance_type: str = "powermove",
) -> BraceSegment:
    """Select one segment to diagnose."""
    if segment_uid is not None:
        for segment in segments:
            if segment.uid == segment_uid:
                return segment
        raise ValueError(f"Segment not found: {segment_uid}")

    candidates = [segment for segment in segments if segment.dance_type == dance_type]
    if not candidates:
        raise ValueError(f"No {dance_type!r} segment found")
    candidates.sort(
        key=lambda segment: (
            -(segment.local_end_frame_exclusive - segment.local_start_frame),
            segment.local_start_frame,
        )
    )
    return candidates[0]


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
    shot_boundary_set = {int(boundary) for boundary in shot_boundaries}
    for local_frame in range(segment.local_start_frame, segment.local_end_frame_exclusive):
        rows.append(
            {
                "segment_uid": segment.uid,
                "local_frame": local_frame,
                "segment_frame": local_frame - segment.local_start_frame,
                "global_frame": sequence.start_frame + local_frame,
                "josh_valid": bool(valid_mask[local_frame]),
                "brace_gt_available": bool(local_frame in gt_frames),
                "shot_boundary": bool(local_frame in shot_boundary_set),
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
        ladder.append(
            {
                "min_frames": int(min_frames),
                "candidate_count": len(windows),
                "best_n_frames": max((int(window.n_frames) for window in windows), default=0),
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


def _frame_overlap_count(frames: dict[int, np.ndarray], start: int, end_exclusive: int) -> int:
    return sum(1 for frame in frames if start <= frame < end_exclusive)


def _track_ids_for_slice(
    source_track_ids: np.ndarray | None,
    valid_mask: np.ndarray,
    start: int,
    end_exclusive: int,
) -> list[int]:
    if source_track_ids is None:
        return []
    window_valid = valid_mask[start:end_exclusive]
    if not np.any(window_valid):
        return []
    ids = np.asarray(source_track_ids[start:end_exclusive])[window_valid]
    return sorted({int(track_id) for track_id in ids})


def _josh_loses_2d(candidate: dict[str, Any]) -> bool:
    josh_2d = candidate.get("josh_2d")
    gvhmr_2d = candidate.get("gvhmr_2d")
    if josh_2d is None or gvhmr_2d is None:
        return False
    return (
        josh_2d["mean_error_bbox_diag_frac"] > gvhmr_2d["mean_error_bbox_diag_frac"] + 0.05
        or josh_2d["pck_0.2"] + 0.05 < gvhmr_2d["pck_0.2"]
    )


def _candidate_window_report(
    *,
    window: Any,
    segment: BraceSegment,
    sequence: BraceSequence,
    josh_joints: np.ndarray,
    gvhmr_joints: np.ndarray,
    fps: float,
    min_window_frames: int,
    gt_status: str,
    gt_frames: dict[int, np.ndarray],
    manual_gt_frames: dict[int, np.ndarray],
    interpolated_gt_frames: dict[int, np.ndarray],
    josh_2d: np.ndarray | None,
    gvhmr_2d: np.ndarray | None,
    source_track_ids: np.ndarray | None,
    josh_valid_mask: np.ndarray,
) -> dict[str, Any]:
    start = int(window.local_start_frame)
    end = int(window.local_end_frame_exclusive)
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
        segment.dance_type,
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
    return {
        **asdict(window),
        "global_start_frame": int(sequence.start_frame + start),
        "global_end_frame_exclusive": int(sequence.start_frame + end),
        "segment_frame_offset_start": int(start - segment.local_start_frame),
        "segment_frame_offset_end_exclusive": int(end - segment.local_end_frame_exclusive),
        "frames_short_of_benchmark_gate": max(0, min_window_frames - (end - start)),
        "is_benchmarkable": bool((end - start) >= min_window_frames),
        "source_track_ids": _track_ids_for_slice(source_track_ids, josh_valid_mask, start, end),
        "gt_status": window_gt_status,
        "manual_gt_frames": _frame_overlap_count(manual_gt_frames, start, end),
        "interpolated_gt_frames": _frame_overlap_count(interpolated_gt_frames, start, end),
        "josh": josh_summary,
        "gvhmr": gvhmr_summary,
        "comparison": comparison,
        "josh_2d": josh_2d_metrics,
        "gvhmr_2d": gvhmr_2d_metrics,
        "failure_tags": failure_tags,
        "recommendation": recommendation,
    }


def build_segment_diagnostics_report(
    *,
    josh_joints: np.ndarray,
    josh_meta: dict[str, Any],
    josh_valid_mask: np.ndarray,
    gvhmr_joints: np.ndarray,
    sequence: BraceSequence,
    segment: BraceSegment,
    shot_boundaries: list[int],
    fps: float = 29.97,
    min_window_frames: int = 45,
    candidate_min_frames: int = 8,
    source_track_ids: np.ndarray | None = None,
    gt_status: str = "unavailable",
    gt_frames: dict[int, np.ndarray] | None = None,
    manual_gt_frames: dict[int, np.ndarray] | None = None,
    interpolated_gt_frames: dict[int, np.ndarray] | None = None,
    josh_2d: np.ndarray | None = None,
    gvhmr_2d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a diagnostics report for one BRACE segment."""
    gt_frames = gt_frames or {}
    manual_gt_frames = manual_gt_frames or {}
    interpolated_gt_frames = interpolated_gt_frames or {}
    if josh_joints.shape[0] != gvhmr_joints.shape[0]:
        raise ValueError(
            f"Expected JOSH and GVHMR clips to have the same frame count, got {josh_joints.shape[0]} and {gvhmr_joints.shape[0]}"
        )
    if josh_valid_mask.shape[0] != josh_joints.shape[0]:
        raise ValueError(
            f"Expected JOSH valid mask to have {josh_joints.shape[0]} frames, got {josh_valid_mask.shape[0]}"
        )
    if source_track_ids is not None and source_track_ids.shape[0] != josh_joints.shape[0]:
        raise ValueError(
            f"Expected JOSH source track ids to have {josh_joints.shape[0]} frames, got {source_track_ids.shape[0]}"
        )

    _, josh_windows = load_josh_windows_from_meta(josh_meta)
    raw_windows = intersect_segment_windows(
        segment,
        josh_windows,
        shot_boundaries,
        min_window_frames=1,
    )
    candidate_windows = [
        window for window in raw_windows
        if window.n_frames >= candidate_min_frames
    ]
    candidate_reports = [
        _candidate_window_report(
            window=window,
            segment=segment,
            sequence=sequence,
            josh_joints=josh_joints,
            gvhmr_joints=gvhmr_joints,
            fps=fps,
            min_window_frames=min_window_frames,
            gt_status=gt_status,
            gt_frames=gt_frames,
            manual_gt_frames=manual_gt_frames,
            interpolated_gt_frames=interpolated_gt_frames,
            josh_2d=josh_2d,
            gvhmr_2d=gvhmr_2d,
            source_track_ids=source_track_ids,
            josh_valid_mask=josh_valid_mask,
        )
        for window in candidate_windows
    ]
    candidate_reports.sort(key=lambda item: (-item["n_frames"], item["local_start_frame"]))

    segment_track_ids = _track_ids_for_slice(
        source_track_ids,
        josh_valid_mask,
        segment.local_start_frame,
        segment.local_end_frame_exclusive,
    )
    raw_overlap_frames = [int(window.n_frames) for window in raw_windows]
    max_raw_overlap_frames = max(raw_overlap_frames, default=0)
    coverage_pct = segment_coverage_pct(
        josh_valid_mask,
        segment.local_start_frame,
        segment.local_end_frame_exclusive,
    )
    benchmarkable_candidates = [
        candidate for candidate in candidate_reports
        if candidate["is_benchmarkable"]
    ]
    manual_segment_frames = _frame_overlap_count(
        manual_gt_frames,
        segment.local_start_frame,
        segment.local_end_frame_exclusive,
    )
    interpolated_segment_frames = _frame_overlap_count(
        interpolated_gt_frames,
        segment.local_start_frame,
        segment.local_end_frame_exclusive,
    )
    merged_segment_frames = _frame_overlap_count(
        gt_frames,
        segment.local_start_frame,
        segment.local_end_frame_exclusive,
    )

    notes = [
        (
            f"Longest contiguous JOSH overlap inside the segment is {max_raw_overlap_frames} frames; "
            f"the benchmark gate is {min_window_frames} frames."
        ),
    ]
    best_candidate = candidate_reports[0] if candidate_reports else None
    if merged_segment_frames > 0:
        notes.append(
            "BRACE 2D overlap exists locally on "
            f"{merged_segment_frames} segment frames "
            f"({manual_segment_frames} manual, {interpolated_segment_frames} interpolated)."
        )
    else:
        notes.append("No BRACE 2D overlap exists for this segment locally.")
    if segment_track_ids:
        if len(segment_track_ids) == 1:
            notes.append(
                f"All valid JOSH frames in this segment come from track {segment_track_ids[0]}, "
                "so the immediate failure is early termination / sparsity, not an in-segment identity handoff."
            )
        else:
            notes.append(
                "Valid JOSH frames in this segment span multiple source tracks: "
                f"{segment_track_ids}."
            )
    else:
        notes.append("JOSH contributes no valid frames inside this segment.")

    if benchmarkable_candidates:
        primary_bottleneck = "benchmarkable"
        next_actions = [
            "Benchmark the surviving candidate directly against GVHMR before changing capture assumptions.",
            "Use the candidate render strip to inspect inversion/contact plausibility and decide whether tuning is still needed.",
        ]
    elif best_candidate and _josh_loses_2d(best_candidate):
        primary_bottleneck = "coverage_and_pose_quality"
        notes.append(
            "On the best available candidate window, JOSH is objectively worse than GVHMR on BRACE 2D, "
            "so the blocker is not coverage alone."
        )
        next_actions = [
            "Inspect the short candidate strip and its 2D reprojection before scheduling a full rerun.",
            "Treat the immediate problem as mixed: JOSH must both extend coverage and improve pose quality on the surviving powermove slice.",
            "If local assembly/tuning cannot improve the short slice, test a stronger prior before escalating to richer capture.",
        ]
    elif merged_segment_frames > 0:
        primary_bottleneck = "coverage_continuity"
        next_actions = [
            "Inspect JOSH track/assembly behavior around the raw candidate window before scheduling a rerun.",
            "Use the short candidate strip for visual debugging and compare it directly against GVHMR on the same frames.",
            "Only consider stronger priors or richer capture if local continuity tuning cannot extend the run toward the benchmark gate.",
        ]
    else:
        primary_bottleneck = "ground_truth_gap"
        next_actions = [
            "Fetch or recover BRACE 2D for the failing segment before making a model-level decision.",
            "Keep any tuning local until the segment can be evaluated objectively.",
        ]

    report = {
        "sequence": asdict(sequence),
        "segment": {
            **asdict(segment),
            "duration_frames": int(segment.local_end_frame_exclusive - segment.local_start_frame),
            "duration_s": round((segment.local_end_frame_exclusive - segment.local_start_frame) / fps, 3),
        },
        "inputs": {
            "fps": fps,
            "min_window_frames": min_window_frames,
            "candidate_min_frames": candidate_min_frames,
            "josh_renderability": josh_meta.get("stats", {}).get("renderability"),
            "josh_recommended_windows": josh_meta.get("stats", {}).get("recommended_windows", []),
        },
        "window_ladder": build_window_ladder(
            segment=segment,
            josh_windows=josh_windows,
            shot_boundaries=shot_boundaries,
        ),
        "ground_truth_2d": {
            "status": gt_status,
            "segment_frames_available": merged_segment_frames,
            "segment_manual_frames": manual_segment_frames,
            "segment_interpolated_frames": interpolated_segment_frames,
        },
        "segment_summary": {
            "josh_valid_coverage_pct": coverage_pct,
            "josh_valid_frames": int(josh_valid_mask[segment.local_start_frame:segment.local_end_frame_exclusive].sum()),
            "raw_overlap_windows": len(raw_windows),
            "candidate_window_count": len(candidate_reports),
            "benchmarkable_candidate_count": len(benchmarkable_candidates),
            "max_raw_overlap_frames": max_raw_overlap_frames,
            "frames_short_of_benchmark_gate": max(0, min_window_frames - max_raw_overlap_frames),
            "source_track_ids": segment_track_ids,
            "best_candidate_frames": int(best_candidate["n_frames"]) if best_candidate else 0,
            "best_candidate_is_benchmarkable": bool(best_candidate["is_benchmarkable"]) if best_candidate else False,
        },
        "candidate_windows": candidate_reports,
        "diagnosis": {
            "primary_bottleneck": primary_bottleneck,
            "notes": notes,
            "next_actions": next_actions,
        },
        "frame_diagnostics": build_frame_diagnostics(
            sequence=sequence,
            segment=segment,
            valid_mask=josh_valid_mask,
            gt_frames=gt_frames,
            shot_boundaries=shot_boundaries,
        ),
        "review_renders": [],
    }
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a readable Markdown summary for one segment diagnostics pass."""
    seq = report["sequence"]
    segment = report["segment"]
    summary = report["segment_summary"]
    gt = report["ground_truth_2d"]
    diagnosis = report["diagnosis"]
    lines = [
        f"# Powermove Diagnostics — {segment['uid']}",
        "",
        "## Summary",
        "",
        f"- Sequence: `{seq['uid']}` ({seq['dancer']})",
        f"- Segment type: `{segment['dance_type']}`",
        f"- Frames: local `{segment['local_start_frame']}–{segment['local_end_frame_exclusive']}` | global `{segment['global_start_frame']}–{segment['global_end_frame_exclusive']}` (end exclusive)",
        f"- JOSH renderability: `{report['inputs']['josh_renderability']}`",
        f"- JOSH coverage in segment: `{summary['josh_valid_coverage_pct']}%` ({summary['josh_valid_frames']} valid / {segment['duration_frames']} total frames)",
        f"- Longest contiguous JOSH overlap: `{summary['max_raw_overlap_frames']}` frames",
        f"- Frames short of `{report['inputs']['min_window_frames']}`-frame benchmark gate: `{summary['frames_short_of_benchmark_gate']}`",
        f"- BRACE 2D status on segment: `{gt['status']}` ({gt['segment_frames_available']} overlapping frames)",
        f"- Primary bottleneck: `{diagnosis['primary_bottleneck']}`",
        "",
        "## Why It Fails Now",
        "",
    ]
    for note in diagnosis["notes"]:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Window Ladder",
            "",
            "| Min Frames | Candidate Windows | Best Window |",
            "|------------|-------------------|-------------|",
        ]
    )
    for row in report.get("window_ladder", []):
        lines.append(
            f"| `{row['min_frames']}` | `{row['candidate_count']}` | `{row['best_n_frames']}` |"
        )

    lines.extend(
        [
            "",
            "## Candidate Windows",
            "",
            "| Local Frames | Global Frames | Length | Track IDs | Benchmarkable | GT | JOSH 2D | GVHMR 2D | Recommendation | Failure Tags |",
            "|-------------|---------------|--------|-----------|---------------|----|---------|----------|----------------|--------------|",
        ]
    )
    if report["candidate_windows"]:
        for candidate in report["candidate_windows"]:
            failure_tags = ", ".join(candidate["failure_tags"])
            josh_2d = candidate["josh_2d"]
            gvhmr_2d = candidate["gvhmr_2d"]
            josh_2d_label = (
                f"{josh_2d['mean_error_bbox_diag_frac']:.4f} / {josh_2d['pck_0.2']:.4f}"
                if josh_2d else "n/a"
            )
            gvhmr_2d_label = (
                f"{gvhmr_2d['mean_error_bbox_diag_frac']:.4f} / {gvhmr_2d['pck_0.2']:.4f}"
                if gvhmr_2d else "n/a"
            )
            lines.append(
                f"| `{candidate['local_start_frame']}–{candidate['local_end_frame_exclusive']}` | "
                f"`{candidate['global_start_frame']}–{candidate['global_end_frame_exclusive']}` | "
                f"{candidate['n_frames']} | "
                f"{candidate['source_track_ids']} | "
                f"{'yes' if candidate['is_benchmarkable'] else 'no'} | "
                f"{candidate['gt_status']} | "
                f"{josh_2d_label} | "
                f"{gvhmr_2d_label} | "
                f"{candidate['recommendation']} | "
                f"{failure_tags} |"
            )
    else:
        lines.append("| none | none | 0 | [] | no | unavailable | n/a | n/a | n/a | no contiguous JOSH overlap |")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in diagnosis["next_actions"]:
        lines.append(f"- {action}")

    if report.get("artifacts"):
        lines.extend(
            [
                "",
                "## Local Artifacts",
                "",
            ]
        )
        for key, value in sorted(report["artifacts"].items()):
            if value:
                lines.append(f"- `{key}`: `{value}`")

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
                f"- `{render['local_start_frame']}–{render['local_end_frame_exclusive']}`: `{render['path']}`"
            )

    return "\n".join(lines) + "\n"


def write_diagnostics_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and CSV outputs for one segment diagnostics pass."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "powermove_report.json"
    md_path = output_dir / "powermove_report.md"
    csv_path = output_dir / "candidate_windows.csv"
    frame_csv_path = output_dir / "frame_diagnostics.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write(render_markdown_report(report))

    fieldnames = [
        "local_start_frame",
        "local_end_frame_exclusive",
        "global_start_frame",
        "global_end_frame_exclusive",
        "n_frames",
        "source_track_ids",
        "gt_status",
        "manual_gt_frames",
        "interpolated_gt_frames",
        "is_benchmarkable",
        "frames_short_of_benchmark_gate",
        "josh_renderability",
        "gvhmr_renderability",
        "mpjpe_mean_mm",
        "josh_2d_mean_error_bbox_diag_frac",
        "gvhmr_2d_mean_error_bbox_diag_frac",
        "recommendation",
        "failure_tags",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in report["candidate_windows"]:
            writer.writerow(
                {
                    "local_start_frame": candidate["local_start_frame"],
                    "local_end_frame_exclusive": candidate["local_end_frame_exclusive"],
                    "global_start_frame": candidate["global_start_frame"],
                    "global_end_frame_exclusive": candidate["global_end_frame_exclusive"],
                    "n_frames": candidate["n_frames"],
                    "source_track_ids": "|".join(map(str, candidate["source_track_ids"])),
                    "gt_status": candidate["gt_status"],
                    "manual_gt_frames": candidate["manual_gt_frames"],
                    "interpolated_gt_frames": candidate["interpolated_gt_frames"],
                    "is_benchmarkable": candidate["is_benchmarkable"],
                    "frames_short_of_benchmark_gate": candidate["frames_short_of_benchmark_gate"],
                    "josh_renderability": candidate["josh"]["renderability"],
                    "gvhmr_renderability": candidate["gvhmr"]["renderability"],
                    "mpjpe_mean_mm": candidate["comparison"]["mpjpe_mean_mm"] if candidate["comparison"] else None,
                    "josh_2d_mean_error_bbox_diag_frac": (
                        candidate["josh_2d"]["mean_error_bbox_diag_frac"]
                        if candidate["josh_2d"] else None
                    ),
                    "gvhmr_2d_mean_error_bbox_diag_frac": (
                        candidate["gvhmr_2d"]["mean_error_bbox_diag_frac"]
                        if candidate["gvhmr_2d"] else None
                    ),
                    "recommendation": candidate["recommendation"],
                    "failure_tags": "|".join(candidate["failure_tags"]),
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
        for row in report.get("frame_diagnostics", []):
            writer.writerow(row)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "frame_csv": str(frame_csv_path),
    }


def build_powermove_report(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias with a clearer public name."""
    return build_segment_diagnostics_report(**kwargs)


def render_powermove_markdown(report: dict[str, Any]) -> str:
    """Compatibility alias with a clearer public name."""
    return render_markdown_report(report)


def write_powermove_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Compatibility alias with a clearer public name."""
    return write_diagnostics_outputs(report, output_dir)
