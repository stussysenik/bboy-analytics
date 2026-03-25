"""BRACE-aligned structural benchmark for JOSH vs GVHMR outputs.

The benchmark is intentionally useful in two modes:

1. Structural mode: runs from joints + BRACE annotations only.
2. 2D eval mode: additionally uses per-frame projected 2D joints and BRACE
   keypoints when those artifacts are available locally.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compare import (
    identity_tracking_test,
    inversion_test,
    run_comparison,
    stage_bounds_test,
)

try:
    from extreme_motion_reimpl.recap.validate import summarize_joint_sequence
except ImportError:  # pragma: no cover - direct import fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from extreme_motion_reimpl.recap.validate import summarize_joint_sequence


@dataclass
class BraceSequence:
    video_id: str
    seq_idx: int
    start_frame: int
    end_frame_exclusive: int
    dancer: str
    dancer_id: int
    year: int
    uid: str


@dataclass
class BraceSegment:
    uid: str
    dance_type: str
    dancer: str
    year: int
    global_start_frame: int
    global_end_frame_exclusive: int
    local_start_frame: int
    local_end_frame_exclusive: int


@dataclass
class EvalWindow:
    local_start_frame: int
    local_end_frame_exclusive: int
    n_frames: int
    source_window_start_frame: int
    source_window_end_frame_inclusive: int
    shot_boundary_count: int


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_brace_sequence(brace_dir: str | Path, video_id: str, seq_idx: int) -> BraceSequence:
    """Load one BRACE sequence row."""
    rows = _load_csv_rows(Path(brace_dir) / "annotations" / "sequences.csv")
    for row in rows:
        if row["video_id"] == video_id and int(row["seq_idx"]) == int(seq_idx):
            return BraceSequence(
                video_id=row["video_id"],
                seq_idx=int(row["seq_idx"]),
                start_frame=int(row["start_frame"]),
                end_frame_exclusive=int(row["end_frame"]),
                dancer=row["dancer"],
                dancer_id=int(row["dancer_id"]),
                year=int(row["year"]),
                uid=row["uid"],
            )
    raise ValueError(f"Sequence not found for video_id={video_id!r}, seq_idx={seq_idx}")


def load_brace_segments(
    brace_dir: str | Path,
    sequence: BraceSequence,
) -> list[BraceSegment]:
    """Load BRACE segments mapped into sequence-local frame coordinates."""
    rows = _load_csv_rows(Path(brace_dir) / "annotations" / "segments.csv")
    segments = []
    for row in rows:
        if row["video_id"] != sequence.video_id or int(row["seq_idx"]) != sequence.seq_idx:
            continue
        global_start = int(row["start_frame"])
        global_end_exclusive = int(row["end_frame"])
        segments.append(
            BraceSegment(
                uid=row["uid"],
                dance_type=row["dance_type"],
                dancer=row["dancer"],
                year=int(row["year"]),
                global_start_frame=global_start,
                global_end_frame_exclusive=global_end_exclusive,
                local_start_frame=global_start - sequence.start_frame,
                local_end_frame_exclusive=global_end_exclusive - sequence.start_frame,
            )
        )
    return segments


def load_brace_shot_boundaries(
    brace_dir: str | Path,
    sequence: BraceSequence,
) -> tuple[list[int], str]:
    """Load shot boundaries local to a sequence plus availability status."""
    with open(Path(brace_dir) / "annotations" / "shot_boundaries.json") as f:
        all_boundaries = json.load(f)
    if sequence.video_id not in all_boundaries:
        return [], "missing_local_annotations"
    boundaries = []
    for boundary in all_boundaries.get(sequence.video_id, []):
        if sequence.start_frame <= boundary < sequence.end_frame_exclusive:
            boundaries.append(int(boundary - sequence.start_frame))
    return boundaries, "available"


def load_josh_windows(meta_path: str | Path) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Load all JOSH contiguous windows from metadata using explicit semantics."""
    with open(meta_path) as f:
        meta = json.load(f)
    windows = []
    for window in meta.get("stats", {}).get("windows", []):
        start = int(window["start_frame"])
        end_inclusive = int(window["end_frame"])
        windows.append(
            {
                "start_frame": start,
                "end_frame_inclusive": end_inclusive,
                "end_frame_exclusive": end_inclusive + 1,
                "n_frames": int(window["n_frames"]),
            }
        )
    return meta, windows


def infer_josh_valid_mask(joints: np.ndarray, valid_mask_path: str | Path | None = None) -> np.ndarray:
    """Load or infer the JOSH valid mask."""
    if valid_mask_path is not None and Path(valid_mask_path).exists():
        return np.load(valid_mask_path).astype(bool)
    return np.isfinite(joints).all(axis=(1, 2))


def intersect_segment_windows(
    segment: BraceSegment,
    josh_windows: list[dict[str, int]],
    shot_boundaries: list[int],
    min_window_frames: int = 45,
) -> list[EvalWindow]:
    """Intersect JOSH contiguous windows with one BRACE segment."""
    out = []
    for window in josh_windows:
        start = max(segment.local_start_frame, window["start_frame"])
        end_exclusive = min(segment.local_end_frame_exclusive, window["end_frame_exclusive"])
        n_frames = end_exclusive - start
        if n_frames < min_window_frames:
            continue
        shot_count = sum(start <= boundary < end_exclusive for boundary in shot_boundaries)
        out.append(
            EvalWindow(
                local_start_frame=start,
                local_end_frame_exclusive=end_exclusive,
                n_frames=n_frames,
                source_window_start_frame=window["start_frame"],
                source_window_end_frame_inclusive=window["end_frame_inclusive"],
                shot_boundary_count=shot_count,
            )
        )
    out.sort(key=lambda item: (-item.n_frames, item.local_start_frame))
    return out


def segment_coverage_pct(valid_mask: np.ndarray, start: int, end_exclusive: int) -> float:
    """Return local coverage percentage over a sequence slice."""
    if end_exclusive <= start:
        return 0.0
    return round(100.0 * float(valid_mask[start:end_exclusive].mean()), 1)


def summarize_model_window(joints: np.ndarray, fps: float) -> dict[str, Any]:
    """Summarize one finite or sparse joint window using shared validators."""
    summary = summarize_joint_sequence(joints, fps=fps)
    return {
        "frames": int(joints.shape[0]),
        "renderability": summary["renderability"],
        "coverage_pct": summary["coverage_pct"],
        "max_root_displacement_m": summary["max_root_displacement_m"],
        "mean_root_displacement_m": summary["mean_root_displacement_m"],
        "max_bone_drift_m": summary["max_bone_drift_m"],
        "identity": identity_tracking_test(joints),
        "inversion": inversion_test(joints, y_down=False),
        "bounds": stage_bounds_test(joints),
    }


def _bbox_diag_from_keypoints(keypoints: np.ndarray) -> float:
    xy = np.asarray(keypoints, dtype=np.float32)[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(diag, 1.0)


def load_projected_2d(path: str | Path | None) -> np.ndarray | None:
    """Load optional 2D keypoints aligned to the local clip."""
    if path is None:
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] < 2:
        raise ValueError(f"Expected (F, 17, 2/3) projected 2D array, got {arr.shape}")
    return arr[:, :, :2].astype(np.float32)


def _manual_keypoint_candidates(brace_dir: Path, sequence: BraceSequence) -> list[Path]:
    candidates = []
    for root_name in ("manual_keypoints", "manual", "mk"):
        candidate = brace_dir / root_name / str(sequence.year) / sequence.video_id
        if candidate.is_dir():
            candidates.append(candidate)
    if not candidates:
        matches = list(brace_dir.rglob(f"{sequence.video_id}/img-*.npz"))
        if matches:
            candidates.append(matches[0].parent)
    return candidates


def load_manual_brace_keypoints(
    brace_dir: str | Path,
    sequence: BraceSequence,
    segments: list[BraceSegment],
) -> dict[int, np.ndarray]:
    """Load per-frame manual BRACE keypoints keyed by local frame."""
    roots = _manual_keypoint_candidates(Path(brace_dir), sequence)
    if not roots:
        return {}
    frames: dict[int, np.ndarray] = {}
    for root in roots:
        for segment in segments:
            for global_frame in range(segment.global_start_frame, segment.global_end_frame_exclusive):
                path = root / f"img-{global_frame:06d}.npz"
                if not path.exists():
                    continue
                keypoints = np.load(path)["coco_joints2d"][:, :2].astype(np.float32)
                frames[global_frame - sequence.start_frame] = keypoints
        if frames:
            break
    return frames


def _find_interpolated_segment_json(brace_dir: Path, sequence: BraceSequence, segment: BraceSegment) -> Path | None:
    filename = (
        f"{sequence.video_id}.{segment.global_start_frame}.{segment.global_end_frame_exclusive}"
    )
    # Prefer the exact BRACE release layout if present.
    for root_name in ("dataset", "keypoints", "segments"):
        for suffix in (".json", ".json.json"):
            candidate = (
                brace_dir / root_name / str(sequence.year) / sequence.video_id /
                f"{sequence.video_id}_{segment.global_start_frame}-{segment.global_end_frame_exclusive}_{segment.dance_type}{suffix}"
            )
            if candidate.exists():
                return candidate
    matches = list(
        brace_dir.rglob(
            f"{sequence.video_id}_{segment.global_start_frame}-{segment.global_end_frame_exclusive}_{segment.dance_type}.json*"
        )
    )
    return matches[0] if matches else None


def load_interpolated_brace_keypoints(
    brace_dir: str | Path,
    sequence: BraceSequence,
    segments: list[BraceSegment],
) -> dict[int, np.ndarray]:
    """Load interpolated BRACE keypoints keyed by local frame."""
    brace_dir = Path(brace_dir)
    frames: dict[int, np.ndarray] = {}
    for segment in segments:
        path = _find_interpolated_segment_json(brace_dir, sequence, segment)
        if path is None:
            continue
        with open(path) as f:
            data = json.load(f)
        for frame_id, payload in data.items():
            frame_name = frame_id.split("/")[-1]
            global_frame = int(frame_name.replace("img-", "").replace(".png", ""))
            local_frame = global_frame - sequence.start_frame
            frames[local_frame] = np.asarray(payload["keypoints"], dtype=np.float32)[:, :2]
    return frames


def load_brace_ground_truth_2d(
    brace_dir: str | Path,
    sequence: BraceSequence,
    segments: list[BraceSegment],
) -> tuple[str, dict[int, np.ndarray]]:
    """Load BRACE 2D keypoints, preferring manual annotations per frame."""
    manual = load_manual_brace_keypoints(brace_dir, sequence, segments)
    interpolated = load_interpolated_brace_keypoints(brace_dir, sequence, segments)
    if manual and interpolated:
        merged = dict(interpolated)
        merged.update(manual)
        return "manual+interpolated", merged
    if manual:
        return "manual", manual
    if interpolated:
        return "interpolated", interpolated
    return "unavailable", {}


def compute_2d_metrics(
    pred_2d: np.ndarray | None,
    gt_frames: dict[int, np.ndarray],
    start: int,
    end_exclusive: int,
) -> dict[str, Any] | None:
    """Compare projected 2D joints against BRACE keypoints over one window."""
    if pred_2d is None or not gt_frames:
        return None
    frame_ids = [frame for frame in range(start, end_exclusive) if frame in gt_frames and frame < pred_2d.shape[0]]
    if not frame_ids:
        return None
    per_joint_errors = []
    diag_norm_errors = []
    pck_hits = 0
    pck_total = 0
    for frame in frame_ids:
        gt = gt_frames[frame]
        pred = pred_2d[frame]
        errors = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
        diag = _bbox_diag_from_keypoints(gt)
        per_joint_errors.extend(errors.tolist())
        diag_norm_errors.extend((errors / diag).tolist())
        pck_hits += int(np.sum(errors < (0.2 * diag)))
        pck_total += int(errors.shape[0])
    return {
        "frames_compared": len(frame_ids),
        "mean_error_px": round(float(np.mean(per_joint_errors)), 2),
        "median_error_px": round(float(np.median(per_joint_errors)), 2),
        "mean_error_bbox_diag_frac": round(float(np.mean(diag_norm_errors)), 4),
        "pck_0.2": round(float(pck_hits / max(pck_total, 1)), 4),
    }


def _window_gt_status(
    gt_status: str,
    gt_frames: dict[int, np.ndarray],
    start: int,
    end_exclusive: int,
) -> str:
    """Return GT status scoped to one evaluated window."""
    if gt_status == "unavailable":
        return "unavailable"
    has_overlap = any(start <= frame < end_exclusive for frame in gt_frames)
    return gt_status if has_overlap else "unavailable"


def classify_failure(
    dance_type: str,
    josh_summary: dict[str, Any],
    comparison: dict[str, Any] | None,
    gt_status: str,
    josh_2d_metrics: dict[str, Any] | None = None,
    gvhmr_2d_metrics: dict[str, Any] | None = None,
) -> list[str]:
    """Deterministic failure tags from the currently available signals."""
    tags = []
    if gt_status == "unavailable":
        tags.append("insufficient_ground_truth")
    if josh_summary["coverage_pct"] < 90.0 or not josh_summary["identity"]["pass"]:
        tags.append("tracking_failure")
    if not josh_summary["bounds"]["pass"]:
        tags.append("scene_scale_failure")
    if dance_type == "powermove" and josh_summary["coverage_pct"] < 95.0:
        tags.append("information_limited")
    if comparison is not None and comparison["mpjpe_mean_mm"] > 450.0 and "tracking_failure" not in tags:
        tags.append("pose_prior_failure")
    if josh_2d_metrics is not None:
        if josh_2d_metrics["mean_error_bbox_diag_frac"] > 0.2 or josh_2d_metrics["pck_0.2"] < 0.5:
            tags.append("pose_prior_failure")
    if josh_2d_metrics is not None and gvhmr_2d_metrics is not None:
        josh_err = josh_2d_metrics["mean_error_bbox_diag_frac"]
        gvhmr_err = gvhmr_2d_metrics["mean_error_bbox_diag_frac"]
        if josh_err > gvhmr_err + 0.05:
            tags.append("pose_prior_failure")
    # Contact remains a placeholder until dense contact metrics are integrated.
    return sorted(set(tags))


def recommend_action(
    josh_summary: dict[str, Any],
    failure_tags: list[str],
    comparison: dict[str, Any] | None = None,
    josh_2d_metrics: dict[str, Any] | None = None,
    gvhmr_2d_metrics: dict[str, Any] | None = None,
) -> str:
    """Map structural failures to the next action recommendation."""
    if "information_limited" in failure_tags:
        return "needs_richer_capture"
    if "pose_prior_failure" in failure_tags:
        if josh_2d_metrics is not None and gvhmr_2d_metrics is not None:
            josh_err = josh_2d_metrics["mean_error_bbox_diag_frac"]
            gvhmr_err = gvhmr_2d_metrics["mean_error_bbox_diag_frac"]
            if josh_err > gvhmr_err + 0.02:
                return "keep_gvhmr_baseline"
        return "needs_stronger_prior"
    if "tracking_failure" in failure_tags or "scene_scale_failure" in failure_tags or "contact_failure" in failure_tags:
        return "needs_josh_tuning"
    if "insufficient_ground_truth" in failure_tags:
        return "keep_gvhmr_baseline"
    if josh_2d_metrics is not None and gvhmr_2d_metrics is not None:
        josh_err = josh_2d_metrics["mean_error_bbox_diag_frac"]
        gvhmr_err = gvhmr_2d_metrics["mean_error_bbox_diag_frac"]
        josh_pck = josh_2d_metrics["pck_0.2"]
        gvhmr_pck = gvhmr_2d_metrics["pck_0.2"]
        if josh_err + 0.02 < gvhmr_err or josh_pck > gvhmr_pck + 0.05:
            return "keep_josh"
        if gvhmr_err + 0.02 < josh_err or gvhmr_pck > josh_pck + 0.05:
            return "keep_gvhmr_baseline"
    if josh_2d_metrics is not None and josh_2d_metrics["mean_error_bbox_diag_frac"] <= 0.2:
        return "keep_josh"
    if josh_summary["renderability"] != "not_renderable":
        return "keep_josh"
    return "keep_gvhmr_baseline"


def classify_segment_without_window(
    dance_type: str,
    josh_coverage_pct: float,
    max_intersection_frames: int,
    min_window_frames: int,
) -> list[str]:
    """Tag why a segment cannot be benchmarked under the current gate."""
    tags = []
    if josh_coverage_pct < 35.0:
        tags.append("tracking_failure")
    if max_intersection_frames < min_window_frames:
        tags.append("information_limited")
    if not tags and dance_type == "powermove":
        tags.append("information_limited")
    if not tags:
        tags.append("information_limited")
    return tags


def recommend_segment_without_window(dance_type: str, failure_tags: list[str]) -> str:
    """Map non-benchmarkable segments to the next action recommendation."""
    if "tracking_failure" in failure_tags or "scene_scale_failure" in failure_tags or "contact_failure" in failure_tags:
        return "needs_josh_tuning"
    if "information_limited" in failure_tags and dance_type == "powermove":
        return "needs_richer_capture"
    return "keep_gvhmr_baseline"


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item[key]
        counts[value] = counts.get(value, 0) + 1
    return counts


def _validate_benchmark_inputs(
    *,
    josh_joints: np.ndarray,
    josh_meta: dict[str, Any],
    josh_valid_mask: np.ndarray,
    gvhmr_joints: np.ndarray,
    segments: list[BraceSegment],
    josh_2d: np.ndarray | None,
    gvhmr_2d: np.ndarray | None,
) -> None:
    """Validate that all benchmark inputs share the same local clip semantics."""
    n_frames = int(josh_joints.shape[0])
    if gvhmr_joints.shape[0] != n_frames:
        raise ValueError(
            f"Expected JOSH and GVHMR clips to have the same frame count, got {n_frames} and {gvhmr_joints.shape[0]}"
        )
    if josh_valid_mask.shape[0] != n_frames:
        raise ValueError(
            f"Expected JOSH valid mask to have {n_frames} frames, got {josh_valid_mask.shape[0]}"
        )
    if josh_2d is not None and josh_2d.shape[0] != n_frames:
        raise ValueError(
            f"Expected JOSH 2D projections to have {n_frames} frames, got {josh_2d.shape[0]}"
        )
    if gvhmr_2d is not None and gvhmr_2d.shape[0] != gvhmr_joints.shape[0]:
        raise ValueError(
            f"Expected GVHMR 2D projections to have {gvhmr_joints.shape[0]} frames, got {gvhmr_2d.shape[0]}"
        )

    for window in josh_meta.get("stats", {}).get("windows", []):
        start = int(window["start_frame"])
        end_inclusive = int(window["end_frame"])
        if start < 0 or end_inclusive < start or end_inclusive >= n_frames:
            raise ValueError(
                f"JOSH metadata window out of bounds for {n_frames}-frame clip: start={start}, end={end_inclusive}"
            )
        expected_n = end_inclusive - start + 1
        if int(window["n_frames"]) != expected_n:
            raise ValueError(
                f"JOSH metadata window frame count mismatch: start={start}, end={end_inclusive}, "
                f"n_frames={window['n_frames']}, expected={expected_n}"
            )

    for segment in segments:
        if segment.local_start_frame < 0 or segment.local_end_frame_exclusive > n_frames:
            raise ValueError(
                f"BRACE segment {segment.uid} is out of bounds for {n_frames}-frame clip: "
                f"{segment.local_start_frame}:{segment.local_end_frame_exclusive}"
            )


def _max_intersection_frames(segment: BraceSegment, josh_windows: list[dict[str, int]]) -> int:
    """Return the largest raw overlap between a segment and any JOSH window."""
    max_frames = 0
    for window in josh_windows:
        start = max(segment.local_start_frame, window["start_frame"])
        end_exclusive = min(segment.local_end_frame_exclusive, window["end_frame_exclusive"])
        max_frames = max(max_frames, max(0, end_exclusive - start))
    return max_frames


def build_benchmark_report(
    *,
    josh_joints: np.ndarray,
    josh_meta: dict[str, Any],
    josh_valid_mask: np.ndarray,
    gvhmr_joints: np.ndarray,
    sequence: BraceSequence,
    segments: list[BraceSegment],
    shot_boundaries: list[int],
    fps: float = 29.97,
    min_window_frames: int = 45,
    josh_2d: np.ndarray | None = None,
    gvhmr_2d: np.ndarray | None = None,
    gt_status: str = "unavailable",
    gt_frames: dict[int, np.ndarray] | None = None,
    shot_boundaries_status: str = "available",
) -> dict[str, Any]:
    """Build a BRACE-aligned structural benchmark report."""
    gt_frames = gt_frames or {}
    _, josh_windows = load_josh_windows_from_meta(josh_meta)
    _validate_benchmark_inputs(
        josh_joints=josh_joints,
        josh_meta=josh_meta,
        josh_valid_mask=josh_valid_mask,
        gvhmr_joints=gvhmr_joints,
        segments=segments,
        josh_2d=josh_2d,
        gvhmr_2d=gvhmr_2d,
    )
    segment_reports = []
    flattened_rows = []

    for segment in segments:
        coverage_pct = segment_coverage_pct(
            josh_valid_mask,
            segment.local_start_frame,
            segment.local_end_frame_exclusive,
        )
        windows = intersect_segment_windows(
            segment,
            josh_windows,
            shot_boundaries,
            min_window_frames=min_window_frames,
        )
        evaluated = []
        for window in windows:
            start = window.local_start_frame
            end = window.local_end_frame_exclusive
            josh_slice = josh_joints[start:end]
            gvhmr_slice = gvhmr_joints[start:end]
            josh_summary = summarize_model_window(josh_slice, fps=fps)
            gvhmr_summary = summarize_model_window(gvhmr_slice, fps=fps)
            comparison = None
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
            evaluated_window = {
                **asdict(window),
                "global_start_frame": sequence.start_frame + start,
                "global_end_frame_exclusive": sequence.start_frame + end,
                "gt_status": window_gt_status,
                "josh": josh_summary,
                "gvhmr": gvhmr_summary,
                "comparison": comparison,
                "josh_2d": josh_2d_metrics,
                "gvhmr_2d": gvhmr_2d_metrics,
                "failure_tags": failure_tags,
                "recommendation": recommendation,
            }
            evaluated.append(evaluated_window)
            flattened_rows.append(
                {
                    "segment_uid": segment.uid,
                    "dance_type": segment.dance_type,
                    "local_start_frame": start,
                    "local_end_frame_exclusive": end,
                    "n_frames": end - start,
                    "josh_coverage_pct_for_segment": coverage_pct,
                    "josh_renderability": josh_summary["renderability"],
                    "gvhmr_renderability": gvhmr_summary["renderability"],
                    "mpjpe_mean_mm": comparison["mpjpe_mean_mm"] if comparison else None,
                    "gt_status": window_gt_status,
                    "failure_tags": "|".join(failure_tags),
                    "recommendation": recommendation,
                }
            )

        best = evaluated[0] if evaluated else None
        if best is None:
            max_intersection = _max_intersection_frames(segment, josh_windows)
            best_failure_tags = classify_segment_without_window(
                segment.dance_type,
                coverage_pct,
                max_intersection,
                min_window_frames,
            )
            best_recommendation = recommend_segment_without_window(
                segment.dance_type,
                best_failure_tags,
            )
        else:
            best_failure_tags = best["failure_tags"]
            best_recommendation = best["recommendation"]
        segment_reports.append(
            {
                **asdict(segment),
                "duration_frames": segment.local_end_frame_exclusive - segment.local_start_frame,
                "duration_s": round((segment.local_end_frame_exclusive - segment.local_start_frame) / fps, 3),
                "josh_valid_coverage_pct": coverage_pct,
                "window_count": len(evaluated),
                "best_recommendation": best_recommendation,
                "best_failure_tags": best_failure_tags,
                "evaluated_windows": evaluated,
            }
        )

    benchmarkable_segments = sum(int(segment["window_count"] > 0) for segment in segment_reports)
    evaluated_windows = [window for segment in segment_reports for window in segment["evaluated_windows"]]
    report = {
        "sequence": asdict(sequence),
        "ground_truth_2d": {
            "status": gt_status,
            "frames_available": len(gt_frames),
        },
        "shot_boundaries": {
            "status": shot_boundaries_status,
            "count": len(shot_boundaries),
        },
        "inputs": {
            "josh_frames": int(josh_joints.shape[0]),
            "gvhmr_frames": int(gvhmr_joints.shape[0]),
            "fps": fps,
            "min_window_frames": min_window_frames,
            "josh_renderability": josh_meta.get("stats", {}).get("renderability"),
            "josh_recommended_windows": josh_meta.get("stats", {}).get("recommended_windows", []),
        },
        "summary": {
            "segments_total": len(segment_reports),
            "benchmarkable_segments": benchmarkable_segments,
            "evaluated_windows_total": len(evaluated_windows),
            "recommendation_counts": _count_by(evaluated_windows, "recommendation"),
            "dance_type_counts": _count_by(segment_reports, "dance_type"),
        },
        "segments": segment_reports,
        "windows": flattened_rows,
    }
    return report


def load_josh_windows_from_meta(meta: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Like load_josh_windows() but works from an already-loaded metadata dict."""
    windows = []
    for window in meta.get("stats", {}).get("windows", []):
        start = int(window["start_frame"])
        end_inclusive = int(window["end_frame"])
        windows.append(
            {
                "start_frame": start,
                "end_frame_inclusive": end_inclusive,
                "end_frame_exclusive": end_inclusive + 1,
                "n_frames": int(window["n_frames"]),
            }
        )
    return meta, windows


def render_markdown_report(report: dict[str, Any]) -> str:
    """Create a readable Markdown summary."""
    seq = report["sequence"]
    summary = report["summary"]
    gt = report["ground_truth_2d"]
    shot_boundaries = report.get("shot_boundaries", {"status": "available", "count": 0})
    lines = [
        f"# BRACE Benchmark — {seq['video_id']} seq {seq['seq_idx']} ({seq['dancer']})",
        "",
        "## Summary",
        "",
        f"- Sequence UID: `{seq['uid']}`",
        f"- Frames: `{seq['start_frame']}–{seq['end_frame_exclusive']}` (end exclusive)",
        f"- JOSH renderability: `{report['inputs']['josh_renderability']}`",
        f"- Benchmarkable segments: `{summary['benchmarkable_segments']}/{summary['segments_total']}`",
        f"- Evaluated windows: `{summary['evaluated_windows_total']}`",
        f"- BRACE 2D status: `{gt['status']}` ({gt['frames_available']} frames loaded)",
        f"- Shot-boundary annotations: `{shot_boundaries['status']}` ({shot_boundaries['count']} local boundaries)",
        "",
        "## Segment Table",
        "",
        "| Segment | Local Frames | Type | JOSH Coverage | Windows | Recommendation | Failure Tags |",
        "|---------|--------------|------|---------------|---------|----------------|--------------|",
    ]
    for segment in report["segments"]:
        tags = ", ".join(segment["best_failure_tags"])
        lines.append(
            f"| `{segment['uid']}` | "
            f"`{segment['local_start_frame']}–{segment['local_end_frame_exclusive']}` | "
            f"{segment['dance_type']} | "
            f"{segment['josh_valid_coverage_pct']}% | "
            f"{segment['window_count']} | "
            f"{segment['best_recommendation']} | "
            f"{tags} |"
        )
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- `window_ready` means JOSH is usable only on selected contiguous windows, not across the whole sequence.",
            (
                "- BRACE 2D keypoints are available locally and are used whenever they overlap the evaluated window."
                if gt["status"] != "unavailable"
                else "- BRACE 2D keypoints are unavailable locally, so the report is structural-only."
            ),
            "- Benchmark recommendations are intended to decide between JOSH tuning, stronger priors, or richer capture before further reruns.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_benchmark_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and CSV outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    md_path = output_dir / "benchmark.md"
    csv_path = output_dir / "windows.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write(render_markdown_report(report))

    fieldnames = [
        "segment_uid",
        "dance_type",
        "local_start_frame",
        "local_end_frame_exclusive",
        "n_frames",
        "josh_coverage_pct_for_segment",
        "josh_renderability",
        "gvhmr_renderability",
        "mpjpe_mean_mm",
        "gt_status",
        "failure_tags",
        "recommendation",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["windows"]:
            writer.writerow(row)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
    }
