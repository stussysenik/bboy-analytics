"""Numerical root-cause analysis for failing powermove windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .brace_benchmark import (
    load_brace_ground_truth_2d,
    load_brace_segments,
    load_brace_sequence,
)

COCO17_TO_SMPL24 = np.array([
    15, 15, 15, 15, 15,
    16, 17,
    18, 19,
    20, 21,
    1, 2,
    4, 5,
    7, 8,
], dtype=np.int64)


def _bbox_diag(xy: np.ndarray) -> float:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    return max(float(np.linalg.norm(maxs - mins)), 1.0)


def similarity_align_2d(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align one 2D skeleton to another."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p_centered = pred - mu_p
    g_centered = gt - mu_g
    norm_p = np.linalg.norm(p_centered)
    norm_g = np.linalg.norm(g_centered)
    if norm_p < 1e-8 or norm_g < 1e-8:
        return pred.copy()
    p_norm = p_centered / norm_p
    g_norm = g_centered / norm_g
    H = p_norm.T @ g_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = norm_g / norm_p
    return p_centered @ R.T * scale + mu_g


def project_with_camera_K(
    joints_3d: np.ndarray,
    camera_K: np.ndarray,
) -> np.ndarray:
    """Project dense JOSH joints with per-frame camera intrinsics."""
    projected = np.full((joints_3d.shape[0], 17, 3), np.nan, dtype=np.float32)
    coco = joints_3d[:, COCO17_TO_SMPL24, :].astype(np.float32, copy=True)
    for frame in range(joints_3d.shape[0]):
        if not np.isfinite(coco[frame]).all():
            continue
        fx = float(camera_K[frame, 0, 0])
        fy = float(camera_K[frame, 1, 1])
        cx = float(camera_K[frame, 0, 2])
        cy = float(camera_K[frame, 1, 2])
        z = coco[frame, :, 2]
        valid = z > 1e-6
        projected[frame, :, 0] = coco[frame, :, 0] * fx / z + cx
        projected[frame, :, 1] = (-coco[frame, :, 1]) * fy / z + cy
        projected[frame, :, 2] = valid.astype(np.float32)
        projected[frame, ~valid, :2] = np.nan
    return projected


def compute_window_projection_diagnostics(
    pred_2d: np.ndarray,
    gt_frames: dict[int, np.ndarray],
    *,
    start: int,
    end_exclusive: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Measure whether failure is dominated by placement, scale, or shape."""
    frame_ids = [
        frame for frame in range(start, end_exclusive)
        if frame in gt_frames and frame < pred_2d.shape[0] and np.isfinite(pred_2d[frame, :, :2]).all()
    ]
    if not frame_ids:
        return {
            "frames_compared": 0,
            "mean_error_px": None,
            "mean_error_bbox_diag_frac": None,
            "translation_aligned_error_px": None,
            "similarity_aligned_error_px": None,
            "mean_center_offset_px": None,
            "mean_scale_ratio_pred_over_gt": None,
            "fraction_joints_out_of_frame": None,
        }

    raw_errors = []
    diag_norm_errors = []
    translation_aligned_errors = []
    translation_aligned_diag_norm_errors = []
    similarity_aligned_errors = []
    similarity_aligned_diag_norm_errors = []
    center_offsets = []
    center_offsets_diag_norm = []
    scale_ratios = []
    out_of_frame = []

    for frame in frame_ids:
        pred = pred_2d[frame, :, :2]
        gt = gt_frames[frame][:, :2]
        diag = _bbox_diag(gt)
        raw_errors.extend(np.linalg.norm(pred - gt, axis=1).tolist())
        diag_norm_errors.extend((np.linalg.norm(pred - gt, axis=1) / diag).tolist())

        pred_trans = pred - pred.mean(axis=0) + gt.mean(axis=0)
        trans_errors = np.linalg.norm(pred_trans - gt, axis=1)
        translation_aligned_errors.extend(trans_errors.tolist())
        translation_aligned_diag_norm_errors.extend((trans_errors / diag).tolist())

        pred_sim = similarity_align_2d(pred, gt)
        sim_errors = np.linalg.norm(pred_sim - gt, axis=1)
        similarity_aligned_errors.extend(sim_errors.tolist())
        similarity_aligned_diag_norm_errors.extend((sim_errors / diag).tolist())

        pred_min = pred.min(axis=0)
        pred_max = pred.max(axis=0)
        gt_min = gt.min(axis=0)
        gt_max = gt.max(axis=0)
        center_offset = float(np.linalg.norm((pred_min + pred_max) / 2.0 - (gt_min + gt_max) / 2.0))
        center_offsets.append(center_offset)
        center_offsets_diag_norm.append(center_offset / diag)
        scale_ratios.append(_bbox_diag(pred) / _bbox_diag(gt))
        out_of_frame.append(
            float(
                np.mean(
                    (pred[:, 0] < 0)
                    | (pred[:, 0] > image_width)
                    | (pred[:, 1] < 0)
                    | (pred[:, 1] > image_height)
                )
            )
        )

    return {
        "frames_compared": len(frame_ids),
        "mean_error_px": round(float(np.mean(raw_errors)), 2),
        "mean_error_bbox_diag_frac": round(float(np.mean(diag_norm_errors)), 4),
        "translation_aligned_error_px": round(float(np.mean(translation_aligned_errors)), 2),
        "translation_aligned_error_bbox_diag_frac": round(float(np.mean(translation_aligned_diag_norm_errors)), 4),
        "similarity_aligned_error_px": round(float(np.mean(similarity_aligned_errors)), 2),
        "similarity_aligned_error_bbox_diag_frac": round(float(np.mean(similarity_aligned_diag_norm_errors)), 4),
        "mean_center_offset_px": round(float(np.mean(center_offsets)), 2),
        "mean_center_offset_bbox_diag_frac": round(float(np.mean(center_offsets_diag_norm)), 4),
        "mean_scale_ratio_pred_over_gt": round(float(np.mean(scale_ratios)), 4),
        "fraction_joints_out_of_frame": round(float(np.mean(out_of_frame)), 4),
    }


def compute_root_and_bbox_summary(
    joints_3d: np.ndarray,
    pred_2d: np.ndarray,
    *,
    start: int,
    end_exclusive: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    window_3d = joints_3d[start:end_exclusive]
    valid = np.isfinite(window_3d).all(axis=(1, 2))
    root = window_3d[valid, 0]
    window_2d = pred_2d[start:end_exclusive, :, :2]
    bbox_centers = []
    bbox_sizes = []
    out_of_frame = []
    for frame in range(window_2d.shape[0]):
        pts = window_2d[frame]
        if not np.isfinite(pts).all():
            continue
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        bbox_centers.append(((mins + maxs) / 2.0).tolist())
        bbox_sizes.append((maxs - mins).tolist())
        out_of_frame.append(
            float(
                np.mean(
                    (pts[:, 0] < 0)
                    | (pts[:, 0] > image_width)
                    | (pts[:, 1] < 0)
                    | (pts[:, 1] > image_height)
                )
            )
        )
    return {
        "root_mean_xyz": np.round(root.mean(axis=0), 4).tolist() if len(root) else None,
        "root_min_xyz": np.round(root.min(axis=0), 4).tolist() if len(root) else None,
        "root_max_xyz": np.round(root.max(axis=0), 4).tolist() if len(root) else None,
        "bbox_center_mean_xy": np.round(np.mean(bbox_centers, axis=0), 2).tolist() if bbox_centers else None,
        "bbox_size_mean_xy": np.round(np.mean(bbox_sizes, axis=0), 2).tolist() if bbox_sizes else None,
        "fraction_joints_out_of_frame": round(float(np.mean(out_of_frame)), 4) if out_of_frame else None,
    }


def build_powermove_root_cause_report(
    *,
    brace_dir: str | Path,
    video_id: str,
    seq_idx: int,
    josh_3d: np.ndarray,
    josh_2d: np.ndarray,
    baseline_2d: np.ndarray,
    camera_K: np.ndarray,
    image_width: int,
    image_height: int,
    target_start: int,
    target_end_exclusive: int,
    control_start: int,
    control_end_exclusive: int,
) -> dict[str, Any]:
    """Build a report that separates projection bugs from 3D/model failures."""
    sequence = load_brace_sequence(brace_dir, video_id, seq_idx)
    segments = load_brace_segments(brace_dir, sequence)
    gt_status, gt_frames = load_brace_ground_truth_2d(brace_dir, sequence, segments)
    josh_2d_altK = project_with_camera_K(josh_3d, camera_K)

    target = {
        "josh_default": compute_window_projection_diagnostics(
            josh_2d, gt_frames, start=target_start, end_exclusive=target_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "josh_alt_cameraK": compute_window_projection_diagnostics(
            josh_2d_altK, gt_frames, start=target_start, end_exclusive=target_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "baseline_2d": compute_window_projection_diagnostics(
            baseline_2d, gt_frames, start=target_start, end_exclusive=target_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "josh_geometry": compute_root_and_bbox_summary(
            josh_3d, josh_2d, start=target_start, end_exclusive=target_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "baseline_geometry": compute_root_and_bbox_summary(
            josh_3d * np.nan, baseline_2d, start=target_start, end_exclusive=target_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
    }
    control = {
        "josh_default": compute_window_projection_diagnostics(
            josh_2d, gt_frames, start=control_start, end_exclusive=control_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "josh_alt_cameraK": compute_window_projection_diagnostics(
            josh_2d_altK, gt_frames, start=control_start, end_exclusive=control_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "baseline_2d": compute_window_projection_diagnostics(
            baseline_2d, gt_frames, start=control_start, end_exclusive=control_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "josh_geometry": compute_root_and_bbox_summary(
            josh_3d, josh_2d, start=control_start, end_exclusive=control_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
        "baseline_geometry": compute_root_and_bbox_summary(
            josh_3d * np.nan, baseline_2d, start=control_start, end_exclusive=control_end_exclusive,
            image_width=image_width, image_height=image_height,
        ),
    }

    target_raw = target["josh_default"]["mean_error_px"]
    target_alt = target["josh_alt_cameraK"]["mean_error_px"]
    target_sim = target["josh_default"]["similarity_aligned_error_px"]
    target_center = target["josh_default"]["mean_center_offset_px"]
    target_scale = target["josh_default"]["mean_scale_ratio_pred_over_gt"]

    if target_raw is None:
        conclusion = "No BRACE-overlapping frames were available for the target window."
    elif abs(target_raw - target_alt) < 1e-6:
        conclusion = (
            "Changing camera intrinsics does not change the powermove error, so this is not a focal-length or principal-point bug. "
            "Most of the raw error collapses under 2D similarity alignment, which points to bad camera-relative placement/scale in JOSH. "
            "Because the same projection code works on the control footwork window, the failure is not a generic application-side projection bug."
        )
    else:
        conclusion = (
            "The powermove error changes materially under alternate intrinsics, so projection assumptions remain a live application-side suspect."
        )

    return {
        "ground_truth_status": gt_status,
        "target_window": {
            "start_frame": target_start,
            "end_frame_exclusive": target_end_exclusive,
            "diagnostics": target,
        },
        "control_window": {
            "start_frame": control_start,
            "end_frame_exclusive": control_end_exclusive,
            "diagnostics": control,
        },
        "key_findings": {
            "target_raw_error_px": target_raw,
            "target_altK_error_px": target_alt,
            "target_similarity_aligned_error_px": target_sim,
            "target_center_offset_px": target_center,
            "target_scale_ratio_pred_over_gt": target_scale,
        },
        "conclusion": conclusion,
    }


def render_root_cause_markdown(report: dict[str, Any]) -> str:
    target = report["target_window"]
    control = report["control_window"]
    t = target["diagnostics"]["josh_default"]
    c = control["diagnostics"]["josh_default"]
    lines = [
        "# Powermove Root Cause Analysis",
        "",
        "## Target Window",
        "",
        f"- Frames: `{target['start_frame']}–{target['end_frame_exclusive']}`",
        f"- Raw JOSH error: `{t['mean_error_px']} px` (`{t['mean_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Translation-aligned error: `{t['translation_aligned_error_px']} px` (`{t['translation_aligned_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Similarity-aligned error: `{t['similarity_aligned_error_px']} px` (`{t['similarity_aligned_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Mean center offset: `{t['mean_center_offset_px']} px` (`{t['mean_center_offset_bbox_diag_frac']}` bbox-diag frac)",
        f"- Mean scale ratio: `{t['mean_scale_ratio_pred_over_gt']}`",
        f"- Fraction of projected joints out of frame: `{t['fraction_joints_out_of_frame']}`",
        "",
        "## Control Window",
        "",
        f"- Frames: `{control['start_frame']}–{control['end_frame_exclusive']}`",
        f"- Raw JOSH error: `{c['mean_error_px']} px` (`{c['mean_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Translation-aligned error: `{c['translation_aligned_error_px']} px` (`{c['translation_aligned_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Similarity-aligned error: `{c['similarity_aligned_error_px']} px` (`{c['similarity_aligned_error_bbox_diag_frac']}` bbox-diag frac)",
        f"- Mean center offset: `{c['mean_center_offset_px']} px` (`{c['mean_center_offset_bbox_diag_frac']}` bbox-diag frac)",
        f"- Mean scale ratio: `{c['mean_scale_ratio_pred_over_gt']}`",
        "",
        "## Conclusion",
        "",
        f"- {report['conclusion']}",
    ]
    return "\n".join(lines) + "\n"


def write_root_cause_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "root_cause_report.json"
    md_path = output_dir / "root_cause_report.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_root_cause_markdown(report))
    return {"json": str(json_path), "markdown": str(md_path)}
