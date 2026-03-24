"""JOSH vs GVHMR comparison — Procrustes alignment, MPJPE, and diagnostic tests."""
import argparse
import json
import os

import numpy as np

from .config import HEAD_IDX, IDENTITY_TRACKING_THRESHOLD_M, PELVIS_IDX


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid Procrustes alignment (rotation + translation, no scale)."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_c = source - mu_s
    t_c = target - mu_t
    H = s_c.T @ t_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return (source - mu_s) @ R.T + mu_t


def align_sequences(josh: np.ndarray, gvhmr: np.ndarray, ref_frames: int = 30) -> np.ndarray:
    """Align JOSH to GVHMR using Procrustes on first N frames."""
    n = min(len(josh), len(gvhmr), ref_frames)
    src = josh[:n].reshape(-1, 3)
    tgt = gvhmr[:n].reshape(-1, 3)
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    H = (src - mu_s).T @ (tgt - mu_t)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    aligned = np.zeros_like(josh)
    for f in range(len(josh)):
        aligned[f] = (josh[f] - mu_s) @ R.T + mu_t
    return aligned


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame Mean Per-Joint Position Error in mm."""
    n = min(len(pred), len(gt))
    errors = np.linalg.norm(pred[:n] - gt[:n], axis=-1)  # (F, J)
    return errors.mean(axis=1) * 1000  # mm


def inversion_test(joints: np.ndarray, y_down: bool = False) -> dict:
    """Detect frames where head is below pelvis."""
    head_y = joints[:, HEAD_IDX, 1]
    pelvis_y = joints[:, PELVIS_IDX, 1]
    if y_down:
        inverted = int(np.sum(head_y > pelvis_y))
    else:
        inverted = int(np.sum(head_y < pelvis_y))
    return {
        "inverted_frames": inverted,
        "total_frames": len(joints),
        "inverted_pct": round(100 * inverted / len(joints), 1),
    }


def identity_tracking_test(joints: np.ndarray) -> dict:
    """Detect large root displacement jumps (identity swap indicator)."""
    root = joints[:, PELVIS_IDX, :]
    displacements = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    max_disp = float(np.max(displacements))
    return {
        "max_root_displacement_m": round(max_disp, 3),
        "max_displacement_frame": int(np.argmax(displacements)),
        "pass": max_disp < IDENTITY_TRACKING_THRESHOLD_M,
    }


def stage_bounds_test(joints: np.ndarray, max_range_m: float = 10.0) -> dict:
    """Check if trajectory stays within reasonable stage bounds."""
    root = joints[:, PELVIS_IDX, :]
    ranges = {ax: float(root[:, i].ptp()) for i, ax in enumerate("xyz")}
    return {
        "ranges_m": {k: round(v, 2) for k, v in ranges.items()},
        "pass": all(v < max_range_m for v in ranges.values()),
    }


def run_comparison(josh_joints: np.ndarray, gvhmr_joints: np.ndarray, fps: float = 30.0) -> dict:
    """Run full comparison suite. Returns summary dict."""
    aligned = align_sequences(josh_joints, gvhmr_joints)
    mpjpe = compute_mpjpe(aligned, gvhmr_joints)

    return {
        "mpjpe_mean_mm": round(float(mpjpe.mean()), 1),
        "mpjpe_median_mm": round(float(np.median(mpjpe)), 1),
        # TODO: verify y_down — TRAM pred_trans is Y-up; JOSH SMPL output may be too
        "josh_inversion": inversion_test(josh_joints, y_down=True),
        "gvhmr_inversion": inversion_test(gvhmr_joints),
        "josh_identity": identity_tracking_test(josh_joints),
        "gvhmr_identity": identity_tracking_test(gvhmr_joints),
        "josh_bounds": stage_bounds_test(josh_joints),
        "gvhmr_bounds": stage_bounds_test(gvhmr_joints),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare JOSH vs GVHMR joints")
    parser.add_argument("--josh", required=True, help="JOSH joints .npy")
    parser.add_argument("--gvhmr", required=True, help="GVHMR joints .npy")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default="comparison.json")
    args = parser.parse_args()

    josh = np.load(args.josh)
    gvhmr = np.load(args.gvhmr)
    result = run_comparison(josh, gvhmr, fps=args.fps)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
