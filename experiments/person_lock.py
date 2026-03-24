"""
Person Lock: split multi-person GVHMR output into stable single-person segments.

Detects frame discontinuities (>threshold pelvis jump) and outputs
per-segment .npy files with metadata.

Usage:
    python experiments/person_lock.py [--input path] [--threshold 0.15]
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def detect_segments(
    joints: np.ndarray,
    fps: float = 30.0,
    threshold_m: float = 0.15,
    min_duration_s: float = 1.0,
) -> list[dict]:
    """
    Split joint sequence into stable single-person segments.

    Args:
        joints: (F, J, 3) joint positions in meters
        fps: frame rate
        threshold_m: max pelvis displacement between frames (meters)
        min_duration_s: minimum segment duration to keep

    Returns:
        List of segment dicts with start/end frames and metadata
    """
    pelvis = joints[:, 0, :]  # joint 0 = pelvis
    frame_diff = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)

    # Find discontinuity frames
    switches = np.where(frame_diff > threshold_m)[0]
    min_frames = int(min_duration_s * fps)

    segments = []
    start = 0
    for sw in switches:
        if sw - start >= min_frames:
            segments.append(_make_segment(joints, start, sw, fps))
        start = sw + 1

    # Final segment
    if joints.shape[0] - start >= min_frames:
        segments.append(_make_segment(joints, start, joints.shape[0] - 1, fps))

    return segments


def _make_segment(joints: np.ndarray, start: int, end: int, fps: float) -> dict:
    """Build segment metadata."""
    seg_joints = joints[start:end + 1]
    pelvis_y = seg_joints[:, 0, 1]
    pelvis_xz = seg_joints[:, 0, [0, 2]]

    # Smoothness: mean frame-to-frame pelvis displacement
    p = seg_joints[:, 0, :]
    smoothness = float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())

    return {
        "start_frame": int(start),
        "end_frame": int(end),
        "n_frames": int(end - start + 1),
        "duration_s": round((end - start + 1) / fps, 2),
        "height_min": round(float(pelvis_y.min()), 3),
        "height_max": round(float(pelvis_y.max()), 3),
        "height_range": round(float(pelvis_y.max() - pelvis_y.min()), 3),
        "xz_range_x": round(float(pelvis_xz[:, 0].max() - pelvis_xz[:, 0].min()), 3),
        "xz_range_z": round(float(pelvis_xz[:, 1].max() - pelvis_xz[:, 1].min()), 3),
        "smoothness": round(smoothness, 4),
    }


def lock_and_save(
    input_path: str,
    output_dir: str,
    fps: float = 30.0,
    threshold_m: float = 0.15,
) -> list[dict]:
    """Run person lock and save segments."""
    joints = np.load(input_path)
    print(f"Loaded {input_path}: {joints.shape}")

    segments = detect_segments(joints, fps, threshold_m)
    print(f"Found {len(segments)} stable segments (threshold={threshold_m}m, min=1s)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        s, e = seg["start_frame"], seg["end_frame"]
        seg_joints = joints[s:e + 1]
        fname = f"seg_{i:02d}_f{s}-{e}.npy"
        np.save(out / fname, seg_joints)
        seg["file"] = fname
        print(f"  Seg {i:2d}: frames {s:5d}-{e:5d} ({seg['duration_s']:5.1f}s) "
              f"h=[{seg['height_min']:.2f},{seg['height_max']:.2f}]m "
              f"smooth={seg['smoothness']:.4f}")

    with open(out / "segments.json", "w") as f:
        json.dump({"source": str(input_path), "fps": fps, "threshold_m": threshold_m,
                    "n_segments": len(segments), "segments": segments}, f, indent=2)

    print(f"\nSaved to {out}/")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Lock — split GVHMR output into segments")
    parser.add_argument("--input", default="experiments/results/joints_3d_REAL.npy")
    parser.add_argument("--output", default="experiments/results/locked")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    lock_and_save(args.input, args.output, args.fps, args.threshold)
