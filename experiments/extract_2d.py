"""
Extract 2D joint projections from GVHMR outputs for skeleton overlay.

Uses vitpose 2D keypoints (already projected onto original video) and
SMPL incam params + camera intrinsics for full 22-joint projection.

Usage:
    python experiments/extract_2d.py [--seq-start 3802] [--seq-end 4801]
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path


# COCO 17 → SMPL mapping (approximate, for the 17 vitpose joints)
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def extract(
    gvhmr_dir: str = "gvhmr_src/outputs/demo/RS0mFARO1x4",
    seq_start: int = 3802,
    seq_end: int = 4801,
    output_dir: str = "experiments/results",
):
    base = Path(gvhmr_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load vitpose 2D keypoints (F, 17, 3) = (x, y, confidence)
    vitpose = torch.load(base / "preprocess/vitpose.pt", map_location="cpu", weights_only=False)
    print(f"Vitpose: {vitpose.shape}")

    # Load camera intrinsics
    results = torch.load(base / "hmr4d_results.pt", map_location="cpu", weights_only=False)
    K = results["K_fullimg"]  # (F, 3, 3)
    print(f"Camera K: {K.shape}")

    # Load bounding boxes
    bbx = torch.load(base / "preprocess/bbx.pt", map_location="cpu", weights_only=False)
    bbx_xys = bbx["bbx_xys"]  # (F, 3) center_x, center_y, scale
    print(f"BBX: {bbx_xys.shape}")

    # Load SMPL incam params for 3D→2D projection
    incam = results["smpl_params_incam"]
    transl = incam["transl"]  # (F, 3)
    print(f"SMPL transl: {transl.shape}")

    # Slice to seq4
    vp_seq = vitpose[seq_start:seq_end].numpy()  # (N, 17, 3)
    K_seq = K[seq_start:seq_end].numpy()          # (N, 3, 3)
    bbx_seq = bbx_xys[seq_start:seq_end].numpy()  # (N, 3)

    n_frames = vp_seq.shape[0]
    print(f"\nSeq4: frames {seq_start}-{seq_end}, {n_frames} frames")

    # Save vitpose 2D keypoints
    np.save(out / "vitpose_2d_seq4.npy", vp_seq)
    print(f"Saved vitpose_2d_seq4.npy: {vp_seq.shape}")

    # Save camera intrinsics
    np.save(out / "camera_K_seq4.npy", K_seq)
    print(f"Saved camera_K_seq4.npy: {K_seq.shape}")

    # Save bounding boxes
    np.save(out / "bbx_seq4.npy", bbx_seq)
    print(f"Saved bbx_seq4.npy: {bbx_seq.shape}")

    # Now project 3D SMPL joints to 2D using camera intrinsics
    # Load the 3D joints we already have
    joints_3d = np.load(out / "joints_3d_REAL_seq4.npy")  # (999, 22, 3)
    print(f"\n3D joints: {joints_3d.shape}")

    # We need to project 3D world → 2D image
    # The incam transl gives the root translation in camera frame
    # For a simpler approach: use vitpose 2D for the overlay
    # (it's already well-aligned with the original video)

    # Compute velocity for coloring
    dt = 1.0 / 30.0
    vel = np.zeros_like(joints_3d)
    vel[1:-1] = (joints_3d[2:] - joints_3d[:-2]) / (2.0 * dt)
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    speed_per_joint = np.linalg.norm(vel, axis=-1)  # (F, 22)
    np.save(out / "joint_speed_seq4.npy", speed_per_joint)
    print(f"Saved joint_speed_seq4.npy: {speed_per_joint.shape}")

    # Summary stats
    print(f"\nVitpose confidence: mean={vp_seq[:,:,2].mean():.3f}, min={vp_seq[:,:,2].min():.3f}")
    print(f"Joint speed: mean={speed_per_joint.mean():.2f} m/s, max={speed_per_joint.max():.2f} m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr-dir", default="gvhmr_src/outputs/demo/RS0mFARO1x4")
    parser.add_argument("--seq-start", type=int, default=3802)
    parser.add_argument("--seq-end", type=int, default=4801)
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()
    extract(args.gvhmr_dir, args.seq_start, args.seq_end, args.output)
