"""
Extract 3D joint positions from GVHMR output.

Usage: python extract-joints.py <video_basename> [--fps 30]
Output: results/joints_3d.npy, results/metadata.json
"""

import argparse
import json
import os
import sys

import numpy as np
import smplx
import torch

def main():
    parser = argparse.ArgumentParser(description="Extract 3D joints from GVHMR output")
    parser.add_argument("video_basename", nargs="?", default="test")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS for duration calculation")
    args = parser.parse_args()

    video_basename = args.video_basename
    fps = args.fps
    gvhmr_dir = os.environ.get("GVHMR_DIR", os.path.expanduser("~/gvhmr"))
    results_dir = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))

    output_dir = os.path.join(gvhmr_dir, "outputs", "demo", video_basename)
    results_path = os.path.join(output_dir, "hmr4d_results.pt")

    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    # Load GVHMR results
    results = torch.load(results_path, map_location="cpu")
    params = results["smpl_params_global"]
    n_frames = params["body_pose"].shape[0]
    print(f"  Loaded {n_frames} frames from GVHMR output")

    # Load SMPL body model
    body_model_path = os.path.join(gvhmr_dir, "inputs", "checkpoints", "body_models")

    for model_type in ["smplx", "smpl"]:
        try:
            body_model = smplx.create(
                body_model_path,
                model_type=model_type,
                gender="neutral",
                batch_size=n_frames,
            )
            print(f"  Using {model_type} body model")
            break
        except Exception as e:
            if model_type == "smpl":
                print(f"ERROR: Could not load any body model: {e}")
                sys.exit(1)

    # Forward kinematics
    pose_dim = 63 if model_type == "smpl" else min(params["body_pose"].shape[1], 63)
    with torch.no_grad():
        output = body_model(
            global_orient=params["global_orient"],
            body_pose=params["body_pose"][:, :pose_dim],
            betas=params["betas"],
            transl=params["transl"],
        )

    joints_3d = output.joints.numpy()  # (F, J, 3)
    print(f"  Joint positions: {joints_3d.shape} (frames × joints × xyz)")

    # Save joints
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "joints_3d.npy"), joints_3d)

    # Compute stats
    velocities = np.diff(joints_3d, axis=0)  # (F-1, J, 3)
    speeds = np.linalg.norm(velocities, axis=-1)  # (F-1, J)

    root_trajectory = joints_3d[:, 0, :]  # root joint (pelvis)

    metadata = {
        "video": video_basename,
        "n_frames": int(n_frames),
        "n_joints": int(joints_3d.shape[1]),
        "joint_unit": "meters",
        "coordinate_system": "gravity-view (Y=up, metric scale)",
        "model": "GVHMR (SIGGRAPH Asia 2024)",
        "body_model": model_type,
        "stats": {
            "mean_speed_m_per_frame": float(np.mean(speeds)),
            "max_speed_m_per_frame": float(np.max(speeds)),
            "root_trajectory_range_m": {
                "x": float(root_trajectory[:, 0].ptp()),
                "y": float(root_trajectory[:, 1].ptp()),
                "z": float(root_trajectory[:, 2].ptp()),
            },
            "duration_seconds": float(n_frames / fps),
            "fps": fps,
        },
    }

    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: joints_3d.npy ({joints_3d.nbytes / 1024:.1f} KB)")
    print(f"  Saved: metadata.json")
    print(f"  Root trajectory range: x={metadata['stats']['root_trajectory_range_m']['x']:.2f}m, "
          f"y={metadata['stats']['root_trajectory_range_m']['y']:.2f}m, "
          f"z={metadata['stats']['root_trajectory_range_m']['z']:.2f}m")


if __name__ == "__main__":
    main()
