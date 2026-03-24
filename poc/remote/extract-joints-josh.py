"""
Extract 3D joint positions from JOSH output.

JOSH saves per-track .npy dicts with:
  pred_rotmat: (N, 24, 3, 3) — rotation matrices
  pred_shape:  (N, 10) — SMPL betas
  pred_trans:  (N, 3) — world translation
  frame:       list[int] — frame indices

This script runs SMPL forward kinematics and outputs joints_3d.npy
in the same (F, 22, 3) format as GVHMR's extract-joints.py.

Usage:
  python extract-joints-josh.py --josh-dir ~/josh/outputs/bcone_seq2 [--fps 25]
  python extract-joints-josh.py --josh-dir ~/josh/outputs/bcone_seq2 --output results/joints_3d_josh.npy
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import smplx
import torch


def find_josh_outputs(josh_dir: str) -> list[str]:
    """Find JOSH per-track .npy output files.

    JOSH saves optimized results to {input_folder}/josh/*.npy
    (same filenames as tram/ but in josh/ subfolder).
    """
    patterns = [
        os.path.join(josh_dir, "josh", "hps_track_*.npy"),
        os.path.join(josh_dir, "hps_track_*.npy"),
        os.path.join(josh_dir, "josh", "*.npy"),
        os.path.join(josh_dir, "*.npy"),
    ]
    files = []
    for p in patterns:
        found = glob.glob(p, recursive=True)
        files.extend(f for f in found if "tram" not in f and "deco" not in f and "sam3" not in f)
    # Deduplicate and sort
    files = sorted(set(files))
    return files


def load_josh_track(npy_path: str) -> dict:
    """Load a single JOSH track .npy file."""
    data = np.load(npy_path, allow_pickle=True).item()
    required_keys = ["pred_rotmat", "pred_shape", "pred_trans"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npy_path}. Found: {list(data.keys())}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Extract 3D joints from JOSH output")
    parser.add_argument("--josh-dir", required=True, help="JOSH output directory")
    parser.add_argument("--output", default=None, help="Output .npy path (default: <josh-dir>/joints_3d_josh.npy)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS")
    parser.add_argument("--body-model-path", default=None,
                        help="Path to SMPL body models (default: ~/gvhmr/inputs/checkpoints/body_models)")
    parser.add_argument("--track-id", type=int, default=0,
                        help="Which track to extract (0 = primary dancer)")
    args = parser.parse_args()

    josh_dir = os.path.expanduser(args.josh_dir)
    output_path = args.output or os.path.join(josh_dir, "joints_3d_josh.npy")
    # JOSH uses data/smpl/ for SMPL models. Try JOSH path first, then GVHMR fallback.
    body_model_path = args.body_model_path
    if not body_model_path:
        josh_smpl = os.path.expanduser("~/josh/data/smpl")
        gvhmr_models = os.path.expanduser("~/gvhmr/inputs/checkpoints/body_models")
        body_model_path = josh_smpl if os.path.isdir(josh_smpl) else gvhmr_models

    # ─── Find JOSH output files ──────────────────────────────
    print("╔══════════════════════════════════════════╗")
    print("║  JOSH Joint Extraction                    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  JOSH dir: {josh_dir}")

    npy_files = find_josh_outputs(josh_dir)
    if not npy_files:
        print(f"\nERROR: No .npy files found in {josh_dir}")
        print("  JOSH output files expected: hps_track_*.npy")
        print("  Listing directory contents:")
        for root, dirs, files in os.walk(josh_dir):
            for f in files:
                path = os.path.join(root, f)
                print(f"    {path} ({os.path.getsize(path) / 1024:.1f} KB)")
        sys.exit(1)

    print(f"  Found {len(npy_files)} output file(s):")
    for f in npy_files:
        print(f"    {f} ({os.path.getsize(f) / 1024:.1f} KB)")

    # ─── Load track data ─────────────────────────────────────
    if args.track_id >= len(npy_files):
        print(f"\nERROR: Track {args.track_id} not found. Only {len(npy_files)} tracks available.")
        sys.exit(1)

    track_file = npy_files[args.track_id]
    print(f"\n▸ Loading track {args.track_id}: {os.path.basename(track_file)}")

    data = load_josh_track(track_file)
    pred_rotmat = torch.tensor(data["pred_rotmat"], dtype=torch.float32)  # (N, 24, 3, 3)
    pred_shape = torch.tensor(data["pred_shape"], dtype=torch.float32)    # (N, 10)
    pred_trans = torch.tensor(data["pred_trans"], dtype=torch.float32)    # (N, 1, 3) or (N, 3)
    if pred_trans.ndim == 3:
        pred_trans = pred_trans.squeeze(1)  # (N, 3) — JOSH saves with extra dim
    frame_indices = data.get("frame", list(range(pred_rotmat.shape[0])))
    if isinstance(frame_indices, np.ndarray):
        frame_indices = frame_indices.tolist()

    n_frames = pred_rotmat.shape[0]
    print(f"  Frames: {n_frames}")
    print(f"  Rotmat shape: {pred_rotmat.shape}")
    print(f"  Shape params: {pred_shape.shape}")
    print(f"  Translation: {pred_trans.shape}")
    if isinstance(frame_indices, (list, np.ndarray)) and len(frame_indices) > 0:
        print(f"  Frame range: {frame_indices[0]} - {frame_indices[-1]}")

    # ─── Load SMPL body model ────────────────────────────────
    print(f"\n▸ Loading SMPL body model from {body_model_path}")

    # JOSH uses SMPL (not SMPLX)
    for model_type in ["smpl", "smplx"]:
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
            if model_type == "smplx":
                print(f"ERROR: Could not load body model from {body_model_path}: {e}")
                print("  JOSH needs SMPL .pkl files. Check:")
                print(f"    {body_model_path}/smpl/SMPL_NEUTRAL.pkl")
                print(f"    {body_model_path}/smplx/SMPLX_NEUTRAL.npz")
                sys.exit(1)

    # ─── Forward kinematics ──────────────────────────────────
    print("\n▸ Running forward kinematics...")

    # JOSH stores rotation matrices: global_orient = rotmat[:, [0]], body_pose = rotmat[:, 1:]
    global_orient = pred_rotmat[:, [0]]  # (N, 1, 3, 3)
    body_pose = pred_rotmat[:, 1:]       # (N, 23, 3, 3)

    with torch.no_grad():
        output = body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=pred_shape,
            transl=pred_trans,
            pose2rot=False,  # JOSH provides rotation matrices, not axis-angle
        )

    # Take first 22 joints to match GVHMR convention
    # SMPL returns 45 joints (24 body + 21 extra regressed)
    # SMPLX returns 55+ joints
    # First 22 are anatomically equivalent in both
    joints_3d = output.joints[:, :22, :].numpy()  # (F, 22, 3)
    print(f"  Joint positions: {joints_3d.shape} (frames x joints x xyz)")

    # ─── Save ────────────────────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, joints_3d)

    # Compute stats
    velocities = np.diff(joints_3d, axis=0)
    speeds = np.linalg.norm(velocities, axis=-1)
    root_trajectory = joints_3d[:, 0, :]

    # Root displacement test (identity tracking)
    root_displacements = np.linalg.norm(np.diff(root_trajectory, axis=0), axis=-1)
    max_displacement = float(np.max(root_displacements))
    max_disp_frame = int(np.argmax(root_displacements))

    # Inversion test (head below pelvis)
    head_y = joints_3d[:, 15, 1]   # head joint, y coordinate
    pelvis_y = joints_3d[:, 0, 1]  # pelvis joint, y coordinate
    inverted_frames = int(np.sum(head_y < pelvis_y))

    metadata = {
        "video": os.path.basename(josh_dir),
        "n_frames": int(n_frames),
        "n_joints": 22,
        "joint_unit": "meters",
        "coordinate_system": "world (JOSH, first camera as origin, RDF)",
        "model": "JOSH (ICLR 2026)",
        "body_model": model_type,
        "fps": args.fps,
        "frame_indices": [int(f) for f in frame_indices] if isinstance(frame_indices, (list, np.ndarray)) else [],
        "stats": {
            "mean_speed_m_per_frame": float(np.mean(speeds)),
            "max_speed_m_per_frame": float(np.max(speeds)),
            "root_trajectory_range_m": {
                "x": float(root_trajectory[:, 0].ptp()),
                "y": float(root_trajectory[:, 1].ptp()),
                "z": float(root_trajectory[:, 2].ptp()),
            },
            "duration_seconds": float(n_frames / args.fps),
            "max_root_displacement_m": max_displacement,
            "max_root_displacement_frame": max_disp_frame,
            "identity_tracking": "PASS" if max_displacement < 0.3 else "FAIL",
            "inverted_frames": inverted_frames,
            "inverted_fraction": round(inverted_frames / n_frames, 3),
        },
    }

    meta_path = output_path.replace(".npy", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved: {output_path} ({joints_3d.nbytes / 1024:.1f} KB)")
    print(f"  Saved: {meta_path}")
    print(f"\n  Root trajectory range: "
          f"x={metadata['stats']['root_trajectory_range_m']['x']:.2f}m, "
          f"y={metadata['stats']['root_trajectory_range_m']['y']:.2f}m, "
          f"z={metadata['stats']['root_trajectory_range_m']['z']:.2f}m")
    print(f"  Max root displacement: {max_displacement:.3f}m at frame {max_disp_frame} "
          f"({'PASS' if max_displacement < 0.3 else 'FAIL — possible identity swap'})")
    print(f"  Inverted frames (head < pelvis): {inverted_frames}/{n_frames} "
          f"({inverted_frames/n_frames*100:.1f}%)")


if __name__ == "__main__":
    main()
