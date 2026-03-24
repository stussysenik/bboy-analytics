"""Joint extraction from JOSH and GVHMR SMPL output.

Both produce (F, 24, 3) numpy arrays in meters via SMPL forward kinematics.
"""
import argparse
import glob
import json
import os

import numpy as np
import smplx
import torch

from .config import (
    BODY_MODELS_DIR, HEAD_IDX, IDENTITY_TRACKING_THRESHOLD_M,
    PELVIS_IDX, SMPL_JOINT_COUNT, resolve_body_model_path,
)


def _load_body_model(body_model_path: str, n_frames: int, prefer: str = "smpl"):
    """Load SMPL or SMPLX body model with fallback."""
    order = ["smpl", "smplx"] if prefer == "smpl" else ["smplx", "smpl"]
    for model_type in order:
        try:
            model = smplx.create(
                str(body_model_path), model_type=model_type,
                gender="neutral", batch_size=n_frames,
            )
            return model, model_type
        except Exception:
            if model_type == order[-1]:
                raise
    raise RuntimeError("unreachable")


def _compute_stats(joints_3d: np.ndarray, fps: float) -> dict:
    """Compute velocity, displacement, and inversion stats."""
    velocities = np.diff(joints_3d, axis=0)
    speeds = np.linalg.norm(velocities, axis=-1)
    root = joints_3d[:, PELVIS_IDX, :]
    root_displacements = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    max_disp = float(np.max(root_displacements))
    head_y = joints_3d[:, HEAD_IDX, 1]
    pelvis_y = joints_3d[:, PELVIS_IDX, 1]
    inverted = int(np.sum(head_y < pelvis_y))
    n = joints_3d.shape[0]
    return {
        "mean_speed_m_per_frame": float(np.mean(speeds)),
        "max_speed_m_per_frame": float(np.max(speeds)),
        "root_trajectory_range_m": {
            "x": float(root[:, 0].ptp()),
            "y": float(root[:, 1].ptp()),
            "z": float(root[:, 2].ptp()),
        },
        "duration_seconds": float(n / fps),
        "max_root_displacement_m": max_disp,
        "max_root_displacement_frame": int(np.argmax(root_displacements)),
        "identity_tracking": "PASS" if max_disp < IDENTITY_TRACKING_THRESHOLD_M else "FAIL",
        "inverted_frames": inverted,
        "inverted_fraction": round(inverted / n, 3),
    }


def extract_josh_joints(
    josh_dir: str,
    body_model_path: str | None = None,
    fps: float = 25.0,
    track_id: int = 0,
) -> tuple[np.ndarray, dict]:
    """Extract (F, 24, 3) joints from JOSH output via SMPL FK.

    Returns (joints_3d, metadata).
    """
    josh_dir = os.path.expanduser(josh_dir)
    bm_path = resolve_body_model_path(body_model_path)

    # Find output files
    patterns = [
        os.path.join(josh_dir, "josh", "hps_track_*.npy"),
        os.path.join(josh_dir, "hps_track_*.npy"),
        os.path.join(josh_dir, "josh", "*.npy"),
        os.path.join(josh_dir, "*.npy"),
    ]
    files = []
    for p in patterns:
        found = glob.glob(p)
        files.extend(f for f in found if "tram" not in f and "deco" not in f and "sam3" not in f)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No JOSH .npy output in {josh_dir}")
    if track_id >= len(files):
        raise IndexError(f"Track {track_id} not found, only {len(files)} tracks")

    data = np.load(files[track_id], allow_pickle=True).item()
    pred_rotmat = torch.tensor(data["pred_rotmat"], dtype=torch.float32)
    pred_shape = torch.tensor(data["pred_shape"], dtype=torch.float32)
    pred_trans = torch.tensor(data["pred_trans"], dtype=torch.float32)
    if pred_trans.ndim == 3:
        pred_trans = pred_trans.squeeze(1)
    frame_indices = data.get("frame", list(range(pred_rotmat.shape[0])))

    n_frames = pred_rotmat.shape[0]
    body_model, model_type = _load_body_model(str(bm_path), n_frames, prefer="smpl")

    with torch.no_grad():
        output = body_model(
            global_orient=pred_rotmat[:, [0]],
            body_pose=pred_rotmat[:, 1:],
            betas=pred_shape,
            transl=pred_trans,
            pose2rot=False,
        )

    joints_3d = output.joints[:, :SMPL_JOINT_COUNT, :].numpy()
    stats = _compute_stats(joints_3d, fps)

    metadata = {
        "source": os.path.basename(josh_dir),
        "n_frames": int(n_frames),
        "n_joints": SMPL_JOINT_COUNT,
        "joint_unit": "meters",
        "coordinate_system": "world (JOSH)",
        "model": "JOSH (ICLR 2026)",
        "body_model": model_type,
        "fps": fps,
        "stats": stats,
    }
    return joints_3d, metadata


def extract_gvhmr_joints(
    gvhmr_output: str,
    body_model_path: str | None = None,
    fps: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Extract (F, 24, 3) joints from GVHMR hmr4d_results.pt.

    Returns (joints_3d, metadata).
    """
    bm_path = resolve_body_model_path(body_model_path)
    results = torch.load(gvhmr_output, map_location="cpu")
    params = results["smpl_params_global"]
    n_frames = params["body_pose"].shape[0]

    body_model, model_type = _load_body_model(str(bm_path), n_frames, prefer="smpl")

    # GVHMR outputs 21 body joints (63 axis-angle params) but the full SMPL
    # model expects 23 body joints (69 params).  Pad with zeros for the two
    # hand joints (left_hand, right_hand) so FK runs on the full 24-joint
    # skeleton.
    body_pose = params["body_pose"]  # (F, 63)
    expected_dim = body_model.NUM_BODY_JOINTS * 3  # 23 * 3 = 69
    if body_pose.shape[1] < expected_dim:
        pad = torch.zeros(n_frames, expected_dim - body_pose.shape[1])
        body_pose = torch.cat([body_pose, pad], dim=1)

    with torch.no_grad():
        output = body_model(
            global_orient=params["global_orient"],
            body_pose=body_pose,
            betas=params["betas"],
            transl=params["transl"],
        )

    joints_3d = output.joints[:, :SMPL_JOINT_COUNT, :].numpy()
    stats = _compute_stats(joints_3d, fps)

    metadata = {
        "source": os.path.basename(gvhmr_output),
        "n_frames": int(n_frames),
        "n_joints": SMPL_JOINT_COUNT,
        "joint_unit": "meters",
        "coordinate_system": "gravity-view (Y=up)",
        "model": "GVHMR (SIGGRAPH Asia 2024)",
        "body_model": model_type,
        "fps": fps,
        "stats": stats,
    }
    return joints_3d, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D joints from JOSH or GVHMR output")
    sub = parser.add_subparsers(dest="source", required=True)

    josh_p = sub.add_parser("josh")
    josh_p.add_argument("--dir", required=True, help="JOSH output directory")
    josh_p.add_argument("--fps", type=float, default=25.0)
    josh_p.add_argument("--output", default=None)

    gvhmr_p = sub.add_parser("gvhmr")
    gvhmr_p.add_argument("--results", required=True, help="Path to hmr4d_results.pt")
    gvhmr_p.add_argument("--fps", type=float, default=30.0)
    gvhmr_p.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.source == "josh":
        joints, meta = extract_josh_joints(args.dir, fps=args.fps)
        out = args.output or os.path.join(args.dir, "joints_3d_josh.npy")
    else:
        joints, meta = extract_gvhmr_joints(args.results, fps=args.fps)
        out = args.output or "joints_3d_gvhmr.npy"

    np.save(out, joints)
    meta_path = out.replace(".npy", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {joints.shape} to {out}")
    print(f"Stats: {json.dumps(meta['stats'], indent=2)}")
