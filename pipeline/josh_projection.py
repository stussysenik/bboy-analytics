"""Project dense clip-aligned JOSH joints into COCO-17 image coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


COCO17_TO_SMPL24 = np.array([
    15, 15, 15, 15, 15,  # nose/eyes/ears -> head
    16, 17,              # shoulders
    18, 19,              # elbows
    20, 21,              # wrists
    1, 2,                # hips
    4, 5,                # knees
    7, 8,                # ankles
], dtype=np.int64)

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def infer_video_geometry(video_path: str | Path) -> dict[str, float]:
    """Infer full-frame geometry using TRAM's default focal heuristic."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions for {video_path}: {width}x{height}")
    focal = float(np.sqrt(width ** 2 + height ** 2))
    return {
        "width": width,
        "height": height,
        "focal": focal,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }


def project_dense_josh_to_coco17(
    joints_3d: np.ndarray,
    *,
    focal: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project dense JOSH joints into full-frame COCO-17 image coordinates.

    The dense JOSH artifact is stored Y-up. TRAM/JOSH camera projection expects
    image-space Y-down, so the vertical axis is flipped during projection.
    """
    if joints_3d.ndim != 3 or joints_3d.shape[1:] != (24, 3):
        raise ValueError(f"Expected dense JOSH joints with shape (F, 24, 3), got {joints_3d.shape}")

    coco_joints = joints_3d[:, COCO17_TO_SMPL24, :].astype(np.float32, copy=True)
    projected = np.full((joints_3d.shape[0], 17, 3), np.nan, dtype=np.float32)

    z = coco_joints[:, :, 2]
    valid = np.isfinite(coco_joints).all(axis=-1) & (z > 1e-6)
    projected[:, :, 0] = coco_joints[:, :, 0] * focal / z + cx
    projected[:, :, 1] = (-coco_joints[:, :, 1]) * focal / z + cy
    projected[:, :, 2] = valid.astype(np.float32)
    projected[~valid, :2] = np.nan
    return projected


def export_josh_projected_2d(
    *,
    joints_path: str | Path,
    video_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """Export dense JOSH COCO-17 projections plus projection metadata."""
    joints_path = Path(joints_path)
    video_path = Path(video_path)
    output_path = Path(output_path) if output_path is not None else joints_path.with_name("joints_2d_josh_coco.npy")

    joints_3d = np.load(joints_path)
    geometry = infer_video_geometry(video_path)
    projected = project_dense_josh_to_coco17(
        joints_3d,
        focal=geometry["focal"],
        cx=geometry["cx"],
        cy=geometry["cy"],
    )
    np.save(output_path, projected)

    meta = {
        "source_joints": str(joints_path),
        "source_video": str(video_path),
        "projection_model": "tram_full_frame_pinhole_from_dense_camera_space_joints",
        "coordinate_system": "input joints are Y-up camera-space; projection flips Y to image-down",
        "coco_names": COCO17_NAMES,
        "coco_to_smpl24": COCO17_TO_SMPL24.tolist(),
        "video_geometry": geometry,
        "frames": int(projected.shape[0]),
        "valid_frames": int(np.isfinite(projected[:, :, :2]).all(axis=(1, 2)).sum()),
    }
    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "joints_2d": str(output_path),
        "metadata": str(meta_path),
    }

