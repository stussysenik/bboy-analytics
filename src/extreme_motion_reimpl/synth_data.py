"""Synthetic breakdance data generator for paper reimplementation experiments.

Generates 17-joint COCO-topology skeletons performing parametric breakdance
moves, point tracks, RGBD sequences, and correlated audio — all numpy-only.
"""
from __future__ import annotations

import numpy as np

# COCO 17-joint topology: indices and parent-child connectivity
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]
N_JOINTS = 17


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _t_pose() -> np.ndarray:
    """Return a (17, 3) T-pose skeleton in metres, Y-up, centred at origin."""
    joints = np.zeros((N_JOINTS, 3))
    joints[0] = [0, 1.7, 0]        # nose
    joints[1] = [-0.03, 1.75, 0]   # left_eye
    joints[2] = [0.03, 1.75, 0]    # right_eye
    joints[3] = [-0.08, 1.73, 0]   # left_ear
    joints[4] = [0.08, 1.73, 0]    # right_ear
    joints[5] = [-0.2, 1.5, 0]     # left_shoulder
    joints[6] = [0.2, 1.5, 0]      # right_shoulder
    joints[7] = [-0.45, 1.5, 0]    # left_elbow
    joints[8] = [0.45, 1.5, 0]     # right_elbow
    joints[9] = [-0.65, 1.5, 0]    # left_wrist
    joints[10] = [0.65, 1.5, 0]    # right_wrist
    joints[11] = [-0.1, 1.0, 0]    # left_hip
    joints[12] = [0.1, 1.0, 0]     # right_hip
    joints[13] = [-0.1, 0.5, 0]    # left_knee
    joints[14] = [0.1, 0.5, 0]     # right_knee
    joints[15] = [-0.1, 0.0, 0]    # left_ankle
    joints[16] = [0.1, 0.0, 0]     # right_ankle
    return joints


def _apply_move(base: np.ndarray, t: float, move_type: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a parametric breakdance move at normalised time t ∈ [0, 1]."""
    joints = base.copy()
    hip_centre = (joints[11] + joints[12]) / 2

    if move_type == "inversion":
        angle = np.pi * np.clip(2 * t, 0, 1)
        rot = _rotation_matrix("x", angle)
        joints = (joints - hip_centre) @ rot.T + hip_centre
        joints[:, 1] = np.maximum(joints[:, 1], 0.0)
    elif move_type == "rotation":
        rot = _rotation_matrix("y", 2 * np.pi * t)
        joints = (joints - hip_centre) @ rot.T + hip_centre
    elif move_type == "extreme-articulation":
        leg_angle = np.pi * 0.85 * np.sin(2 * np.pi * t)
        for idx in [13, 15]:
            pivot = joints[11]
            joints[idx] = pivot + _rotation_matrix("z", leg_angle) @ (joints[idx] - pivot)
        for idx in [14, 16]:
            pivot = joints[12]
            joints[idx] = pivot + _rotation_matrix("z", -leg_angle) @ (joints[idx] - pivot)
    elif move_type == "freeze":
        pass  # static pose, no modification
    elif move_type == "floor-contact":
        drop = 0.9 * np.sin(np.pi * t)
        joints[:, 1] = np.maximum(joints[:, 1] - drop, 0.0)

    noise = rng.normal(0, 0.005, joints.shape)
    return joints + noise


def generate_breakdance_sequence(
    n_frames: int = 128,
    fps: float = 30.0,
    move_type: str = "inversion",
    seed: int = 42,
) -> dict:
    """Generate a synthetic breakdance joint sequence.

    Returns dict with keys:
        joints_3d  (T, 17, 3) — 3D positions in metres
        joints_2d  (T, 17, 2) — perspective-projected 2D pixel coords
        visibility (T, 17)    — binary visibility per joint
        audio      (S,)       — synthetic audio waveform
        fps, sample_rate, move_type, inversion_mask (T,)
    """
    rng = np.random.default_rng(seed)
    base = _t_pose()
    joints_3d = np.zeros((n_frames, N_JOINTS, 3))
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        joints_3d[i] = _apply_move(base, t, move_type, rng)

    # Perspective projection (simple pinhole: focal=200, cx=cy=128)
    fx, fy, cx, cy = 200.0, 200.0, 128.0, 128.0
    cam_z_offset = 3.0
    pts = joints_3d.copy()
    pts[:, :, 2] += cam_z_offset
    depth = np.clip(pts[:, :, 2], 0.1, None)
    joints_2d = np.zeros((n_frames, N_JOINTS, 2))
    joints_2d[:, :, 0] = fx * pts[:, :, 0] / depth + cx
    joints_2d[:, :, 1] = fy * (-pts[:, :, 1]) / depth + cy  # flip Y for image coords

    # Visibility: occluded when behind camera or out of [0,256] frame
    visibility = np.ones((n_frames, N_JOINTS), dtype=np.float64)
    visibility[joints_2d[:, :, 0] < 0] = 0.0
    visibility[joints_2d[:, :, 0] > 256] = 0.0
    visibility[joints_2d[:, :, 1] < 0] = 0.0
    visibility[joints_2d[:, :, 1] > 256] = 0.0

    # Inversion mask: head below hip centre
    hip_y = (joints_3d[:, 11, 1] + joints_3d[:, 12, 1]) / 2
    head_y = joints_3d[:, 0, 1]
    inversion_mask = (head_y < hip_y).astype(np.float64)

    # Synthetic audio correlated with motion energy
    sample_rate = 16000
    velocity = np.diff(joints_3d, axis=0, prepend=joints_3d[:1]) * fps
    motion_energy = np.linalg.norm(velocity, axis=-1).mean(axis=1)
    n_audio = int(n_frames * sample_rate / fps)
    envelope = np.interp(np.linspace(0, 1, n_audio), np.linspace(0, 1, n_frames), motion_energy)
    envelope = envelope / (envelope.max() + 1e-8)
    carrier = np.sin(2 * np.pi * 220 * np.arange(n_audio) / sample_rate)
    audio = envelope * carrier + rng.normal(0, 0.05, n_audio)

    return {
        "joints_3d": joints_3d,
        "joints_2d": joints_2d,
        "visibility": visibility,
        "audio": audio,
        "fps": fps,
        "sample_rate": sample_rate,
        "move_type": move_type,
        "inversion_mask": inversion_mask,
    }


def generate_point_tracks(
    joints_2d: np.ndarray,
    visibility: np.ndarray,
    n_extra: int = 48,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dense point tracks from joint trajectories.

    Returns (T, N, 2) tracks and (T, N) visibility.
    """
    rng = np.random.default_rng(seed)
    T = joints_2d.shape[0]
    tracks_list = [joints_2d]
    vis_list = [visibility]

    # Interpolated limb midpoints
    for i, j in SKELETON_EDGES:
        mid = (joints_2d[:, i:i + 1] + joints_2d[:, j:j + 1]) / 2
        mid_vis = np.minimum(visibility[:, i:i + 1], visibility[:, j:j + 1])
        tracks_list.append(mid)
        vis_list.append(mid_vis)

    # Extra surface points: small random offsets from random joints
    for _ in range(n_extra):
        idx = rng.integers(0, N_JOINTS)
        offset = rng.normal(0, 3.0, (T, 1, 2))
        tracks_list.append(joints_2d[:, idx:idx + 1] + offset)
        vis_list.append(visibility[:, idx:idx + 1])

    return np.concatenate(tracks_list, axis=1), np.concatenate(vis_list, axis=1)


def generate_rgbd_sequence(
    n_frames: int = 32,
    H: int = 64,
    W: int = 64,
    seed: int = 42,
) -> dict:
    """Generate synthetic RGBD frames with ground-truth dancer masks.

    Returns dict with keys:
        rgb       (T, H, W, 3) — float64 [0, 1]
        depth     (T, H, W)    — float64, metres
        masks_gt  (T, H, W)    — binary dancer mask
        joints_3d (T, 17, 3)
        audio     (S,)
        intrinsics (3, 3)
        fps, sample_rate
    """
    data = generate_breakdance_sequence(n_frames=n_frames, fps=30.0, move_type="inversion", seed=seed)
    joints_3d = data["joints_3d"]

    fx, fy, cx, cy = 50.0, 50.0, W / 2, H / 2
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cam_z = 3.0

    rgb = np.zeros((n_frames, H, W, 3))
    depth = np.full((n_frames, H, W), cam_z + 2.0)  # background depth
    masks_gt = np.zeros((n_frames, H, W), dtype=np.float64)

    # Background gradient
    bg_y = np.linspace(0.15, 0.25, H)[:, None]
    for t in range(n_frames):
        rgb[t, :, :, 0] = bg_y
        rgb[t, :, :, 1] = bg_y * 0.8
        rgb[t, :, :, 2] = bg_y * 1.2

        # Project each joint as a small disc onto the image
        for j in range(N_JOINTS):
            pt = joints_3d[t, j].copy()
            pt[2] += cam_z
            if pt[2] < 0.1:
                continue
            px = int(fx * pt[0] / pt[2] + cx)
            py = int(fy * (-pt[1]) / pt[2] + cy)
            r = max(int(4.0 / pt[2] * fx / 50), 1)
            y_lo, y_hi = max(py - r, 0), min(py + r + 1, H)
            x_lo, x_hi = max(px - r, 0), min(px + r + 1, W)
            if y_lo >= y_hi or x_lo >= x_hi:
                continue
            rgb[t, y_lo:y_hi, x_lo:x_hi] = [0.8, 0.3, 0.2]
            depth[t, y_lo:y_hi, x_lo:x_hi] = pt[2]
            masks_gt[t, y_lo:y_hi, x_lo:x_hi] = 1.0

    return {
        "rgb": np.clip(rgb, 0, 1),
        "depth": depth,
        "masks_gt": masks_gt,
        "joints_3d": joints_3d,
        "audio": data["audio"],
        "intrinsics": intrinsics,
        "fps": data["fps"],
        "sample_rate": data["sample_rate"],
        "inversion_mask": data["inversion_mask"],
    }
