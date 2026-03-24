"""Typed data model for breaking analysis visualization.

Every visualization element maps to a typed structure.
Computes ALL derived state from raw joints_3d.npy — no ad-hoc rendering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import (
    BONE_PAIRS,
    CATEGORY_NAMES,
    JOINT_GROUPS,
    JOINT_WEIGHTS,
    validate_joints,
)

from .color_system import (
    CONTACT_SPEED_THRESHOLD,
    CONTACT_Y_THRESHOLD,
    JOINT_CLUSTER,
    TRAIL_LENGTH,
)

# ──────────────────────────────────────────────────────────────────────
# Kinetic chain layers (from BREAKING_KINETIC_CHAIN.md)
# ──────────────────────────────────────────────────────────────────────

KINETIC_LAYERS: Dict[str, List[int]] = {
    "engine":     [0, 1, 2, 3, 6, 9, 13, 14, 16, 17],
    "foundation": [7, 8, 10, 11, 12, 15, 20, 21, 22, 23],
    "expression": [4, 5, 18, 19],
}

_JOINT_LAYER: Dict[int, str] = {}
for _layer, _indices in KINETIC_LAYERS.items():
    for _idx in _indices:
        _JOINT_LAYER[_idx] = _layer

SMOOTH_SIGMA = 1.5  # Gaussian smoothing for velocity (frames)


# ──────────────────────────────────────────────────────────────────────
# Typed data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class JointState:
    idx: int
    position: np.ndarray       # (3,) world coords meters
    velocity: np.ndarray       # (3,) m/s
    speed: float               # ||velocity||
    cluster: str               # "legs"|"torso"|"arms"|"hands"|"head"
    layer: str                 # "engine"|"foundation"|"expression"
    is_contact: bool           # on ground?
    beat_aligned: bool = False # set by beat alignment pass


@dataclass
class BodyState:
    frame_idx: int
    timestamp: float           # seconds
    joints: List[JointState]   # len=24
    com: np.ndarray            # (3,) mass-weighted center
    com_velocity: np.ndarray   # (3,) m/s
    hip_com_offset: np.ndarray # com - pelvis
    hip_com_magnitude: float   # ||hip_com_offset||
    phase: str                 # current phase
    total_energy: float        # Σ(mass * speed²)/2
    musicality: float = 0.0   # set by musicality pass


@dataclass
class TemporalWindow:
    """Sliding window — the 'swarm' view of trailing positions."""
    maxlen: int = TRAIL_LENGTH
    _positions: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    _coms: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))

    def append(self, state: BodyState) -> None:
        positions = np.array([j.position for j in state.joints])  # (24, 3)
        self._positions.append(positions)
        self._coms.append(state.com.copy())

    @property
    def joint_trails(self) -> np.ndarray:
        """(N, 24, 3) — position history, oldest first."""
        if not self._positions:
            return np.empty((0, 24, 3))
        return np.array(list(self._positions))

    @property
    def com_trail(self) -> np.ndarray:
        """(N, 3) — COM trajectory, oldest first."""
        if not self._coms:
            return np.empty((0, 3))
        return np.array(list(self._coms))

    @property
    def length(self) -> int:
        return len(self._positions)


# ──────────────────────────────────────────────────────────────────────
# State computation — single pass from raw joints
# ──────────────────────────────────────────────────────────────────────

def compute_velocities(joints: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-joint velocities via smoothed finite differences.

    Args:
        joints: (T, 24, 3) world coordinates
        fps: frames per second

    Returns:
        (T, 24, 3) velocity in m/s
    """
    # Central differences, clamped at edges
    dt = 1.0 / fps
    vel = np.zeros_like(joints)
    if joints.shape[0] >= 3:
        vel[1:-1] = (joints[2:] - joints[:-2]) / (2 * dt)
        vel[0] = (joints[1] - joints[0]) / dt
        vel[-1] = (joints[-1] - joints[-2]) / dt
    elif joints.shape[0] == 2:
        v = (joints[1] - joints[0]) / dt
        vel[0] = v
        vel[1] = v
    # Gaussian smooth each joint axis
    for j in range(24):
        for ax in range(3):
            vel[:, j, ax] = gaussian_filter1d(vel[:, j, ax], sigma=SMOOTH_SIGMA)
    return vel


def compute_com(joints: np.ndarray) -> np.ndarray:
    """Mass-weighted center of mass. (T, 3)"""
    weights = JOINT_WEIGHTS[:joints.shape[1]]  # handle 22 vs 24
    weights = weights / weights.sum()
    return np.einsum("tjc,j->tc", joints, weights)


def classify_phase_simple(joints: np.ndarray, velocities: np.ndarray, fps: float) -> List[str]:
    """Simple phase classification from joint data.

    Heuristics based on analyze_motion.py CATEGORY_NAMES:
    - freeze: COM speed < 0.04 m/s for sustained period
    - power: high distal expressivity + high hip angular momentum
    - footwork: hip below 0.6m AND feet moving fast
    - toprock: upright stance (hip > 0.8m)
    - transition: everything else
    """
    T = joints.shape[0]
    phases: List[str] = []
    com = compute_com(joints)
    com_vel = np.zeros_like(com)
    dt = 1.0 / fps
    if T >= 3:
        com_vel[1:-1] = (com[2:] - com[:-2]) / (2 * dt)
        com_vel[0] = (com[1] - com[0]) / dt
        com_vel[-1] = (com[-1] - com[-2]) / dt

    com_speed = np.linalg.norm(com_vel, axis=1)
    speeds = np.linalg.norm(velocities, axis=2)  # (T, 24)

    # Ground-relative height (Y-up convention, portable across coordinate origins)
    feet_indices = [7, 8, 10, 11]  # ankles + feet
    ground_y = float(joints[:, feet_indices, 1].min())

    # Per-frame classification
    for t in range(T):
        pelvis_height = joints[t, 0, 1] - ground_y  # height above ground
        cs = com_speed[t]

        # Distal vs proximal speed ratio
        distal_idx = [10, 11, 15, 20, 21, 22, 23]
        proximal_idx = [0, 1, 2, 3, 6, 9]
        distal_speed = speeds[t, distal_idx].mean() if len(distal_idx) > 0 else 0
        proximal_speed = speeds[t, proximal_idx].mean() if len(proximal_idx) > 0 else 0

        # Freeze: very low COM speed
        if cs < 0.04:
            phases.append("freeze")
        # Power: high distal expressivity (hands/head moving fast relative to torso)
        elif distal_speed > 2.0 and (proximal_speed < 0.01 or distal_speed / max(proximal_speed, 0.01) > 3.0):
            phases.append("power")
        # Footwork: hip is low (ground-relative)
        elif pelvis_height < 0.6:
            phases.append("footwork")
        # Toprock: upright
        elif pelvis_height > 0.8:
            phases.append("toprock")
        else:
            phases.append("transition")

    # Smooth: median filter over 15-frame window to remove single-frame noise
    if T > 15:
        phase_to_int = {p: i for i, p in enumerate(CATEGORY_NAMES)}
        int_to_phase = {i: p for p, i in phase_to_int.items()}
        arr = np.array([phase_to_int.get(p, 4) for p in phases])
        from scipy.ndimage import median_filter
        arr_smooth = median_filter(arr, size=15).astype(int)
        phases = [int_to_phase.get(int(v), "transition") for v in arr_smooth]

    return phases


def compute_beat_alignment(
    velocities: np.ndarray,
    beats: np.ndarray,
    fps: float,
    tolerance_ms: float = 70.0,
) -> List[List[int]]:
    """For each beat, find which joints have velocity peaks within ±tolerance.

    Returns: list of length len(beats), each entry is list of joint indices that hit.
    """
    tolerance_frames = int(tolerance_ms / 1000.0 * fps)
    T = velocities.shape[0]
    speeds = np.linalg.norm(velocities, axis=2)  # (T, 24)

    results: List[List[int]] = []
    for beat_sec in beats:
        beat_frame = int(beat_sec * fps)
        if beat_frame < 0 or beat_frame >= T:
            results.append([])
            continue

        window_start = max(0, beat_frame - tolerance_frames)
        window_end = min(T, beat_frame + tolerance_frames + 1)
        hit_joints: List[int] = []

        for j in range(24):
            window = speeds[window_start:window_end, j]
            if len(window) == 0:
                continue
            peak_in_window = window.max()
            # Check if this is a local peak (higher than neighbors outside window)
            context_start = max(0, window_start - tolerance_frames)
            context_end = min(T, window_end + tolerance_frames)
            context = speeds[context_start:context_end, j]
            if peak_in_window >= np.percentile(context, 75):
                hit_joints.append(j)

        results.append(hit_joints)
    return results


def compute_body_states(
    joints: np.ndarray,
    fps: float,
    beats: Optional[np.ndarray] = None,
) -> List[BodyState]:
    """Compute fully typed BodyState for every frame.

    Single pass, single source of truth. Everything derived from joints_3d.
    """
    joints = validate_joints(joints)
    T = joints.shape[0]

    # Velocities
    velocities = compute_velocities(joints, fps)
    speeds = np.linalg.norm(velocities, axis=2)  # (T, 24)

    # COM
    com_all = compute_com(joints)  # (T, 3)
    com_vel = np.zeros_like(com_all)
    dt = 1.0 / fps
    if T >= 3:
        com_vel[1:-1] = (com_all[2:] - com_all[:-2]) / (2 * dt)
        com_vel[0] = (com_all[1] - com_all[0]) / dt
        com_vel[-1] = (com_all[-1] - com_all[-2]) / dt

    # Phase classification
    phases = classify_phase_simple(joints, velocities, fps)

    # Beat alignment
    beat_hits: List[List[int]] = []
    if beats is not None and len(beats) > 0:
        beat_hits = compute_beat_alignment(velocities, beats, fps)

    # Build per-beat lookup: frame → set of aligned joints
    beat_frame_joints: Dict[int, set] = {}
    if beats is not None:
        for i, beat_sec in enumerate(beats):
            bf = int(beat_sec * fps)
            tolerance_frames = int(70.0 / 1000.0 * fps)
            hit_set = set(beat_hits[i]) if i < len(beat_hits) else set()
            for f in range(max(0, bf - tolerance_frames), min(T, bf + tolerance_frames + 1)):
                if f not in beat_frame_joints:
                    beat_frame_joints[f] = set()
                beat_frame_joints[f].update(hit_set)

    # Ground level for contact detection (Y-up, ground-relative)
    feet_indices = [7, 8, 10, 11]
    ground_y = float(joints[:, feet_indices, 1].min())

    # Build states
    states: List[BodyState] = []
    for t in range(T):
        timestamp = t / fps
        pelvis_pos = joints[t, 0]
        com = com_all[t]
        hip_com = com - pelvis_pos
        aligned_joints = beat_frame_joints.get(t, set())

        joint_states: List[JointState] = []
        total_energy = 0.0
        for j in range(24):
            pos = joints[t, j]
            vel = velocities[t, j]
            spd = float(speeds[t, j])
            cluster = JOINT_CLUSTER.get(j, "torso")
            layer = _JOINT_LAYER.get(j, "expression")
            height_above_ground = pos[1] - ground_y
            is_contact = (height_above_ground < CONTACT_Y_THRESHOLD and spd < CONTACT_SPEED_THRESHOLD)
            beat_aligned = j in aligned_joints

            joint_states.append(JointState(
                idx=j,
                position=pos,
                velocity=vel,
                speed=spd,
                cluster=cluster,
                layer=layer,
                is_contact=is_contact,
                beat_aligned=beat_aligned,
            ))

            mass = JOINT_WEIGHTS[j]
            total_energy += 0.5 * mass * spd * spd

        states.append(BodyState(
            frame_idx=t,
            timestamp=timestamp,
            joints=joint_states,
            com=com,
            com_velocity=com_vel[t],
            hip_com_offset=hip_com,
            hip_com_magnitude=float(np.linalg.norm(hip_com)),
            phase=phases[t],
            total_energy=total_energy,
        ))

    return states
