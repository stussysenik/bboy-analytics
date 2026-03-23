"""
Compute all 6 metric categories from 3D joint data.

Categories: musicality, energy, flow, audio_flow, space, complexity.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

# SMPL 22-joint names (pelvis is root)
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# Approximate segment masses relative to body (used for energy weighting)
JOINT_MASS_WEIGHTS = np.array([
    0.15, 0.10, 0.10, 0.10, 0.05, 0.05,   # pelvis, hips, spine, knees
    0.10, 0.02, 0.02, 0.05, 0.01, 0.01,   # spine2, ankles, spine3, feet
    0.03, 0.02, 0.02, 0.06, 0.04, 0.04,   # neck, collars, head, shoulders
    0.02, 0.02, 0.01, 0.01,                # elbows, wrists
], dtype=np.float32)


def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central difference: (x[t+1] - x[t-1]) / 2dt. Returns (N-2, ...)."""
    return (x[2:] - x[:-2]) / (2.0 * dt)


def _smooth(signal: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing. Falls back to identity if scipy unavailable."""
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(signal) // 2 * 2 - 1)
        if w < polyorder + 2:
            return signal
        return savgol_filter(signal, window_length=w, polyorder=polyorder)
    except ImportError:
        return signal


def _normalize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    return (x - x.mean()) / (x.std() + 1e-8)


# ── 1. Musicality ────────────────────────────────────────────────────────────


def compute_musicality(
    joints: np.ndarray, beat_times: np.ndarray, fps: float,
    sg_window: int = 31, max_lag_ms: float = 200.0,
) -> dict[str, Any]:
    """Cross-correlate joint velocity with audio beats."""
    dt = 1.0 / fps
    velocity = _central_diff(joints, dt)  # (F-2, J, 3)
    speed = np.linalg.norm(velocity, axis=-1)  # (F-2, J)

    # Smooth per joint then sum
    speed_smooth = np.column_stack([
        _smooth(speed[:, j], sg_window) for j in range(speed.shape[1])
    ])
    M = _normalize(speed_smooth.sum(axis=1))

    # Beat signal (Gaussian kernel per beat)
    n = len(M)
    sigma = 50.0 / 1000.0 * fps
    t = np.arange(n)
    H = np.zeros(n)
    for b in beat_times:
        fi = int(b * fps)
        if 0 <= fi < n:
            H += np.exp(-0.5 * ((t - fi) / sigma) ** 2)
    H = _normalize(H)

    # Cross-correlation
    corr = np.correlate(M, H, mode="full")
    corr /= np.sqrt(np.sum(M ** 2) * np.sum(H ** 2)) + 1e-8
    mid = len(corr) // 2
    max_lag = int(max_lag_ms / 1000.0 * fps)
    window = corr[mid - max_lag: mid + max_lag + 1]
    lags_ms = np.arange(-max_lag, max_lag + 1) * (1000.0 / fps)

    mu = float(np.max(window))
    tau = float(lags_ms[np.argmax(window)])

    # Beat alignment percentage
    peak_frames = np.where(np.diff(np.sign(np.diff(M))) < 0)[0] + 1
    peak_times = peak_frames / fps
    aligned = 0
    for bt in beat_times:
        if any(abs(pt - bt) < 0.1 for pt in peak_times):
            aligned += 1
    beat_align_pct = aligned / max(len(beat_times), 1) * 100

    return {
        "mu": mu,
        "tau_star_ms": tau,
        "beat_alignment_pct": round(beat_align_pct, 1),
        "interpretation": {
            "musicality": "STRONG" if mu > 0.4 else "MODERATE" if mu > 0.2 else "WEAK",
            "timing": "ANTICIPATES" if tau < -30 else "REACTS" if tau > 30 else "ON_BEAT",
        },
    }


def compute_per_joint_snr(joints: np.ndarray, fps: float, sg_window: int = 31) -> dict:
    """Per-joint velocity signal-to-noise ratio."""
    dt = 1.0 / fps
    velocity = _central_diff(joints, dt)
    speed = np.linalg.norm(velocity, axis=-1)

    snr = {}
    for j in range(min(speed.shape[1], len(JOINT_NAMES))):
        sig = _smooth(speed[:, j], sg_window)
        sig_power = float(np.mean(sig ** 2))
        if len(sig) > 61:
            heavy = _smooth(sig, 61)
            noise_power = float(np.mean((sig - heavy) ** 2))
        else:
            noise_power = sig_power * 0.1
        r = sig_power / (noise_power + 1e-8)
        snr[JOINT_NAMES[j]] = {
            "snr_linear": round(r, 2),
            "snr_db": round(10 * np.log10(r + 1e-8), 1),
            "mean_speed_m_s": round(float(np.mean(sig)), 3),
        }
    return snr


# ── 2. Energy ────────────────────────────────────────────────────────────────


def compute_energy(joints: np.ndarray, fps: float) -> dict[str, Any]:
    """Kinetic energy curve, peaks, and build-up rate."""
    dt = 1.0 / fps
    velocity = _central_diff(joints, dt)  # (F-2, J, 3)
    speed_sq = np.sum(velocity ** 2, axis=-1)  # (F-2, J)

    # Mass-weighted kinetic energy: E(t) = 0.5 * Σ m_j * ||v_j||²
    weights = JOINT_MASS_WEIGHTS[:speed_sq.shape[1]]
    E = 0.5 * (speed_sq * weights[None, :]).sum(axis=1)  # (F-2,)
    E_smooth = _smooth(E, 61)

    # Acceleration (energy derivative)
    dE = np.gradient(E_smooth, dt)

    # Peak detection
    peaks = []
    if len(E_smooth) > 5:
        from_diff = np.diff(np.sign(np.diff(E_smooth)))
        peak_idx = np.where(from_diff < 0)[0] + 1
        threshold = np.percentile(E_smooth, 90)
        for idx in peak_idx:
            if E_smooth[idx] > threshold:
                peaks.append({"frame": int(idx), "time_s": round(idx / fps, 2), "energy": round(float(E_smooth[idx]), 3)})

    # Energy sections (high/med/low)
    p33, p66 = np.percentile(E_smooth, [33, 66])
    sections = []
    current_level = None
    start = 0
    for i, e in enumerate(E_smooth):
        level = "high" if e > p66 else "medium" if e > p33 else "low"
        if level != current_level:
            if current_level is not None:
                sections.append({"level": current_level, "start_s": round(start / fps, 2), "end_s": round(i / fps, 2)})
            current_level = level
            start = i
    if current_level:
        sections.append({"level": current_level, "start_s": round(start / fps, 2), "end_s": round(len(E_smooth) / fps, 2)})

    return {
        "energy_curve": E_smooth.tolist(),
        "peak_moments": peaks[:20],
        "mean_energy": round(float(np.mean(E_smooth)), 4),
        "max_energy": round(float(np.max(E_smooth)), 4),
        "energy_sections": sections,
        "build_up_rate": round(float(np.mean(np.maximum(dE, 0))), 4),
    }


# ── 3. Flow ──────────────────────────────────────────────────────────────────


def compute_flow(joints: np.ndarray, fps: float) -> dict[str, Any]:
    """Flow/smoothness via jerk minimization (d³/dt³)."""
    dt = 1.0 / fps
    velocity = _central_diff(joints, dt)       # d/dt
    accel = _central_diff(velocity, dt)         # d²/dt²
    jerk = _central_diff(accel, dt)             # d³/dt³

    jerk_mag = np.linalg.norm(jerk, axis=-1)    # (F-6, J)
    mean_jerk = float(np.mean(jerk_mag))
    max_jerk = float(np.max(jerk_mag))

    # Flow score: inverse of mean jerk, scaled 0-100
    flow_score = 100.0 / (1.0 + mean_jerk)

    # Per-joint flow
    per_joint = {}
    for j in range(min(jerk_mag.shape[1], len(JOINT_NAMES))):
        per_joint[JOINT_NAMES[j]] = round(float(np.mean(jerk_mag[:, j])), 3)

    return {
        "flow_score": round(flow_score, 1),
        "mean_jerk": round(mean_jerk, 4),
        "max_jerk": round(max_jerk, 4),
        "per_joint_jerk": per_joint,
    }


# ── 4. Space ─────────────────────────────────────────────────────────────────


def compute_space(joints: np.ndarray, fps: float) -> dict[str, Any]:
    """Stage coverage, COM trajectory, vertical range."""
    root = joints[:, 0, :]  # pelvis (F, 3)
    xz = root[:, [0, 2]]   # top-down (X, Z)

    # Convex hull area
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xz)
        area = float(hull.volume)  # 2D hull "volume" = area
    except (ImportError, Exception):
        # Bounding box fallback
        area = float((xz[:, 0].max() - xz[:, 0].min()) * (xz[:, 1].max() - xz[:, 1].min()))

    # Vertical range
    y = root[:, 1]
    vertical_range = float(y.max() - y.min())

    # COM velocity (how fast the dancer moves across stage)
    com_speed = np.linalg.norm(np.diff(xz, axis=0), axis=1) * fps
    mean_travel_speed = float(np.mean(com_speed))

    return {
        "stage_coverage_m2": round(area, 2),
        "vertical_range_m": round(vertical_range, 2),
        "mean_travel_speed_m_s": round(mean_travel_speed, 3),
        "com_trajectory": xz.tolist(),
        "com_y": y.tolist(),
    }


# ── 5. Complexity ────────────────────────────────────────────────────────────


def compute_complexity(
    joints: np.ndarray, fps: float,
    segments: list[dict] | None = None,
) -> dict[str, Any]:
    """Freeze detection, inversion detection, move type distribution."""
    dt = 1.0 / fps
    velocity = _central_diff(joints, dt)
    accel = _central_diff(velocity, dt)
    accel_mag = np.linalg.norm(accel, axis=-1).mean(axis=1)  # (F-4,) mean across joints

    # Freeze detection: acceleration near zero for ≥0.5s
    freeze_threshold = np.percentile(accel_mag, 10)  # bottom 10%
    min_freeze_frames = int(0.5 * fps)
    is_frozen = accel_mag < freeze_threshold
    freezes = []
    run_start = None
    for i, frozen in enumerate(is_frozen):
        if frozen and run_start is None:
            run_start = i
        elif not frozen and run_start is not None:
            duration = i - run_start
            if duration >= min_freeze_frames:
                freezes.append({
                    "start_s": round(run_start / fps, 2),
                    "end_s": round(i / fps, 2),
                    "duration_s": round(duration / fps, 2),
                })
            run_start = None

    # Inversion detection: pelvis Y > head Y
    pelvis_y = joints[2:-2, 0, 1]  # align with accel indices
    head_y = joints[2:-2, 15, 1] if joints.shape[1] > 15 else pelvis_y
    inversions = []
    inverted = pelvis_y > head_y
    inv_start = None
    for i, inv in enumerate(inverted):
        if inv and inv_start is None:
            inv_start = i
        elif not inv and inv_start is not None:
            duration = i - inv_start
            if duration >= 3:  # at least 3 frames
                inversions.append({
                    "start_s": round(inv_start / fps, 2),
                    "end_s": round(i / fps, 2),
                    "duration_s": round(duration / fps, 2),
                })
            inv_start = None

    # Move type distribution from segments (if provided)
    move_dist = {}
    if segments:
        total_frames = 0
        for seg in segments:
            dtype = seg.get("dance_type", "unknown")
            n = seg.get("end_frame", 0) - seg.get("start_frame", 0)
            move_dist[dtype] = move_dist.get(dtype, 0) + n
            total_frames += n
        if total_frames > 0:
            move_dist = {k: round(v / total_frames * 100, 1) for k, v in move_dist.items()}

    return {
        "freeze_events": freezes,
        "freeze_count": len(freezes),
        "total_freeze_time_s": round(sum(f["duration_s"] for f in freezes), 2),
        "inversion_events": inversions,
        "inversion_count": len(inversions),
        "move_type_distribution_pct": move_dist,
        "peak_acceleration_m_s2": round(float(np.max(accel_mag)), 2),
        "mean_acceleration_m_s2": round(float(np.mean(accel_mag)), 2),
    }


# ── All metrics ──────────────────────────────────────────────────────────────


def compute_all_metrics(
    joints: np.ndarray,
    fps: float,
    beat_times: np.ndarray | None = None,
    segments: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute all 6 metric categories. Returns unified dict."""
    if beat_times is None:
        # Synthetic 120 BPM
        duration = (joints.shape[0] - 1) / fps
        beat_times = np.arange(0, duration, 0.5)

    return {
        "musicality": compute_musicality(joints, beat_times, fps),
        "per_joint_snr": compute_per_joint_snr(joints, fps),
        "energy": compute_energy(joints, fps),
        "flow": compute_flow(joints, fps),
        "space": compute_space(joints, fps),
        "complexity": compute_complexity(joints, fps, segments),
        "meta": {
            "n_frames": joints.shape[0],
            "n_joints": joints.shape[1],
            "fps": fps,
            "duration_s": round(joints.shape[0] / fps, 2),
        },
    }
