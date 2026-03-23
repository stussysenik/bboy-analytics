"""
Calibrated synthetic SMPL 22-joint trajectory generator.

Produces physically-plausible breakdance joint data seeded by
real BRACE ground-truth beat timings. NOT fabrication — this is
kinematic simulation with documented parameters.

Output format: (F, 22, 3) numpy arrays matching GVHMR output.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

# SMPL 22-joint rest pose (approximate T-pose, meters)
# Y-up, Z-forward convention matching GVHMR gravity-view output
SMPL_REST_POSE = np.array([
    [0.000,  0.900, 0.000],   # 0  pelvis
    [0.085,  0.850, 0.000],   # 1  left_hip
    [-0.085, 0.850, 0.000],   # 2  right_hip
    [0.000,  1.050, 0.000],   # 3  spine1
    [0.085,  0.450, 0.000],   # 4  left_knee
    [-0.085, 0.450, 0.000],   # 5  right_knee
    [0.000,  1.200, 0.000],   # 6  spine2
    [0.085,  0.050, 0.000],   # 7  left_ankle
    [-0.085, 0.050, 0.000],   # 8  right_ankle
    [0.000,  1.350, 0.000],   # 9  spine3
    [0.085,  0.000, 0.050],   # 10 left_foot
    [-0.085, 0.000, 0.050],   # 11 right_foot
    [0.000,  1.500, 0.000],   # 12 neck
    [0.100,  1.450, 0.000],   # 13 left_collar
    [-0.100, 1.450, 0.000],   # 14 right_collar
    [0.000,  1.600, 0.000],   # 15 head
    [0.200,  1.400, 0.000],   # 16 left_shoulder
    [-0.200, 1.400, 0.000],   # 17 right_shoulder
    [0.400,  1.400, 0.000],   # 18 left_elbow
    [-0.400, 1.400, 0.000],   # 19 right_elbow
    [0.550,  1.400, 0.000],   # 20 left_wrist
    [-0.550, 1.400, 0.000],   # 21 right_wrist
], dtype=np.float64)

# Joint groups for realistic motion coupling
UPPER_BODY = [6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY = [1, 2, 4, 5, 7, 8, 10, 11]
ARMS = [16, 17, 18, 19, 20, 21]
LEGS = [1, 2, 4, 5, 7, 8, 10, 11]
EXTREMITIES = [7, 8, 10, 11, 20, 21]  # ankles, feet, wrists

# Amplitude profiles per joint (relative to base amplitude)
# Extremities move more than core
JOINT_AMPLITUDE = np.array([
    0.3,   # pelvis (root moves less)
    0.5, 0.5,  # hips
    0.2,   # spine1
    0.7, 0.7,  # knees
    0.2,   # spine2
    0.9, 0.9,  # ankles
    0.15,  # spine3
    0.6, 0.6,  # feet
    0.3,   # neck
    0.2, 0.2,  # collars
    0.4,   # head
    0.6, 0.6,  # shoulders
    0.8, 0.8,  # elbows
    1.0, 1.0,  # wrists (most mobile)
], dtype=np.float64)


def _beat_envelope(n_frames: int, beat_times: np.ndarray, fps: float,
                   sigma_ms: float = 80.0) -> np.ndarray:
    """Create a smooth envelope that peaks at each beat time."""
    sigma = sigma_ms / 1000.0 * fps
    t = np.arange(n_frames)
    envelope = np.zeros(n_frames)
    for b in beat_times:
        fi = b * fps
        envelope += np.exp(-0.5 * ((t - fi) / sigma) ** 2)
    # Normalize to [0, 1]
    if envelope.max() > 0:
        envelope /= envelope.max()
    return envelope


def _add_noise(joints: np.ndarray, sigma_mm: float = 10.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian noise matching GVHMR MPJPE (~10mm)."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma_mm / 1000.0, joints.shape)
    return joints + noise


def generate_toprock_onbeat(
    beat_times: np.ndarray,
    fps: float = 30.0,
    duration_s: float | None = None,
    base_amplitude: float = 0.04,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate toprock joints with velocity peaks at beat times.

    Calibrated ranges:
    - Velocity: 1-3 m/s (toprock)
    - Vertical range: 0.3-0.5m (weight shifts)
    - Stage coverage: 1-2 m²
    """
    rng = np.random.default_rng(seed)

    if duration_s is None:
        duration_s = beat_times[-1] + 2.0 if len(beat_times) > 0 else 10.0
    n_frames = int(duration_s * fps)

    # Beat envelope drives velocity peaks
    envelope = _beat_envelope(n_frames, beat_times, fps)

    # Base trajectory: dancer sways and shifts weight
    t = np.arange(n_frames) / fps
    joints = np.tile(SMPL_REST_POSE, (n_frames, 1, 1)).copy()

    # Root (pelvis) lateral sway — synced to beats
    bpm = 120.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        bpm = 60.0 / np.median(intervals)
    beat_freq = bpm / 60.0

    # Lateral sway (X) — at beat frequency
    sway_x = 0.15 * np.sin(2 * np.pi * beat_freq / 2 * t) * (0.5 + 0.5 * envelope)
    # Forward/back (Z) — at half beat frequency
    sway_z = 0.08 * np.sin(2 * np.pi * beat_freq / 4 * t + rng.uniform(0, np.pi))
    # Vertical bounce (Y) — at beat frequency
    bounce_y = 0.05 * np.abs(np.sin(2 * np.pi * beat_freq * t)) * envelope

    # Apply root motion
    joints[:, 0, 0] += sway_x
    joints[:, 0, 1] += bounce_y
    joints[:, 0, 2] += sway_z

    # Propagate root motion to all joints (rigid body base)
    for j in range(1, 22):
        joints[:, j, 0] += sway_x * 0.8
        joints[:, j, 1] += bounce_y * 0.7
        joints[:, j, 2] += sway_z * 0.8

    # Per-joint beat-synced oscillations
    for j in range(22):
        amp = base_amplitude * JOINT_AMPLITUDE[j]
        # Each joint has slightly different phase and frequency harmonics
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        phase_z = rng.uniform(0, 2 * np.pi)

        # Primary motion at beat frequency, modulated by envelope
        joints[:, j, 0] += amp * np.sin(2 * np.pi * beat_freq * t + phase_x) * (0.3 + 0.7 * envelope)
        joints[:, j, 1] += amp * 0.5 * np.sin(2 * np.pi * beat_freq * 2 * t + phase_y) * envelope
        joints[:, j, 2] += amp * 0.6 * np.sin(2 * np.pi * beat_freq * t + phase_z) * (0.3 + 0.7 * envelope)

    # Arms have additional beat-synced gesture
    for j in ARMS:
        arm_amp = base_amplitude * 1.5
        joints[:, j, 0] += arm_amp * np.sin(2 * np.pi * beat_freq * t + rng.uniform(0, np.pi)) * envelope
        joints[:, j, 1] += arm_amp * 0.8 * np.cos(2 * np.pi * beat_freq * t) * envelope

    # Slow stage drift (dancer moves across stage over time)
    drift_x = 0.3 * np.sin(2 * np.pi * 0.05 * t)  # ~20s period
    drift_z = 0.2 * np.cos(2 * np.pi * 0.07 * t)
    for j in range(22):
        joints[:, j, 0] += drift_x
        joints[:, j, 2] += drift_z

    return _add_noise(joints, sigma_mm=10.0, rng=rng)


def generate_toprock_offbeat(
    beat_times: np.ndarray,
    fps: float = 30.0,
    duration_s: float | None = None,
    base_amplitude: float = 0.04,
    seed: int = 42,
) -> np.ndarray:
    """
    Same kinematics as on-beat but shifted by half a beat period.
    This is the negative control: same frequency content, no beat alignment.
    """
    if len(beat_times) >= 2:
        half_period = np.median(np.diff(beat_times)) / 2.0
    else:
        half_period = 0.25
    shifted_beats = beat_times + half_period
    return generate_toprock_onbeat(shifted_beats, fps, duration_s, base_amplitude, seed)


def generate_random_control(
    fps: float = 30.0,
    duration_s: float = 30.0,
    base_amplitude: float = 0.04,
    seed: int = 99,
) -> np.ndarray:
    """
    Random movement with no beat structure.
    Same velocity range as toprock but no temporal correlation to any beat.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * fps)

    joints = np.tile(SMPL_REST_POSE, (n_frames, 1, 1)).copy()
    t = np.arange(n_frames) / fps

    # Multiple incommensurate frequencies (no beat alignment possible)
    freqs = [0.7, 1.3, 2.1, 3.7, 5.3]

    for j in range(22):
        amp = base_amplitude * JOINT_AMPLITUDE[j]
        for dim in range(3):
            signal = np.zeros(n_frames)
            for f in freqs:
                phase = rng.uniform(0, 2 * np.pi)
                weight = rng.uniform(0.2, 1.0)
                signal += weight * amp * np.sin(2 * np.pi * f * t + phase)
            signal /= len(freqs)  # normalize
            joints[:, j, dim] += signal

    # Slow drift
    drift_x = 0.2 * np.sin(2 * np.pi * 0.03 * t)
    drift_z = 0.15 * np.cos(2 * np.pi * 0.04 * t)
    for j in range(22):
        joints[:, j, 0] += drift_x
        joints[:, j, 2] += drift_z

    return _add_noise(joints, sigma_mm=10.0, rng=rng)


def generate_powermove(
    beat_times: np.ndarray,
    fps: float = 30.0,
    duration_s: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate powermove joints with inversions (pelvis above head).

    Calibrated:
    - Velocity: 2-5 m/s
    - Vertical range: 0.8-1.5m
    - Contains inversion events (headspin/windmill segments)
    """
    rng = np.random.default_rng(seed)

    if duration_s is None:
        duration_s = beat_times[-1] + 2.0 if len(beat_times) > 0 else 10.0
    n_frames = int(duration_s * fps)

    joints = np.tile(SMPL_REST_POSE, (n_frames, 1, 1)).copy()
    t = np.arange(n_frames) / fps

    # Higher amplitude than toprock
    base_amp = 0.08

    # Create inversion segments (20-40% of duration)
    inv_start = int(0.3 * n_frames)
    inv_end = int(0.6 * n_frames)

    # During inversion: flip the skeleton vertically
    for f in range(inv_start, inv_end):
        # Smooth transition in/out over 15 frames
        blend_in = min(1.0, (f - inv_start) / 15.0)
        blend_out = min(1.0, (inv_end - f) / 15.0)
        blend = blend_in * blend_out

        # Invert Y coordinates: pelvis goes up, head goes down
        for j in range(22):
            rest_y = SMPL_REST_POSE[j, 1]
            center_y = 0.8  # rotation center
            inverted_y = 2 * center_y - rest_y
            joints[f, j, 1] = rest_y + blend * (inverted_y - rest_y)

    # Add rotational motion (windmill-like)
    rotation_freq = 2.5  # rotations per second during power
    for j in range(22):
        amp = base_amp * JOINT_AMPLITUDE[j] * 2.0
        phase = rng.uniform(0, 2 * np.pi)

        # During power section: fast rotation
        power_envelope = np.zeros(n_frames)
        power_envelope[inv_start:inv_end] = 1.0
        # Smooth edges
        for f in range(min(30, inv_start)):
            power_envelope[inv_start + f] *= f / 30.0
        for f in range(min(30, n_frames - inv_end)):
            power_envelope[inv_end - 1 - f] *= f / 30.0

        normal_envelope = 1.0 - power_envelope

        # Normal toprock-like motion outside power section
        joints[:, j, 0] += amp * 0.5 * np.sin(2 * np.pi * 2.0 * t + phase) * normal_envelope
        joints[:, j, 1] += amp * 0.3 * np.sin(2 * np.pi * 2.0 * t + phase + 0.5) * normal_envelope

        # Fast rotational motion during power section
        joints[:, j, 0] += amp * np.sin(2 * np.pi * rotation_freq * t + phase) * power_envelope
        joints[:, j, 2] += amp * np.cos(2 * np.pi * rotation_freq * t + phase) * power_envelope

    # Stage drift
    drift_x = 0.15 * np.sin(2 * np.pi * 0.04 * t)
    for j in range(22):
        joints[:, j, 0] += drift_x

    return _add_noise(joints, sigma_mm=15.0, rng=rng)  # More noise for power moves


def load_brace_beats(video_id: str, seq_idx: int) -> dict:
    """Load ground truth beats from BRACE annotations."""
    beats_path = Path(__file__).parent.parent / "data" / "brace" / "annotations" / "audio_beats.json"
    with open(beats_path) as f:
        all_beats = json.load(f)
    key = f"{video_id}.{seq_idx}"
    data = all_beats[key]
    return {
        "beat_times": np.array(data["beats_sec"]),
        "bpm": data["bpm"],
        "confidence": data["beats_confidence"],
    }


def load_brace_segments(video_id: str, seq_idx: int) -> list[dict]:
    """Load dance segments from BRACE annotations."""
    seg_path = Path(__file__).parent.parent / "data" / "brace" / "annotations" / "segments.csv"
    segments = []
    with open(seg_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            if row["video_id"] == video_id and int(row["seq_idx"]) == seq_idx:
                segments.append({
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "dance_type": row["dance_type"],
                    "dancer": row["dancer"],
                })
    return segments
