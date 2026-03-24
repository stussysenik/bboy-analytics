#!/usr/bin/env python3
"""
TRIVIUM Scoring Engine — Deterministic Physical Pattern Matching

Scores breaking performances by counting how many audio beats the dancer's
body physically hit, broken down by joint cluster. No floating coefficients —
raw percentage of beats matched within a 70ms tolerance window.

Hip placement (pelvis joint 0) is tracked as the kinetic chain engine:
  - Standing (Y > 0.85m) → toprock phase
  - Crouching (0.4m < Y < 0.85m) → footwork phase
  - Ground (Y < 0.4m) → power/freeze phase

Each joint cluster gets its own accent envelope and hit rate against the beats:
  - legs:  joints 1,2,4,5,7,8,10,11  (foundation in upright, expression in power)
  - torso: joints 0,3,6,9            (the engine — hips drive everything)
  - arms:  joints 13,14,16,17,18,19  (expression layer)
  - hands: joints 20,21,22,23        (distal expression, become foundation when inverted)
  - head:  joints 12,15              (tracking reference, active in headspins)

Usage:
  python experiments/bboy-battle-analysis/run_trivium.py
  python experiments/bboy-battle-analysis/run_trivium.py --joints path/to/joints.npy --audio path/to/audio.wav
  python experiments/bboy-battle-analysis/run_trivium.py --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Add overnight code to path
OVERNIGHT_DIR = Path(__file__).parent.parent.parent / "autoresearch" / "experiments" / "bboy-battle-analysis"
sys.path.insert(0, str(OVERNIGHT_DIR))

import analyze_motion as motion_mod
import match_beats as beats_mod

EPS = 1e-8

# ── Joint Cluster Definitions ────────────────────────────────────────────────
# From SMPL 24-joint topology. Hips (joint 0) are the engine of the kinetic chain.
CLUSTERS = {
    "legs":  [1, 2, 4, 5, 7, 8, 10, 11],   # foundation (upright) / expression (power)
    "torso": [0, 3, 6, 9],                   # the ENGINE — pelvis drives everything
    "arms":  [13, 14, 16, 17, 18, 19],       # expression layer
    "hands": [20, 21, 22, 23],               # distal tips, foundation when inverted
    "head":  [12, 15],                        # reference + headspin axis
}

# Mass weights per joint (from biomechanics literature, same as analyze_motion.py)
JOINT_MASSES_KG = {
    0: 11.17, 1: 2.78, 2: 2.78, 3: 5.0, 4: 3.28, 5: 3.28,
    6: 3.0, 7: 0.61, 8: 0.61, 9: 2.5, 10: 0.97, 11: 0.97,
    12: 1.5, 13: 0.5, 14: 0.5, 15: 5.0, 16: 2.0, 17: 2.0,
    18: 1.14, 19: 1.14, 20: 0.45, 21: 0.45, 22: 0.41, 23: 0.41,
}


def compute_cluster_accents(joints_3d: np.ndarray, fps: float, delta_s: float = 0.070):
    """
    For each joint cluster, compute its own movement energy and detect accents.

    Returns dict mapping cluster_name → {
        "energy": (F,) movement energy signal,
        "accents": list of {frame, time, strength},
        "acceleration": (F,) acceleration magnitude signal,
    }
    """
    joints = np.asarray(joints_3d, dtype=np.float64)
    F = joints.shape[0]
    cluster_data = {}

    for name, indices in CLUSTERS.items():
        # Per-cluster velocity: central difference
        subset = joints[:, indices, :]  # (F, n_joints, 3)
        velocities = np.gradient(subset, axis=0) * fps  # (F, n_joints, 3)
        speed = np.linalg.norm(velocities, axis=-1)  # (F, n_joints)

        # Mass-weighted energy for this cluster
        masses = np.array([JOINT_MASSES_KG[j] for j in indices], dtype=np.float64)
        masses /= masses.sum() + EPS
        energy = speed @ masses  # (F,)
        energy_smooth = gaussian_filter1d(energy, sigma=max(1.0, 0.05 * fps))

        # Acceleration = derivative of energy (motion accents are acceleration peaks)
        acceleration = np.gradient(energy_smooth) * fps
        accent_env = np.maximum(0.0, acceleration)

        # Detect accent peaks
        accents = []
        if accent_env.size > 0 and float(np.max(accent_env)) > EPS:
            peak_height = max(
                float(np.quantile(accent_env, 0.75)),
                float(np.max(accent_env) * 0.25),
            )
            peaks, props = find_peaks(
                accent_env,
                height=peak_height,
                distance=max(1, int(round(0.18 * fps))),
            )
            if peaks.size == 0:
                peaks = np.array([int(np.argmax(accent_env))])
            heights = accent_env[peaks]
            for i, p in enumerate(peaks):
                accents.append({
                    "frame": int(p),
                    "time": float(p / fps),
                    "strength": float(heights[i]),
                })

        cluster_data[name] = {
            "energy": energy_smooth,
            "accents": accents,
            "acceleration": accent_env,
        }

    return cluster_data


def match_cluster_to_beats(accents: list, beat_times: np.ndarray, delta_s: float = 0.070):
    """
    For a single cluster's accent list, count how many beats it hit.
    Returns: {hits, total, hit_rate, per_beat: [{time, hit, lag_ms}]}
    """
    beat_times = np.asarray(beat_times, dtype=np.float64)
    accent_times = np.array([a["time"] for a in accents], dtype=np.float64) if accents else np.zeros(0)

    hits = 0
    per_beat = []
    for bt in beat_times:
        if accent_times.size > 0:
            nearest_idx = int(np.argmin(np.abs(accent_times - bt)))
            lag = float(accent_times[nearest_idx] - bt)
            is_hit = abs(lag) <= delta_s
        else:
            lag = float("inf")
            is_hit = False

        if is_hit:
            hits += 1
        per_beat.append({
            "beat_time_s": float(bt),
            "hit": is_hit,
            "lag_ms": round(lag * 1000, 1) if is_hit else None,
        })

    total = len(beat_times)
    return {
        "hits": hits,
        "total": total,
        "hit_rate_pct": round(100.0 * hits / max(total, 1), 1),
        "n_accents": len(accents),
        "per_beat": per_beat,
    }


def compute_hip_tracking(joints_3d: np.ndarray, fps: float):
    """
    Track pelvis (joint 0) position, velocity, and acceleration through time.
    Hips are the ENGINE of the kinetic chain in breaking.

    Returns dict with:
      - position: (F, 3) hip xyz in meters
      - velocity: (F, 3) hip velocity m/s
      - acceleration_mag: (F,) acceleration magnitude m/s²
      - height: (F,) hip Y coordinate (phase indicator)
      - phase_labels: (F,) string labels per frame
      - phase_summary: dict of phase → {frames, pct, mean_height}
    """
    pelvis = joints_3d[:, 0, :]  # (F, 3)
    vel = np.gradient(pelvis, axis=0) * fps
    accel = np.gradient(vel, axis=0) * fps
    accel_mag = np.linalg.norm(accel, axis=-1)
    accel_mag_smooth = gaussian_filter1d(accel_mag, sigma=max(1.0, 0.05 * fps))

    # Hip height determines breaking phase
    # Y axis is vertical in SMPL (GVHMR outputs gravity-aligned)
    height = pelvis[:, 1]  # Y coordinate
    height_smooth = gaussian_filter1d(height, sigma=max(1.0, 0.1 * fps))

    # Phase classification using RELATIVE thresholds (quantile-based)
    # GVHMR outputs are camera-relative, not world-grounded, so absolute
    # heights don't work. Instead: standing = top 35%, crouching = middle,
    # ground = bottom 25% of the dancer's own height range.
    F = joints_3d.shape[0]
    h_min, h_max = float(height_smooth.min()), float(height_smooth.max())
    h_range = h_max - h_min
    if h_range < 0.05:
        # Nearly no vertical variation — all one phase
        thresh_low = h_min - 1.0
        thresh_high = h_max + 1.0
    else:
        thresh_low = h_min + 0.25 * h_range   # bottom 25% → power/freeze
        thresh_high = h_max - 0.35 * h_range   # top 35% → toprock

    phase_labels = []
    for i in range(F):
        h = float(height_smooth[i])
        if h > thresh_high:
            phase_labels.append("toprock")
        elif h < thresh_low:
            phase_labels.append("power/freeze")
        else:
            phase_labels.append("footwork")

    # Phase summary
    phases = {}
    for phase_name in ["toprock", "footwork", "power/freeze"]:
        mask = np.array([p == phase_name for p in phase_labels])
        n = int(mask.sum())
        phases[phase_name] = {
            "frames": n,
            "pct": round(100.0 * n / F, 1),
            "mean_height_m": round(float(height_smooth[mask].mean()), 3) if n > 0 else 0.0,
        }

    return {
        "position": pelvis,
        "velocity": vel,
        "acceleration_mag": accel_mag_smooth,
        "height": height_smooth,
        "phase_labels": phase_labels,
        "phase_summary": phases,
    }


def compute_joint_trajectories(joints_3d: np.ndarray, fps: float):
    """
    Compute per-joint 4D trajectories (x, y, z, speed) for visualization.
    Returns dict with per-joint position, velocity, and color-code data.
    """
    F, J, _ = joints_3d.shape
    velocities = np.gradient(joints_3d, axis=0) * fps
    speeds = np.linalg.norm(velocities, axis=-1)  # (F, J)

    # Per-joint trajectory data for visualization
    trajectories = {}
    joint_names = [
        "pelvis", "L.hip", "R.hip", "spine1", "L.knee", "R.knee",
        "spine2", "L.ankle", "R.ankle", "spine3", "L.foot", "R.foot",
        "neck", "L.collar", "R.collar", "head", "L.shoulder", "R.shoulder",
        "L.elbow", "R.elbow", "L.wrist", "R.wrist", "L.hand", "R.hand",
    ]

    for j in range(min(J, 24)):
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        trajectories[name] = {
            "mean_speed_ms": round(float(speeds[:, j].mean()), 4),
            "max_speed_ms": round(float(speeds[:, j].max()), 4),
            "position_range_m": {
                "x": [round(float(joints_3d[:, j, 0].min()), 3), round(float(joints_3d[:, j, 0].max()), 3)],
                "y": [round(float(joints_3d[:, j, 1].min()), 3), round(float(joints_3d[:, j, 1].max()), 3)],
                "z": [round(float(joints_3d[:, j, 2].min()), 3), round(float(joints_3d[:, j, 2].max()), 3)],
            },
        }

    return {
        "positions": joints_3d,       # (F, J, 3) - for rendering
        "speeds": speeds,             # (F, J) - for color coding
        "trajectories": trajectories,  # per-joint summary stats
    }


def run_trivium(
    joints_3d: np.ndarray,
    audio_path: str | None = None,
    beat_times: np.ndarray | None = None,
    fps: float = 30.0,
    delta_s: float = 0.070,
):
    """
    Full TRIVIUM pipeline: deterministic physical matching.

    Returns dict with:
      - cluster_hits: per-cluster beat hit rates (the core metric)
      - total_hits: overall beat hit rate
      - hip_tracking: pelvis position/velocity/acceleration/phase
      - motion_features: 9D feature extraction from analyze_motion
      - trivium_score: BODY/SOUL/MIND composite (from match_beats)
      - joint_trajectories: per-joint 4D data for visualization
    """
    joints = np.asarray(joints_3d, dtype=np.float64)
    if joints.shape[1] > 24:
        joints = joints[:, :24, :]

    F = joints.shape[0]
    duration_s = F / fps
    print(f"  Input: {F} frames, {duration_s:.1f}s @ {fps} fps, {joints.shape[1]} joints")

    # ── 1. Motion feature extraction (9D + accents + phase detection) ────
    print("  [1/5] Extracting motion features...")
    motion_result = motion_mod.extract_features(joints, fps=fps)
    features_9xN = motion_result[0]  # (9, N_segments)
    motion_meta = motion_result[1]   # dict with accents, segments, trivium sub-scores

    # ── 2. Audio beat detection ──────────────────────────────────────────
    print("  [2/5] Loading audio beats...")
    if audio_path is not None:
        audio_payload = beats_mod.analyze_wav(audio_path)
        beat_times_arr = audio_payload["beat_times"]
    elif beat_times is not None:
        beat_times_arr = np.asarray(beat_times, dtype=np.float64)
        audio_payload = {
            "audio_energy": np.ones(int(duration_s * 100)),
            "beat_times": beat_times_arr,
            "sample_hz": 100.0,
            "duration_s": duration_s,
        }
    else:
        raise ValueError("Provide --audio or --beats")

    n_beats = len(beat_times_arr)
    print(f"    {n_beats} beats detected")

    # ── 3. Per-cluster deterministic matching ────────────────────────────
    print("  [3/5] Per-cluster accent→beat matching (±70ms)...")
    cluster_data = compute_cluster_accents(joints, fps, delta_s)
    cluster_hits = {}
    for name, cdata in cluster_data.items():
        result = match_cluster_to_beats(cdata["accents"], beat_times_arr, delta_s)
        cluster_hits[name] = {
            "hits": result["hits"],
            "total": result["total"],
            "hit_rate_pct": result["hit_rate_pct"],
            "n_accents": result["n_accents"],
        }

    # Overall hit rate (from total movement energy, not clusters)
    total_match = beats_mod.match_accents_to_beats(
        joints_3d=joints, fps=fps,
        beat_times=beat_times_arr,
    )

    # ── 4. Hip tracking (kinetic chain engine) ───────────────────────────
    print("  [4/5] Tracking hips (kinetic chain engine)...")
    hip_data = compute_hip_tracking(joints, fps)

    # ── 5. Joint trajectories for 4D visualization ───────────────────────
    print("  [5/5] Computing joint trajectories...")
    traj_data = compute_joint_trajectories(joints, fps)

    # ── 6. Full TRIVIUM score (from overnight match_beats) ───────────────
    if audio_path is not None:
        trivium_result = beats_mod.run_pipeline(joints, audio_payload, fps=fps)
    else:
        trivium_result = None

    # ── Compile results ──────────────────────────────────────────────────
    output = {
        "summary": {
            "n_frames": F,
            "duration_s": round(duration_s, 2),
            "fps": fps,
            "n_beats": n_beats,
            "n_joints": joints.shape[1],
        },

        # Core metric: per-cluster beat hit rates
        "cluster_hits": cluster_hits,
        "total_hit_rate": {
            "hit_rate_pct": round(100.0 * total_match["accent_hit_rate"], 1),
            "hits": total_match["n_hits"],
            "total": total_match["n_beats"],
            "groove_lock": round(total_match["groove_lock"], 3),
            "optimal_lag_ms": round(total_match["optimal_lag_ms"], 1),
        },

        # Hip tracking: position, velocity, phase breakdown
        "hip_tracking": {
            "phase_summary": hip_data["phase_summary"],
            "mean_height_m": round(float(hip_data["height"].mean()), 3),
            "max_acceleration_ms2": round(float(hip_data["acceleration_mag"].max()), 2),
            "mean_acceleration_ms2": round(float(hip_data["acceleration_mag"].mean()), 2),
        },

        # 9D motion features (normalized [0,1])
        "motion_features_9d": {
            name: round(float(np.mean(features_9xN[i])), 3)
            for i, name in enumerate(motion_mod.FEATURE_NAMES)
        },

        # Per-joint trajectory stats
        "joint_trajectories": traj_data["trajectories"],

        # Phase detection from analyze_motion
        "phase_detection": motion_meta.get("phase_distribution", {}),

        # Full TRIVIUM composite (from overnight code)
        "trivium_composite": trivium_result.get("trivium") if trivium_result else None,
    }

    return output, hip_data, traj_data, cluster_data


def print_report(output: dict):
    """Pretty-print the TRIVIUM report."""
    s = output["summary"]
    print(f"\n{'='*60}")
    print(f"  TRIVIUM — Deterministic Physical Pattern Matching")
    print(f"  {s['n_frames']} frames | {s['duration_s']}s | {s['n_beats']} beats")
    print(f"{'='*60}")

    print(f"\n  OVERALL: {output['total_hit_rate']['hits']}/{output['total_hit_rate']['total']} "
          f"beats hit = {output['total_hit_rate']['hit_rate_pct']}%")
    print(f"  Groove lock: {output['total_hit_rate']['groove_lock']}")
    print(f"  Optimal lag: {output['total_hit_rate']['optimal_lag_ms']}ms")

    print(f"\n  PER-CLUSTER BREAKDOWN:")
    for name, data in output["cluster_hits"].items():
        bar_len = int(data["hit_rate_pct"] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    {name:6s}: {data['hits']:3d}/{data['total']:3d} = "
              f"{data['hit_rate_pct']:5.1f}% |{bar}| ({data['n_accents']} accents)")

    print(f"\n  HIP TRACKING (Kinetic Chain Engine):")
    hip = output["hip_tracking"]
    print(f"    Mean height: {hip['mean_height_m']}m")
    print(f"    Peak acceleration: {hip['max_acceleration_ms2']} m/s²")
    for phase, data in hip["phase_summary"].items():
        print(f"    {phase:13s}: {data['pct']:5.1f}% ({data['frames']} frames, "
              f"mean height {data['mean_height_m']}m)")

    print(f"\n  9D MOTION FEATURES:")
    for name, val in output["motion_features_9d"].items():
        bar_len = int(val * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"    {name:30s}: {val:.3f} |{bar}|")

    if output["trivium_composite"]:
        t = output["trivium_composite"]
        print(f"\n  TRIVIUM COMPOSITE: {t.get('score_100', 0):.1f}/100")
        for axis in ["body", "soul", "mind"]:
            if axis in t:
                score = t[axis].get("score", 0)
                print(f"    {axis.upper():5s}: {score:.3f} — {t[axis].get('components', {})}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="TRIVIUM — Deterministic Physical Pattern Matching")
    parser.add_argument("--joints", default="experiments/results/joints_3d_REAL_seq4.npy",
                        help="Path to joints .npy (F, 24, 3)")
    parser.add_argument("--audio", default=None,
                        help="Path to audio .wav file")
    parser.add_argument("--beats", default=None,
                        help="Path to beat times .npy (fallback if no audio)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--delta-ms", type=float, default=70.0,
                        help="Hit tolerance in milliseconds (default: 70)")
    parser.add_argument("--output", default=None,
                        help="Save JSON output to file")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 4D trajectory visualization")
    args = parser.parse_args()

    # Resolve paths
    root = Path(__file__).parent.parent.parent
    joints_path = root / args.joints
    if not joints_path.exists():
        joints_path = Path(args.joints)

    print(f"Loading joints: {joints_path}")
    joints = np.load(str(joints_path))
    print(f"  Shape: {joints.shape}")

    # Audio or beats
    audio_path = None
    beat_times = None
    if args.audio:
        audio_path = str(root / args.audio) if not Path(args.audio).exists() else args.audio
    else:
        # Try default audio location
        default_audio = root / "josh_input" / "bcone_seq4" / "audio.wav"
        if default_audio.exists():
            audio_path = str(default_audio)
            print(f"Using audio: {audio_path}")
        elif args.beats:
            beats_path = root / args.beats if not Path(args.beats).exists() else Path(args.beats)
            beat_times = np.load(str(beats_path))
            print(f"Using pre-computed beats: {beats_path} ({len(beat_times)} beats)")
        else:
            # Fall back to librosa beats
            default_beats = root / "experiments" / "results" / "librosa_beats.npy"
            if default_beats.exists():
                beat_times = np.load(str(default_beats))
                print(f"Using librosa beats: {default_beats} ({len(beat_times)} beats)")
            else:
                raise FileNotFoundError("No audio or beat file found")

    # Run pipeline
    delta_s = args.delta_ms / 1000.0
    output, hip_data, traj_data, cluster_data = run_trivium(
        joints, audio_path=audio_path, beat_times=beat_times,
        fps=args.fps, delta_s=delta_s,
    )

    # Print report
    print_report(output)

    # Save JSON
    if args.output:
        out_path = root / args.output if not Path(args.output).is_absolute() else Path(args.output)
        # Strip non-serializable arrays
        serializable = json.loads(json.dumps(output, default=lambda o: None if isinstance(o, np.ndarray) else o))
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved to {out_path}")

    # Save numpy arrays for visualization
    viz_dir = root / "experiments" / "results" / "trivium_viz"
    viz_dir.mkdir(exist_ok=True)
    np.save(viz_dir / "hip_height.npy", hip_data["height"])
    np.save(viz_dir / "hip_acceleration.npy", hip_data["acceleration_mag"])
    np.save(viz_dir / "hip_position.npy", hip_data["position"])
    np.save(viz_dir / "joint_speeds.npy", traj_data["speeds"])
    for name, cdata in cluster_data.items():
        np.save(viz_dir / f"cluster_{name}_energy.npy", cdata["energy"])
        np.save(viz_dir / f"cluster_{name}_accel.npy", cdata["acceleration"])
    print(f"Visualization data saved to {viz_dir}/")


if __name__ == "__main__":
    main()
