"""
World State: compute all per-frame scalars from 3D joint data.

All math is deterministic and documented inline.
Given the same joints + fps + beat_times, output is identical.

Usage:
    python experiments/world_state.py --joints path/to/joints.npy [--beats path/to/beats.npy]
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

try:
    from scipy.signal import savgol_filter, correlate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class WorldState:
    """All per-frame scalars for a joint sequence. Fully deterministic."""

    # Input
    frames: int
    fps: float
    joints: np.ndarray          # (F, 22, 3)

    # Per-frame scalars (F-length or F-1 / F-2 padded to F)
    kinetic_energy: np.ndarray  # K(t) = Σⱼ ‖vⱼ(t)‖²
    energy_accel: np.ndarray    # dK/dt
    com_pos: np.ndarray         # (F, 3) pelvis position
    com_height: np.ndarray      # h(t) = pelvis_y
    compactness: np.ndarray     # C(t) = mean ‖pⱼ - centroid‖
    cyclic_score: np.ndarray    # P(t) = windowed autocorrelation peak
    composite: np.ndarray       # weighted [0,1] scalar

    # Musicality (optional — requires beat_times)
    mu: float = 0.0
    tau_star_ms: float = 0.0
    beat_times: np.ndarray | None = None

    # Pattern detection
    cyclic_regions: list = field(default_factory=list)
    dominant_freq_hz: float = 0.0

    # Per-joint velocity for data point panels
    joint_velocities: np.ndarray | None = None  # (F, 22) speed per joint

    # Beat-level analysis (insight-first renderer)
    beat_hits: list = field(default_factory=list)     # [{time_s, hit: bool, energy_at_beat}]
    beat_hit_pct: float = 0.0                         # e.g. 85%
    local_mu: np.ndarray | None = None                # (F,) sliding-window μ
    movement_energy: np.ndarray | None = None         # (F,) M(t) for musicality

    # Contact detection
    contact_feet: np.ndarray | None = None   # (F, 4) confidence for [L.ankle, R.ankle, L.foot, R.foot]
    contact_hands: np.ndarray | None = None  # (F, 2) confidence for [L.wrist, R.wrist]


def compute_world_state(
    joints: np.ndarray,
    fps: float = 30.0,
    beat_times: np.ndarray | None = None,
    sg_window: int = 15,
    cyclic_window_s: float = 2.0,
    cyclic_threshold: float = 0.5,
) -> WorldState:
    """
    Compute all world state scalars from joint data.

    Args:
        joints: (F, J, 3) in meters, SMPL 22-joint
        fps: frame rate
        beat_times: optional beat times in seconds for musicality
        sg_window: Savitzky-Golay smoothing window (frames)
        cyclic_window_s: autocorrelation window size in seconds
        cyclic_threshold: minimum autocorrelation peak to count as cyclic
    """
    F, J, _ = joints.shape
    dt = 1.0 / fps

    # ── Velocity: central difference v(t) = (p(t+1) - p(t-1)) / (2Δt) ──
    vel = np.zeros_like(joints)
    vel[1:-1] = (joints[2:] - joints[:-2]) / (2.0 * dt)
    vel[0] = (joints[1] - joints[0]) / dt
    vel[-1] = (joints[-1] - joints[-2]) / dt

    # Per-joint speed ‖vⱼ(t)‖
    joint_speed = np.linalg.norm(vel, axis=-1)  # (F, J)

    # Smooth per-joint speed
    if HAS_SCIPY and F > sg_window:
        w = min(sg_window, F // 2 * 2 - 1)
        if w >= 5:
            for j in range(J):
                joint_speed[:, j] = savgol_filter(joint_speed[:, j], w, 3)

    # ── Kinetic energy: K(t) = Σⱼ ‖vⱼ(t)‖² ────────────────────────
    kinetic_energy = (joint_speed ** 2).sum(axis=1)  # (F,)

    # ── Energy acceleration: dK/dt ───────────────────────────────────
    energy_accel = np.zeros(F)
    energy_accel[1:-1] = (kinetic_energy[2:] - kinetic_energy[:-2]) / (2.0 * dt)
    energy_accel[0] = (kinetic_energy[1] - kinetic_energy[0]) / dt
    energy_accel[-1] = (kinetic_energy[-1] - kinetic_energy[-2]) / dt

    # ── COM: pelvis (joint 0) ────────────────────────────────────────
    com_pos = joints[:, 0, :]     # (F, 3)
    com_height = com_pos[:, 1]    # (F,)

    # ── Compactness: C(t) = (1/J) Σⱼ ‖pⱼ - centroid‖ ───────────────
    centroid = joints.mean(axis=1, keepdims=True)  # (F, 1, 3)
    compactness = np.linalg.norm(joints - centroid, axis=-1).mean(axis=1)  # (F,)

    # ── Cyclic score: windowed autocorrelation ───────────────────────
    cyclic_window = int(cyclic_window_s * fps)
    cyclic_score = np.zeros(F)
    cyclic_regions = []

    # Use pelvis + wrists for cyclic detection (most periodic in powermoves)
    detect_joints = [0, 20, 21]  # pelvis, L.wrist, R.wrist
    detect_signal = joints[:, detect_joints, :].reshape(F, -1)  # (F, 9)

    if F > cyclic_window * 2:
        half = cyclic_window // 2
        for t in range(half, F - half):
            window = detect_signal[t - half:t + half]
            # Normalize
            w_centered = window - window.mean(axis=0)
            w_norm = np.linalg.norm(w_centered)
            if w_norm < 1e-8:
                continue
            w_centered /= w_norm

            # Autocorrelation via dot product at different lags
            # Search lags from 0.2s to 1.5s (typical powermove period)
            min_lag = max(6, int(0.2 * fps))
            max_lag = min(half - 1, int(1.5 * fps))
            best_corr = 0.0
            for lag in range(min_lag, max_lag):
                if t - half + lag + half > F:
                    break
                shifted = detect_signal[t - half + lag:t + half + lag]
                if shifted.shape[0] != window.shape[0]:
                    continue
                s_centered = shifted - shifted.mean(axis=0)
                s_norm = np.linalg.norm(s_centered)
                if s_norm < 1e-8:
                    continue
                s_centered /= s_norm
                corr = float(np.sum(w_centered * s_centered))
                if corr > best_corr:
                    best_corr = corr
            cyclic_score[t] = best_corr

        # Detect cyclic regions
        in_region = False
        region_start = 0
        for t in range(F):
            if cyclic_score[t] >= cyclic_threshold and not in_region:
                in_region = True
                region_start = t
            elif cyclic_score[t] < cyclic_threshold and in_region:
                in_region = False
                if (t - region_start) / fps >= 0.5:  # min 0.5s region
                    cyclic_regions.append({
                        "start_frame": int(region_start),
                        "end_frame": int(t),
                        "start_s": round(region_start / fps, 2),
                        "end_s": round(t / fps, 2),
                        "mean_score": round(float(cyclic_score[region_start:t].mean()), 3),
                    })
        if in_region and (F - region_start) / fps >= 0.5:
            cyclic_regions.append({
                "start_frame": int(region_start),
                "end_frame": int(F - 1),
                "start_s": round(region_start / fps, 2),
                "end_s": round((F - 1) / fps, 2),
                "mean_score": round(float(cyclic_score[region_start:].mean()), 3),
            })

    # Dominant frequency via FFT of pelvis Y
    y_centered = com_height - com_height.mean()
    spectrum = np.abs(np.fft.fft(y_centered))[:F // 2]
    freqs = np.fft.fftfreq(F, 1.0 / fps)[:F // 2]
    if len(spectrum) > 2:
        # Skip DC component
        dominant_idx = np.argmax(spectrum[1:]) + 1
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = 0.0

    # ── Composite scalar: weighted combination, normalized [0,1] ─────
    def _normalize(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-8 else np.zeros_like(x)

    composite = (
        0.4 * _normalize(kinetic_energy) +
        0.2 * _normalize(np.abs(energy_accel)) +
        0.2 * _normalize(1.0 - _normalize(compactness)) +  # inverted: compact = high
        0.2 * _normalize(cyclic_score)
    )

    # ── Musicality (if beats provided) ───────────────────────────────
    mu = 0.0
    tau_star_ms = 0.0
    if beat_times is not None and len(beat_times) > 0 and HAS_SCIPY:
        # Movement energy: M(t) = Σⱼ ‖vⱼ(t)‖ (SG-smoothed total speed)
        M = joint_speed.sum(axis=1)
        M = (M - M.mean()) / (M.std() + 1e-8)

        # Beat signal: Gaussian-convolved onset times
        sigma_frames = 0.05 * fps  # 50ms Gaussian
        H = np.zeros(F)
        for b in beat_times:
            fi = int(b * fps)
            if 0 <= fi < F:
                H += np.exp(-0.5 * ((np.arange(F) - fi) / sigma_frames) ** 2)
        H = (H - H.mean()) / (H.std() + 1e-8)

        # Cross-correlation: μ = max_τ corr(M, H(t-τ))
        max_lag_frames = int(0.2 * fps)  # ±200ms
        corr = correlate(M, H, mode="full")
        corr /= np.sqrt(np.sum(M ** 2) * np.sum(H ** 2)) + 1e-8
        mid = len(corr) // 2
        window = corr[mid - max_lag_frames:mid + max_lag_frames + 1]
        mu = float(np.max(window))
        tau_idx = np.argmax(window)
        tau_star_ms = float((tau_idx - max_lag_frames) * (1000.0 / fps))

    # ── Beat hit/miss classification ─────────────────────────────────
    beat_hits = []
    beat_hit_pct = 0.0
    local_mu = np.zeros(F)
    movement_energy_signal = joint_speed.sum(axis=1) if joint_speed is not None else np.zeros(F)

    if beat_times is not None and len(beat_times) > 0:
        # For each beat, check if movement energy peaks within ±100ms
        beat_window_frames = int(0.1 * fps)  # ±100ms
        M_raw = movement_energy_signal.copy()

        for bt in beat_times:
            fi = int(bt * fps)
            if fi < 0 or fi >= F:
                continue
            lo = max(0, fi - beat_window_frames)
            hi = min(F, fi + beat_window_frames + 1)
            local_energy = M_raw[lo:hi]
            # "Hit" = energy in this window is above median (dancer moving on beat)
            is_hit = float(local_energy.max()) > float(np.median(M_raw))
            beat_hits.append({
                "time_s": round(float(bt), 3),
                "frame": fi,
                "hit": bool(is_hit),
                "energy_at_beat": round(float(M_raw[fi]) if fi < F else 0, 2),
            })

        n_hits = sum(1 for b in beat_hits if b["hit"])
        beat_hit_pct = 100.0 * n_hits / len(beat_hits) if beat_hits else 0.0

        # Sliding-window local μ (2s window)
        if HAS_SCIPY:
            # Shrink the local correlation window for short diagnostic clips.
            win_frames = min(int(2.0 * fps), F if F % 2 == 0 else max(F - 1, 2))
            M_norm = (M_raw - M_raw.mean()) / (M_raw.std() + 1e-8)
            sigma_frames = 0.05 * fps
            H_full = np.zeros(F)
            for b in beat_times:
                fi = int(b * fps)
                if 0 <= fi < F:
                    H_full += np.exp(-0.5 * ((np.arange(F) - fi) / sigma_frames) ** 2)
            H_norm = (H_full - H_full.mean()) / (H_full.std() + 1e-8)

            if win_frames >= 2:
                half = win_frames // 2
                for t in range(half, F - half):
                    m_win = M_norm[t - half:t + half]
                    h_win = H_norm[t - half:t + half]
                    corr_val = correlate(m_win, h_win, mode="full")
                    corr_val /= np.sqrt(np.sum(m_win ** 2) * np.sum(h_win ** 2)) + 1e-8
                    max_lag = int(0.15 * fps)  # ±150ms window
                    mid_c = len(corr_val) // 2
                    local_window = corr_val[max(0, mid_c - max_lag):mid_c + max_lag + 1]
                    local_mu[t] = float(np.max(local_window)) if len(local_window) > 0 else 0
                if half < F:
                    local_mu[:half] = local_mu[half]
                    local_mu[F - half:] = local_mu[F - half - 1]

    # ── Contact detection: foot/hand proximity to ground ─────────────
    # Joints: 7=L.ankle, 8=R.ankle, 10=L.foot, 11=R.foot, 20=L.wrist, 21=R.wrist
    ground_y = float(joints[:, [7, 8, 10, 11], 1].min())  # estimated ground level
    contact_threshold = 0.12  # 12cm above ground = contact

    foot_indices = [7, 8, 10, 11]  # L.ankle, R.ankle, L.foot, R.foot
    hand_indices = [20, 21]         # L.wrist, R.wrist

    contact_feet = np.zeros((F, 4))
    for i, ji in enumerate(foot_indices):
        y_above_ground = joints[:, ji, 1] - ground_y
        # Confidence: 1.0 at ground, 0.0 at threshold
        contact_feet[:, i] = np.clip(1.0 - y_above_ground / contact_threshold, 0, 1)

    contact_hands = np.zeros((F, 2))
    for i, ji in enumerate(hand_indices):
        y_above_ground = joints[:, ji, 1] - ground_y
        contact_hands[:, i] = np.clip(1.0 - y_above_ground / (contact_threshold * 2), 0, 1)

    return WorldState(
        frames=F,
        fps=fps,
        joints=joints,
        kinetic_energy=kinetic_energy,
        energy_accel=energy_accel,
        com_pos=com_pos,
        com_height=com_height,
        compactness=compactness,
        cyclic_score=cyclic_score,
        composite=composite,
        mu=mu,
        tau_star_ms=tau_star_ms,
        beat_times=beat_times,
        cyclic_regions=cyclic_regions,
        dominant_freq_hz=dominant_freq,
        joint_velocities=joint_speed,
        beat_hits=beat_hits,
        beat_hit_pct=beat_hit_pct,
        local_mu=local_mu,
        movement_energy=movement_energy_signal,
        contact_feet=contact_feet,
        contact_hands=contact_hands,
    )


def classify_phases(ws: WorldState, min_duration_s: float = 0.5) -> list[dict]:
    """Auto-detect dance phases from WorldState signals.

    Returns list of {start_s, end_s, dance_type} segments compatible with MoveBar.

    Detection rules (with standing-height gate to avoid mislabeling):
      FREEZE:    kinetic_energy < 15th percentile for 0.5s+
      POWERMOVE: cyclic_score > 0.5 AND com_height drops significantly below standing
      FOOTWORK:  com_height significantly below standing AND not freeze/powermove
      TOPROCK:   everything else (standing, rhythmic movement)

    The standing-height gate prevents toprock-only clips from being mislabeled.
    A significant height drop = COM drops > 30% of standing height range below
    the 75th percentile (i.e., the dancer actually goes down, not just bobbing).
    """
    F = ws.frames
    fps = ws.fps
    labels = np.full(F, 0, dtype=int)  # 0=toprock, 1=footwork, 2=powermove, 3=freeze

    # Standing height gate: estimate if/when the dancer actually goes down
    # Standing height = 75th percentile of COM height (most of the time standing)
    # "Going down" = COM drops > 30% of the total height range below standing
    height_p75 = np.percentile(ws.com_height, 75)
    height_range = ws.com_height.max() - ws.com_height.min()
    drop_threshold = height_p75 - 0.30 * max(height_range, 0.1)
    # Also require the drop to be at least 0.25m (25cm) below standing
    # This prevents small bobbing from triggering footwork
    min_absolute_drop = 0.25  # meters
    effective_threshold = min(drop_threshold, height_p75 - min_absolute_drop)
    is_below_standing = ws.com_height < effective_threshold

    # Check if the dancer EVER goes down significantly
    went_down = np.any(is_below_standing)

    # Freeze: very low energy for sustained period
    ke_thresh = np.percentile(ws.kinetic_energy, 15)
    is_low_energy = ws.kinetic_energy < ke_thresh
    # Require 0.3s sustained low energy to count as freeze
    freeze_min_frames = int(0.3 * fps)
    count = 0
    for t in range(F):
        if is_low_energy[t]:
            count += 1
        else:
            if count >= freeze_min_frames:
                labels[t - count:t] = 3
            count = 0
    if count >= freeze_min_frames:
        labels[F - count:F] = 3

    # Powermove: from cyclic_regions BUT only if COM actually drops
    if went_down:
        for region in ws.cyclic_regions:
            sf = region["start_frame"]
            ef = min(region["end_frame"], F)
            # Only label as powermove if COM is actually low during this region
            region_low = np.mean(is_below_standing[sf:ef]) > 0.3
            if region_low:
                labels[sf:ef] = 2

    # Footwork: COM below standing threshold, not already freeze/power
    # Only applies if the dancer actually went down
    if went_down:
        for t in range(F):
            if labels[t] == 0 and is_below_standing[t]:
                labels[t] = 1

    # Convert frame labels to contiguous segments
    type_names = {0: "toprock", 1: "footwork", 2: "powermove", 3: "freeze"}
    segments = []
    if F == 0:
        return segments

    current_label = labels[0]
    start_frame = 0
    for t in range(1, F):
        if labels[t] != current_label:
            dur = (t - start_frame) / fps
            if dur >= min_duration_s:
                segments.append({
                    "start_s": round(start_frame / fps, 2),
                    "end_s": round(t / fps, 2),
                    "dance_type": type_names[current_label],
                })
            current_label = labels[t]
            start_frame = t

    # Final segment
    dur = (F - start_frame) / fps
    if dur >= min_duration_s:
        segments.append({
            "start_s": round(start_frame / fps, 2),
            "end_s": round(F / fps, 2),
            "dance_type": type_names[current_label],
        })

    return segments


def print_summary(ws: WorldState) -> None:
    """Print human-readable summary of world state."""
    print(f"\n{'='*55}")
    print(f"  WORLD STATE — {ws.frames} frames ({ws.frames/ws.fps:.1f}s @ {ws.fps} fps)")
    print(f"{'='*55}")
    print(f"  Kinetic Energy K(t): mean={ws.kinetic_energy.mean():.1f}  max={ws.kinetic_energy.max():.1f}")
    print(f"  Energy Accel dK/dt:  max_abs={np.abs(ws.energy_accel).max():.1f}")
    print(f"  COM Height h(t):     [{ws.com_height.min():.2f}, {ws.com_height.max():.2f}]m  range={ws.com_height.max()-ws.com_height.min():.2f}m")
    print(f"  Compactness C(t):    [{ws.compactness.min():.3f}, {ws.compactness.max():.3f}]m")
    print(f"  Cyclic Score P(t):   max={ws.cyclic_score.max():.3f}  mean={ws.cyclic_score.mean():.3f}")
    print(f"  Cyclic Regions:      {len(ws.cyclic_regions)}")
    for r in ws.cyclic_regions:
        print(f"    {r['start_s']:.1f}s — {r['end_s']:.1f}s  score={r['mean_score']:.3f}")
    print(f"  Dominant Freq:       {ws.dominant_freq_hz:.2f} Hz ({1/ws.dominant_freq_hz:.2f}s period)" if ws.dominant_freq_hz > 0 else "  Dominant Freq:       none")
    print(f"  Composite Scalar:    [{ws.composite.min():.3f}, {ws.composite.max():.3f}]")
    if ws.beat_times is not None:
        print(f"  Musicality μ:        {ws.mu:.4f}")
        print(f"  Optimal Lag τ*:      {ws.tau_star_ms:.1f}ms")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World State computation")
    parser.add_argument("--joints", default="experiments/results/joints_3d_REAL_seq4.npy")
    parser.add_argument("--beats", default=None, help="Optional .npy of beat times (seconds)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="Save WorldState arrays to this dir")
    args = parser.parse_args()

    joints = np.load(args.joints)
    print(f"Loaded {args.joints}: {joints.shape}")

    beat_times = None
    if args.beats and Path(args.beats).exists():
        beat_times = np.load(args.beats)
        print(f"Loaded beats: {len(beat_times)} times")

    ws = compute_world_state(joints, args.fps, beat_times)
    print_summary(ws)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "kinetic_energy.npy", ws.kinetic_energy)
        np.save(out / "energy_accel.npy", ws.energy_accel)
        np.save(out / "com_pos.npy", ws.com_pos)
        np.save(out / "cyclic_score.npy", ws.cyclic_score)
        np.save(out / "composite.npy", ws.composite)
        print(f"Saved arrays to {out}/")
