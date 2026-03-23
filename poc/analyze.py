"""
Bboy Musicality Analysis — POC v0.1

Computes the audio-motion cross-correlation from:
  - joints_3d.npy (from GVHMR via extract-joints.py)
  - audio extracted from the original video

Usage:
  python analyze.py [--video path/to/video.mp4] [--fps 30] [--no-audio]

Output:
  results/musicality_score.json
  results/correlation_plot.png (if matplotlib available)
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Install with: pip install scipy")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not installed. Install with: pip install librosa")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


RESULTS_DIR = Path(__file__).parent / "results"


def _beats_to_signal(beat_times, n_frames: int, fps: int, sigma_ms: float = 50.0):
    """Convert beat timestamps to a normalized Gaussian beat signal."""
    sigma_frames = sigma_ms / 1000.0 * fps
    t = np.arange(n_frames)
    H = np.zeros(n_frames)
    for b in beat_times:
        fi = int(b * fps)
        if 0 <= fi < n_frames:
            H += np.exp(-0.5 * ((t - fi) / sigma_frames) ** 2)
    H = (H - H.mean()) / (H.std() + 1e-8)
    return H


def _synthetic_beats(n_frames: int, fps: int, bpm: int = 120, sigma_ms: float = 50.0):
    """Generate a synthetic Gaussian beat signal at the given BPM."""
    duration = (n_frames - 1) / fps
    beat_times = np.arange(0, duration, 60.0 / bpm)
    return _beats_to_signal(beat_times, n_frames, fps, sigma_ms), beat_times


def compute_movement_signal(joints_3d: np.ndarray, fps: int = 30, sg_window: int = 31) -> np.ndarray:
    """
    Compute total movement energy M(t) from 3D joint positions.

    joints_3d: (F, J, 3) in meters
    Returns: (F-2,) normalized movement energy
    """
    # Central difference velocity: v(t) = (p(t+1) - p(t-1)) / (2Δt)
    dt = 1.0 / fps
    velocity = (joints_3d[2:] - joints_3d[:-2]) / (2.0 * dt)  # (F-2, J, 3) m/s

    # Per-joint speed (magnitude)
    speed = np.linalg.norm(velocity, axis=-1)  # (F-2, J)

    # Savitzky-Golay smoothing per joint
    if HAS_SCIPY and sg_window > 1:
        speed_smooth = np.zeros_like(speed)
        for j in range(speed.shape[1]):
            speed_smooth[:, j] = savgol_filter(
                speed[:, j], window_length=min(sg_window, speed.shape[0] // 2 * 2 - 1), polyorder=3
            )
    else:
        speed_smooth = speed

    # Total movement energy
    M = speed_smooth.sum(axis=1)  # (F-2,)

    # Normalize to zero mean, unit variance
    M = (M - M.mean()) / (M.std() + 1e-8)

    return M, speed_smooth


def compute_audio_signal(video_path: str, n_frames: int, fps: int = 30, sigma_ms: float = 50.0) -> np.ndarray:
    """
    Extract beat signal H(t) from audio.

    Returns: (n_frames,) normalized audio hotness signal
    """
    if not HAS_LIBROSA:
        print("  librosa not available. Generating synthetic beat signal for testing.")
        return _synthetic_beats(n_frames, fps, sigma_ms=sigma_ms)

    # Extract audio from video
    audio_path = str(RESULTS_DIR / "audio_temp.wav")
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-ar", "44100", "-ac", "1", audio_path, "-y"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.decode()[:500]}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio extraction produced no output: {audio_path}")

    # Load audio and detect onsets
    y, sr = librosa.load(audio_path, sr=44100)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    beat_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

    print(f"  Detected {len(beat_times)} beats ({60.0 / np.median(np.diff(beat_times)):.0f} BPM estimate)")

    # Create smooth beat signal at video frame rate
    H = _beats_to_signal(beat_times, n_frames, fps, sigma_ms)

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return H, beat_times


def compute_musicality(M: np.ndarray, H: np.ndarray, fps: int = 30, max_lag_ms: float = 200.0) -> dict:
    """
    Compute audio-motion cross-correlation.

    M: (N,) movement energy signal
    H: (N,) audio hotness signal
    fps: video frame rate
    max_lag_ms: maximum lag to search (±)

    Returns: dict with mu, tau_star_ms, correlation curve
    """
    n = min(len(M), len(H))
    M, H = M[:n], H[:n]
    max_lag_frames = int(max_lag_ms / 1000.0 * fps)

    corr = np.correlate(M, H, mode="full")
    corr /= np.sqrt(np.sum(M ** 2) * np.sum(H ** 2)) + 1e-8

    mid = len(corr) // 2
    window = corr[mid - max_lag_frames : mid + max_lag_frames + 1]
    lags_ms = np.arange(-max_lag_frames, max_lag_frames + 1) * (1000.0 / fps)

    mu = float(np.max(window))
    tau_star_ms = float(lags_ms[np.argmax(window)])

    return {
        "mu": mu,
        "tau_star_ms": tau_star_ms,
        "max_lag_ms": max_lag_ms,
        "fps": fps,
        "n_frames": n,
        "correlation_window": window.tolist(),
        "lags_ms": lags_ms.tolist(),
    }


def compute_per_joint_snr(speed_smooth: np.ndarray, fps: int = 30) -> dict:
    """
    Estimate per-joint velocity SNR.
    Uses high-frequency residual after SG smoothing as noise estimate.
    """
    snr_per_joint = {}

    # Joint names (SMPL 22-joint convention, first 22)
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]

    for j in range(min(speed_smooth.shape[1], len(joint_names))):
        signal = speed_smooth[:, j]
        signal_power = float(np.mean(signal ** 2))

        # Estimate noise as residual from heavier smoothing
        if HAS_SCIPY and len(signal) > 61:
            heavy_smooth = savgol_filter(signal, window_length=61, polyorder=3)
            noise = signal - heavy_smooth
            noise_power = float(np.mean(noise ** 2))
        else:
            noise_power = signal_power * 0.1  # rough estimate

        snr = signal_power / (noise_power + 1e-8)
        snr_per_joint[joint_names[j]] = {
            "snr_linear": round(snr, 2),
            "snr_db": round(10 * np.log10(snr + 1e-8), 1),
            "mean_speed_m_s": round(float(np.mean(signal)), 3),
        }

    return snr_per_joint


def main():
    parser = argparse.ArgumentParser(description="Bboy Musicality Analysis — POC v0.1")
    parser.add_argument("--video", type=str, default=None, help="Path to original video (for audio extraction)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--sg-window", type=int, default=31, help="Savitzky-Golay smoothing window")
    parser.add_argument("--no-audio", action="store_true", help="Use synthetic beat signal instead of real audio")
    parser.add_argument("--joints", type=str, default=str(RESULTS_DIR / "joints_3d.npy"), help="Path to joints .npy")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║  Bboy Musicality Analysis — POC v0.1     ║")
    print("╚══════════════════════════════════════════╝")

    # Load joint positions
    print("\n▸ Loading joint positions...")
    joints_path = args.joints
    if not os.path.exists(joints_path):
        print(f"ERROR: {joints_path} not found. Run GVHMR inference first.")
        sys.exit(1)

    joints_3d = np.load(joints_path)
    print(f"  Shape: {joints_3d.shape} (frames × joints × xyz)")
    print(f"  Duration: {joints_3d.shape[0] / args.fps:.1f}s at {args.fps} FPS")

    # Stage 2: Movement signal
    print("\n▸ Computing movement signal...")
    M, speed_smooth = compute_movement_signal(joints_3d, fps=args.fps, sg_window=args.sg_window)
    print(f"  Movement energy: {len(M)} samples")
    print(f"  SG window: {args.sg_window} frames ({args.sg_window / args.fps * 1000:.0f}ms)")

    # Per-joint SNR
    print("\n▸ Computing per-joint velocity SNR...")
    snr = compute_per_joint_snr(speed_smooth, fps=args.fps)
    usable_joints = sum(1 for j in snr.values() if j["snr_linear"] > 3)
    print(f"  {usable_joints}/{len(snr)} joints have SNR > 3:1 (usable)")

    # Stage 3: Audio signal
    print("\n▸ Computing audio signal...")
    if args.no_audio or args.video is None:
        H, beat_times = _synthetic_beats(len(M), args.fps)
        audio_source = "synthetic_120bpm"
    else:
        H, beat_times = compute_audio_signal(args.video, len(M), fps=args.fps)
        audio_source = args.video

    # Stage 4: Cross-correlation
    print("\n▸ Computing audio-motion cross-correlation...")
    result = compute_musicality(M, H, fps=args.fps)

    mu = result["mu"]
    tau = result["tau_star_ms"]

    # Build final output
    output = {
        "musicality_score": mu,
        "optimal_lag_ms": tau,
        "audio_source": audio_source,
        "fps": args.fps,
        "sg_window": args.sg_window,
        "n_frames": joints_3d.shape[0],
        "n_joints": joints_3d.shape[1],
        "duration_s": joints_3d.shape[0] / args.fps,
        "per_joint_snr": snr,
        "interpretation": {
            "musicality": "STRONG" if mu > 0.4 else "MODERATE" if mu > 0.2 else "WEAK",
            "timing": "ANTICIPATES" if tau < -30 else "REACTS" if tau > 30 else "ON_BEAT",
        },
        "correlation_curve": result,
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "musicality_score.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  MUSICALITY SCORE")
    print(f"{'=' * 50}")
    print(f"  μ = {mu:.3f}")
    print(f"  τ* = {tau:.0f} ms")
    print(f"")
    if mu > 0.4:
        print(f"  → STRONG musicality (dancer hits beats consistently)")
    elif mu > 0.2:
        print(f"  → MODERATE musicality")
    else:
        print(f"  → WEAK or no detectable musicality")
    print(f"")
    if tau < -30:
        print(f"  → Dancer ANTICIPATES beats by {abs(tau):.0f}ms (elite)")
    elif tau > 30:
        print(f"  → Dancer REACTS to beats with {tau:.0f}ms delay")
    else:
        print(f"  → Dancer is ON the beat (±30ms)")
    print(f"")
    print(f"  Usable joints (SNR > 3): {usable_joints}/{len(snr)}")
    print(f"  Saved: results/musicality_score.json")
    print(f"{'=' * 50}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        # Plot 1: Movement + Audio signals
        t = np.arange(len(M)) / args.fps
        axes[0].plot(t, M, label="Movement Energy M(t)", alpha=0.8, linewidth=0.8)
        axes[0].plot(t[:len(H)], H[:len(M)], label="Audio Hotness H(t)", alpha=0.8, linewidth=0.8)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Normalized signal")
        axes[0].set_title(f"Movement vs Audio — μ = {mu:.3f}, τ* = {tau:.0f}ms")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Cross-correlation
        lags = np.array(result["lags_ms"])
        corr_window = np.array(result["correlation_window"])
        axes[1].plot(lags, corr_window, "b-", linewidth=1.5)
        axes[1].axvline(x=tau, color="r", linestyle="--", label=f"τ* = {tau:.0f}ms")
        axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[1].set_xlabel("Lag (ms)")
        axes[1].set_ylabel("Correlation")
        axes[1].set_title("Audio-Motion Cross-Correlation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Per-joint SNR
        joints = list(snr.keys())
        snr_values = [snr[j]["snr_db"] for j in joints]
        colors = ["green" if snr[j]["snr_linear"] > 3 else "red" for j in joints]
        axes[2].bar(range(len(joints)), snr_values, color=colors, alpha=0.7)
        axes[2].set_xticks(range(len(joints)))
        axes[2].set_xticklabels(joints, rotation=45, ha="right", fontsize=7)
        axes[2].axhline(y=10 * np.log10(3), color="orange", linestyle="--", label="SNR = 3 threshold")
        axes[2].set_ylabel("SNR (dB)")
        axes[2].set_title("Per-Joint Velocity SNR (green = usable)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "correlation_plot.png", dpi=150)
        print(f"\n  Saved: results/correlation_plot.png")
        plt.close()


if __name__ == "__main__":
    main()
