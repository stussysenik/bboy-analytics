"""Musicality analysis — cross-correlation of 3D joint velocities with audio beats."""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

from .config import JOINT_NAMES, RESULTS_DIR

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def compute_movement_signal(joints_3d: np.ndarray, fps: float = 30.0, sg_window: int = 31):
    """Compute total movement energy M(t) from 3D joint positions.

    Args:
        joints_3d: (F, J, 3) in meters
    Returns:
        M: (F-2,) normalized movement energy
        speed_smooth: (F-2, J) per-joint smoothed speeds
    """
    dt = 1.0 / fps
    velocity = (joints_3d[2:] - joints_3d[:-2]) / (2.0 * dt)
    speed = np.linalg.norm(velocity, axis=-1)

    if HAS_SCIPY and sg_window > 1:
        w = min(sg_window, speed.shape[0] // 2 * 2 - 1)
        speed_smooth = np.column_stack([
            savgol_filter(speed[:, j], window_length=w, polyorder=3)
            for j in range(speed.shape[1])
        ])
    else:
        speed_smooth = speed

    M = speed_smooth.sum(axis=1)
    M = (M - M.mean()) / (M.std() + 1e-8)
    return M, speed_smooth


def compute_audio_signal(beat_times: np.ndarray, n_frames: int, fps: float = 30.0, sigma_ms: float = 50.0):
    """Convert beat timestamps to a normalized Gaussian beat signal H(t)."""
    sigma_frames = sigma_ms / 1000.0 * fps
    t = np.arange(n_frames)
    H = np.zeros(n_frames)
    for b in beat_times:
        fi = int(b * fps)
        if 0 <= fi < n_frames:
            H += np.exp(-0.5 * ((t - fi) / sigma_frames) ** 2)
    H = (H - H.mean()) / (H.std() + 1e-8)
    return H


def extract_beats_from_video(video_path: str) -> np.ndarray:
    """Extract beat times from video audio using librosa."""
    if not HAS_LIBROSA:
        raise ImportError("librosa required: pip install librosa")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-ar", "44100", "-ac", "1", tmp_path, "-y"],
            capture_output=True, check=True,
        )
        y, sr = librosa.load(tmp_path, sr=44100)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        beat_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
        return librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def compute_musicality(M: np.ndarray, H: np.ndarray, fps: float = 30.0, max_lag_ms: float = 200.0) -> dict:
    """Compute audio-motion cross-correlation.

    Returns dict with mu, tau_star_ms, and correlation curve.
    """
    n = min(len(M), len(H))
    M, H = M[:n], H[:n]
    max_lag_frames = int(max_lag_ms / 1000.0 * fps)

    corr = np.correlate(M, H, mode="full")
    corr /= np.sqrt(np.sum(M ** 2) * np.sum(H ** 2)) + 1e-8

    mid = len(corr) // 2
    window = corr[mid - max_lag_frames: mid + max_lag_frames + 1]
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


def compute_per_joint_snr(speed_smooth: np.ndarray, fps: float = 30.0) -> dict:
    """Per-joint velocity SNR using heavy SG residual as noise estimate."""
    result = {}
    for j in range(min(speed_smooth.shape[1], len(JOINT_NAMES))):
        signal = speed_smooth[:, j]
        signal_power = float(np.mean(signal ** 2))
        if HAS_SCIPY and len(signal) > 61:
            heavy = savgol_filter(signal, window_length=61, polyorder=3)
            noise_power = float(np.mean((signal - heavy) ** 2))
        else:
            noise_power = signal_power * 0.1
        snr = signal_power / (noise_power + 1e-8)
        result[JOINT_NAMES[j]] = {
            "snr_linear": round(snr, 2),
            "snr_db": round(10 * np.log10(snr + 1e-8), 1),
            "mean_speed_m_s": round(float(np.mean(signal)), 3),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Musicality analysis")
    parser.add_argument("--joints", required=True, help="Path to joints_3d.npy")
    parser.add_argument("--video", default=None, help="Video for beat extraction")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    joints = np.load(args.joints)
    M, speed = compute_movement_signal(joints, fps=args.fps)

    if args.video:
        beats = extract_beats_from_video(args.video)
        H = compute_audio_signal(beats, len(M), fps=args.fps)
    else:
        # Synthetic 120 BPM
        duration = (len(M) - 1) / args.fps
        beats = np.arange(0, duration, 0.5)
        H = compute_audio_signal(beats, len(M), fps=args.fps)

    result = compute_musicality(M, H, fps=args.fps)
    snr = compute_per_joint_snr(speed, fps=args.fps)
    result["per_joint_snr"] = snr

    out = args.output or str(RESULTS_DIR / "musicality_score.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"mu={result['mu']:.3f}, tau*={result['tau_star_ms']:.0f}ms -> {out}")
