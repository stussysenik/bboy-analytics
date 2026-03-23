from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioMotionMetrics:
    alignment_peak: float
    alignment_lag_ms: float
    alignment_stability: float
    derivative_snr_db: float
    velocity_energy: float
    acceleration_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "alignment_peak": round(self.alignment_peak, 4),
            "alignment_lag_ms": round(self.alignment_lag_ms, 2),
            "alignment_stability": round(self.alignment_stability, 4),
            "derivative_snr_db": round(self.derivative_snr_db, 4),
            "velocity_energy": round(self.velocity_energy, 4),
            "acceleration_energy": round(self.acceleration_energy, 4),
        }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, [(pad, pad)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    smoothed = np.apply_along_axis(lambda axis: np.convolve(axis, kernel, mode="valid"), 0, padded)
    return smoothed[: values.shape[0]]


def smooth_pose_sequence(joints: np.ndarray, window: int = 5) -> np.ndarray:
    return _moving_average(np.asarray(joints, dtype=np.float64), window)


def kinematic_derivatives(joints: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float64)
    velocity = np.diff(joints, axis=0, prepend=joints[:1]) * fps
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * fps
    return velocity, acceleration


def movement_energy_signal(joints: np.ndarray, fps: float, smoothing_window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = smooth_pose_sequence(joints, window=smoothing_window)
    velocity, acceleration = kinematic_derivatives(smoothed, fps)
    velocity_mag = np.linalg.norm(velocity, axis=-1).mean(axis=1)
    acceleration_mag = np.linalg.norm(acceleration, axis=-1).mean(axis=1)
    motion_signal = velocity_mag + (0.35 * acceleration_mag)
    return motion_signal, velocity_mag, acceleration_mag


def _frame_rms(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    frames = []
    for start in range(0, max(len(audio) - frame_size + 1, 1), hop_size):
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        frames.append(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    return np.asarray(frames, dtype=np.float64)


def movement_spectrogram(signal: np.ndarray, frame_size: int = 32, hop_size: int = 16) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    frames = []
    window = np.hanning(frame_size)
    for start in range(0, max(len(signal) - frame_size + 1, 1), hop_size):
        frame = signal[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        frames.append(power)
    return np.asarray(frames, dtype=np.float64)


def _normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - signal.mean()
    scale = centered.std() or 1.0
    return centered / scale


def _best_cross_correlation(signal_a: np.ndarray, signal_b: np.ndarray, max_lag: int) -> tuple[float, int]:
    best_score = -1.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = signal_a[-lag:]
            right = signal_b[: len(left)]
        elif lag > 0:
            left = signal_a[:-lag]
            right = signal_b[lag:]
        else:
            left = signal_a
            right = signal_b

        if len(left) < 4 or len(right) < 4:
            continue

        score = float(np.corrcoef(left, right)[0, 1])
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_score, best_lag


def derivative_snr(raw_joints: np.ndarray, smoothed_joints: np.ndarray, fps: float) -> float:
    raw_velocity, _ = kinematic_derivatives(raw_joints, fps)
    clean_velocity, _ = kinematic_derivatives(smoothed_joints, fps)
    signal_energy = np.mean(np.square(clean_velocity))
    noise_energy = np.mean(np.square(raw_velocity - clean_velocity))
    if noise_energy <= 1e-12:
        return 60.0
    return float(10.0 * np.log10((signal_energy + 1e-12) / noise_energy))


def audio_motion_alignment(
    joints: np.ndarray,
    audio: np.ndarray,
    fps: float,
    sample_rate: int,
    lag_window_ms: int = 200,
    smoothing_window: int = 5,
) -> AudioMotionMetrics:
    joints = np.asarray(joints, dtype=np.float64)
    audio = np.asarray(audio, dtype=np.float64)

    smoothed = smooth_pose_sequence(joints, smoothing_window)
    motion_signal, velocity_mag, acceleration_mag = movement_energy_signal(
        smoothed,
        fps,
        smoothing_window=1,
    )

    frame_size = max(int(sample_rate / fps), 8)
    hop_size = frame_size
    audio_envelope = _frame_rms(audio, frame_size=frame_size, hop_size=hop_size)

    steps = min(len(motion_signal), len(audio_envelope))
    motion_signal = _normalize(motion_signal[:steps])
    audio_envelope = _normalize(audio_envelope[:steps])

    max_lag_frames = max(int(round((lag_window_ms / 1000.0) * fps)), 1)
    peak, lag = _best_cross_correlation(motion_signal, audio_envelope, max_lag=max_lag_frames)

    window = max(steps // 4, 8)
    local_scores = []
    for start in range(0, max(steps - window + 1, 1), max(window // 2, 1)):
        motion_chunk = motion_signal[start : start + window]
        audio_chunk = audio_envelope[start : start + window]
        if len(motion_chunk) < 8 or len(audio_chunk) < 8:
            continue
        local_peak, _ = _best_cross_correlation(motion_chunk, audio_chunk, max_lag=max_lag_frames)
        local_scores.append(max(local_peak, 0.0))

    stability = 0.0
    if local_scores:
        stability = float(np.clip(np.mean(local_scores) - np.std(local_scores), 0.0, 1.0))

    snr_db = derivative_snr(joints, smoothed, fps)

    return AudioMotionMetrics(
        alignment_peak=float(max(peak, 0.0)),
        alignment_lag_ms=float((lag / fps) * 1000.0),
        alignment_stability=stability,
        derivative_snr_db=snr_db,
        velocity_energy=float(np.mean(velocity_mag)),
        acceleration_energy=float(np.mean(acceleration_mag)),
    )


def synthetic_alignment_payload(samples: int = 256) -> dict[str, object]:
    fps = 32.0
    sample_rate = 3200
    t = np.arange(samples) / fps
    movement = np.sin(2 * np.pi * 1.5 * t)
    joints = np.stack(
        [
            np.stack([movement, np.cos(2 * np.pi * 1.5 * t), movement * 0.25], axis=-1),
            np.stack([movement * 0.8, np.sin(2 * np.pi * 1.5 * t + 0.3), movement * 0.2], axis=-1),
        ],
        axis=1,
    )
    audio_t = np.arange(int(samples * (sample_rate / fps))) / sample_rate
    audio = np.sin(2 * np.pi * 1.5 * audio_t)
    metrics = audio_motion_alignment(joints=joints, audio=audio, fps=fps, sample_rate=sample_rate)
    return {
        "fps": fps,
        "sample_rate": sample_rate,
        "metrics": metrics.to_dict(),
    }


def synthetic_alignment_payload_json(samples: int = 256) -> str:
    return json.dumps(synthetic_alignment_payload(samples), indent=2)
