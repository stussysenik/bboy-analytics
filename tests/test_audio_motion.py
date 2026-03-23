from __future__ import annotations

import numpy as np

from extreme_motion_reimpl.audio_motion import audio_motion_alignment, synthetic_alignment_payload


def _signals(samples: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    fps = 32.0
    sample_rate = 3200
    t = np.arange(samples) / fps
    movement = np.sin(2 * np.pi * 1.25 * t)
    joints = np.stack(
        [
            np.stack([movement, np.cos(2 * np.pi * 1.25 * t), movement * 0.2], axis=-1),
            np.stack([movement * 0.9, np.sin(2 * np.pi * 1.25 * t + 0.2), movement * 0.1], axis=-1),
        ],
        axis=1,
    )
    audio_t = np.arange(int(samples * (sample_rate / fps))) / sample_rate
    aligned_audio = np.sin(2 * np.pi * 1.25 * audio_t)
    shifted_audio = np.sin(2 * np.pi * 1.25 * (audio_t + 0.25))
    return joints, aligned_audio, shifted_audio, fps, sample_rate


def test_synthetic_payload_is_well_aligned() -> None:
    payload = synthetic_alignment_payload(samples=256)
    metrics = payload["metrics"]

    assert metrics["alignment_peak"] > 0.7
    assert abs(metrics["alignment_lag_ms"]) < 40.0
    assert metrics["alignment_stability"] > 0.3
    assert metrics["derivative_snr_db"] > 5.0


def test_alignment_detects_shifted_audio_as_worse() -> None:
    joints, aligned_audio, shifted_audio, fps, sample_rate = _signals()
    aligned = audio_motion_alignment(joints, aligned_audio, fps=fps, sample_rate=sample_rate)
    shifted = audio_motion_alignment(joints, shifted_audio, fps=fps, sample_rate=sample_rate)

    assert aligned.alignment_peak > shifted.alignment_peak
    assert abs(aligned.alignment_lag_ms) < abs(shifted.alignment_lag_ms)
