"""Audio analysis: beats, BPM, sections, flow."""

from __future__ import annotations

import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Any


def load_brace_beats(audio_beats_path: Path, video_id: str, seq_idx: int = 0) -> dict | None:
    """Load ground truth beats from BRACE audio_beats.json."""
    if not audio_beats_path.exists():
        return None
    with open(audio_beats_path) as f:
        all_beats = json.load(f)
    key = f"{video_id}.{seq_idx}"
    return all_beats.get(key)


def extract_beats_librosa(video_path: Path, sr: int = 44100) -> dict[str, Any]:
    """Extract beats from video audio using librosa."""
    import librosa

    # Extract audio via ffmpeg
    audio_path = video_path.parent / f".{video_path.stem}_audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-ar", str(sr), "-ac", "1", str(audio_path), "-y"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:300]}")

    try:
        y, _ = librosa.load(str(audio_path), sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        beat_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512).tolist()

        # BPM
        if len(beat_times) >= 2:
            intervals = np.diff(beat_times)
            bpm = 60.0 / float(np.median(intervals))
            bpm_stability = 1.0 - float(np.std(intervals) / (np.mean(intervals) + 1e-8))
        else:
            bpm, bpm_stability = 0.0, 0.0

        # Onset density curve (smoothed onset strength resampled to 1 Hz)
        onset_density = onset_env.tolist()

        return {
            "beat_times": beat_times,
            "bpm": round(bpm, 1),
            "bpm_stability": round(max(0, bpm_stability), 3),
            "n_beats": len(beat_times),
            "onset_density": onset_density,
            "source": "librosa",
        }
    finally:
        audio_path.unlink(missing_ok=True)


def analyze_audio(
    video_path: Path | None = None,
    brace_beats_path: Path | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Full audio analysis. Tries BRACE ground truth first, falls back to librosa."""
    # Try BRACE ground truth
    if brace_beats_path and video_id:
        gt = load_brace_beats(brace_beats_path, video_id)
        if gt:
            beat_times = gt.get("beats_sec", [])
            bpm = gt.get("bpm", 0)
            intervals = np.diff(beat_times) if len(beat_times) >= 2 else [0.5]
            bpm_stability = 1.0 - float(np.std(intervals) / (np.mean(intervals) + 1e-8))
            return {
                "beat_times": beat_times,
                "bpm": round(bpm, 1),
                "bpm_stability": round(max(0, bpm_stability), 3),
                "n_beats": len(beat_times),
                "source": "brace_ground_truth",
            }

    # Try librosa
    if video_path and video_path.exists():
        try:
            return extract_beats_librosa(video_path)
        except (ImportError, RuntimeError) as e:
            print(f"  WARNING: librosa extraction failed: {e}")

    # Synthetic fallback
    print("  Using synthetic 120 BPM beats")
    beat_times = np.arange(0, 120, 0.5).tolist()  # 2 min at 120 BPM
    return {
        "beat_times": beat_times,
        "bpm": 120.0,
        "bpm_stability": 1.0,
        "n_beats": len(beat_times),
        "source": "synthetic_120bpm",
    }
