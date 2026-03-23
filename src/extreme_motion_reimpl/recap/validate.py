"""Validate inputs and outputs at each pipeline step."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


class ValidationError(Exception):
    """Raised when pipeline data fails validation."""


def validate_joints(path: Path) -> np.ndarray:
    """Load and validate joints_3d.npy."""
    if not path.exists():
        raise ValidationError(f"joints file not found: {path}")
    joints = np.load(path)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValidationError(f"Expected (F, J, 3), got {joints.shape}")
    if joints.shape[0] < 10:
        raise ValidationError(f"Too few frames: {joints.shape[0]}")
    if joints.shape[1] < 22:
        raise ValidationError(f"Expected ≥22 joints, got {joints.shape[1]}")
    if np.any(np.isnan(joints)):
        raise ValidationError("NaN values in joint data")
    return joints


def validate_metadata(path: Path) -> dict:
    """Load and validate metadata.json."""
    if not path.exists():
        raise ValidationError(f"metadata not found: {path}")
    with open(path) as f:
        meta = json.load(f)
    for key in ("n_frames", "n_joints"):
        if key not in meta and key not in meta.get("stats", {}):
            raise ValidationError(f"Missing key in metadata: {key}")
    return meta


def validate_audio(path: Path) -> dict:
    """Load and validate audio_analysis.json."""
    if not path.exists():
        raise ValidationError(f"audio analysis not found: {path}")
    with open(path) as f:
        audio = json.load(f)
    if "beat_times" not in audio:
        raise ValidationError("Missing 'beat_times' in audio analysis")
    return audio


def validate_metrics(path: Path) -> dict:
    """Load and validate metrics.json."""
    if not path.exists():
        raise ValidationError(f"metrics not found: {path}")
    with open(path) as f:
        metrics = json.load(f)
    required = ["musicality", "energy", "flow", "space", "complexity"]
    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValidationError(f"Missing metric categories: {missing}")
    return metrics


def validate_output_dir(output_dir: Path) -> dict[str, bool]:
    """Check which outputs exist in a recap directory."""
    expected = [
        "joints_3d.npy", "metadata.json", "audio_analysis.json",
        "metrics.json", "timeline.json", "freeze_events.json",
        "energy_flow.png", "spatial_heatmap.png", "com_trajectory.png",
        "recap.mp4", "summary.txt",
    ]
    return {f: (output_dir / f).exists() for f in expected}
