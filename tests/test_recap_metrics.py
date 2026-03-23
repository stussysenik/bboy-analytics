"""Tests for battle recap metric computation."""

import numpy as np
from extreme_motion_reimpl.recap.metrics import (
    compute_musicality,
    compute_energy,
    compute_flow,
    compute_space,
    compute_complexity,
    compute_all_metrics,
    compute_per_joint_snr,
)


def _make_joints(n_frames: int = 300, n_joints: int = 22) -> np.ndarray:
    """Generate synthetic joint data with rhythmic movement."""
    np.random.seed(42)
    t = np.linspace(0, 10, n_frames)
    joints = np.zeros((n_frames, n_joints, 3))
    for j in range(n_joints):
        joints[:, j, 0] = np.sin(t * 2 + j * 0.3) * 0.3
        joints[:, j, 1] = 1.0 + j * 0.05 + np.sin(t * 4 + j) * 0.05
        joints[:, j, 2] = t * 0.05 + np.sin(t + j * 0.5) * 0.1
    joints += np.random.randn(*joints.shape) * 0.005
    return joints


def test_musicality_returns_valid_score():
    joints = _make_joints()
    beats = np.arange(0, 10, 0.5)  # 120 BPM
    result = compute_musicality(joints, beats, fps=30.0)
    assert "mu" in result
    assert 0.0 <= result["mu"] <= 1.0
    assert "tau_star_ms" in result
    assert result["interpretation"]["musicality"] in ("STRONG", "MODERATE", "WEAK")


def test_energy_detects_peaks():
    joints = _make_joints()
    result = compute_energy(joints, fps=30.0)
    assert "energy_curve" in result
    assert len(result["energy_curve"]) > 0
    assert result["mean_energy"] > 0
    assert len(result["energy_sections"]) > 0


def test_flow_score_positive():
    joints = _make_joints()
    result = compute_flow(joints, fps=30.0)
    assert result["flow_score"] > 0
    assert result["mean_jerk"] > 0
    assert len(result["per_joint_jerk"]) > 0


def test_space_coverage():
    joints = _make_joints()
    result = compute_space(joints, fps=30.0)
    assert result["stage_coverage_m2"] > 0
    assert result["vertical_range_m"] > 0
    assert len(result["com_trajectory"]) == joints.shape[0]


def test_complexity_freeze_detection():
    # Create joints with a freeze segment (stationary for 1 second)
    joints = _make_joints(n_frames=300)
    joints[100:130, :, :] = joints[100, :, :]  # 30 frames = 1s freeze at 30fps
    result = compute_complexity(joints, fps=30.0)
    assert result["freeze_count"] >= 1
    assert result["total_freeze_time_s"] > 0


def test_complexity_inversion_detection():
    joints = _make_joints(n_frames=300)
    # Force inversion: pelvis Y > head Y for 10 frames
    joints[50:60, 0, 1] = 2.0   # pelvis high
    joints[50:60, 15, 1] = 0.5  # head low
    result = compute_complexity(joints, fps=30.0)
    assert result["inversion_count"] >= 1


def test_per_joint_snr():
    joints = _make_joints()
    snr = compute_per_joint_snr(joints, fps=30.0)
    assert "pelvis" in snr
    assert snr["pelvis"]["snr_linear"] > 0


def test_compute_all_metrics():
    joints = _make_joints()
    result = compute_all_metrics(joints, fps=30.0)
    assert "musicality" in result
    assert "energy" in result
    assert "flow" in result
    assert "space" in result
    assert "complexity" in result
    assert "meta" in result
    assert result["meta"]["n_frames"] == 300
