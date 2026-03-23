"""Tests for recap validation."""

import json
import numpy as np
import pytest
from pathlib import Path
from extreme_motion_reimpl.recap.validate import (
    validate_joints,
    validate_metadata,
    validate_output_dir,
    ValidationError,
)


def test_validate_joints_good(tmp_path):
    joints = np.random.randn(100, 22, 3).astype(np.float32)
    path = tmp_path / "joints_3d.npy"
    np.save(path, joints)
    result = validate_joints(path)
    assert result.shape == (100, 22, 3)


def test_validate_joints_missing(tmp_path):
    with pytest.raises(ValidationError, match="not found"):
        validate_joints(tmp_path / "missing.npy")


def test_validate_joints_wrong_shape(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((10, 5)))
    with pytest.raises(ValidationError, match="Expected"):
        validate_joints(path)


def test_validate_joints_nan(tmp_path):
    joints = np.full((100, 22, 3), np.nan)
    path = tmp_path / "nan.npy"
    np.save(path, joints)
    with pytest.raises(ValidationError, match="NaN"):
        validate_joints(path)


def test_validate_metadata_good(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"n_frames": 100, "n_joints": 22}))
    result = validate_metadata(path)
    assert result["n_frames"] == 100


def test_validate_output_dir(tmp_path):
    (tmp_path / "metrics.json").write_text("{}")
    (tmp_path / "joints_3d.npy").write_bytes(b"fake")
    status = validate_output_dir(tmp_path)
    assert status["metrics.json"] is True
    assert status["recap.mp4"] is False
