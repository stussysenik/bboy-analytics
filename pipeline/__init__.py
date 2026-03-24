"""Battle analytics pipeline — clean, modular interface."""
from .config import (
    PROJECT_ROOT, JOSH_DIR, BODY_MODELS_DIR, DEFAULT_FPS,
    SMPL_JOINT_COUNT, JOINT_NAMES, resolve_body_model_path,
)
from .extract import extract_josh_joints, extract_gvhmr_joints
from .analyze import compute_movement_signal, compute_musicality, compute_per_joint_snr
from .inference import run_josh_pipeline

__all__ = [
    "extract_josh_joints", "extract_gvhmr_joints",
    "compute_movement_signal", "compute_musicality", "compute_per_joint_snr",
    "run_josh_pipeline",
    "PROJECT_ROOT", "JOSH_DIR", "BODY_MODELS_DIR", "DEFAULT_FPS",
    "SMPL_JOINT_COUNT", "JOINT_NAMES", "resolve_body_model_path",
]
