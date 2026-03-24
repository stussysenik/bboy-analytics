"""Centralized paths and constants for the battle analytics pipeline."""
from pathlib import Path

PROJECT_ROOT = Path("/teamspace/studios/this_studio")
JOSH_DIR = PROJECT_ROOT / "josh"
JOSH_INPUT_DIR = PROJECT_ROOT / "josh_input"
GVHMR_SRC_DIR = PROJECT_ROOT / "gvhmr_src"
BODY_MODELS_DIR = GVHMR_SRC_DIR / "inputs" / "checkpoints" / "body_models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_FPS = 30.0
SMPL_JOINT_COUNT = 24

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Joint indices for quick tests
HEAD_IDX = 15
PELVIS_IDX = 0

# Thresholds
IDENTITY_TRACKING_THRESHOLD_M = 0.3
MUSICALITY_STRONG_THRESHOLD = 0.4
MUSICALITY_MODERATE_THRESHOLD = 0.2


def resolve_body_model_path(preferred: str | None = None) -> Path:
    """Resolve body model path with JOSH-first fallback."""
    if preferred and Path(preferred).is_dir():
        return Path(preferred)
    josh_smpl = JOSH_DIR / "data" / "smpl"
    if josh_smpl.is_dir():
        return josh_smpl.parent
    if BODY_MODELS_DIR.is_dir():
        return BODY_MODELS_DIR
    raise FileNotFoundError(f"No body models found. Checked: {josh_smpl}, {BODY_MODELS_DIR}")
