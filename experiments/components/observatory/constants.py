"""Constants extracted from autoresearch analyze_motion.py.

Single source of truth for joint masses, bone pairs, groups, and phase categories.
"""

from __future__ import annotations

import numpy as np

CATEGORY_NAMES = ["toprock", "footwork", "power", "freeze", "transition"]

JOINT_MASSES_KG = {
    0: 11.17, 1: 2.78, 2: 2.78, 3: 5.0, 4: 3.28, 5: 3.28,
    6: 3.0, 7: 0.61, 8: 0.61, 9: 2.5, 10: 0.97, 11: 0.97,
    12: 1.5, 13: 0.5, 14: 0.5, 15: 5.0, 16: 2.0, 17: 2.0,
    18: 1.14, 19: 1.14, 20: 0.45, 21: 0.45, 22: 0.41, 23: 0.41,
}

JOINT_WEIGHTS = np.array([JOINT_MASSES_KG[j] for j in range(24)], dtype=np.float64)
JOINT_WEIGHTS /= JOINT_WEIGHTS.sum()

JOINT_GROUPS = {
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "torso": [0, 3, 6, 9],
    "arms": [13, 14, 16, 17, 18, 19],
    "hands": [20, 21, 22, 23],
    "head": [12, 15],
}

BONE_PAIRS = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def validate_joints(joints_3d: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_3d, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError("Expected joints_3d with shape [T, 24, 3].")
    if joints.shape[0] < 8:
        raise ValueError("Need at least 8 frames of motion.")
    return joints
