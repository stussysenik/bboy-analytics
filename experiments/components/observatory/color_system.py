"""TRIVIUM Palette — single source of truth for ALL colors.

Every property has ONE color, used everywhere: skeleton, graphs, legends, dashboards.
Import this module in every renderer. Never hardcode hex values elsewhere.
"""

from __future__ import annotations
from typing import Dict, Tuple

# ──────────────────────────────────────────────────────────────────────
# Body Clusters (anatomical, warm→cool bottom-up)
# ──────────────────────────────────────────────────────────────────────

CLUSTER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "legs":  (0xFF, 0x6B, 0x35),   # #FF6B35 Burnt Orange — grounded
    "torso": (0xE6, 0x39, 0x46),   # #E63946 Imperial Red — the ENGINE
    "arms":  (0x45, 0x7B, 0x9D),   # #457B9D Steel Blue — expression
    "hands": (0xA8, 0x55, 0xF7),   # #A855F7 Violet — foundation (inversions)
    "head":  (0xE0, 0xF2, 0xFE),   # #E0F2FE Ice White — pivot point
}

# Joint index → cluster name (from analyze_motion.py JOINT_GROUPS)
JOINT_CLUSTER: Dict[int, str] = {}
_GROUPS = {
    "legs":  [1, 2, 4, 5, 7, 8, 10, 11],
    "torso": [0, 3, 6, 9],
    "arms":  [13, 14, 16, 17, 18, 19],
    "hands": [20, 21, 22, 23],
    "head":  [12, 15],
}
for _cluster, _indices in _GROUPS.items():
    for _idx in _indices:
        JOINT_CLUSTER[_idx] = _cluster


def joint_color(joint_idx: int) -> Tuple[int, int, int]:
    """Get RGB color for a joint by its SMPL index."""
    return CLUSTER_COLORS[JOINT_CLUSTER[joint_idx]]


def joint_color_alpha(joint_idx: int, alpha: float) -> Tuple[int, int, int, int]:
    """Get RGBA color for a joint with specified alpha (0.0–1.0)."""
    r, g, b = joint_color(joint_idx)
    return (r, g, b, int(alpha * 255))


# ──────────────────────────────────────────────────────────────────────
# Dance Phases (energy-coded)
# ──────────────────────────────────────────────────────────────────────

PHASE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "toprock":    (0x3B, 0x82, 0xF6),  # #3B82F6 Royal Blue
    "footwork":   (0x22, 0xC5, 0x5E),  # #22C55E Emerald
    "power":      (0xF4, 0x3F, 0x5E),  # #F43F5E Hot Magenta
    "freeze":     (0x94, 0xA3, 0xB8),  # #94A3B8 Slate
    "transition": (0xFB, 0xBF, 0x24),  # #FBBF24 Amber
}


def phase_color(phase: str) -> Tuple[int, int, int]:
    """Get RGB color for a dance phase."""
    return PHASE_COLORS.get(phase, (0x94, 0xA3, 0xB8))


# ──────────────────────────────────────────────────────────────────────
# Dynamics & Metrics
# ──────────────────────────────────────────────────────────────────────

COM_COLOR = (0xFF, 0xFF, 0xFF)          # White
HIP_COM_COLOR = (0xFF, 0xD7, 0x00)     # Gold #FFD700
ENERGY_COLOR = (0xF9, 0x73, 0x16)      # Orange #F97316
MUSICALITY_COLOR = (0xEC, 0x48, 0x99)  # Pink #EC4899
BEAT_HIT_COLOR = (0x4A, 0xDE, 0x80)    # Bright Green #4ADE80
BEAT_MISS_COLOR = (0x7F, 0x1D, 0x1D)   # Dim Red #7F1D1D

# ──────────────────────────────────────────────────────────────────────
# UI Chrome
# ──────────────────────────────────────────────────────────────────────

BG_COLOR = (0x0A, 0x0A, 0x0F)          # Near Black #0A0A0F
PANEL_BG = (0x11, 0x11, 0x18)          # Dark Navy #111118
GRID_COLOR = (0x1E, 0x1E, 0x2A)        # Subtle #1E1E2A
TEXT_PRIMARY = (0xE2, 0xE8, 0xF0)      # Slate 200
TEXT_SECONDARY = (0x94, 0xA3, 0xB8)    # Slate 400
ACCENT_COLOR = (0xF5, 0x9E, 0x0B)      # Amber #F59E0B

# ──────────────────────────────────────────────────────────────────────
# Rendering constants
# ──────────────────────────────────────────────────────────────────────

JOINT_DOT_RADIUS = 5          # Normal joint dot (10px diameter)
JOINT_DOT_EMPHASIS = 7        # Emphasized joint (14px diameter)
JOINT_DOT_OUTLINE = 1         # White outline width
COM_DOT_RADIUS = 6            # COM indicator (12px diameter)
BONE_WIDTH = 2                # Wireframe bone line width
TRAIL_LENGTH = 30             # Frames of trail (1s at 30fps)
TRAIL_OPACITY_START = 0.30    # Trail start opacity
HIP_COM_WIDTH = 2             # Hip→COM vector line width

# 2D overlay (left panel — subtle)
OVERLAY_DOT_RADIUS = 3        # 6px diameter
OVERLAY_BONE_WIDTH = 1
OVERLAY_OPACITY = 0.50        # 50% opacity for 2D overlay
OVERLAY_BONE_OPACITY = 0.30   # 30% for bone lines

# Contact detection thresholds
CONTACT_Y_THRESHOLD = 0.15    # meters above ground_y — feet within 15cm of lowest point
CONTACT_SPEED_THRESHOLD = 0.5 # m/s — below this = near-stationary

# Phase-emphasized joint groups (from BREAKING_KINETIC_CHAIN.md)
PHASE_EMPHASIS: Dict[str, list] = {
    "power":    [0, 1, 2, 3, 6, 9, 16, 17, 13, 14],  # ENGINE: hips+spine+shoulders
    "freeze":   [20, 21, 22, 23, 12, 15, 7, 8, 10, 11],  # FOUNDATION: hands+head+feet
    "footwork": [1, 2, 4, 5, 7, 8, 10, 11],  # LEGS
    "toprock":  [],  # all normal
    "transition": [],
}

# Joint labels (only the 6 most important, always visible)
JOINT_LABELS: Dict[int, str] = {
    0: "PELVIS",
    15: "HEAD",
    20: "L_WRIST",
    21: "R_WRIST",
    7: "L_ANKLE",
    8: "R_ANKLE",
}
