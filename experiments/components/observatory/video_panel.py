"""Left panel renderer: original video + subtle 2D joint overlay.

Shows where the tracker THINKS joints are vs the real dancer.
Drift is immediately visible — dot floating in air = tracking broken.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_state import BONE_PAIRS, BodyState
from .color_system import (
    OVERLAY_BONE_OPACITY,
    OVERLAY_BONE_WIDTH,
    OVERLAY_DOT_RADIUS,
    OVERLAY_OPACITY,
    joint_color,
    phase_color,
)

try:
    _FONT_BADGE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except OSError:
    _FONT_BADGE = ImageFont.load_default()


def _project_to_video(
    pos_3d: np.ndarray,
    video_w: int,
    video_h: int,
    cam_center: np.ndarray,
    cam_scale: float,
) -> Tuple[int, int]:
    """Project 3D joint to 2D video coordinates.

    Orthographic (Y-up): X→screen X, Y→screen Y (inverted).
    For proper projection, replace with camera intrinsics when available.
    """
    x = (pos_3d[0] - cam_center[0]) * cam_scale + video_w / 2
    y = video_h - ((pos_3d[1] - cam_center[1]) * cam_scale + video_h * 0.15)
    return int(x), int(y)


def render_video_panel(
    video_frame: np.ndarray,
    state: BodyState,
    panel_w: int = 960,
    panel_h: int = 820,
    vitpose_2d: np.ndarray | None = None,
) -> Image.Image:
    """Render video frame with 2D joint overlay.

    Args:
        video_frame: (H, W, 3) RGB uint8 from source video
        state: Current frame's typed body state
        panel_w: Target panel width
        panel_h: Target panel height
        vitpose_2d: Optional (24, 2) or (17, 2) 2D keypoints in pixel coords.
                    If provided, uses these instead of 3D projection.

    Returns:
        PIL Image (RGBA)
    """
    # Resize video frame to panel size
    pil_frame = Image.fromarray(video_frame).convert("RGBA")
    pil_frame = pil_frame.resize((panel_w, panel_h), Image.LANCZOS)

    overlay = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Compute projection params from 3D joints
    positions = np.array([j.position for j in state.joints])
    cam_center = positions.mean(axis=0)
    span = max(positions[:, 0].max() - positions[:, 0].min(),
               positions[:, 1].max() - positions[:, 1].min(), 0.5)
    cam_scale = min(panel_w, panel_h) * 0.7 / span

    def get_2d(joint_idx: int) -> Tuple[int, int]:
        if vitpose_2d is not None and joint_idx < len(vitpose_2d):
            x, y = vitpose_2d[joint_idx]
            # Scale to panel size (assuming vitpose is in original video coords)
            return int(x * panel_w / video_frame.shape[1]), int(y * panel_h / video_frame.shape[0])
        return _project_to_video(
            state.joints[joint_idx].position,
            panel_w, panel_h, cam_center, cam_scale,
        )

    # Draw bone lines (subtle)
    bone_alpha = int(OVERLAY_BONE_OPACITY * 255)
    for parent_idx, child_idx in BONE_PAIRS:
        if parent_idx < 24 and child_idx < 24:
            p1 = get_2d(parent_idx)
            p2 = get_2d(child_idx)
            color = joint_color(child_idx)
            draw.line([p1, p2], fill=color + (bone_alpha,), width=OVERLAY_BONE_WIDTH)

    # Draw joint dots (subtle)
    dot_alpha = int(OVERLAY_OPACITY * 255)
    for js in state.joints:
        px, py = get_2d(js.idx)
        if 0 <= px < panel_w and 0 <= py < panel_h:
            color = joint_color(js.idx)
            r = OVERLAY_DOT_RADIUS
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=color + (dot_alpha,),
            )

    # Composite overlay onto video
    result = Image.alpha_composite(pil_frame, overlay)

    # Phase badge (bottom-left corner)
    phase_name = state.phase.upper()
    pc = phase_color(state.phase)
    bbox = draw.textbbox((0, 0), phase_name, font=_FONT_BADGE)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    bx, by = 12, panel_h - th - 20
    badge_draw = ImageDraw.Draw(result)
    badge_draw.rounded_rectangle(
        [bx, by, bx + tw + 16, by + th + 8],
        radius=4,
        fill=pc + (200,),
    )
    badge_draw.text(
        (bx + 8, by + 4),
        phase_name,
        fill=(255, 255, 255, 240),
        font=_FONT_BADGE,
    )

    return result
