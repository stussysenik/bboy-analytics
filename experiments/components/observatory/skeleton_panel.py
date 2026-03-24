"""Right panel renderer: analytical skeleton system on dark background.

Draws wireframe + cluster-colored joints + 30-frame temporal trails +
COM + hip-COM vector + contact indicators + phase labels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .body_state import BONE_PAIRS, BodyState, TemporalWindow
from .color_system import (
    BG_COLOR,
    BONE_WIDTH,
    COM_COLOR,
    COM_DOT_RADIUS,
    HIP_COM_COLOR,
    HIP_COM_WIDTH,
    JOINT_DOT_EMPHASIS,
    JOINT_DOT_OUTLINE,
    JOINT_DOT_RADIUS,
    JOINT_LABELS,
    PHASE_EMPHASIS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TRAIL_OPACITY_START,
    joint_color,
    joint_color_alpha,
    phase_color,
)

# Font setup (fallback to default if specific font unavailable)
try:
    _FONT_PHASE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    _FONT_LABEL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    _FONT_CONTACT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except OSError:
    _FONT_PHASE = ImageFont.load_default()
    _FONT_LABEL = ImageFont.load_default()
    _FONT_CONTACT = ImageFont.load_default()


def _project_3d_to_2d(
    point: np.ndarray,
    panel_w: int,
    panel_h: int,
    center_offset: np.ndarray,
    scale: float,
) -> Tuple[int, int]:
    """Project 3D world point to 2D panel coordinates.

    Orthographic projection (Y-up): X → screen X, Y → screen Y (inverted).
    Z axis (depth) is ignored for the front view.
    """
    x = (point[0] - center_offset[0]) * scale + panel_w / 2
    y = panel_h - ((point[1] - center_offset[1]) * scale + panel_h * 0.15)
    return int(x), int(y)


def _compute_projection_params(
    state: BodyState,
    window: TemporalWindow,
    panel_w: int,
    panel_h: int,
) -> Tuple[np.ndarray, float]:
    """Compute center offset and scale to fit skeleton in panel."""
    # Use pelvis as center reference
    positions = np.array([j.position for j in state.joints])
    center = positions.mean(axis=0)

    # Compute bounding box of current skeleton (X=horizontal, Y=vertical)
    span_x = positions[:, 0].max() - positions[:, 0].min()
    span_y = positions[:, 1].max() - positions[:, 1].min()
    max_span = max(span_x, span_y, 0.5)  # at least 0.5m

    # Scale to fill ~70% of panel
    scale = min(panel_w, panel_h) * 0.7 / max_span

    return center, scale


def render_skeleton_panel(
    state: BodyState,
    window: TemporalWindow,
    panel_w: int = 960,
    panel_h: int = 820,
    contact_flash_frames: Optional[dict] = None,
) -> Image.Image:
    """Render the analytical skeleton panel.

    Args:
        state: Current frame's typed body state
        window: Temporal window for trails
        panel_w: Panel width in pixels
        panel_h: Panel height in pixels
        contact_flash_frames: {joint_idx: frames_remaining} for contact labels

    Returns:
        PIL Image (RGBA)
    """
    img = Image.new("RGBA", (panel_w, panel_h), BG_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    center, scale = _compute_projection_params(state, window, panel_w, panel_h)

    def proj(pos: np.ndarray) -> Tuple[int, int]:
        return _project_3d_to_2d(pos, panel_w, panel_h, center, scale)

    # ── 1. Temporal trails (oldest to newest, fading) ──
    trails = window.joint_trails  # (N, 24, 3)
    n_trail = trails.shape[0] if trails.size > 0 else 0

    if n_trail > 1:
        for t_idx in range(n_trail - 1):  # skip current frame (drawn separately)
            alpha = TRAIL_OPACITY_START * (t_idx / max(n_trail - 1, 1))
            for j in range(24):
                pos = trails[t_idx, j]
                px, py = proj(pos)
                if 0 <= px < panel_w and 0 <= py < panel_h:
                    rgba = joint_color_alpha(j, alpha)
                    r = max(2, int(JOINT_DOT_RADIUS * 0.5 * (t_idx / max(n_trail - 1, 1))))
                    draw.ellipse(
                        [px - r, py - r, px + r, py + r],
                        fill=rgba,
                    )

    # COM trail
    com_trail = window.com_trail  # (N, 3)
    if com_trail.shape[0] > 1:
        for t_idx in range(com_trail.shape[0] - 1):
            alpha = TRAIL_OPACITY_START * (t_idx / max(com_trail.shape[0] - 1, 1))
            px, py = proj(com_trail[t_idx])
            if 0 <= px < panel_w and 0 <= py < panel_h:
                r = max(2, int(COM_DOT_RADIUS * 0.4))
                draw.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=COM_COLOR + (int(alpha * 255),),
                )

    # ── 2. Wireframe bones (current frame) ──
    positions = {j.idx: j.position for j in state.joints}
    for parent_idx, child_idx in BONE_PAIRS:
        if parent_idx in positions and child_idx in positions:
            p1 = proj(positions[parent_idx])
            p2 = proj(positions[child_idx])
            color = joint_color(child_idx)
            draw.line([p1, p2], fill=color + (200,), width=BONE_WIDTH)

    # ── 3. Joint dots (current frame, full opacity) ──
    emphasized = set(PHASE_EMPHASIS.get(state.phase, []))
    for js in state.joints:
        px, py = proj(js.position)
        if not (0 <= px < panel_w and 0 <= py < panel_h):
            continue

        radius = JOINT_DOT_EMPHASIS if js.idx in emphasized else JOINT_DOT_RADIUS
        color = joint_color(js.idx)

        # White outline
        draw.ellipse(
            [px - radius - JOINT_DOT_OUTLINE, py - radius - JOINT_DOT_OUTLINE,
             px + radius + JOINT_DOT_OUTLINE, py + radius + JOINT_DOT_OUTLINE],
            fill=(255, 255, 255, 180),
        )
        # Filled dot
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=color + (255,),
        )

        # Beat alignment glow
        if js.beat_aligned:
            glow_r = radius + 4
            draw.ellipse(
                [px - glow_r, py - glow_r, px + glow_r, py + glow_r],
                outline=(255, 255, 255, 120),
                width=2,
            )

        # Joint label (only 6 key joints)
        if js.idx in JOINT_LABELS:
            label = JOINT_LABELS[js.idx]
            draw.text(
                (px + radius + 4, py - 6),
                label,
                fill=TEXT_SECONDARY + (180,),
                font=_FONT_LABEL,
            )

    # ── 4. COM indicator ──
    com_px, com_py = proj(state.com)
    if 0 <= com_px < panel_w and 0 <= com_py < panel_h:
        r = COM_DOT_RADIUS
        draw.ellipse(
            [com_px - r, com_py - r, com_px + r, com_py + r],
            fill=COM_COLOR + (255,),
        )

        # Hip→COM vector (gold line from pelvis to COM)
        pelvis_px, pelvis_py = proj(state.joints[0].position)
        draw.line(
            [(pelvis_px, pelvis_py), (com_px, com_py)],
            fill=HIP_COM_COLOR + (200,),
            width=HIP_COM_WIDTH,
        )

    # ── 5. Contact indicators ──
    for js in state.joints:
        if js.is_contact:
            px, py = proj(js.position)
            if 0 <= px < panel_w and 0 <= py < panel_h:
                color = joint_color(js.idx)
                # Radiating circle
                for ring_r in [12, 18, 24]:
                    alpha = max(0, 180 - ring_r * 6)
                    draw.ellipse(
                        [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                        outline=color + (alpha,),
                        width=1,
                    )
                # Contact label
                cluster = js.cluster.upper()
                draw.text(
                    (px + 14, py - 8),
                    f"{cluster} ▼",
                    fill=color + (220,),
                    font=_FONT_CONTACT,
                )

    # ── 6. Phase label (large, top-left, always visible) ──
    phase_name = state.phase.upper()
    pc = phase_color(state.phase)
    # Semi-transparent background pill
    text_bbox = draw.textbbox((0, 0), phase_name, font=_FONT_PHASE)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    pill_x, pill_y = 16, 12
    draw.rounded_rectangle(
        [pill_x, pill_y, pill_x + tw + 24, pill_y + th + 12],
        radius=6,
        fill=pc + (150,),
    )
    draw.text(
        (pill_x + 12, pill_y + 6),
        phase_name,
        fill=(255, 255, 255, 240),
        font=_FONT_PHASE,
    )

    return img
