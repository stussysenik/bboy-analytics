"""Header bar: phase label + timecode + BPM + hip-COM readout.

Phase transitions flash the header background in the new phase color.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .body_state import BodyState
from .color_system import (
    HIP_COM_COLOR,
    PANEL_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    phase_color,
)

try:
    _FONT_PHASE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    _FONT_INFO = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    _FONT_PHASE = ImageFont.load_default()
    _FONT_INFO = ImageFont.load_default()

HEADER_HEIGHT = 40


def render_header(
    state: BodyState,
    total_duration: float,
    bpm: float = 125.0,
    header_w: int = 1920,
    transition_flash: int = 0,
) -> Image.Image:
    """Render header bar.

    Args:
        state: Current body state
        total_duration: Total clip duration in seconds
        bpm: Beats per minute
        header_w: Width in pixels
        transition_flash: Frames remaining in phase transition flash (0 = none)

    Returns:
        PIL Image (RGBA)
    """
    pc = phase_color(state.phase)

    # Background: flash during transition, else dark
    if transition_flash > 0:
        alpha = min(255, int(200 * (transition_flash / 10)))
        bg = pc + (alpha,)
    else:
        bg = PANEL_BG + (255,)

    img = Image.new("RGBA", (header_w, HEADER_HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    # Left: phase dot + name
    dot_x, dot_y = 16, HEADER_HEIGHT // 2
    draw.ellipse(
        [dot_x - 6, dot_y - 6, dot_x + 6, dot_y + 6],
        fill=pc + (255,),
    )
    draw.text(
        (dot_x + 14, dot_y - 10),
        state.phase.upper(),
        fill=TEXT_PRIMARY + (255,),
        font=_FONT_PHASE,
    )

    # Center: timecode
    t = state.timestamp
    timecode = f"{t:.1f}s / {total_duration:.1f}s"
    tc_bbox = draw.textbbox((0, 0), timecode, font=_FONT_INFO)
    tc_w = tc_bbox[2] - tc_bbox[0]
    draw.text(
        (header_w // 2 - tc_w // 2, dot_y - 7),
        timecode,
        fill=TEXT_SECONDARY + (255,),
        font=_FONT_INFO,
    )

    # Right: BPM + Hip→COM
    hc_dist = f"{state.hip_com_magnitude:.3f}m"
    right_text = f"{bpm:.0f} BPM  |  hip→COM: {hc_dist}"
    rt_bbox = draw.textbbox((0, 0), right_text, font=_FONT_INFO)
    rt_w = rt_bbox[2] - rt_bbox[0]
    draw.text(
        (header_w - rt_w - 16, dot_y - 7),
        right_text,
        fill=HIP_COM_COLOR + (220,),
        font=_FONT_INFO,
    )

    return img
