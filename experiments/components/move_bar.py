"""
MoveBar: animated segment classification bar with smooth transitions.

Shows TOPROCK → FOOTWORK → POWERMOVE as color-coded regions with
animated fade transitions at boundaries.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, TXT, TXT_DIM


SEG_COLORS = {
    "toprock":   (33, 150, 243),
    "footwork":  (76, 175, 80),
    "powermove": (220, 40, 40),
    "freeze":    (156, 39, 176),
}


class MoveBar(Panel):
    """Animated segment bar with transitions."""

    def __init__(self, width: int, height: int, state, segments: list[dict]):
        """
        Args:
            segments: list of {start_s, end_s, dance_type}
        """
        super().__init__(width, height, state)
        self.segments = segments
        self.transition_s = 0.3  # 300ms transition fade

    def prerender(self) -> None:
        """Pre-render the static segment background."""
        ws = self.state
        total_dur = ws.frames / ws.fps

        self._bg = Image.new("RGB", (self.w, self.h), (18, 18, 24))
        d = ImageDraw.Draw(self._bg)

        bar_y = 4
        bar_h = self.h - 24

        # Draw segment blocks
        for seg in self.segments:
            x1 = int(seg["start_s"] / total_dur * self.w)
            x2 = int(seg["end_s"] / total_dur * self.w)
            color = SEG_COLORS.get(seg["dance_type"], (100, 100, 100))

            # Fill with slight gradient
            for x in range(x1, x2):
                # Fade in at start, fade out at end
                edge_dist = min(x - x1, x2 - x) / max(1, (x2 - x1) * 0.1)
                fade = min(1.0, edge_dist)
                r = int(color[0] * fade + 18 * (1 - fade))
                g = int(color[1] * fade + 18 * (1 - fade))
                b = int(color[2] * fade + 24 * (1 - fade))
                d.line([(x, bar_y), (x, bar_y + bar_h)], fill=(r, g, b))

            # Label
            label = seg["dance_type"].upper()
            tw = FontCache.get(bold=True, size=13).getlength(label)
            mid_x = (x1 + x2) // 2
            if x2 - x1 > tw + 10:
                d.text((int(mid_x - tw / 2), bar_y + bar_h // 2 - 7),
                       label, fill=TXT, font=FontCache.get(bold=True, size=13))

        # Label
        d.text((8, self.h - 18), "MOVE TYPE", fill=TXT_DIM, font=FontCache.get(size=10))

    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        ws = self.state
        total_dur = ws.frames / ws.fps
        t = frame_idx / ws.fps

        bar_y = 4
        bar_h = self.h - 24

        # Playhead
        ph_x = int(t / total_dur * self.w)
        d.line([(ph_x, 0), (ph_x, self.h)], fill=(255, 255, 255), width=2)

        # Transition flash at segment boundaries
        for seg in self.segments:
            for boundary in [seg["start_s"], seg["end_s"]]:
                dt = abs(t - boundary)
                if dt < self.transition_s:
                    flash = 1.0 - dt / self.transition_s
                    bx = int(boundary / total_dur * self.w)
                    flash_w = int(30 * flash)
                    alpha_val = int(180 * flash)
                    color = SEG_COLORS.get(seg["dance_type"], (200, 200, 200))
                    for dx in range(-flash_w, flash_w + 1):
                        px = bx + dx
                        if 0 <= px < self.w:
                            dist_frac = abs(dx) / max(1, flash_w)
                            a = int(alpha_val * (1 - dist_frac))
                            d.line([(px, bar_y), (px, bar_y + bar_h)],
                                   fill=(min(255, color[0] + 80), min(255, color[1] + 80),
                                         min(255, color[2] + 80)))

        # Current move type label (bottom right)
        current_type = None
        for seg in self.segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                current_type = seg["dance_type"]
                break

        if current_type:
            color = SEG_COLORS.get(current_type, (200, 200, 200))
            label = current_type.upper()
            tw = FontCache.get(bold=True, size=14).getlength(label)
            d.rounded_rectangle([self.w - tw - 24, self.h - 20, self.w - 4, self.h - 2],
                                radius=4, fill=color)
            d.text((self.w - tw - 16, self.h - 19), label, fill=TXT,
                   font=FontCache.get(bold=True, size=14))

        return img
