"""VideoOverlay v6: minimal badges only — timer, phase pill, grade badge.

All heavy visualization (skeleton, data points, COM trail) has been moved
to dedicated panels (MultiViewPanel, MetricsSidebar) for clear separation
between original footage and computed analytics.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, TXT, TXT_DIM, SEG_COLORS
from .musicality_grade import grade_mu


class VideoOverlay(Panel):
    """Minimal overlay on original video — timer, phase pill, dancer name."""

    def __init__(self, width: int, height: int, state,
                 segments: list[dict] | None = None, dancer_name: str = ""):
        super().__init__(width, height, state)
        self._segments = segments or []
        self._dancer_name = dancer_name

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank_rgba()
        f = FontCache.get
        ws = self.state

        # ── Timer badge (top-left, away from Instagram right-edge buttons) ──
        t = frame_idx / ws.fps
        total = ws.frames / ws.fps
        d.rounded_rectangle([12, 12, 175, 42], radius=6, fill=(0, 0, 0, 160))
        d.text((22, 16), f"{t:.1f}s / {total:.1f}s",
               fill=(*TXT, 230), font=f(bold=True, size=16))

        # ── Dancer name badge (top-left, below timer) ───────────────
        if self._dancer_name:
            name_text = self._dancer_name.upper()
            name_font = f(bold=True, size=16)
            nw = int(name_font.getlength(name_text))
            d.rounded_rectangle([12, 50, 12 + nw + 20, 80], radius=6,
                                fill=(0, 0, 0, 180))
            d.text((22, 53), name_text,
                   fill=(*TXT, 230), font=name_font)
        elif ws.beat_hit_pct > 0:
            letter, label, color, _ = grade_mu(ws.beat_hit_pct)
            d.rounded_rectangle([12, 50, 100, 80], radius=6,
                                fill=(*color, 220))
            d.text((22, 53), f"{letter} {label}",
                   fill=(0, 0, 0, 255), font=f(bold=True, size=16))

        # ── Dance phase label (top-center) ──────────────────────────
        if frame_idx < ws.frames and self._segments:
            t_s = frame_idx / ws.fps
            for seg in self._segments:
                if seg["start_s"] <= t_s <= seg["end_s"]:
                    phase_label = seg["dance_type"].upper()
                    color = SEG_COLORS.get(seg["dance_type"], (200, 200, 200))
                    lbl_font = f(bold=True, size=20)
                    bb = d.textbbox((0, 0), phase_label, font=lbl_font)
                    tw = bb[2] - bb[0]
                    cx = (self.w - tw) // 2
                    d.rounded_rectangle([cx - 8, 12, cx + tw + 8, 40],
                                        radius=6, fill=(*color, 220))
                    d.text((cx, 15), phase_label,
                           fill=(255, 255, 255, 255), font=lbl_font)
                    break

        # ── Subtle energy border glow (kept, it's non-intrusive) ────
        if hasattr(ws, 'composite') and frame_idx < ws.frames:
            val = float(ws.composite[frame_idx])
            r = int(50 + 205 * val)
            g = int(130 - 80 * val)
            b = int(200 - 180 * val)
            alpha = int(40 + 100 * val)
            d.rectangle([0, 0, self.w - 1, self.h - 1],
                         outline=(r, g, b, alpha), width=2)

        return img
