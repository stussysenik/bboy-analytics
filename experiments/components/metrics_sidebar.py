"""
MetricsSidebar: live metrics dashboard sidebar panel.

Vertical sidebar showing grade badge, beat hit %, BPM, energy sparkline,
COM mini-map, and data source badge.  Static elements are pre-rendered
once; only the sparkline and COM map update per frame.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, BG, TXT, TXT_DIM, C_ENERGY_LO, C_ENERGY_HI
from .musicality_grade import grade_mu


def _lerp_color(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class MetricsSidebar(Panel):
    """Vertical sidebar with key analytics metrics."""

    def __init__(self, width: int, height: int, state,
                 bpm: float = 0.0, data_source: str = "GVHMR"):
        super().__init__(width, height, state)
        self.bpm = bpm
        self.data_source = data_source

    # ── prerender: all static elements ──────────────────────────────
    def prerender(self) -> None:
        ws = self.state
        # Layout constants — computed once, responsive to panel size
        self._ly = dict(
            grade_y=16, beat_y=100, bpm_y=164,
            spark_y=210, spark_h=40,
            badge_y=self.h - 36,
        )
        ly = self._ly
        self._bg, d = self._blank()
        f = FontCache.get
        cx = self.w // 2

        # 1. Grade badge — 48x48 rounded rectangle with letter
        letter, label, color, _ = grade_mu(ws.beat_hit_pct)
        bx = cx - 24
        by = ly["grade_y"]
        d.rounded_rectangle([bx, by, bx + 48, by + 48], radius=8, fill=color)
        bb = d.textbbox((0, 0), letter, font=f(bold=True, size=32))
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        d.text((bx + (48 - lw) // 2, by + (48 - lh) // 2 - 2),
               letter, fill=(0, 0, 0), font=f(bold=True, size=32))
        d.text((cx - int(f(bold=True, size=12).getlength(label) / 2), by + 54),
               label, fill=color, font=f(bold=True, size=12))

        # 2. Beat hit %
        pct_text = f"{ws.beat_hit_pct:.0f}%"
        hit_color = (80, 220, 100) if ws.beat_hit_pct > 60 else (255, 180, 80)
        tw = f(bold=True, size=28).getlength(pct_text)
        d.text((cx - int(tw / 2), ly["beat_y"]),
               pct_text, fill=hit_color, font=f(bold=True, size=28))
        lbl = "ON BEAT"
        d.text((cx - int(f(size=11).getlength(lbl) / 2), ly["beat_y"] + 34),
               lbl, fill=TXT_DIM, font=f(size=11))

        # 3. BPM readout
        bpm_text = f"{self.bpm:.0f} BPM" if self.bpm > 0 else "— BPM"
        tw = f(bold=True, size=16).getlength(bpm_text)
        d.text((cx - int(tw / 2), ly["bpm_y"]),
               bpm_text, fill=TXT, font=f(bold=True, size=16))

        # 4–5 are dynamic (sparkline + COM) — drawn in draw()

        # Sparkline / COM labels (static)
        lbl = "ENERGY"
        d.text((cx - int(f(size=10).getlength(lbl) / 2),
                ly["spark_y"] + ly["spark_h"] + 4),
               lbl, fill=TXT_DIM, font=f(size=10))

        # 6. Data source badge
        self._draw_source_badge(d, f, cx, ly["badge_y"])

    def _draw_source_badge(self, d: ImageDraw.ImageDraw, f, cx: int, y: int):
        label = self.data_source.upper()
        pill_color = (40, 180, 80) if label == "JOSH" else (50, 120, 220)
        tw = f(bold=True, size=12).getlength(label)
        pw = int(tw) + 16
        px = cx - pw // 2
        d.rounded_rectangle([px, y, px + pw, y + 22], radius=11, fill=pill_color)
        check = " \u2713" if label == "JOSH" else ""
        full = label + check
        ftw = f(bold=True, size=12).getlength(full)
        d.text((cx - int(ftw / 2), y + 4), full, fill=TXT, font=f(bold=True, size=12))

    # ── draw: copy static bg + dynamic sparkline & COM ──────────────
    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        ly = self._ly
        ws = self.state

        self._draw_sparkline(d, frame_idx, ly)
        return img

    def _draw_sparkline(self, d: ImageDraw.ImageDraw, frame_idx: int, ly: dict):
        """Energy sparkline: trailing 2s window of K(t)."""
        ws = self.state
        win = int(2.0 * ws.fps)  # ~60 frames
        lo = max(0, frame_idx - win)
        hi = frame_idx + 1
        ke = ws.kinetic_energy[lo:hi]
        if len(ke) < 2:
            return

        sx = 8
        sw = self.w - 16
        sy = ly["spark_y"]
        sh = ly["spark_h"]

        ke_min, ke_max = float(ke.min()), float(ke.max())
        ke_range = ke_max - ke_min if ke_max - ke_min > 1e-8 else 1.0

        points = []
        for i, val in enumerate(ke):
            x = sx + int(i / max(1, len(ke) - 1) * (sw - 1))
            norm = (float(val) - ke_min) / ke_range
            y = sy + sh - 1 - int(norm * (sh - 2))
            points.append((x, y))

        # Draw line segments with color gradient (blue→red by energy)
        for i in range(1, len(points)):
            norm = (float(ke[i]) - ke_min) / ke_range
            color = _lerp_color(C_ENERGY_LO, C_ENERGY_HI, norm)
            d.line([points[i - 1], points[i]], fill=color, width=2)

