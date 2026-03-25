"""
MusalityGradePanel: letter grade + beat dots for musicality scoring.

Shows D/C/B/A/S grade with colored badge, percentage readout,
and a timeline of beat dots (green=hit, red=miss).
Simple enough for a six year old: colored letter + colored dots.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, BG, TXT, TXT_DIM


# Grade thresholds based on beat_hit_pct (deterministic, not correlation-based)
# Per feedback_deterministic_scoring.md: use hit rates, NOT coefficients
GRADES = [
    (40,  "D", "Off-Beat",    (220, 50, 50)),
    (55,  "C", "Catching It", (255, 160, 40)),
    (70,  "B", "Grooving",    (255, 220, 50)),
    (85,  "A", "Locked In",   (80, 220, 100)),
    (999, "S", "Surgical",    (255, 215, 0)),
]


def grade_mu(beat_hit_pct: float) -> tuple[str, str, tuple[int, int, int], int]:
    """Return (letter, label, color, percentage) for a given beat hit percentage.

    Despite the legacy name 'grade_mu', this now grades on beat_hit_pct
    (deterministic physical matching) rather than cross-correlation mu.
    """
    pct = int(min(100, max(0, beat_hit_pct)))
    for threshold, letter, label, color in GRADES:
        if beat_hit_pct < threshold:
            return letter, label, color, pct
    return "S", "Surgical", (255, 215, 0), pct


class CompactMusalityPanel(Panel):
    """Single-row compact musicality: grade + μ + beat dots + stats."""

    def __init__(self, width: int, height: int, state, bpm: float = 0.0):
        super().__init__(width, height, state)
        self.bpm = bpm

    def prerender(self) -> None:
        ws = self.state
        self._has_data = ws.mu > 0.001 or (ws.beat_hits and len(ws.beat_hits) > 0)

        # Beat dot positions (middle 50% of width)
        self._dot_x = []
        self._dot_hit = []
        if ws.beat_hits:
            dot_start = int(self.w * 0.28)
            dot_end = int(self.w * 0.72)
            dot_w = dot_end - dot_start
            n = len(ws.beat_hits)
            for i, bh in enumerate(ws.beat_hits):
                x = dot_start + int((i + 0.5) / n * dot_w)
                self._dot_x.append(x)
                self._dot_hit.append(bh["hit"])

        # Static background
        self._bg = Image.new("RGB", (self.w, self.h), (18, 18, 24))
        d = ImageDraw.Draw(self._bg)
        f = FontCache.get

        if not self._has_data:
            d.text((self.w // 2 - 30, (self.h - 12) // 2),
                   "NO AUDIO DATA", fill=TXT_DIM, font=f(bold=True, size=14))
            return

        letter, label, color, pct = grade_mu(ws.beat_hit_pct)
        cy = self.h // 2

        # Left section: grade badge (compact)
        badge_w = 36
        badge_x = 10
        d.rounded_rectangle([badge_x, cy - 16, badge_x + badge_w, cy + 16],
                            radius=6, fill=color, outline=color)
        bb = d.textbbox((0, 0), letter, font=f(bold=True, size=22))
        lw = bb[2] - bb[0]
        d.text((badge_x + (badge_w - lw) // 2, cy - 14),
               letter, fill=(0, 0, 0), font=f(bold=True, size=22))

        # Label + beat hit %
        d.text((badge_x + badge_w + 8, cy - 14),
               label, fill=color, font=f(bold=True, size=13))
        d.text((badge_x + badge_w + 8, cy + 2),
               f"{pct}% on beat",
               fill=TXT_DIM, font=f(size=11))

        # Beat dots (middle)
        dot_y = cy
        for x, hit in zip(self._dot_x, self._dot_hit):
            clr = (80, 220, 100) if hit else (120, 40, 40)
            r = 4 if hit else 2
            d.ellipse([x - r, dot_y - r, x + r, dot_y + r], fill=clr)

        # Right section: beat stats + BPM
        n_hits = sum(1 for h in self._dot_hit if h)
        total = len(self._dot_hit)
        duration = ws.frames / ws.fps
        hit_color = (80, 220, 100) if ws.beat_hit_pct > 60 else (255, 180, 80)

        stats = f"{n_hits}/{total} BEATS ({ws.beat_hit_pct:.0f}%)"
        d.text((int(self.w * 0.74), cy - 14),
               stats, fill=hit_color, font=f(bold=True, size=12))

        bpm_text = f"BPM:{self.bpm:.0f}" if self.bpm > 0 else ""
        dur_text = f"{duration:.1f}s"
        d.text((int(self.w * 0.74), cy + 2),
               f"{bpm_text}  {dur_text}  @bboy_analytics",
               fill=TXT_DIM, font=f(size=10))

    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        if not self._has_data:
            return img

        d = ImageDraw.Draw(img)
        ws = self.state

        # Playhead over beat dots
        if self._dot_x:
            progress = frame_idx / max(1, ws.frames - 1)
            dot_start = int(self.w * 0.28)
            dot_end = int(self.w * 0.72)
            ph_x = dot_start + int(progress * (dot_end - dot_start))
            cy = self.h // 2
            d.line([(ph_x, cy - 16), (ph_x, cy + 16)],
                   fill=(255, 255, 255), width=2)

        return img


# Keep old panel available for backward compatibility
MusalityGradePanel = CompactMusalityPanel
