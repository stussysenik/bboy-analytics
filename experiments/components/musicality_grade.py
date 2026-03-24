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


# Grade thresholds: (max_mu, letter, label, color)
GRADES = [
    (0.10, "D", "Off-Beat",    (220, 50, 50)),
    (0.25, "C", "Catching It", (255, 160, 40)),
    (0.40, "B", "Grooving",    (255, 220, 50)),
    (0.60, "A", "Locked In",   (80, 220, 100)),
    (9.99, "S", "Surgical",    (255, 215, 0)),
]


def grade_mu(mu: float) -> tuple[str, str, tuple[int, int, int], int]:
    """Return (letter, label, color, percentage) for a given μ value."""
    pct = int(min(100, mu / 0.6 * 100))
    for threshold, letter, label, color in GRADES:
        if mu < threshold:
            return letter, label, color, pct
    return "S", "Surgical", (255, 215, 0), 100


class MusalityGradePanel(Panel):
    """Large musicality grade badge + beat dot timeline."""

    def prerender(self) -> None:
        ws = self.state
        self._has_data = ws.mu > 0.001 or (ws.beat_hits and len(ws.beat_hits) > 0)

        # Pre-compute beat dot positions
        self._dot_x = []
        self._dot_hit = []
        if ws.beat_hits:
            dot_area_x = int(self.w * 0.32)
            dot_area_w = self.w - dot_area_x - 16
            n = len(ws.beat_hits)
            for i, bh in enumerate(ws.beat_hits):
                x = dot_area_x + int((i + 0.5) / n * dot_area_w)
                self._dot_x.append(x)
                self._dot_hit.append(bh["hit"])

        # Pre-render static background
        self._bg = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(self._bg)

        if not self._has_data:
            f = FontCache.get
            d.text((self.w // 2 - 40, self.h // 2 - 10),
                   "NO AUDIO", fill=TXT_DIM, font=f(bold=True, size=18))
            return

        # Draw beat dots (static — playhead is dynamic)
        dot_y = self.h // 2
        r_hit, r_miss = 5, 3
        for i, (x, hit) in enumerate(zip(self._dot_x, self._dot_hit)):
            color = (80, 220, 100) if hit else (180, 50, 50)
            r = r_hit if hit else r_miss
            d.ellipse([x - r, dot_y - r, x + r, dot_y + r], fill=color)

        # "BEATS" label above dots
        if self._dot_x:
            f = FontCache.get
            d.text((self._dot_x[0], dot_y - 24),
                   "BEATS", fill=TXT_DIM, font=f(size=11))

    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        f = FontCache.get
        ws = self.state

        if not self._has_data:
            return img

        # Use local_mu for animated grade (changes per frame)
        if ws.local_mu is not None and frame_idx < ws.frames:
            current_mu = float(ws.local_mu[frame_idx])
        else:
            current_mu = ws.mu

        letter, label, color, pct = grade_mu(ws.mu)

        # --- Badge area (left 30%) ---
        badge_w = int(self.w * 0.28)
        badge_h = min(self.h - 16, 180)
        badge_x, badge_y = 12, (self.h - badge_h) // 2

        # Badge background
        d.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=10, fill=color + (40,))
        d.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=10, outline=color, width=2)

        # Large letter
        letter_font = f(bold=True, size=min(72, badge_h - 50))
        bb = d.textbbox((0, 0), letter, font=letter_font)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        d.text((badge_x + (badge_w - lw) // 2, badge_y + 8),
               letter, fill=color, font=letter_font)

        # Label below letter
        label_font = f(bold=True, size=13)
        bb2 = d.textbbox((0, 0), label, font=label_font)
        lw2 = bb2[2] - bb2[0]
        d.text((badge_x + (badge_w - lw2) // 2, badge_y + 8 + lh + 4),
               label, fill=TXT, font=label_font)

        # Percentage
        pct_text = f"MUSICALITY: {letter} ({pct}%)"
        pct_font = f(bold=True, size=12)
        bb3 = d.textbbox((0, 0), pct_text, font=pct_font)
        lw3 = bb3[2] - bb3[0]
        d.text((badge_x + (badge_w - lw3) // 2, badge_y + badge_h - 22),
               pct_text, fill=TXT_DIM, font=pct_font)

        # --- Beat dot playhead ---
        if self._dot_x and ws.beat_hits:
            progress = frame_idx / max(1, ws.frames - 1)
            ph_x = int(self.w * 0.32 + progress * (self.w - self.w * 0.32 - 16))
            dot_y = self.h // 2
            d.line([(ph_x, dot_y - 20), (ph_x, dot_y + 20)],
                   fill=(255, 255, 255), width=2)

            # Beat score text (bottom right of dot area)
            n_hits = sum(1 for h in self._dot_hit if h)
            total = len(self._dot_hit)
            score_text = f"{n_hits}/{total} ON BEAT ({ws.beat_hit_pct:.0f}%)"
            score_color = (80, 220, 100) if ws.beat_hit_pct > 60 else (255, 180, 80)
            d.text((self.w - 220, self.h - 24),
                   score_text, fill=score_color, font=f(bold=True, size=12))

        return img
