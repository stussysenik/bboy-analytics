"""
MusalityRibbon: live correlation strip that breathes with the music.

Green when correlation is high (dancer on beat), fading to red when
it drops. Replaces the static μ badge.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, TXT, TXT_DIM


class MusalityRibbon(Panel):
    """Thin horizontal ribbon showing live musicality."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)

    def prerender(self) -> None:
        """Pre-render the static ribbon background."""
        ws = self.state
        F = ws.frames

        self._bg = Image.new("RGB", (self.w, self.h), (12, 12, 18))
        d = ImageDraw.Draw(self._bg)

        if ws.local_mu is None or ws.local_mu.max() < 0.001:
            # No musicality data — draw empty ribbon
            d.rectangle([0, 0, self.w, self.h], fill=(30, 30, 40))
            return

        # Draw ribbon: each column = one frame's local μ
        ribbon_y = 0
        ribbon_h = self.h - 22  # leave room for label

        for x in range(self.w):
            fi = int(x / self.w * F)
            fi = min(fi, F - 1)
            val = float(ws.local_mu[fi])
            val = min(1.0, max(0.0, val * 2.5))  # scale up for visibility

            # Green (on beat) → Yellow → Red (off beat)
            if val > 0.5:
                t = (val - 0.5) / 0.5
                r = int(80 * (1 - t))
                g = int(180 + 75 * t)
                b = int(40 * (1 - t))
            else:
                t = val / 0.5
                r = int(180 - 100 * t)
                g = int(40 + 140 * t)
                b = int(30)

            d.line([(x, ribbon_y), (x, ribbon_y + ribbon_h)],
                   fill=(r, g, b))

        # Beat markers on ribbon
        if ws.beat_hits:
            for bh in ws.beat_hits:
                bx = int(bh["frame"] / F * self.w)
                if bh["hit"]:
                    d.line([(bx, ribbon_y), (bx, ribbon_y + ribbon_h)],
                           fill=(120, 255, 120), width=1)
                else:
                    d.line([(bx, ribbon_y), (bx, ribbon_y + ribbon_h)],
                           fill=(255, 80, 80), width=1)

    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        f = FontCache.get
        ws = self.state

        # Playhead
        progress = min(1.0, frame_idx / max(1, ws.frames - 1))
        ph_x = int(progress * self.w)
        ribbon_h = self.h - 22
        d.line([(ph_x, 0), (ph_x, ribbon_h)], fill=(255, 255, 255), width=2)

        # Labels
        label_y = ribbon_h + 3
        d.text((8, label_y), "MUSICALITY", fill=TXT_DIM, font=f(bold=True, size=12))

        # Current local μ
        if ws.local_mu is not None and frame_idx < ws.frames:
            local_val = float(ws.local_mu[frame_idx])
            d.text((120, label_y), f"μ={local_val:.3f}", fill=TXT, font=f(bold=True, size=12))

        # Global μ
        d.text((230, label_y), f"global μ={ws.mu:.3f}", fill=TXT_DIM, font=f(size=11))

        # Beat score
        if ws.beat_hits:
            n_hits = sum(1 for b in ws.beat_hits if b["hit"])
            total = len(ws.beat_hits)
            d.text((self.w - 180, label_y),
                   f"{n_hits}/{total} ON BEAT ({ws.beat_hit_pct:.0f}%)",
                   fill=(120, 255, 120) if ws.beat_hit_pct > 60 else (255, 180, 80),
                   font=f(bold=True, size=13))

        return img
