"""ScalarStrip: thin horizontal heat strip — one pixel column per frame."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, TXT_DIM


class ScalarStrip(Panel):
    """Per-frame composite scalar as a color-coded strip."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)

    def prerender(self) -> None:
        """Pre-render the static heat strip."""
        ws = self.state
        F = ws.frames

        self._bg = Image.new("RGB", (self.w, self.h), (12, 12, 18))
        d = ImageDraw.Draw(self._bg)

        strip_y = 4
        strip_h = self.h - 8

        for x in range(self.w):
            fi = int(x / self.w * F)
            fi = min(fi, F - 1)
            val = float(ws.composite[fi])

            # Blue → Cyan → Yellow → Red
            if val < 0.33:
                t = val / 0.33
                r, g, b = int(30 + 20 * t), int(60 + 140 * t), int(180 + 40 * t)
            elif val < 0.66:
                t = (val - 0.33) / 0.33
                r, g, b = int(50 + 180 * t), int(200 - 20 * t), int(220 - 180 * t)
            else:
                t = (val - 0.66) / 0.34
                r, g, b = int(230 + 25 * t), int(180 - 140 * t), int(40 - 30 * t)

            d.line([(x, strip_y), (x, strip_y + strip_h)], fill=(r, g, b))

    def draw(self, frame_idx: int) -> Image.Image:
        img = self._bg.copy()
        d = ImageDraw.Draw(img)

        # Playhead
        progress = min(1.0, frame_idx / max(1, self.state.frames - 1))
        ph_x = int(progress * self.w)
        d.line([(ph_x, 0), (ph_x, self.h)], fill=(255, 255, 255), width=2)

        return img
