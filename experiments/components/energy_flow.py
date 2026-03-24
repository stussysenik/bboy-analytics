"""EnergyPanel: K(t) curve + dK/dt acceleration overlay with spike markers."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, C_ENERGY_LO, C_ENERGY_HI, TXT, TXT_DIM, BG


class EnergyPanel(Panel):
    """Energy timeline with acceleration overlay."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)

    def prerender(self) -> None:
        """Pre-render the static energy curve background."""
        ws = self.state
        F = ws.frames

        # Normalize energy for rendering
        ke = ws.kinetic_energy
        self.ke_norm = (ke - ke.min()) / (ke.max() - ke.min() + 1e-8)

        # Normalize acceleration
        accel = ws.energy_accel
        accel_abs = np.abs(accel)
        self.accel_norm = accel_abs / (accel_abs.max() + 1e-8)
        self.accel_sign = np.sign(accel)  # +1 = accelerating, -1 = decelerating

        # Find spike frames (top 2% acceleration)
        threshold = np.percentile(accel_abs, 98)
        self.spike_frames = np.where(accel_abs > threshold)[0]

        # Pre-render static background with energy bars
        chart_h = self.h - 35  # leave room for labels
        chart_y = 22
        self._bg = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(self._bg)

        # Title
        d.text((10, 3), "ENERGY K(t)", fill=TXT_DIM, font=FontCache.get(bold=True, size=13))
        d.text((140, 5), "dK/dt", fill=(80, 200, 220), font=FontCache.get(size=11))

        # Energy bars
        for x in range(self.w):
            fi = int(x / self.w * F)
            fi = min(fi, F - 1)

            # K(t) bar
            val = float(self.ke_norm[fi])
            bar_h = max(1, int(val * chart_h * 0.85))
            r = int(C_ENERGY_LO[0] + (C_ENERGY_HI[0] - C_ENERGY_LO[0]) * val)
            g = int(C_ENERGY_LO[1] + (C_ENERGY_HI[1] - C_ENERGY_LO[1]) * val)
            b = int(C_ENERGY_LO[2] + (C_ENERGY_HI[2] - C_ENERGY_LO[2]) * val)
            d.line([(x, chart_y + chart_h - bar_h), (x, chart_y + chart_h)],
                   fill=(r, g, b))

            # dK/dt overlay (thin cyan line on top)
            accel_val = float(self.accel_norm[fi])
            if accel_val > 0.1:
                accel_h = int(accel_val * chart_h * 0.4)
                color = (80, 200, 220) if self.accel_sign[fi] > 0 else (220, 100, 80)
                d.line([(x, chart_y + chart_h - bar_h - accel_h),
                        (x, chart_y + chart_h - bar_h)],
                       fill=color)

        # Spike markers
        for sf in self.spike_frames:
            sx = int(sf / F * self.w)
            d.line([(sx, chart_y), (sx, chart_y + 5)], fill=(255, 80, 80), width=2)

        self._chart_y = chart_y
        self._chart_h = chart_h

    def draw(self, frame_idx: int) -> Image.Image:
        # Copy pre-rendered background
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        f = FontCache.get
        ws = self.state

        # Playhead
        progress = min(1.0, frame_idx / max(1, ws.frames - 1))
        ph_x = int(progress * self.w)
        d.line([(ph_x, self._chart_y), (ph_x, self._chart_y + self._chart_h)],
               fill=(255, 255, 255), width=2)

        # Current values readout
        if frame_idx < ws.frames:
            ke = ws.kinetic_energy[frame_idx]
            accel = ws.energy_accel[frame_idx]
            sign = "+" if accel >= 0 else ""
            d.text((self.w - 200, 3),
                   f"K={ke:.0f}  dK/dt={sign}{accel:.0f}",
                   fill=TXT, font=f(bold=True, size=12))

        return img
