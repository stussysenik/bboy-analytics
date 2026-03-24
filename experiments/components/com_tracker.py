"""COMPanel: single COM dot + trail + height bar + travel speed."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, C_COM, TXT, TXT_DIM, BG


class COMPanel(Panel):
    """Top-down COM map with trail and height indicator."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)
        self.trail_frames = int(2.0 * state.fps)
        self.map_margin = 15

    def prerender(self) -> None:
        """Pre-compute COM bounds for stable mapping."""
        com = self.state.com_pos
        self.x_min = float(com[:, 0].min())
        self.x_max = float(com[:, 0].max())
        self.z_min = float(com[:, 2].min())
        self.z_max = float(com[:, 2].max())
        self.y_min = float(com[:, 1].min())
        self.y_max = float(com[:, 1].max())

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank()
        f = FontCache.get
        ws = self.state

        # Title
        d.text((10, 4), "CENTER OF MASS", fill=TXT_DIM, font=f(bold=True, size=13))

        # Layout: map on left, height bar on right
        map_size = min(self.w - 100, self.h - 55)
        map_x, map_y = 10, 28
        bar_x = map_x + map_size + 20
        bar_w = 25

        # ── Map background ────────────────────────────────────────
        d.rectangle([map_x, map_y, map_x + map_size, map_y + map_size],
                    fill=(28, 28, 38), outline=(50, 50, 60))

        # Grid lines
        for i in range(1, 4):
            frac = i / 4
            gx = map_x + int(frac * map_size)
            gy = map_y + int(frac * map_size)
            d.line([(gx, map_y), (gx, map_y + map_size)], fill=(40, 40, 50))
            d.line([(map_x, gy), (map_x + map_size, gy)], fill=(40, 40, 50))

        # Map world XZ → screen
        m = self.map_margin
        x_range = max(self.x_max - self.x_min, 0.1)
        z_range = max(self.z_max - self.z_min, 0.1)

        def to_map(pos):
            mx = map_x + m + int((pos[0] - self.x_min) / x_range * (map_size - 2 * m))
            my = map_y + m + int((pos[2] - self.z_min) / z_range * (map_size - 2 * m))
            return mx, my

        # Trail
        com = ws.com_pos
        trail_start = max(0, frame_idx - self.trail_frames)
        for i in range(trail_start, min(frame_idx, ws.frames)):
            mx, my = to_map(com[i])
            frac = (i - trail_start) / max(1, frame_idx - trail_start)
            r = int(80 + 175 * frac)
            a = max(60, int(255 * frac))
            d.ellipse([mx - 2, my - 2, mx + 2, my + 2], fill=(r, 80, 80))

        # Current position
        if frame_idx < ws.frames:
            mx, my = to_map(com[frame_idx])
            d.ellipse([mx - 6, my - 6, mx + 6, my + 6],
                      fill=C_COM, outline=(255, 255, 255))

            # Position readout
            pos = com[frame_idx]
            d.text((map_x, map_y + map_size + 4),
                   f"x={pos[0]:+.2f}  z={pos[2]:+.2f}",
                   fill=TXT, font=f(bold=True, size=11))

            # Speed
            if frame_idx > 0:
                dx = np.linalg.norm(com[frame_idx] - com[max(0, frame_idx - 1)])
                speed = dx * ws.fps
                d.text((map_x + map_size - 70, map_y + map_size + 4),
                       f"{speed:.1f} m/s", fill=TXT_DIM, font=f(size=11))

        # ── Height bar ────────────────────────────────────────────
        bar_h = map_size
        d.text((bar_x, map_y - 2), "H", fill=TXT_DIM, font=f(bold=True, size=11))
        d.rectangle([bar_x, map_y + 12, bar_x + bar_w, map_y + 12 + bar_h],
                    fill=(28, 28, 38), outline=(50, 50, 60))

        if frame_idx < ws.frames:
            y_val = ws.com_height[frame_idx]
            y_range = max(self.y_max - self.y_min, 0.01)
            frac = (y_val - self.y_min) / y_range
            fill_h = int(frac * bar_h)
            # Gradient fill
            for py in range(fill_h):
                pf = py / max(1, fill_h)
                r = int(50 + 205 * pf)
                g = int(130 + 70 * (1 - pf))
                b = int(200 - 150 * pf)
                d.line([(bar_x + 1, map_y + 12 + bar_h - py),
                        (bar_x + bar_w - 1, map_y + 12 + bar_h - py)],
                       fill=(r, g, b))

            # Value
            marker_y = map_y + 12 + bar_h - fill_h
            d.text((bar_x + bar_w + 4, marker_y - 6),
                   f"{y_val:.2f}m", fill=TXT, font=f(bold=True, size=11))

        # Scale labels
        d.text((bar_x + bar_w + 4, map_y + 12),
               f"{self.y_max:.1f}", fill=TXT_DIM, font=f(size=9))
        d.text((bar_x + bar_w + 4, map_y + 12 + bar_h - 10),
               f"{self.y_min:.1f}", fill=TXT_DIM, font=f(size=9))

        return img
