"""VideoOverlay: draws COM dot, joint labels, border glow on the video frame."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, KEY_JOINTS, C_COM, TXT, TXT_DIM, BG


class VideoOverlay(Panel):
    """Overlay drawn directly on top of the video frame (RGBA)."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)
        self.trail_frames = int(2.0 * state.fps)  # 2s trail

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank_rgba()
        f = FontCache.get
        ws = self.state

        # ── Border glow from composite scalar ─────────────────────
        val = float(ws.composite[frame_idx]) if frame_idx < ws.frames else 0
        # Blue (low) → Red (high)
        r = int(50 + 205 * val)
        g = int(130 - 80 * val)
        b = int(200 - 180 * val)
        alpha = int(60 + 140 * val)
        d.rectangle([0, 0, self.w - 1, self.h - 1],
                     outline=(r, g, b, alpha), width=3)

        # ── Timer ─────────────────────────────────────────────────
        t = frame_idx / ws.fps
        total = ws.frames / ws.fps
        d.rounded_rectangle([self.w - 175, 12, self.w - 12, 42],
                            radius=6, fill=(0, 0, 0, 160))
        d.text((self.w - 165, 16), f"{t:.1f}s / {total:.1f}s",
               fill=(*TXT, 230), font=f(bold=True, size=16))

        # ── Composite scalar badge ────────────────────────────────
        d.rounded_rectangle([self.w - 175, 50, self.w - 12, 82],
                            radius=6, fill=(0, 0, 0, 160))
        d.text((self.w - 165, 54), f"state={val:.3f}",
               fill=(r, g, b, 255), font=f(bold=True, size=16))

        # ── μ badge (top-right) ───────────────────────────────────
        if ws.mu > 0:
            d.rounded_rectangle([self.w - 175, 90, self.w - 12, 118],
                                radius=6, fill=(0, 0, 0, 160))
            d.text((self.w - 165, 94), f"μ={ws.mu:.3f}",
                   fill=(*TXT_DIM, 200), font=f(bold=True, size=14))

        # ── Single COM dot with trail ─────────────────────────────
        # Project pelvis XZ to screen (centered in video)
        com = ws.com_pos  # (F, 3)
        # Map world XZ to screen, anchored to middle of video
        x_range = com[:, 0].max() - com[:, 0].min()
        z_range = com[:, 2].max() - com[:, 2].min()
        scale = min(self.w * 0.3, self.h * 0.3) / max(x_range, z_range, 0.1)

        cx_world = (com[:, 0].max() + com[:, 0].min()) / 2
        cz_world = (com[:, 2].max() + com[:, 2].min()) / 2
        screen_cx, screen_cy = self.w // 2, self.h - 80

        def world_to_screen(pos):
            sx = screen_cx + int((pos[0] - cx_world) * scale)
            sy = screen_cy + int((pos[2] - cz_world) * scale)
            return sx, sy

        # Trail
        trail_start = max(0, frame_idx - self.trail_frames)
        for i in range(trail_start, frame_idx):
            sx, sy = world_to_screen(com[i])
            alpha_frac = (i - trail_start) / max(1, frame_idx - trail_start)
            a = int(30 + 120 * alpha_frac)
            d.ellipse([sx - 2, sy - 2, sx + 2, sy + 2],
                      fill=(255, 80, 80, a))

        # Current position
        if frame_idx < ws.frames:
            sx, sy = world_to_screen(com[frame_idx])
            d.ellipse([sx - 6, sy - 6, sx + 6, sy + 6],
                      fill=(255, 80, 80, 220), outline=(255, 255, 255, 180))
            d.text((sx + 10, sy - 8), "COM", fill=(255, 80, 80, 200),
                   font=f(bold=True, size=12))

        # ── Coordinate units legend ───────────────────────────────
        d.rounded_rectangle([10, 10, 340, 36], radius=6, fill=(0, 0, 0, 140))
        d.text((20, 14), "Coords: meters | Y=up | GVHMR world frame",
               fill=(*TXT_DIM, 180), font=f(size=12))

        # ── Cyclic indicator ──────────────────────────────────────
        if frame_idx < ws.frames and ws.cyclic_score[frame_idx] > 0.5:
            score = ws.cyclic_score[frame_idx]
            pulse_alpha = int(100 + 155 * min(1.0, score))
            d.rounded_rectangle([10, self.h - 45, 140, self.h - 15],
                                radius=8, fill=(80, 220, 140, pulse_alpha))
            d.text((20, self.h - 42), f"CYCLIC {score:.2f}",
                   fill=(255, 255, 255, 240), font=f(bold=True, size=16))

        return img
