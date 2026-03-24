"""VideoOverlay: draws COM dot, joint labels, border glow on the video frame."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, KEY_JOINTS, C_COM, TXT, TXT_DIM, BG


# Labeled data points: (joint_idx, name, label_side, color)
_DATA_JOINTS = [
    (15, "HEAD",    "above", (224, 242, 254)),
    (0,  "PELVIS",  "below", (230, 57, 70)),
    (1,  "L.HIP",   "left",  (255, 107, 53)),
    (2,  "R.HIP",   "right", (255, 107, 53)),
    (20, "L.WRIST", "left",  (168, 85, 247)),
    (21, "R.WRIST", "right", (168, 85, 247)),
    (7,  "L.ANKLE", "left",  (255, 107, 53)),
    (8,  "R.ANKLE", "right", (255, 107, 53)),
]


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

        # ── TouchDesigner-style data points on skeleton ────────────
        if frame_idx < ws.frames:
            self._draw_data_points(d, f, frame_idx)

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

    def _draw_data_points(self, d: ImageDraw.ImageDraw, f, frame_idx: int):
        """Draw TouchDesigner-style data points: joint dots + XYZ coords + speed."""
        ws = self.state
        joints = ws.joints[frame_idx]  # (22, 3)
        n_j = joints.shape[0]

        # Orthographic projection: fit skeleton into video center
        positions = joints[:n_j]
        center = positions.mean(axis=0)
        span_x = positions[:, 0].max() - positions[:, 0].min()
        span_y = positions[:, 1].max() - positions[:, 1].min()
        max_span = max(span_x, span_y, 0.5)
        scale = min(self.w, self.h) * 0.6 / max_span

        def proj_2d(pos):
            sx = self.w // 2 + int((pos[0] - center[0]) * scale)
            sy = self.h - int(self.h * 0.15 + (pos[1] - center[1]) * scale)
            return sx, sy

        # Draw skeleton wireframe (subtle)
        from .panel import BONES
        for j1, j2 in BONES:
            if j1 >= n_j or j2 >= n_j:
                continue
            p1 = proj_2d(joints[j1])
            p2 = proj_2d(joints[j2])
            d.line([p1, p2], fill=(255, 255, 255, 40), width=1)

        lbl_font = f(size=10)
        coord_font = f(size=9)

        # Draw each labeled joint
        for ji, name, side, color in _DATA_JOINTS:
            if ji >= n_j:
                continue
            pos = joints[ji]
            sx, sy = proj_2d(pos)

            # Joint dot
            r = 5
            d.ellipse([sx - r, sy - r, sx + r, sy + r],
                      fill=(*color, 220), outline=(255, 255, 255, 120))

            # Speed
            speed = 0.0
            if ws.joint_velocities is not None and frame_idx < ws.frames:
                speed = float(ws.joint_velocities[frame_idx, ji])

            # Label text
            coord_text = f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
            speed_text = f"{speed:.1f} m/s"

            # Compute label position (no overlap)
            if side == "above":
                lx, ly = sx - 30, sy - 38
            elif side == "below":
                lx, ly = sx - 30, sy + 10
            elif side == "left":
                lx, ly = sx - 130, sy - 10
            elif side == "right":
                lx, ly = sx + 12, sy - 10

            # Clamp to overlay bounds
            lx = max(4, min(lx, self.w - 140))
            ly = max(4, min(ly, self.h - 30))

            # Pill background
            pill_w = 132
            pill_h = 26
            d.rounded_rectangle([lx - 2, ly - 2, lx + pill_w, ly + pill_h],
                                radius=4, fill=(0, 0, 0, 170))

            # Name + speed on line 1
            d.text((lx, ly), f"{name} {speed_text}", fill=(*color, 240), font=lbl_font)
            # Coordinates on line 2
            d.text((lx, ly + 13), coord_text, fill=(200, 200, 210, 180), font=coord_font)

        # COM star (★) — always most prominent
        com = ws.com_pos[frame_idx]
        cx, cy = proj_2d(com)
        # Gold star
        d.text((cx - 6, cy - 8), "★", fill=(255, 215, 0, 255), font=f(bold=True, size=16))
        # COM label
        com_speed = 0.0
        if ws.joint_velocities is not None and frame_idx < ws.frames:
            # COM speed approximated from pelvis velocity
            com_speed = float(ws.joint_velocities[frame_idx, 0])
        com_text = f"COM ({com[0]:+.2f}, {com[1]:+.2f}, {com[2]:+.2f}) {com_speed:.1f}m/s"
        d.rounded_rectangle([cx + 10, cy - 6, cx + 10 + len(com_text) * 6 + 8, cy + 12],
                            radius=4, fill=(0, 0, 0, 180))
        d.text((cx + 14, cy - 4), com_text, fill=(255, 215, 0, 255), font=f(bold=True, size=10))
