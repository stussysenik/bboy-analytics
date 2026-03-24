"""DataPointsPanel: 6 tracked joints with live x,y,z coordinates and sparklines."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, KEY_JOINTS, TXT, TXT_DIM, BG


class DataPointsPanel(Panel):
    """Shows 6 key joints as tracked data point systems with coordinates and velocity."""

    def __init__(self, width: int, height: int, state):
        super().__init__(width, height, state)
        self.trail_s = 3.0  # seconds of sparkline trail
        self.trail_frames = int(self.trail_s * state.fps)

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank()
        f = FontCache.get
        ws = self.state
        joints = ws.joints  # (F, 22, 3)

        # Title
        d.text((10, 4), "TRACKED SYSTEMS", fill=TXT_DIM, font=f(bold=True, size=13))

        # Layout: 2 rows x 3 columns
        col_w = self.w // 3
        row_h = (self.h - 25) // 2
        y_offset = 25

        for idx, (ji, name, color) in enumerate(KEY_JOINTS):
            col = idx % 3
            row = idx // 3
            px = col * col_w + 8
            py = y_offset + row * row_h + 4

            if frame_idx >= ws.frames:
                continue

            pos = joints[frame_idx, ji]  # (3,)
            vel_mag = float(ws.joint_velocities[frame_idx, ji]) if ws.joint_velocities is not None else 0

            # Joint name + color dot
            d.ellipse([px, py + 2, px + 8, py + 10], fill=color)
            d.text((px + 12, py - 1), name, fill=color, font=f(bold=True, size=12))

            # Velocity bar
            vel_norm = min(1.0, vel_mag / 5.0)  # normalize to ~5 m/s max
            bar_w = int(vel_norm * 60)
            bar_y = py + 1
            d.rectangle([px + 80, bar_y, px + 80 + bar_w, bar_y + 8],
                        fill=(*color[:2], min(255, color[2] + 80)))
            d.text((px + 145, py - 1), f"{vel_mag:.1f}m/s", fill=TXT_DIM, font=f(size=10))

            # x, y, z coordinates
            cy = py + 16
            for axis, label, val in [(0, "x", pos[0]), (1, "y", pos[1]), (2, "z", pos[2])]:
                sign = "+" if val >= 0 else ""
                d.text((px + 4, cy), f"{label}={sign}{val:.3f}", fill=TXT, font=f(bold=True, size=11))
                cy += 13

            # Sparkline (last 3s of this joint's trajectory, Y axis = height)
            spark_x = px + 4
            spark_y = cy + 2
            spark_w = col_w - 20
            spark_h = row_h - (cy - (y_offset + row * row_h)) - 10

            if spark_h > 10 and frame_idx > 5:
                trail_start = max(0, frame_idx - self.trail_frames)
                trail = joints[trail_start:frame_idx + 1, ji, :]

                # Draw sparkline background
                d.rectangle([spark_x, spark_y, spark_x + spark_w, spark_y + spark_h],
                            fill=(30, 30, 40))

                # Plot all 3 axes as thin lines
                axis_colors = [
                    (*color[:2], min(255, color[2] + 40)),  # x - slightly shifted
                    (200, 200, 200),                         # y - white (height)
                    (*color[:1], min(255, color[1] + 60), color[2]),  # z
                ]
                for ax in range(3):
                    vals = trail[:, ax]
                    if len(vals) < 2:
                        continue
                    v_min, v_max = vals.min(), vals.max()
                    v_range = max(v_max - v_min, 0.01)

                    points = []
                    for i, v in enumerate(vals):
                        sx = spark_x + int(i / max(1, len(vals) - 1) * spark_w)
                        sy = spark_y + spark_h - int((v - v_min) / v_range * spark_h)
                        points.append((sx, sy))

                    if len(points) > 1:
                        c = axis_colors[ax] if ax < len(axis_colors) else color
                        for i in range(len(points) - 1):
                            alpha = int(80 + 175 * (i / len(points)))
                            d.line([points[i], points[i + 1]],
                                   fill=c, width=1)

        return img
