"""
SkeletonOverlay: velocity-colored wireframe skeleton with beat pulses.

The core insight-first visualization. Draws a clean wireframe over the
original video, colored by velocity (blue=still → white-hot=fast),
with green/red beat pulses.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, BONES, TXT


def velocity_color(speed: float, max_speed: float = 5.0) -> tuple[int, int, int]:
    """Map velocity to color: blue (still) → cyan → white → yellow → red (fast)."""
    t = min(1.0, speed / max_speed)
    if t < 0.25:
        # Blue → Cyan
        f = t / 0.25
        return (int(30 + 20 * f), int(80 + 175 * f), 255)
    elif t < 0.5:
        # Cyan → White
        f = (t - 0.25) / 0.25
        return (int(50 + 205 * f), 255, 255)
    elif t < 0.75:
        # White → Yellow
        f = (t - 0.5) / 0.25
        return (255, 255, int(255 - 200 * f))
    else:
        # Yellow → Red
        f = (t - 0.75) / 0.25
        return (255, int(255 - 200 * f), int(55 - 55 * f))


class SkeletonOverlay(Panel):
    """Draws velocity-colored wireframe skeleton with beat pulses over video."""

    def __init__(self, width: int, height: int, state, vitpose_2d: np.ndarray):
        """
        Args:
            vitpose_2d: (F, 17, 3) — 2D keypoints (x, y, confidence) on original video
        """
        super().__init__(width, height, state)
        self.vitpose = vitpose_2d
        self.max_speed = 5.0  # m/s for color normalization

        # COCO 17 skeleton connections (for vitpose)
        self.coco_bones = [
            (0, 1), (0, 2), (1, 3), (2, 4),     # face
            (5, 6),                                # shoulders
            (5, 7), (7, 9),                        # left arm
            (6, 8), (8, 10),                       # right arm
            (5, 11), (6, 12),                      # torso
            (11, 12),                              # hips
            (11, 13), (13, 15),                    # left leg
            (12, 14), (14, 16),                    # right leg
        ]

        # COCO joints → approximate velocity mapping from SMPL joints
        # COCO: 0=nose,1=Leye,2=Reye,3=Lear,4=Rear,5=Lshoulder,6=Rshoulder,
        #       7=Lelbow,8=Relbow,9=Lwrist,10=Rwrist,11=Lhip,12=Rhip,
        #       13=Lknee,14=Rknee,15=Lankle,16=Rankle
        # SMPL: 15=head, 16=Lshoulder, 17=Rshoulder, 18=Lelbow, 19=Relbow,
        #       20=Lwrist, 21=Rwrist, 1=Lhip, 2=Rhip, 4=Lknee, 5=Rknee,
        #       7=Lankle, 8=Rankle
        self.coco_to_smpl = {
            0: 15, 1: 15, 2: 15, 3: 15, 4: 15,  # head area
            5: 16, 6: 17,   # shoulders
            7: 18, 8: 19,   # elbows
            9: 20, 10: 21,  # wrists
            11: 1, 12: 2,   # hips
            13: 4, 14: 5,   # knees
            15: 7, 16: 8,   # ankles
        }

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank_rgba()
        ws = self.state

        if frame_idx >= ws.frames or frame_idx >= self.vitpose.shape[0]:
            return img

        kp = self.vitpose[frame_idx]  # (17, 3) — x, y, conf
        speeds = ws.joint_velocities[frame_idx] if ws.joint_velocities is not None else np.zeros(22)

        # Scale vitpose coords to our overlay size
        # vitpose is in original video resolution — we need to map to our panel size
        # Assume vitpose coords are in the original video pixel space
        # We need the original video resolution to scale properly
        # For now, use the keypoint range to auto-scale
        valid = kp[:, 2] > 0.3  # confidence threshold
        if valid.sum() < 3:
            return img

        # ── Beat pulse effect ─────────────────────────────────────
        beat_glow = 0.0
        beat_is_hit = None
        if ws.beat_hits:
            t = frame_idx / ws.fps
            for bh in ws.beat_hits:
                dt_beat = abs(t - bh["time_s"])
                if dt_beat < 0.15:  # within 150ms of beat
                    beat_glow = max(beat_glow, 1.0 - dt_beat / 0.15)
                    beat_is_hit = bh["hit"]
                    break

        # ── Draw bones ────────────────────────────────────────────
        for j1, j2 in self.coco_bones:
            if kp[j1, 2] < 0.3 or kp[j2, 2] < 0.3:
                continue

            x1, y1 = int(kp[j1, 0]), int(kp[j1, 1])
            x2, y2 = int(kp[j2, 0]), int(kp[j2, 1])

            # Get velocity for both endpoints
            smpl1 = self.coco_to_smpl.get(j1, 0)
            smpl2 = self.coco_to_smpl.get(j2, 0)
            speed1 = float(speeds[smpl1])
            speed2 = float(speeds[smpl2])
            avg_speed = (speed1 + speed2) / 2

            # Velocity color
            r, g, b = velocity_color(avg_speed, self.max_speed)

            # Beat pulse tint
            if beat_glow > 0:
                pulse_alpha = int(beat_glow * 200)
                if beat_is_hit:
                    # Green pulse
                    r = int(r * (1 - beat_glow * 0.5) + 80 * beat_glow)
                    g = int(min(255, g + 180 * beat_glow))
                    b = int(b * (1 - beat_glow * 0.5))
                else:
                    # Red pulse
                    r = int(min(255, r + 180 * beat_glow))
                    g = int(g * (1 - beat_glow * 0.5))
                    b = int(b * (1 - beat_glow * 0.5))

            # Line alpha based on confidence
            conf = min(kp[j1, 2], kp[j2, 2])
            alpha = int(120 + 135 * conf)

            # Draw bone with glow
            line_width = 3 if beat_glow > 0.3 else 2
            d.line([(x1, y1), (x2, y2)], fill=(r, g, b, alpha), width=line_width)

            # Glow effect (wider, dimmer line underneath)
            if avg_speed > 2.0 or beat_glow > 0.3:
                glow_alpha = int(40 + 60 * min(1.0, avg_speed / self.max_speed))
                d.line([(x1, y1), (x2, y2)], fill=(r, g, b, glow_alpha), width=line_width + 3)

        # ── Draw joint dots ───────────────────────────────────────
        for j in range(17):
            if kp[j, 2] < 0.3:
                continue
            x, y = int(kp[j, 0]), int(kp[j, 1])
            smpl_j = self.coco_to_smpl.get(j, 0)
            speed = float(speeds[smpl_j])
            r, g, b = velocity_color(speed, self.max_speed)
            dot_r = 4 if speed > 2.0 else 3
            d.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                      fill=(r, g, b, 200))

        # ── Beat flash border ─────────────────────────────────────
        if beat_glow > 0.2:
            border_alpha = int(beat_glow * 150)
            if beat_is_hit:
                border_color = (80, 255, 120, border_alpha)
            else:
                border_color = (255, 80, 80, border_alpha)
            bw = int(2 + beat_glow * 3)
            d.rectangle([0, 0, self.w - 1, self.h - 1],
                        outline=border_color, width=bw)

        return img
