"""
ContactLight: floor contact visualization as projected light pools.

Makes the invisible (ground contact physics) visible. Renders glowing
ellipses on the floor where the dancer is grounded.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .panel import Panel


# Joint indices for contact detection
CONTACT_JOINTS = {
    "L.ankle": 7,
    "R.ankle": 8,
    "L.foot": 10,
    "R.foot": 11,
    "L.wrist": 20,
    "R.wrist": 21,
}

FOOT_COLOR = (100, 180, 255)   # Blue-white light
HAND_COLOR = (255, 160, 80)     # Warm orange light


class ContactLight(Panel):
    """Draws floor contact light pools on the video overlay."""

    def __init__(self, width: int, height: int, state, vitpose_2d: np.ndarray):
        super().__init__(width, height, state)
        self.vitpose = vitpose_2d

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank_rgba()
        ws = self.state

        if frame_idx >= ws.frames or ws.contact_feet is None:
            return img

        kp = self.vitpose[frame_idx] if frame_idx < self.vitpose.shape[0] else None
        if kp is None:
            return img

        # Foot contacts (COCO indices: 15=Lankle, 16=Rankle)
        foot_coco = [(15, 0), (16, 1)]  # (coco_idx, contact_feet column)
        for coco_j, contact_col in foot_coco:
            if kp[coco_j, 2] < 0.3:
                continue

            # Also check L.foot/R.foot contacts (columns 2, 3)
            conf_ankle = float(ws.contact_feet[frame_idx, contact_col])
            conf_foot = float(ws.contact_feet[frame_idx, contact_col + 2])
            conf = max(conf_ankle, conf_foot)

            if conf < 0.1:
                continue

            x, y = int(kp[coco_j, 0]), int(kp[coco_j, 1])

            # Draw light pool — ellipse at foot position, larger when more confident
            pool_rx = int(15 + 25 * conf)
            pool_ry = int(8 + 12 * conf)
            alpha = int(40 + 160 * conf)

            # Outer glow
            d.ellipse([x - pool_rx - 5, y - pool_ry - 3, x + pool_rx + 5, y + pool_ry + 3],
                      fill=(FOOT_COLOR[0], FOOT_COLOR[1], FOOT_COLOR[2], alpha // 3))
            # Inner pool
            d.ellipse([x - pool_rx, y - pool_ry, x + pool_rx, y + pool_ry],
                      fill=(FOOT_COLOR[0], FOOT_COLOR[1], FOOT_COLOR[2], alpha))

        # Hand contacts (COCO indices: 9=Lwrist, 10=Rwrist)
        if ws.contact_hands is not None:
            hand_coco = [(9, 0), (10, 1)]
            for coco_j, contact_col in hand_coco:
                if kp[coco_j, 2] < 0.3:
                    continue

                conf = float(ws.contact_hands[frame_idx, contact_col])
                if conf < 0.1:
                    continue

                x, y = int(kp[coco_j, 0]), int(kp[coco_j, 1])
                pool_r = int(12 + 20 * conf)
                alpha = int(40 + 160 * conf)

                d.ellipse([x - pool_r - 4, y - pool_r - 2, x + pool_r + 4, y + pool_r + 2],
                          fill=(HAND_COLOR[0], HAND_COLOR[1], HAND_COLOR[2], alpha // 3))
                d.ellipse([x - pool_r, y - pool_r, x + pool_r, y + pool_r],
                          fill=(HAND_COLOR[0], HAND_COLOR[1], HAND_COLOR[2], alpha))

        return img
