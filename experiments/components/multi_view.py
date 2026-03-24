"""
MultiViewPanel: 4 orthographic skeleton views (front/side/back/top).

Shows spatial context of the dancer from all angles. Each view has:
- Skeleton wireframe (cluster-colored)
- Labeled key joints (HEAD, HIP, L.HIP, R.HIP, wrists, ankles)
- COM trail (2s, red fading)
- Grid lines (1m spacing)
- View title

Projection logic reused from render_trails.py TriViewRenderer.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .panel import Panel, FontCache, BG, TXT, TXT_DIM, BONES, KEY_JOINTS

# Cluster colors for bones/joints (same as render_trails.py)
CLUSTER_COLORS = {
    "legs":  (255, 107, 53),
    "torso": (230, 57, 70),
    "arms":  (69, 123, 157),
    "hands": (168, 85, 247),
    "head":  (224, 242, 254),
}

JOINT_CLUSTER = {}
_GROUPS = {
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "torso": [0, 3, 6, 9],
    "arms": [13, 14, 16, 17, 18, 19],
    "hands": [20, 21, 22, 23],
    "head": [12, 15],
}
for _c, _ids in _GROUPS.items():
    for _i in _ids:
        JOINT_CLUSTER[_i] = _c

C_COM = (255, 80, 80)

# Labeled joints: (index, name, label_side)
# label_side: "above", "below", "left", "right"
LABELED_JOINTS = [
    (15, "HEAD",    "above"),
    (0,  "HIP",     "below"),
    (1,  "L.HIP",   "left"),
    (2,  "R.HIP",   "right"),
    (20, "L.WRIST", "left"),
    (21, "R.WRIST", "right"),
    (7,  "L.ANKLE", "left"),
    (8,  "R.ANKLE", "right"),
]

# View definitions: (name, u_axis, v_axis, flip_u)
VIEWS = [
    ("FRONT", 0, 1, False),   # X→horiz, Y→vert
    ("SIDE",  2, 1, False),   # Z→horiz, Y→vert
    ("BACK",  0, 1, True),    # -X→horiz, Y→vert (mirrored)
    ("TOP",   0, 2, False),   # X→horiz, Z→vert (bird's eye)
]


# ── Reusable projection helpers ──

def compute_bounds(joints: np.ndarray, pad: float = 0.4) -> dict:
    """Compute global bounding boxes for each axis pair across all frames.

    Returns dict with 'x', 'y', 'z' min/max values.
    """
    return {
        "x": (float(joints[:, :, 0].min() - pad), float(joints[:, :, 0].max() + pad)),
        "y": (float(joints[:, :, 1].min() - pad), float(joints[:, :, 1].max() + pad)),
        "z": (float(joints[:, :, 2].min() - pad), float(joints[:, :, 2].max() + pad)),
    }


def project_ortho(point: np.ndarray, u_axis: int, v_axis: int, flip_u: bool,
                   bounds: dict, px: int, py: int, pw: int, ph: int,
                   is_top: bool = False) -> tuple[int, int]:
    """Project a 3D point to 2D panel coordinates via orthographic projection.

    Args:
        point: (3,) array with x, y, z
        u_axis: index for horizontal axis (0=x, 1=y, 2=z)
        v_axis: index for vertical axis
        flip_u: mirror horizontal axis (for back view)
        bounds: dict from compute_bounds()
        px, py: panel top-left position
        pw, ph: panel width/height
        is_top: True for top-down view (Z increases downward)
    """
    axis_names = ["x", "y", "z"]
    u = float(point[u_axis])
    v = float(point[v_axis])

    u_min, u_max = bounds[axis_names[u_axis]]
    v_min, v_max = bounds[axis_names[v_axis]]

    ufrac = (u - u_min) / max(u_max - u_min, 0.01)
    vfrac = (v - v_min) / max(v_max - v_min, 0.01)

    if flip_u:
        ufrac = 1.0 - ufrac

    sx = px + int(ufrac * pw)
    if is_top:
        sy = py + int(vfrac * ph)
    else:
        sy = py + ph - int(vfrac * ph)  # Y-up: invert

    return sx, sy


class MultiViewPanel(Panel):
    """4 orthographic skeleton views: front, side, back, top."""

    def prerender(self) -> None:
        ws = self.state
        self._bounds = compute_bounds(ws.joints)
        self._n_joints = ws.joints.shape[1]

    def draw(self, frame_idx: int) -> Image.Image:
        img, d = self._blank()
        ws = self.state
        f = FontCache.get

        pw = self.w // 4  # each sub-panel width
        ph = self.h
        j_now = ws.joints[frame_idx]

        for vi, (view_name, u_ax, v_ax, flip_u) in enumerate(VIEWS):
            vx = vi * pw  # sub-panel x offset
            is_top = (view_name == "TOP")

            def proj(pt):
                return project_ortho(pt, u_ax, v_ax, flip_u,
                                     self._bounds, vx, 0, pw, ph, is_top)

            # Panel border
            d.rectangle([vx, 0, vx + pw - 1, ph - 1], outline=(40, 40, 50))

            # Grid lines (1m spacing)
            axis_names = ["x", "y", "z"]
            u_min, u_max = self._bounds[axis_names[u_ax]]
            v_min, v_max = self._bounds[axis_names[v_ax]]
            for val in np.arange(np.ceil(u_min), u_max, 1.0):
                frac = (val - u_min) / max(u_max - u_min, 0.01)
                if flip_u:
                    frac = 1.0 - frac
                gx = vx + int(frac * pw)
                d.line([(gx, 0), (gx, ph)], fill=(30, 30, 40), width=1)
            for val in np.arange(np.ceil(v_min), v_max, 1.0):
                frac = (val - v_min) / max(v_max - v_min, 0.01)
                gy = int(frac * ph) if is_top else ph - int(frac * ph)
                d.line([(vx, gy), (vx + pw, gy)], fill=(30, 30, 40), width=1)

            # COM trail (2s = 60 frames)
            trail_len = min(60, frame_idx)
            for ti in range(max(0, frame_idx - trail_len), frame_idx):
                prog = (ti - (frame_idx - trail_len)) / max(trail_len, 1)
                alpha = int(30 + 180 * prog)
                cpx, cpy = proj(ws.com_pos[ti])
                r = max(1, int(2 * prog))
                d.ellipse([cpx - r, cpy - r, cpx + r, cpy + r],
                          fill=(*C_COM, min(alpha, 255)))

            # Skeleton wireframe
            for j1, j2 in BONES:
                if j1 >= self._n_joints or j2 >= self._n_joints:
                    continue
                p1 = proj(j_now[j1])
                p2 = proj(j_now[j2])
                c = CLUSTER_COLORS.get(JOINT_CLUSTER.get(j1, "torso"), (150, 150, 150))
                d.line([p1, p2], fill=(*c, 200), width=2)

            # Joint dots
            for ji in range(self._n_joints):
                c = CLUSTER_COLORS.get(JOINT_CLUSTER.get(ji, "torso"), (150, 150, 150))
                px_j, py_j = proj(j_now[ji])
                r = 4 if ji in {0, 1, 2, 7, 8, 15, 20, 21} else 2
                d.ellipse([px_j - r, py_j - r, px_j + r, py_j + r], fill=(*c, 255))

            # COM dot (bright, current frame)
            cpx, cpy = proj(ws.com_pos[frame_idx])
            d.ellipse([cpx - 5, cpy - 5, cpx + 5, cpy + 5],
                      fill=(*C_COM, 255), outline=(255, 255, 255, 180))

            # Joint labels (only labeled joints, smart placement)
            lbl_font = f(bold=True, size=9)
            for ji, name, side in LABELED_JOINTS:
                if ji >= self._n_joints:
                    continue
                px_j, py_j = proj(j_now[ji])

                # Compute label position based on side preference
                if side == "above":
                    lx, ly = px_j - 12, py_j - 16
                elif side == "below":
                    lx, ly = px_j - 12, py_j + 6
                elif side == "left":
                    lx, ly = px_j - 50, py_j - 6
                elif side == "right":
                    lx, ly = px_j + 8, py_j - 6

                # Clamp to sub-panel bounds
                lx = max(vx + 2, min(lx, vx + pw - 50))
                ly = max(2, min(ly, ph - 14))

                # Draw pill background + text
                bb = d.textbbox((lx, ly), name, font=lbl_font)
                d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1],
                            fill=(0, 0, 0, 160))
                c = CLUSTER_COLORS.get(JOINT_CLUSTER.get(ji, "torso"), TXT)
                d.text((lx, ly), name, fill=c, font=lbl_font)

            # COM label
            d.text((cpx + 7, cpy - 4), "COM", fill=(255, 215, 0),
                   font=f(bold=True, size=9))

            # View title (top-left corner)
            d.text((vx + 6, 4), view_name, fill=TXT_DIM, font=f(bold=True, size=12))

        return img
