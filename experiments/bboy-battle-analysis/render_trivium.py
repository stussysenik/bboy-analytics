#!/usr/bin/env python3
"""
TRIVIUM 4D Visualization — Joint trajectories + energy trails + beat matching

Renders a video showing:
  - Color-coded 3D skeleton (speed → color: blue=slow, red=fast)
  - Energy trails: trailing positions for each joint over last 0.5s
  - Beat markers: vertical lines at beat times, GREEN=hit, RED=miss
  - Per-cluster hit rate bars (live updating)
  - Hip height phase indicator
  - Coordinate readouts for key joints (pelvis, wrists, ankles, head)

Output: experiments/exports/trivium_4d.mp4
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS = 29.97
BG = (14, 14, 20)
TXT = (255, 255, 255)
TXT_DIM = (140, 140, 155)

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# SMPL skeleton bones
BONES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(12,13),(12,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),(20,22),(21,23),
]

# Cluster colors
CLUSTER_COLORS = {
    "legs":  (76, 175, 80),    # green
    "torso": (255, 87, 34),    # orange
    "arms":  (33, 150, 243),   # blue
    "hands": (156, 39, 176),   # purple
    "head":  (255, 235, 59),   # yellow
}

CLUSTER_INDICES = {
    "legs":  [1, 2, 4, 5, 7, 8, 10, 11],
    "torso": [0, 3, 6, 9],
    "arms":  [13, 14, 16, 17, 18, 19],
    "hands": [20, 21, 22, 23],
    "head":  [12, 15],
}

JOINT_NAMES = [
    "pelvis", "L.hip", "R.hip", "spine1", "L.knee", "R.knee",
    "spine2", "L.ankle", "R.ankle", "spine3", "L.foot", "R.foot",
    "neck", "L.collar", "R.collar", "head", "L.shoulder", "R.shoulder",
    "L.elbow", "R.elbow", "L.wrist", "R.wrist", "L.hand", "R.hand",
]

# Speed → color mapping (blue=slow → red=fast)
def speed_to_color(speed: float, max_speed: float) -> tuple:
    t = min(speed / max(max_speed, 1e-6), 1.0)
    if t < 0.33:
        s = t / 0.33
        return (int(30 + 50*s), int(80 + 100*s), int(220 - 40*s))
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (int(80 + 150*s), int(180 + 40*s), int(180 - 140*s))
    else:
        s = (t - 0.66) / 0.34
        return (int(230 + 25*s), int(220 - 140*s), int(40 - 30*s))


def joint_to_cluster(j: int) -> str:
    for name, indices in CLUSTER_INDICES.items():
        if j in indices:
            return name
    return "torso"


def project_3d_to_2d(pos_3d: np.ndarray, cx: float, cy: float, scale: float) -> tuple:
    """Simple orthographic projection of 3D joint to 2D screen coords."""
    x = cx + pos_3d[0] * scale
    y = cy - pos_3d[1] * scale  # Y-up → screen Y-down
    return (int(x), int(y))


def render_frame(
    draw: ImageDraw.Draw,
    frame_idx: int,
    joints: np.ndarray,       # (F, J, 3)
    speeds: np.ndarray,        # (F, J)
    beat_times: np.ndarray,
    cluster_hits: dict,
    hip_height: np.ndarray,
    hip_accel: np.ndarray,
    phase_labels: list,
    font_big, font_med, font_sm,
    max_speed: float,
    trail_frames: int = 15,
):
    F, J, _ = joints.shape
    t = frame_idx
    fps = FPS

    # ── Layout ───────────────────────────────────────────────────────
    # Left: 3D skeleton view (1200 x 900)
    # Right top: cluster hit rates (720 x 400)
    # Right mid: beat timeline (720 x 200)
    # Right bottom: hip tracking (720 x 200)
    # Bottom: coordinate readouts (1920 x 180)

    skel_cx, skel_cy = 550, 480
    skel_scale = 380.0

    # ── Draw skeleton with trails ────────────────────────────────────
    # Energy trails: draw previous positions as fading dots
    trail_start = max(0, t - trail_frames)
    for past_t in range(trail_start, t):
        alpha_frac = (past_t - trail_start) / max(trail_frames, 1)
        for j in range(min(J, 24)):
            pos = joints[past_t, j]
            sx, sy = project_3d_to_2d(pos, skel_cx, skel_cy, skel_scale)
            cluster = joint_to_cluster(j)
            base_color = CLUSTER_COLORS[cluster]
            fade = 0.15 + 0.3 * alpha_frac
            color = tuple(int(c * fade) for c in base_color)
            r = max(1, int(2 * alpha_frac))
            draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=color)

    # Current frame: draw bones
    for a, b in BONES:
        if a >= J or b >= J:
            continue
        pa = project_3d_to_2d(joints[t, a], skel_cx, skel_cy, skel_scale)
        pb = project_3d_to_2d(joints[t, b], skel_cx, skel_cy, skel_scale)
        # Bone color = average speed of endpoints
        avg_speed = (speeds[t, a] + speeds[t, b]) / 2
        bone_color = speed_to_color(avg_speed, max_speed)
        draw.line([pa, pb], fill=bone_color, width=2)

    # Current frame: draw joints with speed-based coloring
    for j in range(min(J, 24)):
        pos = joints[t, j]
        sx, sy = project_3d_to_2d(pos, skel_cx, skel_cy, skel_scale)
        spd = speeds[t, j]
        color = speed_to_color(spd, max_speed)
        r = 5
        draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=color, outline=TXT)

    # Label key joints with coordinates
    key_joints = [(0, "PELVIS"), (15, "HEAD"), (20, "L.WR"), (21, "R.WR"), (7, "L.AN"), (8, "R.AN")]
    for j_idx, label in key_joints:
        if j_idx >= J:
            continue
        pos = joints[t, j_idx]
        sx, sy = project_3d_to_2d(pos, skel_cx, skel_cy, skel_scale)
        coord_str = f"{label} ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        cluster = joint_to_cluster(j_idx)
        color = CLUSTER_COLORS[cluster]
        draw.text((sx + 8, sy - 6), coord_str, fill=color, font=font_sm)

    # ── Speed color legend ───────────────────────────────────────────
    legend_x, legend_y = 30, 30
    draw.text((legend_x, legend_y), "SPEED", fill=TXT, font=font_med)
    for i in range(100):
        c = speed_to_color(i / 100 * max_speed, max_speed)
        draw.rectangle([legend_x + i*2, legend_y+22, legend_x + i*2+2, legend_y+32], fill=c)
    draw.text((legend_x, legend_y+34), "slow", fill=TXT_DIM, font=font_sm)
    draw.text((legend_x+170, legend_y+34), "fast", fill=TXT_DIM, font=font_sm)

    # ── Phase label ──────────────────────────────────────────────────
    phase = phase_labels[t] if t < len(phase_labels) else "?"
    phase_colors = {"toprock": (33,150,243), "footwork": (76,175,80), "power/freeze": (255,87,34)}
    pc = phase_colors.get(phase, TXT)
    draw.text((30, 70), f"PHASE: {phase.upper()}", fill=pc, font=font_big)

    # ── Frame / time info ────────────────────────────────────────────
    time_s = t / fps
    draw.text((30, H - 40), f"f{t:04d} | {time_s:.2f}s", fill=TXT_DIM, font=font_sm)

    # ── Right panel: cluster hit rates ───────────────────────────────
    panel_x = 1200
    panel_y = 30
    draw.text((panel_x, panel_y), "CLUSTER HIT RATES", fill=TXT, font=font_med)
    for i, (name, data) in enumerate(cluster_hits.items()):
        y = panel_y + 30 + i * 50
        color = CLUSTER_COLORS[name]
        pct = data["hit_rate_pct"]
        bar_w = int(pct / 100 * 500)
        draw.rectangle([panel_x, y+20, panel_x+500, y+35], fill=(40,40,50))
        draw.rectangle([panel_x, y+20, panel_x+bar_w, y+35], fill=color)
        draw.text((panel_x, y), f"{name:6s}: {pct:5.1f}% ({data['hits']}/{data['total']})",
                  fill=color, font=font_sm)

    # ── Right panel: beat timeline ───────────────────────────────────
    timeline_y = 310
    timeline_w = 680
    draw.text((panel_x, timeline_y), "BEAT TIMELINE", fill=TXT, font=font_med)
    tl_y = timeline_y + 30
    tl_h = 60
    draw.rectangle([panel_x, tl_y, panel_x+timeline_w, tl_y+tl_h], fill=(30,30,40))

    # Draw beat markers
    duration_s = F / fps
    for bt in beat_times:
        bx = panel_x + int(bt / duration_s * timeline_w)
        # Check if this beat was hit (from total_match per_beat)
        draw.line([bx, tl_y, bx, tl_y+tl_h], fill=(60,60,80), width=1)

    # Current time cursor
    cursor_x = panel_x + int(time_s / duration_s * timeline_w)
    draw.line([cursor_x, tl_y, cursor_x, tl_y+tl_h], fill=(255,255,255), width=2)

    # ── Right panel: hip height over time ────────────────────────────
    hip_y_start = 420
    draw.text((panel_x, hip_y_start), "HIP HEIGHT (Kinetic Chain Engine)", fill=TXT, font=font_med)
    graph_y = hip_y_start + 28
    graph_h = 100
    draw.rectangle([panel_x, graph_y, panel_x+timeline_w, graph_y+graph_h], fill=(30,30,40))

    # Draw hip height trace up to current frame
    h_min, h_max = float(hip_height.min()), float(hip_height.max())
    h_range = max(h_max - h_min, 0.01)
    for i in range(1, min(t+1, F)):
        x0 = panel_x + int((i-1) / F * timeline_w)
        x1 = panel_x + int(i / F * timeline_w)
        y0 = graph_y + graph_h - int((hip_height[i-1] - h_min) / h_range * graph_h)
        y1 = graph_y + graph_h - int((hip_height[i] - h_min) / h_range * graph_h)
        draw.line([x0, y0, x1, y1], fill=(255, 87, 34), width=2)

    # ── Right panel: hip acceleration over time ──────────────────────
    accel_y_start = 560
    draw.text((panel_x, accel_y_start), "HIP ACCELERATION (m/s²)", fill=TXT, font=font_med)
    ag_y = accel_y_start + 28
    ag_h = 100
    draw.rectangle([panel_x, ag_y, panel_x+timeline_w, ag_y+ag_h], fill=(30,30,40))

    a_max = float(hip_accel.max()) + 0.01
    for i in range(1, min(t+1, F)):
        x0 = panel_x + int((i-1) / F * timeline_w)
        x1 = panel_x + int(i / F * timeline_w)
        y0 = ag_y + ag_h - int(hip_accel[i-1] / a_max * ag_h)
        y1 = ag_y + ag_h - int(hip_accel[i] / a_max * ag_h)
        draw.line([x0, y0, x1, y1], fill=(33, 150, 243), width=1)

    # Beat markers on acceleration graph (show physical signature match)
    for bt in beat_times:
        bx = panel_x + int(bt / duration_s * timeline_w)
        frame_at_beat = int(bt * fps)
        if 0 <= frame_at_beat < F:
            accel_at_beat = hip_accel[frame_at_beat]
            is_high = accel_at_beat > float(np.median(hip_accel))
            color = (0, 255, 100) if is_high else (255, 50, 50)
            draw.line([bx, ag_y, bx, ag_y+ag_h], fill=color, width=1)

    # ── Right panel: per-cluster energy right now ────────────────────
    energy_y = 700
    draw.text((panel_x, energy_y), "CLUSTER ENERGY (now)", fill=TXT, font=font_med)
    for i, (name, indices) in enumerate(CLUSTER_INDICES.items()):
        y = energy_y + 28 + i * 28
        color = CLUSTER_COLORS[name]
        # Current cluster speed
        cluster_speed = float(np.mean([speeds[t, j] for j in indices if j < J]))
        bar_w = int(min(cluster_speed / max(max_speed * 0.5, 0.01), 1.0) * 400)
        draw.rectangle([panel_x, y+12, panel_x+400, y+22], fill=(40,40,50))
        draw.rectangle([panel_x, y+12, panel_x+bar_w, y+22], fill=color)
        draw.text((panel_x, y-2), f"{name}: {cluster_speed:.2f} m/s", fill=color, font=font_sm)

    # ── Bottom: TRIVIUM title ────────────────────────────────────────
    draw.text((W//2 - 200, H - 50), "TRIVIUM — 4D Physical Signature Matching",
              fill=TXT_DIM, font=font_sm)


def main():
    root = Path(__file__).parent.parent.parent
    results = root / "experiments" / "results"
    viz_dir = results / "trivium_viz"

    # Load data
    print("Loading data...")
    joints = np.load(results / "joints_3d_REAL_seq4.npy")
    F, J, _ = joints.shape

    speeds = np.load(viz_dir / "joint_speeds.npy")
    hip_height = np.load(viz_dir / "hip_height.npy")
    hip_accel = np.load(viz_dir / "hip_acceleration.npy")

    # Load beat times
    beat_times = np.load(results / "librosa_beats.npy")

    # Load trivium scores
    scores_path = results / "trivium_scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            trivium = json.load(f)
        cluster_hits = trivium.get("cluster_hits", {})
    else:
        cluster_hits = {n: {"hits": 0, "total": 70, "hit_rate_pct": 0} for n in CLUSTER_INDICES}

    # Compute phase labels
    from scipy.ndimage import gaussian_filter1d
    h_smooth = gaussian_filter1d(hip_height, sigma=3.0)
    h_min, h_max = float(h_smooth.min()), float(h_smooth.max())
    h_range = h_max - h_min
    thresh_low = h_min + 0.25 * h_range
    thresh_high = h_max - 0.35 * h_range
    phase_labels = []
    for i in range(F):
        h = float(h_smooth[i])
        if h > thresh_high:
            phase_labels.append("toprock")
        elif h < thresh_low:
            phase_labels.append("power/freeze")
        else:
            phase_labels.append("footwork")

    max_speed = float(np.percentile(speeds, 98))

    # Load fonts
    try:
        font_big = ImageFont.truetype(FONT_B, 22)
        font_med = ImageFont.truetype(FONT_B, 16)
        font_sm = ImageFont.truetype(FONT_R, 12)
    except Exception:
        font_big = font_med = font_sm = ImageFont.load_default()

    # Render frames
    out_dir = root / "experiments" / "exports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "trivium_4d.mp4"

    print(f"Rendering {F} frames → {out_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for t in range(F):
            if t % 100 == 0:
                print(f"  Frame {t}/{F}...")

            img = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(img)

            render_frame(
                draw, t, joints, speeds, beat_times, cluster_hits,
                hip_height, hip_accel, phase_labels,
                font_big, font_med, font_sm, max_speed,
            )

            img.save(tmpdir / f"f{t:05d}.png")

        # Encode to video
        print("Encoding MP4...")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(tmpdir / "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "fast",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True)

    print(f"Done: {out_path}")
    print(f"  {F} frames, {F/FPS:.1f}s @ {FPS} fps")


if __name__ == "__main__":
    main()
