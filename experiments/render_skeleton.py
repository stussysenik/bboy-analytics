"""
Video 1: Skeleton + Joints — Mesh overlay with 6 key joint coordinates + 3 COM centers.

Full 1920x1080. Shows:
- GVHMR mesh overlay (from mesh video)
- 6 key joints with (x,y,z) labels: head, L/R wrist, pelvis, L/R ankle
- 3 COM dots: red=full body, blue=upper, green=lower
- Move type label + units legend
- Coordinate readout panel at bottom

Output: experiments/exports/v2/skeleton_joints.mp4
"""

from __future__ import annotations
import json, subprocess, sys, time, csv, tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 29.97
VID_H = 880  # Large video area
PANEL_H = 200  # Bottom coordinate panel
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BG = (14, 14, 20)
TXT = (255, 255, 255)
TXT_DIM = (140, 140, 155)
C_COM_FULL = (255, 80, 80)    # Red - full body
C_COM_UPPER = (80, 160, 255)  # Blue - upper body
C_COM_LOWER = (80, 220, 100)  # Green - lower body
SEG_COLORS = {"toprock": (33, 150, 243), "footwork": (76, 175, 80), "powermove": (255, 87, 34)}

# Key joints to label (index, name, color)
KEY_JOINTS = [
    (15, "HEAD", (255, 220, 80)),
    (0,  "PELVIS", C_COM_FULL),
    (20, "L.WRIST", (200, 150, 255)),
    (21, "R.WRIST", (200, 150, 255)),
    (7,  "L.ANKLE", (100, 220, 180)),
    (8,  "R.ANKLE", (100, 220, 180)),
]

# SMPL skeleton connections (bone pairs)
BONES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(12,13),(12,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),
]

# COM group indices
UPPER_JOINTS = [9, 12, 16, 17]   # spine3, neck, shoulders
LOWER_JOINTS = [1, 2, 4, 5]      # hips, knees

BASE = Path(__file__).parent


def load_all(joints_path=None, metrics_path=None):
    r = BASE / "results"
    joints = np.load(joints_path or (r / "joints_3d_CLEAN.npy"))
    with open(metrics_path or (r / "CLEAN_metrics.json")) as f:
        metrics = json.load(f)
    segments = []
    seg_file = BASE.parent / "data/brace/annotations/segments.csv"
    if seg_file.exists():
        with open(seg_file) as f:
            for row in csv.DictReader(f):
                if row["video_id"] == "RS0mFARO1x4" and int(row["seq_idx"]) == 4:
                    segments.append({
                        "start_s": (int(row["start_frame"]) - 3802) / FPS,
                        "end_s": (int(row["end_frame"]) - 3802) / FPS,
                        "dance_type": row["dance_type"],
                    })
    return joints, metrics, segments


def project_3d_to_screen(joints_3d, frame_w, frame_h):
    """Simple orthographic projection of 3D joints to screen coordinates.
    Uses XZ plane mapped to screen width/height, with Y as depth."""
    # Normalize to frame coordinates
    # X → screen X, Z → screen Y (top-down-ish), Y = height
    x = joints_3d[:, 0]
    z = joints_3d[:, 2]

    # Scale to frame - use all-frame range for stability
    return x, z  # Return raw 3D coords, we'll overlay as text


class SkeletonRenderer:
    def __init__(self, joints, metrics, segments):
        self.joints = joints
        self.metrics = metrics
        self.segments = segments
        self.n_frames = joints.shape[0]
        self.total_dur = self.n_frames / FPS

        # Compute 3 COM centers for all frames
        self.com_full = joints[:, 0, :]  # Pelvis
        self.com_upper = joints[:, UPPER_JOINTS, :].mean(axis=1)
        self.com_lower = joints[:, LOWER_JOINTS, :].mean(axis=1)

        self.f_big = ImageFont.truetype(FONT_B, 22)
        self.f_med = ImageFont.truetype(FONT_B, 16)
        self.f_sm = ImageFont.truetype(FONT_R, 13)
        self.f_coord = ImageFont.truetype(FONT_B, 14)
        self.f_title = ImageFont.truetype(FONT_B, 18)

    def _current_segment(self, t):
        for seg in self.segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return None

    def draw(self, frame_idx, video_frame):
        canvas = Image.new("RGB", (W, H), BG)
        t = frame_idx / FPS
        j = self.joints[frame_idx]  # (22, 3)

        # Top: video frame
        vf = video_frame.resize((W, VID_H), Image.LANCZOS)
        canvas.paste(vf, (0, 0))

        # Overlay on video
        overlay = Image.new("RGBA", (W, VID_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        # Move type label
        seg = self._current_segment(t)
        if seg:
            c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
            label = seg["dance_type"].upper()
            tw = self.f_big.getlength(label)
            od.rounded_rectangle([20, VID_H - 55, 20 + tw + 24, VID_H - 20],
                                radius=8, fill=(*c, 200))
            od.text((32, VID_H - 50), label, fill=TXT, font=self.f_big)

        # Timer
        od.rounded_rectangle([W - 180, 15, W - 15, 50], radius=8, fill=(0, 0, 0, 180))
        od.text((W - 170, 20), f"{t:.1f}s / {self.total_dur:.1f}s", fill=TXT, font=self.f_med)

        # 3 COM dots on video (projected roughly to center area)
        # These are in 3D space, we show them as colored dots near center
        # with coordinate readouts
        com_data = [
            ("BODY", self.com_full[frame_idx], C_COM_FULL),
            ("UPPER", self.com_upper[frame_idx], C_COM_UPPER),
            ("LOWER", self.com_lower[frame_idx], C_COM_LOWER),
        ]

        # Draw COM dots in a cluster near center-bottom of video
        cx, cy = W // 2, VID_H - 100
        for i, (name, pos, color) in enumerate(com_data):
            # Offset each dot slightly
            dx = (i - 1) * 60
            dot_x = cx + dx
            dot_y = cy
            od.ellipse([dot_x - 8, dot_y - 8, dot_x + 8, dot_y + 8],
                      fill=(*color, 220), outline=(255, 255, 255, 180))
            od.text((dot_x - 15, dot_y + 12), name, fill=(*color, 255), font=self.f_sm)

        # Units legend
        od.rounded_rectangle([15, 15, 380, 45], radius=6, fill=(0, 0, 0, 160))
        od.text((25, 20), "Coordinates: meters | Gravity-view (Y=up) | GVHMR",
                fill=TXT_DIM, font=self.f_sm)

        canvas.paste(Image.alpha_composite(
            canvas.crop((0, 0, W, VID_H)).convert("RGBA"), overlay
        ).convert("RGB"), (0, 0))

        # ── Bottom panel: Joint coordinate readout ──────────────
        d = ImageDraw.Draw(canvas)
        py = VID_H + 5

        # Title
        d.text((20, py), "3D JOINT COORDINATES", fill=TXT_DIM, font=self.f_title)
        d.text((300, py + 2), "(6 key joints, SMPL 22-joint skeleton)", fill=TXT_DIM, font=self.f_sm)

        # 3 COM readouts on the right
        cx_start = W - 500
        d.text((cx_start, py), "CENTER OF MASS", fill=TXT_DIM, font=self.f_title)
        for i, (name, pos, color) in enumerate(com_data):
            ry = py + 25 + i * 22
            d.ellipse([cx_start, ry + 2, cx_start + 12, ry + 14], fill=color)
            d.text((cx_start + 18, ry),
                   f"{name}: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})",
                   fill=TXT, font=self.f_coord)

        # Joint readouts in a grid (2 rows x 3 cols)
        jy = py + 28
        col_w = 240
        for idx, (ji, name, color) in enumerate(KEY_JOINTS):
            col = idx % 3
            row = idx // 3
            jx = 20 + col * col_w
            jry = jy + row * 50

            pos = j[ji]
            d.rectangle([jx, jry, jx + 8, jry + 8], fill=color)
            d.text((jx + 14, jry - 3), name, fill=color, font=self.f_coord)
            d.text((jx + 14, jry + 16),
                   f"x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}",
                   fill=TXT, font=self.f_coord)

        # Divider line
        d.line([(0, VID_H), (W, VID_H)], fill=(60, 60, 70), width=2)

        # Height bars for 3 COMs
        bar_x = W - 140
        bar_h = 150
        bar_y = py + 30
        d.text((bar_x, py + 10), "HEIGHT (m)", fill=TXT_DIM, font=self.f_sm)
        d.rectangle([bar_x, bar_y, bar_x + 100, bar_y + bar_h], outline=(60, 60, 70))

        y_min = min(self.com_full[:, 1].min(), self.com_lower[:, 1].min())
        y_max = max(self.com_full[:, 1].max(), self.com_upper[:, 1].max())
        y_range = max(y_max - y_min, 0.01)

        for i, (name, pos, color) in enumerate(com_data):
            frac = (pos[1] - y_min) / y_range
            bx = bar_x + 10 + i * 30
            fill_h = int(frac * bar_h)
            d.rectangle([bx, bar_y + bar_h - fill_h, bx + 22, bar_y + bar_h], fill=color)
            d.text((bx, bar_y + bar_h + 3), f"{pos[1]:.2f}", fill=color, font=self.f_sm)

        return canvas


def render(joints_path=None, metrics_path=None, mesh_video_path=None, audio_path=None, output_path=None):
    output = output_path or str(BASE / "exports/v2/skeleton_joints.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    joints, metrics, segments = load_all(joints_path, metrics_path)
    n_frames = joints.shape[0]

    renderer = SkeletonRenderer(joints, metrics, segments)

    mesh_video = mesh_video_path or str(BASE / "results/gvhmr_mesh_clean_seq4.mp4")
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                           "-show_streams", mesh_video], capture_output=True, text=True)
    vs = [s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video"][0]
    src_w, src_h = int(vs["width"]), int(vs["height"])
    src_bytes = src_w * src_h * 3

    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_src = audio_path or "/tmp/lilg_round.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", audio_src, "-vn", "-c:a", "aac",
                    "-b:a", "192k", audio_tmp.name], capture_output=True)

    read_proc = subprocess.Popen(
        ["ffmpeg", "-i", mesh_video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=src_bytes * 2)

    write_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-i", audio_tmp.name,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac",
         "-pix_fmt", "yuv420p", "-shortest", "-v", "error", output],
        stdin=subprocess.PIPE, bufsize=W * H * 3 * 2)

    print(f"Rendering {n_frames} frames...")
    t0 = time.time()

    for fi in range(n_frames + 10):
        raw = read_proc.stdout.read(src_bytes)
        if len(raw) < src_bytes:
            break
        if fi >= n_frames:
            continue

        vf = Image.frombytes("RGB", (src_w, src_h), raw)
        canvas = renderer.draw(fi, vf)
        write_proc.stdin.write(canvas.tobytes())

        if fi % 200 == 0 and fi > 0:
            elapsed = time.time() - t0
            print(f"  {fi}/{n_frames} ({100*fi/n_frames:.0f}%) — {fi/elapsed:.1f} fps")

    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()
    print(f"\nDone! {output} ({time.time()-t0:.0f}s)")
    Path(audio_tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", default=None, help="Path to joints_3d.npy")
    parser.add_argument("--metrics", default=None, help="Path to metrics.json")
    parser.add_argument("--mesh-video", default=None, help="Path to mesh overlay video")
    parser.add_argument("--audio", default=None, help="Path to audio source (video or audio file)")
    parser.add_argument("--output", default=None, help="Output MP4 path")
    args = parser.parse_args()
    render(joints_path=args.joints, metrics_path=args.metrics,
           mesh_video_path=args.mesh_video, audio_path=args.audio,
           output_path=args.output)
