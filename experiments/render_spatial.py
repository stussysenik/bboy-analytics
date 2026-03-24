"""
Spatial Renderer (v2) — COM tracking + stage coverage + heatmap.

Full 1920x1080. Shows:
- Battle video with 3 COM trails building up on stage
- Semi-transparent spatial heatmap overlay
- Coverage area counter (m²) growing in real-time
- Height profile sidebar
- Top-down mini map in corner

Output: experiments/exports/spatial/
"""

from __future__ import annotations
import json, subprocess, sys, time, csv, tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 29.97
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BG = (14, 14, 20)
TXT = (255, 255, 255)
TXT_DIM = (140, 140, 155)
C_COM_FULL = (255, 80, 80)
C_COM_UPPER = (80, 160, 255)
C_COM_LOWER = (80, 220, 100)
SEG_COLORS = {"toprock": (33, 150, 243), "footwork": (76, 175, 80), "powermove": (255, 87, 34)}

UPPER_JOINTS = [9, 12, 16, 17]
LOWER_JOINTS = [1, 2, 4, 5]

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


class SpatialRenderer:
    def __init__(self, joints, metrics, segments):
        self.joints = joints
        self.metrics = metrics
        self.segments = segments
        self.n_frames = joints.shape[0]
        self.total_dur = self.n_frames / FPS

        # 3 COM centers
        self.com_full = joints[:, 0, :]
        self.com_upper = joints[:, UPPER_JOINTS, :].mean(axis=1)
        self.com_lower = joints[:, LOWER_JOINTS, :].mean(axis=1)

        # XZ coordinates for all COMs
        self.full_xz = self.com_full[:, [0, 2]]
        self.upper_xz = self.com_upper[:, [0, 2]]
        self.lower_xz = self.com_lower[:, [0, 2]]

        # Global range for consistent mapping
        all_xz = np.vstack([self.full_xz, self.upper_xz, self.lower_xz])
        self.xmin, self.xmax = all_xz[:, 0].min() - 0.2, all_xz[:, 0].max() + 0.2
        self.zmin, self.zmax = all_xz[:, 1].min() - 0.2, all_xz[:, 1].max() + 0.2

        # Pre-compute heatmap (accumulates)
        self.heatmap_size = 200
        self.heatmap = np.zeros((self.heatmap_size, self.heatmap_size))

        self.f_big = ImageFont.truetype(FONT_B, 24)
        self.f_med = ImageFont.truetype(FONT_B, 18)
        self.f_sm = ImageFont.truetype(FONT_R, 14)
        self.f_title = ImageFont.truetype(FONT_B, 20)

    def _map_to_minimap(self, xz, map_x, map_y, map_size):
        """Map XZ coordinates to mini-map pixel coordinates."""
        px = map_x + int((xz[0] - self.xmin) / (self.xmax - self.xmin) * map_size)
        py = map_y + int((xz[1] - self.zmin) / (self.zmax - self.zmin) * map_size)
        return px, py

    def _update_heatmap(self, frame_idx):
        """Add current position to heatmap."""
        xz = self.full_xz[frame_idx]
        hx = int((xz[0] - self.xmin) / (self.xmax - self.xmin) * (self.heatmap_size - 1))
        hz = int((xz[1] - self.zmin) / (self.zmax - self.zmin) * (self.heatmap_size - 1))
        hx = np.clip(hx, 0, self.heatmap_size - 1)
        hz = np.clip(hz, 0, self.heatmap_size - 1)
        # Gaussian splat
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = hz + dy, hx + dx
                if 0 <= ny < self.heatmap_size and 0 <= nx < self.heatmap_size:
                    dist = (dx * dx + dy * dy) ** 0.5
                    self.heatmap[ny, nx] += max(0, 1.0 - dist / 4.0)

    def _current_segment(self, t):
        for seg in self.segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return None

    def _compute_coverage(self, frame_idx):
        """Compute convex hull area of COM path so far."""
        if frame_idx < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull
            pts = self.full_xz[:frame_idx + 1]
            if len(np.unique(pts, axis=0)) < 3:
                return 0.0
            hull = ConvexHull(pts)
            return float(hull.volume)  # 2D hull "volume" = area
        except:
            return 0.0

    def draw(self, frame_idx, video_frame):
        canvas = Image.new("RGB", (W, H), BG)
        t = frame_idx / FPS

        # Update heatmap
        self._update_heatmap(frame_idx)

        # Full-screen video
        vf = video_frame.resize((W, H), Image.LANCZOS)
        canvas.paste(vf, (0, 0))

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        # ── Mini map (top-down, bottom-right) ───────────────────
        map_size = 280
        map_x = W - map_size - 20
        map_y = H - map_size - 20

        # Map background
        od.rounded_rectangle([map_x - 5, map_y - 30, map_x + map_size + 5, map_y + map_size + 5],
                            radius=10, fill=(0, 0, 0, 200))
        od.text((map_x, map_y - 25), "STAGE (top-down)", fill=TXT_DIM, font=self.f_sm)

        # Draw heatmap on mini map
        if self.heatmap.max() > 0:
            hm_norm = self.heatmap / self.heatmap.max()
            hm_img = Image.new("RGBA", (self.heatmap_size, self.heatmap_size), (0, 0, 0, 0))
            for hy in range(self.heatmap_size):
                for hx in range(self.heatmap_size):
                    v = hm_norm[hy, hx]
                    if v > 0.05:
                        r = int(50 + 200 * v)
                        g = int(50 + 100 * (1 - v))
                        b = int(200 * (1 - v))
                        a = int(80 * min(v * 3, 1.0))
                        hm_img.putpixel((hx, hy), (r, g, b, a))
            hm_resized = hm_img.resize((map_size, map_size), Image.NEAREST)
            overlay.paste(Image.alpha_composite(
                overlay.crop((map_x, map_y, map_x + map_size, map_y + map_size)),
                hm_resized), (map_x, map_y))
            od = ImageDraw.Draw(overlay)  # Refresh draw context

        # Draw 3 COM trails (last 90 frames = 3 seconds)
        trail_len = min(90, frame_idx)
        trails = [
            (self.full_xz, C_COM_FULL, "BODY"),
            (self.upper_xz, C_COM_UPPER, "UPPER"),
            (self.lower_xz, C_COM_LOWER, "LOWER"),
        ]

        for xz_data, color, name in trails:
            start = max(0, frame_idx - trail_len)
            for i in range(start, frame_idx):
                px, py = self._map_to_minimap(xz_data[i], map_x, map_y, map_size)
                alpha = int(80 + 175 * (i - start) / max(1, trail_len))
                r, g, b = color
                od.ellipse([px - 2, py - 2, px + 2, py + 2],
                          fill=(r, g, b, min(alpha, 255)))

            # Current position (larger dot)
            if frame_idx < len(xz_data):
                px, py = self._map_to_minimap(xz_data[frame_idx], map_x, map_y, map_size)
                od.ellipse([px - 6, py - 6, px + 6, py + 6],
                          fill=(*color, 255), outline=(255, 255, 255, 200))

        # ── Coverage counter (top-right) ────────────────────────
        coverage = self._compute_coverage(frame_idx)
        od.rounded_rectangle([W - 320, 15, W - 15, 90], radius=10, fill=(0, 0, 0, 200))
        od.text((W - 310, 20), "STAGE COVERAGE", fill=TXT_DIM, font=self.f_sm)
        od.text((W - 310, 40), f"{coverage:.2f} m\u00b2", fill=TXT, font=self.f_big)
        od.text((W - 310, 70), f"Vert: {self.com_full[frame_idx, 1]:.2f}m", fill=TXT_DIM, font=self.f_sm)

        # ── Move type label ─────────────────────────────────────
        seg = self._current_segment(t)
        if seg:
            c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
            label = seg["dance_type"].upper()
            tw = self.f_big.getlength(label)
            od.rounded_rectangle([20, H - 60, 20 + tw + 24, H - 25],
                                radius=8, fill=(*c, 200))
            od.text((32, H - 55), label, fill=TXT, font=self.f_big)

        # Timer
        od.rounded_rectangle([20, 15, 200, 50], radius=8, fill=(0, 0, 0, 180))
        od.text((30, 20), f"{t:.1f}s / {self.total_dur:.1f}s", fill=TXT, font=self.f_med)

        # ── COM legend ──────────────────────────────────────────
        ly = H - 60
        lx = 250
        for name, color in [("BODY", C_COM_FULL), ("UPPER", C_COM_UPPER), ("LOWER", C_COM_LOWER)]:
            od.ellipse([lx, ly + 5, lx + 12, ly + 17], fill=(*color, 255))
            od.text((lx + 16, ly + 2), name, fill=(*color, 255), font=self.f_sm)
            lx += 80

        # Composite overlay onto canvas
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        return canvas


def render(joints_path=None, metrics_path=None, mesh_video_path=None, audio_path=None, output_path=None):
    output = output_path or str(BASE / "exports/spatial/gvhmr.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    joints, metrics, segments = load_all(joints_path, metrics_path)
    n_frames = joints.shape[0]

    renderer = SpatialRenderer(joints, metrics, segments)

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
    parser.add_argument("--joints", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--mesh-video", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    render(joints_path=args.joints, metrics_path=args.metrics,
           mesh_video_path=args.mesh_video, audio_path=args.audio,
           output_path=args.output)
