"""
Combined analytics dashboard video renderer.

Layout (1920x1080):
  Top 756px:    GVHMR mesh overlay + beat pulse + move label + mu badge
  Mid 100px:    Audio waveform + beat markers + playhead
  Bot 224px:    Energy/Flow | COM/Balance map | Metrics panel

Usage:
    python experiments/render_combined.py
"""

from __future__ import annotations
import json, subprocess, sys, time, csv, tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ────────────────────────────────────────────────────────

W, H = 1920, 1080
VID_H = 756        # Main video area
WAVE_H = 100       # Audio waveform strip
PANEL_H = 224      # Bottom panels (H - VID_H - WAVE_H = 224)
PANEL_W = 640      # Each of 3 bottom panels
FPS = 29.97
FRAME_BYTES_SRC = None  # Set after reading source resolution

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Colors (RGBA)
BG = (18, 18, 24, 255)
HUD = (0, 0, 0, 180)
TXT = (255, 255, 255, 230)
TXT_DIM = (160, 160, 170, 200)
BEAT_CLR = (255, 220, 50)
C_TOPROCK = (33, 150, 243)
C_FOOTWORK = (76, 175, 80)
C_POWERMOVE = (255, 87, 34)
C_ENERGY_LO = (50, 130, 200)
C_ENERGY_HI = (255, 80, 40)
SEG_COLORS = {"toprock": C_TOPROCK, "footwork": C_FOOTWORK, "powermove": C_POWERMOVE}


def load_data():
    """Load all pre-computed data."""
    base = Path(__file__).parent

    joints = np.load(base / "results/joints_3d_CLEAN.npy")
    onset = np.load(base / "results/audio_onset.npy")
    waveform = np.load(base / "results/audio_waveform.npy")

    with open(base / "results/CLEAN_metrics.json") as f:
        metrics = json.load(f)

    with open(base.parent / "data/brace/annotations/audio_beats.json") as f:
        beats = np.array(json.load(f)["RS0mFARO1x4.4"]["beats_sec"])

    segments = []
    with open(base.parent / "data/brace/annotations/segments.csv") as f:
        for row in csv.DictReader(f):
            if row["video_id"] == "RS0mFARO1x4" and int(row["seq_idx"]) == 4:
                segments.append({
                    "start_s": (int(row["start_frame"]) - 3802) / FPS,
                    "end_s": (int(row["end_frame"]) - 3802) / FPS,
                    "dance_type": row["dance_type"],
                })

    return joints, onset, waveform, metrics, beats, segments


def prerender_waveform(onset, beats, total_dur, w=1880, h=80):
    """Pre-render audio waveform strip with beat markers."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Normalize onset
    on = onset / (onset.max() + 1e-8)
    x_idx = np.linspace(0, len(on) - 1, w).astype(int)

    # Draw waveform bars
    for x in range(w):
        val = float(on[x_idx[x]])
        bar_h = max(1, int(val * h * 0.8))
        mid = h // 2
        r = int(50 + 180 * val)
        g = int(130 - 80 * val)
        b = int(200 - 150 * val)
        d.line([(x, mid - bar_h//2), (x, mid + bar_h//2)], fill=(r, g, b, 160))

    # Beat markers
    for bt in beats:
        if 0 <= bt < total_dur:
            bx = int(bt / total_dur * w)
            d.line([(bx, 0), (bx, h)], fill=(*BEAT_CLR, 100), width=1)

    return img


def prerender_energy(energy_curve, segments, total_dur, w=600, h=180):
    """Pre-render energy curve for bottom panel."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    e = np.array(energy_curve)
    e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
    x_idx = np.linspace(0, len(e_norm) - 1, w).astype(int)

    for x in range(w):
        val = float(e_norm[x_idx[x]])
        bar_h = max(1, int(val * (h - 30)))
        r = int(C_ENERGY_LO[0] + (C_ENERGY_HI[0] - C_ENERGY_LO[0]) * val)
        g = int(C_ENERGY_LO[1] + (C_ENERGY_HI[1] - C_ENERGY_LO[1]) * val)
        b = int(C_ENERGY_LO[2] + (C_ENERGY_HI[2] - C_ENERGY_LO[2]) * val)
        d.line([(x, h - 25 - bar_h), (x, h - 25)], fill=(r, g, b, 180))

    # Segment strip
    for seg in segments:
        xs = int(seg["start_s"] / total_dur * w)
        xe = int(seg["end_s"] / total_dur * w)
        c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
        d.rectangle([xs, h - 20, xe, h - 8], fill=(*c, 200))

    return img


class DashboardRenderer:
    def __init__(self, joints, onset, waveform, metrics, beats, segments):
        self.joints = joints
        self.onset = onset
        self.metrics = metrics
        self.beats = beats
        self.segments = segments
        self.n_frames = joints.shape[0]
        self.total_dur = self.n_frames / FPS

        # Pre-compute energy from joints
        dt = 1.0 / FPS
        vel = np.diff(joints, axis=0) / dt
        self.speed = np.linalg.norm(vel, axis=-1)  # (F-1, 22)
        self.total_energy = self.speed.sum(axis=1)
        # Smooth energy for display
        from scipy.signal import savgol_filter
        w = min(31, len(self.total_energy) // 2 * 2 - 1)
        if w >= 5:
            self.energy_smooth = savgol_filter(self.total_energy, w, 3)
        else:
            self.energy_smooth = self.total_energy

        # COM trajectory (pelvis = joint 0, XZ plane)
        self.com_xz = joints[:, 0, [0, 2]]  # (F, 2)
        self.com_y = joints[:, 0, 1]         # (F,) height

        # Pre-render static elements
        self.waveform_img = prerender_waveform(onset, beats, self.total_dur)
        self.energy_img = prerender_energy(
            self.energy_smooth.tolist(), segments, self.total_dur)

        # Fonts
        self.f_title = ImageFont.truetype(FONT_B, 20)
        self.f_big = ImageFont.truetype(FONT_B, 28)
        self.f_med = ImageFont.truetype(FONT_B, 18)
        self.f_sm = ImageFont.truetype(FONT_R, 14)
        self.f_label = ImageFont.truetype(FONT_B, 22)

    def _current_segment(self, t):
        for seg in self.segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return None

    def draw(self, frame_idx, video_frame):
        """Composite full 1920x1080 dashboard for one frame."""
        canvas = Image.new("RGB", (W, H), BG[:3])
        t = frame_idx / FPS

        # ── Top: Video (scale to 1920x756) ──────────────────────
        vf = video_frame.resize((W, VID_H), Image.LANCZOS)
        canvas.paste(vf, (0, 0))

        # Beat pulse border
        if len(self.beats) > 0:
            nearest = float(np.min(np.abs(self.beats - t)))
            if nearest < 0.15:
                alpha = int(200 * (1.0 - nearest / 0.15))
                border = Image.new("RGBA", (W, VID_H), (0, 0, 0, 0))
                bd = ImageDraw.Draw(border)
                bd.rectangle([0, 0, W-1, VID_H-1], outline=(*BEAT_CLR, alpha), width=4)
                canvas.paste(Image.alpha_composite(
                    canvas.crop((0, 0, W, VID_H)).convert("RGBA"), border
                ).convert("RGB"), (0, 0))

        # Move type label
        seg = self._current_segment(t)
        if seg:
            c = SEG_COLORS.get(seg["dance_type"], (128, 128, 128))
            overlay = Image.new("RGBA", (W, VID_H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            label = seg["dance_type"].upper()
            tw = self.f_label.getlength(label)
            lx, ly = 30, VID_H - 60
            od.rounded_rectangle([lx, ly, lx + tw + 24, ly + 36], radius=6, fill=(*c, 200))
            od.text((lx + 12, ly + 7), label, fill=TXT, font=self.f_label)
            canvas.paste(Image.alpha_composite(
                canvas.crop((0, 0, W, VID_H)).convert("RGBA"), overlay
            ).convert("RGB"), (0, 0))

        # Mu badge (top-right)
        mu = self.metrics["musicality"]["mu"]
        badge = Image.new("RGBA", (220, 65), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        bd.rounded_rectangle([0, 0, 219, 64], radius=10, fill=(0, 0, 0, 180))
        bd.text((12, 8), f"\u03bc = {mu:.3f}", fill=TXT, font=self.f_big)
        bd.text((12, 40), f"125 BPM | GVHMR", fill=TXT_DIM, font=self.f_sm)
        canvas.paste(Image.alpha_composite(
            canvas.crop((W-235, 15, W-15, 80)).convert("RGBA"), badge
        ).convert("RGB"), (W-235, 15))

        # Timer
        timer_ov = Image.new("RGBA", (160, 30), (0, 0, 0, 0))
        td = ImageDraw.Draw(timer_ov)
        td.rounded_rectangle([0, 0, 159, 29], radius=6, fill=(0, 0, 0, 160))
        td.text((10, 5), f"{t:.1f}s / {self.total_dur:.1f}s", fill=TXT, font=self.f_med)
        canvas.paste(Image.alpha_composite(
            canvas.crop((W-175, VID_H-45, W-15, VID_H-15)).convert("RGBA"), timer_ov
        ).convert("RGB"), (W-175, VID_H-45))

        # ── Middle: Audio waveform ──────────────────────────────
        wave_y = VID_H
        wave_bg = Image.new("RGB", (W, WAVE_H), (12, 12, 18))
        canvas.paste(wave_bg, (0, wave_y))

        # Paste pre-rendered waveform
        canvas.paste(Image.alpha_composite(
            canvas.crop((20, wave_y+10, 20+self.waveform_img.width, wave_y+10+self.waveform_img.height)).convert("RGBA"),
            self.waveform_img
        ).convert("RGB"), (20, wave_y + 10))

        # Playhead
        progress = min(1.0, t / self.total_dur)
        ph_x = 20 + int(progress * self.waveform_img.width)
        draw = ImageDraw.Draw(canvas)
        draw.line([(ph_x, wave_y + 5), (ph_x, wave_y + WAVE_H - 5)], fill="white", width=2)

        # Label
        draw.text((20, wave_y + 2), "\u266a AUDIO + BEATS", fill=TXT_DIM[:3], font=self.f_sm)

        # ── Bottom panels ───────────────────────────────────────
        panel_y = VID_H + WAVE_H

        # Panel backgrounds
        for i in range(3):
            px = i * PANEL_W
            draw.rectangle([px, panel_y, px + PANEL_W - 1, H - 1], fill=(22, 22, 30))
            if i > 0:
                draw.line([(px, panel_y), (px, H)], fill=(60, 60, 70), width=1)

        # ── Panel 1: Energy + Flow ──────────────────────────────
        p1x, p1y = 10, panel_y + 5
        draw.text((p1x, p1y), "ENERGY + FLOW", fill=TXT_DIM[:3], font=self.f_sm)

        # Paste energy chart
        ey = p1y + 20
        canvas.paste(Image.alpha_composite(
            canvas.crop((p1x, ey, p1x + self.energy_img.width, ey + self.energy_img.height)).convert("RGBA"),
            self.energy_img
        ).convert("RGB"), (p1x, ey))

        # Energy playhead
        eph_x = p1x + int(progress * self.energy_img.width)
        draw.line([(eph_x, ey), (eph_x, ey + self.energy_img.height)], fill="white", width=2)

        # Current energy value
        if frame_idx < len(self.total_energy):
            cur_e = self.total_energy[frame_idx]
            draw.text((p1x + self.energy_img.width - 80, p1y),
                      f"E={cur_e:.0f}", fill=TXT[:3], font=self.f_med)

        # ── Panel 2: COM / Balance ──────────────────────────────
        p2x, p2y = PANEL_W + 10, panel_y + 5
        draw.text((p2x, p2y), "CENTER OF MASS", fill=TXT_DIM[:3], font=self.f_sm)

        # Mini COM map (top-down XZ)
        map_size = 190
        map_x, map_y = p2x + 10, p2y + 25
        draw.rectangle([map_x, map_y, map_x + map_size, map_y + map_size],
                       fill=(30, 30, 40), outline=(60, 60, 70))

        # Scale COM to map coordinates
        com = self.com_xz[:frame_idx + 1]
        if len(com) > 1:
            xmin, xmax = self.com_xz[:, 0].min(), self.com_xz[:, 0].max()
            zmin, zmax = self.com_xz[:, 1].min(), self.com_xz[:, 1].max()
            xr = max(xmax - xmin, 0.1)
            zr = max(zmax - zmin, 0.1)
            margin = 10

            # Draw trail (last 2 seconds = 60 frames)
            trail_start = max(0, frame_idx - 60)
            for i in range(trail_start, frame_idx):
                cx = map_x + margin + int((com[i, 0] - xmin) / xr * (map_size - 2 * margin))
                cz = map_y + margin + int((com[i, 1] - zmin) / zr * (map_size - 2 * margin))
                alpha_frac = (i - trail_start) / max(1, frame_idx - trail_start)
                r = int(50 + 200 * alpha_frac)
                draw.ellipse([cx-2, cz-2, cx+2, cz+2], fill=(r, 200, 255))

            # Current position
            cx = map_x + margin + int((com[frame_idx, 0] - xmin) / xr * (map_size - 2 * margin))
            cz = map_y + margin + int((com[frame_idx, 1] - zmin) / zr * (map_size - 2 * margin))
            draw.ellipse([cx-5, cz-5, cx+5, cz+5], fill=(255, 100, 100))

        # Height indicator
        hy = p2y + 25
        hx = p2x + map_size + 40
        draw.text((hx, hy), "HEIGHT", fill=TXT_DIM[:3], font=self.f_sm)
        if frame_idx < len(self.com_y):
            h_val = self.com_y[frame_idx]
            h_min, h_max = self.com_y.min(), self.com_y.max()
            h_bar_h = 160
            h_frac = (h_val - h_min) / max(h_max - h_min, 0.01)
            bar_fill = int(h_frac * h_bar_h)
            draw.rectangle([hx, hy + 18, hx + 20, hy + 18 + h_bar_h],
                          outline=(60, 60, 70))
            draw.rectangle([hx, hy + 18 + h_bar_h - bar_fill, hx + 20, hy + 18 + h_bar_h],
                          fill=(100, 200, 255))
            draw.text((hx + 25, hy + 18 + h_bar_h - bar_fill - 8),
                      f"{h_val:.2f}m", fill=TXT[:3], font=self.f_sm)

        # ── Panel 3: Metrics ────────────────────────────────────
        p3x, p3y = 2 * PANEL_W + 15, panel_y + 5
        draw.text((p3x, p3y), "METRICS", fill=TXT_DIM[:3], font=self.f_sm)

        mu = self.metrics["musicality"]["mu"]
        flow = self.metrics["flow"]["flow_score"]
        coverage = self.metrics["space"]["stage_coverage_m2"]
        vert = self.metrics["space"]["vertical_range_m"]
        freezes = self.metrics["complexity"]["freeze_count"]
        inversions = self.metrics["complexity"]["inversion_count"]

        my = p3y + 25
        metrics_lines = [
            (f"\u03bc = {mu:.3f}", self.f_big, TXT[:3]),
            (f"Musicality: {self.metrics['musicality']['interpretation']['musicality']}", self.f_med, TXT_DIM[:3]),
            ("", self.f_sm, TXT_DIM[:3]),
            (f"Flow Score: {flow:.1f}", self.f_med, TXT[:3]),
            (f"Coverage: {coverage:.2f} m\u00b2", self.f_med, TXT[:3]),
            (f"Vert Range: {vert:.2f} m", self.f_med, TXT[:3]),
            (f"Freezes: {freezes}  |  Inversions: {inversions}", self.f_sm, TXT_DIM[:3]),
        ]
        for text, font, color in metrics_lines:
            if text:
                draw.text((p3x, my), text, fill=color, font=font)
            my += font.size + 6

        return canvas


def render():
    base = Path(__file__).parent
    mesh_video = str(base / "results/gvhmr_mesh_clean_seq4.mp4")
    output = str(base / "results/combined_analytics.mp4")

    print("Loading data...")
    joints, onset, waveform, metrics, beats, segments = load_data()
    n_frames = joints.shape[0]
    total_dur = n_frames / FPS

    renderer = DashboardRenderer(joints, onset, waveform, metrics, beats, segments)

    # Get source video resolution
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", mesh_video],
        capture_output=True, text=True)
    streams = json.loads(probe.stdout)["streams"]
    vs = [s for s in streams if s["codec_type"] == "video"][0]
    src_w, src_h = int(vs["width"]), int(vs["height"])
    print(f"Source: {src_w}x{src_h}, {n_frames} frames, {total_dur:.1f}s")

    # Extract audio from ORIGINAL clip (mesh video has no audio)
    print("Extracting audio from original clip...")
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    original_clip = "/tmp/lilg_round.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", original_clip, "-vn", "-c:a", "aac",
                    "-b:a", "192k", audio_tmp.name], capture_output=True)

    # FFmpeg pipes
    print("Opening FFmpeg pipes...")
    src_frame_bytes = src_w * src_h * 3

    read_cmd = ["ffmpeg", "-i", mesh_video, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-v", "error", "pipe:1"]
    read_proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=src_frame_bytes * 2)

    out_frame_bytes = W * H * 3
    write_cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-r", str(FPS),
                 "-i", "pipe:0", "-i", audio_tmp.name,
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                 "-shortest", "-v", "error", output]
    write_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=out_frame_bytes * 2)

    # Render loop
    print(f"Rendering {n_frames} frames...")
    t0 = time.time()
    frame_idx = 0

    while True:
        raw = read_proc.stdout.read(src_frame_bytes)
        if len(raw) < src_frame_bytes:
            break

        video_frame = Image.frombytes("RGB", (src_w, src_h), raw)
        canvas = renderer.draw(frame_idx, video_frame)
        write_proc.stdin.write(canvas.tobytes())

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps_actual = frame_idx / elapsed
            eta = (n_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"  {frame_idx}/{n_frames} ({100*frame_idx/n_frames:.0f}%) "
                  f"— {fps_actual:.1f} fps, ETA {eta:.0f}s")

    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()

    elapsed = time.time() - t0
    print(f"\nDone! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Output: {output}")

    Path(audio_tmp.name).unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    render()
