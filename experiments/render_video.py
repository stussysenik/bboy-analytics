"""
Render battle analytics overlay onto BRACE video.

Composites beat pulses, musicality badge, move type labels,
energy timeline, and segment markers onto actual battle footage.
Outputs shareable MP4 with audio.

Usage:
    python experiments/render_video.py
    # Defaults: RS0mFARO1x4 seq.4 (lil g, 125 BPM)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Constants ───────────────────────────────────────────────────────────────

W, H = 1920, 1080
FPS = 29.97
FRAME_BYTES = W * H * 3

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Colors (RGBA)
C_HUD_BG = (0, 0, 0, 153)          # 60% black
C_TEXT = (255, 255, 255, 230)
C_TEXT_DIM = (180, 180, 180, 200)
C_BEAT = (255, 220, 50, 255)       # Yellow flash
C_TOPROCK = (33, 150, 243, 220)    # Blue
C_FOOTWORK = (76, 175, 80, 220)    # Green
C_POWERMOVE = (255, 87, 34, 220)   # Red-orange
C_PLAYHEAD = (255, 255, 255, 255)
C_ENERGY_LO = (50, 130, 200)
C_ENERGY_HI = (255, 80, 40)

SEGMENT_COLORS = {
    "toprock": C_TOPROCK,
    "footwork": C_FOOTWORK,
    "powermove": C_POWERMOVE,
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_beats(beats_path: str, video_id: str, seq_idx: int) -> tuple[list[float], float]:
    """Load BRACE beat times. Returns (beat_times_sec, bpm)."""
    with open(beats_path) as f:
        data = json.load(f)
    key = f"{video_id}.{seq_idx}"
    entry = data[key]
    return entry["beats_sec"], entry["bpm"]


def load_segments(csv_path: str, video_id: str, seq_idx: int, fps: float,
                  clip_start_frame: int) -> list[dict]:
    """Load segments and convert to local time (relative to clip start)."""
    segments = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            if row["video_id"] == video_id and int(row["seq_idx"]) == seq_idx:
                start_f = int(row["start_frame"]) - clip_start_frame
                end_f = int(row["end_frame"]) - clip_start_frame
                segments.append({
                    "start_s": max(0, start_f / fps),
                    "end_s": end_f / fps,
                    "dance_type": row["dance_type"],
                    "dancer": row.get("dancer", ""),
                })
    return segments


def load_energy(metrics_path: str) -> np.ndarray:
    """Load and normalize energy curve from metrics.json."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    e = np.array(metrics["energy"]["energy_curve"])
    e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
    return e_norm


# ── Pre-rendered overlay elements ───────────────────────────────────────────

def prerender_badge(mu: float, bpm: float, interpretation: str) -> Image.Image:
    """Static musicality badge (top-right)."""
    bw, bh = 300, 85
    badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(badge)

    # Background
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=12, fill=(0, 0, 0, 180))

    font_big = ImageFont.truetype(FONT_BOLD, 28)
    font_sm = ImageFont.truetype(FONT_REG, 16)

    d.text((15, 12), f"\u03bc = {mu:.3f}", fill=C_TEXT, font=font_big)
    d.text((15, 50), f"{interpretation}  |  {bpm:.0f} BPM", fill=C_TEXT_DIM, font=font_sm)

    return badge


def prerender_energy_bar(energy: np.ndarray, segments: list[dict],
                         total_dur: float, bar_w: int = 1840, bar_h: int = 35) -> Image.Image:
    """Pre-render energy timeline + segment strip."""
    img = Image.new("RGBA", (bar_w, bar_h + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Resample energy to bar width
    x_idx = np.linspace(0, len(energy) - 1, bar_w).astype(int)

    for x in range(bar_w):
        val = float(energy[x_idx[x]])
        h = max(1, int(val * bar_h))
        # Gradient: blue → orange → red
        r = int(C_ENERGY_LO[0] + (C_ENERGY_HI[0] - C_ENERGY_LO[0]) * val)
        g = int(C_ENERGY_LO[1] + (C_ENERGY_HI[1] - C_ENERGY_LO[1]) * val)
        b = int(C_ENERGY_LO[2] + (C_ENERGY_HI[2] - C_ENERGY_LO[2]) * val)
        d.line([(x, bar_h - h), (x, bar_h)], fill=(r, g, b, 180))

    # Segment strip below energy
    for seg in segments:
        x_start = int(seg["start_s"] / total_dur * bar_w)
        x_end = int(seg["end_s"] / total_dur * bar_w)
        color = SEGMENT_COLORS.get(seg["dance_type"], (128, 128, 128, 200))
        d.rectangle([x_start, bar_h + 3, x_end, bar_h + 13], fill=color)

    return img


# ── Per-frame overlay drawing ───────────────────────────────────────────────

class OverlayRenderer:
    def __init__(self, beats: list[float], segments: list[dict],
                 energy: np.ndarray, mu: float, bpm: float,
                 interpretation: str, total_dur: float):
        self.beats = np.array(beats)
        self.segments = segments
        self.energy = energy
        self.mu = mu
        self.bpm = bpm
        self.total_dur = total_dur

        # Pre-render static elements
        self.badge = prerender_badge(mu, bpm, interpretation)
        self.energy_bar = prerender_energy_bar(energy, segments, total_dur)
        self.bar_w = self.energy_bar.width

        # Fonts
        self.font_label = ImageFont.truetype(FONT_BOLD, 24)
        self.font_timer = ImageFont.truetype(FONT_BOLD, 20)
        self.font_move = ImageFont.truetype(FONT_BOLD, 22)

        # Build frame → segment lookup
        self._seg_cache = {}

    def _current_segment(self, t: float) -> dict | None:
        for seg in self.segments:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return None

    def draw(self, frame_idx: int) -> Image.Image:
        """Draw all overlay elements for one frame. Returns RGBA image."""
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        t = frame_idx / FPS

        # ── Beat pulse (top-left) ───────────────────────────
        if len(self.beats) > 0:
            dists = np.abs(self.beats - t)
            nearest = float(np.min(dists))
            fade_dur = 0.18
            if nearest < fade_dur:
                alpha = int(255 * (1.0 - nearest / fade_dur))
                radius = int(22 + 10 * (1.0 - nearest / fade_dur))
                pulse_color = (255, 220, 50, alpha)
                d.ellipse([50 - radius, 45 - radius, 50 + radius, 45 + radius],
                          fill=pulse_color)
                # Outer glow
                if alpha > 100:
                    glow_r = radius + 8
                    d.ellipse([50 - glow_r, 45 - glow_r, 50 + glow_r, 45 + glow_r],
                              outline=(255, 220, 50, alpha // 3), width=2)

        # ── Musicality badge (top-right) ────────────────────
        overlay.paste(self.badge, (W - 320, 18), self.badge)

        # ── Move type label (left, mid-height) ──────────────
        seg = self._current_segment(t)
        if seg:
            dtype = seg["dance_type"]
            color = SEGMENT_COLORS.get(dtype, (128, 128, 128, 200))
            label = dtype.upper()
            tw = self.font_move.getlength(label)
            lw, lh = int(tw + 30), 38
            lx, ly = 30, H // 2 - 19
            d.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=8, fill=color)
            d.text((lx + 15, ly + 7), label, fill=C_TEXT, font=self.font_move)

        # ── Bottom HUD ──────────────────────────────────────
        hud_h = 70
        hud_y = H - hud_h
        d.rectangle([0, hud_y, W, H], fill=C_HUD_BG)

        # Energy bar
        bar_x = 40
        bar_y = hud_y + 10
        overlay.paste(self.energy_bar, (bar_x, bar_y), self.energy_bar)

        # Playhead
        progress = min(1.0, t / self.total_dur)
        ph_x = bar_x + int(progress * self.bar_w)
        d.line([(ph_x, bar_y - 2), (ph_x, bar_y + 50)], fill=C_PLAYHEAD, width=2)

        # Timer
        timer_txt = f"{t:.1f}s / {self.total_dur:.1f}s"
        d.text((W - 180, hud_y + 25), timer_txt, fill=C_TEXT, font=self.font_timer)

        return overlay


# ── Main rendering pipeline ─────────────────────────────────────────────────

def render_overlay_video(
    video_path: str = "data/brace/videos/RS0mFARO1x4.mp4",
    metrics_path: str = "experiments/results/EXP-002_toprock_on-beat_lil_g/metrics.json",
    beats_path: str = "data/brace/annotations/audio_beats.json",
    segments_path: str = "data/brace/annotations/segments.csv",
    video_id: str = "RS0mFARO1x4",
    seq_idx: int = 4,
    clip_start_frame: int = 3802,
    clip_end_frame: int = 4801,
    output_path: str = "experiments/results/overlay_RS0mFARO1x4_seq4.mp4",
) -> str:
    """Render analytics overlay onto BRACE video clip. Returns output path."""

    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_frames = clip_end_frame - clip_start_frame
    start_time = clip_start_frame / FPS
    duration = n_frames / FPS

    print(f"  Video: {video_path}")
    print(f"  Clip: frames {clip_start_frame}-{clip_end_frame} ({duration:.1f}s)")
    print(f"  Output: {output_path}")

    # ── Load data ───────────────────────────────────────────
    print("  Loading data...")
    beats, bpm = load_beats(beats_path, video_id, seq_idx)
    segments = load_segments(segments_path, video_id, seq_idx, FPS, clip_start_frame)
    energy = load_energy(metrics_path)

    # Load musicality score
    with open(metrics_path) as f:
        metrics = json.load(f)
    mu = metrics["musicality"]["mu"]
    interp = metrics["musicality"]["interpretation"]["musicality"]

    print(f"  Beats: {len(beats)} at {bpm:.0f} BPM")
    print(f"  Segments: {len(segments)}")
    print(f"  \u03bc = {mu:.3f} ({interp})")

    # ── Extract audio clip ──────────────────────────────────
    print("  Extracting audio...")
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_path = audio_tmp.name
    audio_tmp.close()

    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
        "-t", str(duration), "-vn", "-c:a", "aac", "-b:a", "192k",
        audio_path,
    ], capture_output=True)

    # ── Create renderer ─────────────────────────────────────
    renderer = OverlayRenderer(
        beats=beats, segments=segments, energy=energy,
        mu=mu, bpm=bpm, interpretation=interp, total_dur=duration,
    )

    # ── Open FFmpeg pipes ───────────────────────────────────
    print("  Opening FFmpeg pipes...")

    # Reader: decode video clip to raw RGB
    read_cmd = [
        "ffmpeg", "-ss", str(start_time), "-i", video_path,
        "-t", str(duration),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error", "pipe:1",
    ]
    read_proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 2)

    # Writer: encode composited frames to H.264 + mux audio
    write_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-v", "error",
        output_path,
    ]
    write_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=FRAME_BYTES * 2)

    # ── Frame-by-frame compositing ──────────────────────────
    print(f"  Rendering {n_frames} frames...")
    t0 = time.time()
    frame_idx = 0

    while True:
        raw = read_proc.stdout.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            break

        # Decode frame
        frame = Image.frombytes("RGB", (W, H), raw)

        # Draw overlay
        overlay = renderer.draw(frame_idx)

        # Composite
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")

        # Write
        write_proc.stdin.write(frame.tobytes())

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps_actual = frame_idx / elapsed
            eta = (n_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"    {frame_idx}/{n_frames} ({frame_idx/n_frames*100:.0f}%) "
                  f"— {fps_actual:.1f} fps, ETA {eta:.0f}s")

    # ── Cleanup ─────────────────────────────────────────────
    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()

    elapsed = time.time() - t0
    print(f"\n  Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"  Output: {output_path}")

    # Cleanup temp audio
    try:
        Path(audio_path).unlink()
    except Exception:
        pass

    return output_path


if __name__ == "__main__":
    render_overlay_video()
