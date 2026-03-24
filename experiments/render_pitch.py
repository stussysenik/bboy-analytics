"""
Pitch Renderer: the insight-first split-screen visualization.

Layout:
  ┌────────────────────────┬────────────────────────┐
  │                        │                        │
  │   ORIGINAL VIDEO       │  SKELETON + BEATS      │
  │   (960 × 840)          │  over original video   │
  │                        │  velocity-colored       │
  │                        │  beat pulses            │
  │                        │  contact light pools    │
  │                        │                        │
  ├────────────────────────┴────────────────────────┤
  │  MOVE BAR (toprock → footwork → powermove)      │  40px
  ├─────────────────────────────────────────────────┤
  │  MUSICALITY RIBBON (green ↔ red) + BEAT SCORE   │  40px
  │              N/M ON BEAT (XX%)                   │
  ├─────────────────────────────────────────────────┤
  │  ENERGY STRIP                                    │  30px
  └─────────────────────────────────────────────────┘

Usage:
    python experiments/render_pitch.py [--output path]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from world_state import compute_world_state, print_summary
from components.panel import FontCache, BG, TXT, TXT_DIM
from components.skeleton_overlay import SkeletonOverlay
from components.contact_light import ContactLight
from components.musicality_ribbon import MusalityRibbon
from components.move_bar import MoveBar
from components.scalar_strip import ScalarStrip
from components.energy_flow import EnergyPanel

W, H = 1920, 1080
VID_H = 840      # video area height
HALF_W = 960     # half width for split screen
MOVE_H = 40      # move bar
RIBBON_H = 40    # musicality ribbon
ENERGY_H = 30    # energy strip (fills remaining)
# Total bottom = 40 + 40 + 30 = 110, leaving 1080 - 840 - 110 = 90px... adjusted:
BOTTOM_START = VID_H  # = 840
# VID_H + MOVE_H + RIBBON_H + ENERGY_H should = H
# 840 + 40 + 40 + 30 = 950... we have 130px extra
# Let's give more to MOVE and ENERGY:
MOVE_H = 50
RIBBON_H = 50
ENERGY_H = H - VID_H - MOVE_H - RIBBON_H  # = 1080 - 840 - 50 - 50 = 140
FPS = 29.97


def load_segments(video_id: str = "RS0mFARO1x4", seq_idx: int = 4, fps: float = 29.97):
    """Load BRACE segment annotations for this sequence."""
    segments = []
    seg_file = Path(__file__).parent.parent / "data/brace/annotations/segments.csv"
    if not seg_file.exists():
        return segments

    seq_start_frame = 3802  # seq4 start in full video

    with open(seg_file) as f:
        for row in csv.DictReader(f):
            if row["video_id"] == video_id and int(row["seq_idx"]) == seq_idx:
                segments.append({
                    "start_s": (int(row["start_frame"]) - seq_start_frame) / fps,
                    "end_s": (int(row["end_frame"]) - seq_start_frame) / fps,
                    "dance_type": row["dance_type"],
                })
    return segments


def load_beats(video_id: str = "RS0mFARO1x4", seq_idx: int = 4):
    """Load beat times from BRACE annotations."""
    beats_file = Path(__file__).parent.parent / "data/brace/annotations/audio_beats.json"
    if not beats_file.exists():
        return None

    with open(beats_file) as f:
        data = json.load(f)

    key = f"{video_id}.{seq_idx}"
    if key in data:
        return np.array(data[key]["beats_sec"])
    return None


def extract_seq_from_video(video_path: str, seq_start: int, seq_end: int, fps: float):
    """Extract sequence frames from original video, return as temp file path."""
    start_s = seq_start / fps
    duration_s = (seq_end - seq_start) / fps

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path,
        "-t", str(duration_s), "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "18", "-an", "-v", "error", tmp.name
    ], capture_output=True)
    return tmp.name


def render_pitch(
    joints_path: str = "experiments/results/joints_3d_REAL_seq4.npy",
    vitpose_path: str = "experiments/results/vitpose_2d_seq4.npy",
    original_video: str = "data/brace/videos/RS0mFARO1x4.mp4",
    audio_source: str | None = None,
    output_path: str = "experiments/results/pitch_v1.mp4",
    seq_start: int = 3802,
    seq_end: int = 4801,
):
    base = Path(__file__).parent.parent

    # Resolve paths
    def resolve(p):
        return str(base / p) if not Path(p).is_absolute() else p

    joints_path = resolve(joints_path)
    vitpose_path = resolve(vitpose_path)
    original_video = resolve(original_video)
    output_path = resolve(output_path)

    # Load data
    print("Loading data...")
    joints = np.load(joints_path)
    vitpose = np.load(vitpose_path)
    segments = load_segments()
    beat_times = load_beats()

    print(f"  Joints: {joints.shape}")
    print(f"  Vitpose: {vitpose.shape}")
    print(f"  Segments: {len(segments)}")
    print(f"  Beats: {len(beat_times) if beat_times is not None else 'none'}")

    n_frames = joints.shape[0]

    # Compute world state
    print("Computing world state...")
    ws = compute_world_state(joints, FPS, beat_times, sg_window=15)
    print_summary(ws)

    # Extract original video sequence
    print(f"Extracting seq4 from original video ({seq_start}-{seq_end})...")
    seq_video = extract_seq_from_video(original_video, seq_start, seq_end, FPS)

    # Probe video
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", seq_video],
        capture_output=True, text=True)
    streams = json.loads(probe.stdout)["streams"]
    vs = [s for s in streams if s["codec_type"] == "video"][0]
    src_w, src_h = int(vs["width"]), int(vs["height"])
    src_bytes = src_w * src_h * 3
    print(f"  Source video: {src_w}x{src_h}")

    # Build panels
    print("Building panels...")
    skeleton = SkeletonOverlay(HALF_W, VID_H, ws, vitpose)
    contact = ContactLight(HALF_W, VID_H, ws, vitpose)
    ribbon = MusalityRibbon(W, RIBBON_H, ws)
    move_bar = MoveBar(W, MOVE_H, ws, segments)
    energy = EnergyPanel(W, ENERGY_H, ws)

    # Prerender
    ribbon.prerender()
    move_bar.prerender()
    energy.prerender()

    # Extract audio
    audio_tmp = None
    audio_args = []
    audio_src = audio_source or original_video
    if Path(resolve(audio_src)).exists():
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
        start_s = seq_start / FPS
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_s), "-i", resolve(audio_src),
            "-t", str(n_frames / FPS), "-vn", "-c:a", "aac", "-b:a", "192k",
            audio_tmp.name
        ], capture_output=True)
        audio_args = ["-i", audio_tmp.name, "-c:a", "aac", "-b:a", "192k", "-shortest"]

    # FFmpeg pipes
    read_proc = subprocess.Popen(
        ["ffmpeg", "-i", seq_video, "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=src_bytes * 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0", *audio_args,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-v", "error", output_path,
    ]
    write_proc = subprocess.Popen(
        write_cmd, stdin=subprocess.PIPE, bufsize=W * H * 3 * 2)

    # ── Render loop ──────────────────────────────────────────────
    print(f"Rendering {n_frames} frames → {output_path}")
    t0 = time.time()

    for fi in range(n_frames):
        raw = read_proc.stdout.read(src_bytes)
        if len(raw) < src_bytes:
            print(f"  Video ended at frame {fi}")
            break

        video_frame = Image.frombytes("RGB", (src_w, src_h), raw)

        # Build canvas
        canvas = Image.new("RGB", (W, H), BG)

        # Left side: original video, scaled to half
        left = video_frame.resize((HALF_W, VID_H), Image.LANCZOS)
        canvas.paste(left, (0, 0))

        # Right side: original video + skeleton overlay + contact light
        right = video_frame.resize((HALF_W, VID_H), Image.LANCZOS).convert("RGBA")

        # Draw skeleton
        skel_img = skeleton.draw(fi)
        right = Image.alpha_composite(right, skel_img)

        # Draw contact lights
        contact_img = contact.draw(fi)
        right = Image.alpha_composite(right, contact_img)

        canvas.paste(right.convert("RGB"), (HALF_W, 0))

        # Divider line
        d = ImageDraw.Draw(canvas)
        d.line([(HALF_W, 0), (HALF_W, VID_H)], fill=(80, 80, 90), width=2)

        # Left label
        d.text((15, 12), "ORIGINAL", fill=TXT_DIM, font=FontCache.get(bold=True, size=14))

        # Right label
        d.text((HALF_W + 15, 12), "ANALYSIS", fill=TXT_DIM, font=FontCache.get(bold=True, size=14))

        # Timer
        t = fi / FPS
        d.text((HALF_W - 90, 12), f"{t:.1f}s", fill=TXT, font=FontCache.get(bold=True, size=14))

        # Beat score badge (hero stat — top right)
        if ws.beat_hits:
            n_hits = sum(1 for b in ws.beat_hits if b["hit"])
            total = len(ws.beat_hits)
            pct = ws.beat_hit_pct
            badge_w = 220
            badge_h = 55
            badge_x = W - badge_w - 15
            badge_y = 15
            # Background
            badge = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
            bd = ImageDraw.Draw(badge)
            bd.rounded_rectangle([0, 0, badge_w - 1, badge_h - 1], radius=10,
                                 fill=(0, 0, 0, 180))
            # Text
            color = (120, 255, 120, 255) if pct > 60 else (255, 180, 80, 255)
            bd.text((12, 5), f"{n_hits}/{total} ON BEAT",
                    fill=color, font=FontCache.get(bold=True, size=20))
            bd.text((12, 30), f"{pct:.0f}% beat alignment",
                    fill=(200, 200, 210, 200), font=FontCache.get(size=14))

            region = canvas.crop((badge_x, badge_y, badge_x + badge_w, badge_y + badge_h)).convert("RGBA")
            composited = Image.alpha_composite(region, badge)
            canvas.paste(composited.convert("RGB"), (badge_x, badge_y))

        # Bottom panels
        move_img = move_bar.draw(fi)
        canvas.paste(move_img, (0, BOTTOM_START))

        ribbon_img = ribbon.draw(fi)
        canvas.paste(ribbon_img, (0, BOTTOM_START + MOVE_H))

        energy_img = energy.draw(fi)
        canvas.paste(energy_img, (0, BOTTOM_START + MOVE_H + RIBBON_H))

        # Write frame
        write_proc.stdin.write(canvas.tobytes())

        if fi % 100 == 0 and fi > 0:
            elapsed = time.time() - t0
            fps_actual = fi / elapsed
            eta = (n_frames - fi) / fps_actual if fps_actual > 0 else 0
            print(f"  {fi}/{n_frames} ({100*fi/n_frames:.0f}%) — {fps_actual:.1f} fps, ETA {eta:.0f}s")

    write_proc.stdin.close()
    write_proc.wait()
    read_proc.wait()

    # Cleanup
    Path(seq_video).unlink(missing_ok=True)
    if audio_tmp:
        Path(audio_tmp.name).unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"\nDone! {fi} frames in {elapsed:.1f}s ({fi/elapsed:.1f} fps)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch Renderer — insight-first split-screen")
    parser.add_argument("--joints", default="experiments/results/joints_3d_REAL_seq4.npy")
    parser.add_argument("--vitpose", default="experiments/results/vitpose_2d_seq4.npy")
    parser.add_argument("--video", default="data/brace/videos/RS0mFARO1x4.mp4")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", default="experiments/results/pitch_v1.mp4")
    parser.add_argument("--seq-start", type=int, default=3802)
    parser.add_argument("--seq-end", type=int, default=4801)
    args = parser.parse_args()

    render_pitch(
        joints_path=args.joints,
        vitpose_path=args.vitpose,
        original_video=args.video,
        audio_source=args.audio,
        output_path=args.output,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
    )
