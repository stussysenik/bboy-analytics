#!/usr/bin/env python3
"""Olympic Observatory — unified breaking analysis renderer.

Single entry point: loads joints_3d.npy + source video + audio,
computes all typed state, renders split-screen observatory, muxes with audio.

Usage:
    python -m observatory.render \
        --joints path/to/joints_3d.npy \
        --video path/to/source_video.mp4 \
        --audio path/to/audio.wav \
        --beats path/to/beats.json \
        --output observatory_output.mp4 \
        --bpm 125
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .body_state import BodyState, TemporalWindow, compute_body_states
from .color_system import BG_COLOR, PANEL_BG, TRAIL_LENGTH
from .header import HEADER_HEIGHT, render_header
from .skeleton_panel import render_skeleton_panel
from .timeline_strip import render_timeline_strip
from .video_panel import render_video_panel

# ──────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────

OUTPUT_W = 1920
OUTPUT_H = 1080
PANEL_W = OUTPUT_W // 2  # 960
TIMELINE_H = 220
PANEL_H = OUTPUT_H - HEADER_HEIGHT - TIMELINE_H  # 820
FPS = 29.97


def load_beats(beats_path: Optional[str]) -> Optional[np.ndarray]:
    """Load beat timestamps from JSON file."""
    if beats_path is None:
        return None
    path = Path(beats_path)
    if not path.exists():
        print(f"Warning: beats file not found: {beats_path}")
        return None
    with open(path) as f:
        data = json.load(f)
    # Support various formats
    if isinstance(data, list):
        return np.array(data, dtype=np.float64)
    if isinstance(data, dict):
        # Try common keys
        for key in ["beats_sec", "beats", "timestamps"]:
            if key in data:
                return np.array(data[key], dtype=np.float64)
        # Try nested (e.g., {"RS0mFARO1x4.4": {"beats_sec": [...]}})
        for v in data.values():
            if isinstance(v, dict) and "beats_sec" in v:
                return np.array(v["beats_sec"], dtype=np.float64)
    print(f"Warning: could not parse beats from {beats_path}")
    return None


def read_video_frames(video_path: str, target_frames: int) -> np.ndarray:
    """Read video frames via ffmpeg as (T, H, W, 3) uint8 array."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    vs = [s for s in streams if s.get("codec_type") == "video"]
    if not vs:
        raise RuntimeError(f"No video stream found in {video_path}")

    src_w = int(vs[0]["width"])
    src_h = int(vs[0]["height"])

    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.wait()

    frame_bytes = src_w * src_h * 3
    n_frames = len(raw) // frame_bytes
    n_frames = min(n_frames, target_frames)

    frames = np.frombuffer(raw[:n_frames * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(n_frames, src_h, src_w, 3)
    return frames


def render_observatory(
    joints_path: str,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    beats_path: Optional[str] = None,
    output_path: str = "observatory_output.mp4",
    bpm: float = 125.0,
):
    """Main rendering pipeline."""
    print("Loading joints...")
    joints = np.load(joints_path)
    T = joints.shape[0]
    total_dur = T / FPS

    print(f"  {T} frames, {total_dur:.1f}s at {FPS} fps")
    print(f"  Joint shape: {joints.shape}")

    # Load beats
    beats = load_beats(beats_path)
    if beats is not None:
        print(f"  Loaded {len(beats)} beats")

    # Compute all typed state
    print("Computing body states...")
    states = compute_body_states(joints, FPS, beats)

    # Compute beat alignment for timeline
    from .body_state import compute_beat_alignment, compute_velocities
    beat_hits = None
    if beats is not None:
        velocities = compute_velocities(joints, FPS)
        beat_hits = compute_beat_alignment(velocities, beats, FPS)
        total_hit = sum(1 for h in beat_hits if len(h) > 0)
        print(f"  Beat alignment: {total_hit}/{len(beats)} ({100*total_hit/max(len(beats),1):.1f}%)")

    # Load video frames (if provided)
    video_frames = None
    if video_path:
        print(f"Loading video: {video_path}")
        video_frames = read_video_frames(video_path, T)
        print(f"  {video_frames.shape[0]} video frames loaded ({video_frames.shape[1]}x{video_frames.shape[2]})")

    # Extract audio
    audio_tmp = None
    if audio_path:
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-vn", "-c:a", "aac", "-b:a", "192k", audio_tmp.name],
            capture_output=True,
        )
        print(f"  Audio extracted to {audio_tmp.name}")

    # Set up ffmpeg output pipe
    output_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{OUTPUT_W}x{OUTPUT_H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
    ]
    if audio_tmp:
        ffmpeg_cmd.extend(["-i", audio_tmp.name, "-c:a", "copy"])
    ffmpeg_cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ])

    print(f"Rendering {T} frames to {output_path}...")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Temporal window for trails
    window = TemporalWindow(maxlen=TRAIL_LENGTH)

    # Phase transition tracking
    prev_phase = ""
    transition_flash = 0

    for t in range(T):
        state = states[t]
        window.append(state)

        # Track phase transitions
        if state.phase != prev_phase and prev_phase != "":
            transition_flash = 10  # 10-frame flash
        prev_phase = state.phase
        if transition_flash > 0:
            transition_flash -= 1

        # Render panels
        header = render_header(state, total_dur, bpm, OUTPUT_W, transition_flash)

        if video_frames is not None and t < len(video_frames):
            left = render_video_panel(video_frames[t], state, PANEL_W, PANEL_H)
        else:
            # No video — show dark placeholder
            left = Image.new("RGBA", (PANEL_W, PANEL_H), PANEL_BG + (255,))

        right = render_skeleton_panel(state, window, PANEL_W, PANEL_H)

        bottom = render_timeline_strip(
            states, t, beats, beat_hits,
            OUTPUT_W, TIMELINE_H,
        )

        # Compose
        canvas = Image.new("RGB", (OUTPUT_W, OUTPUT_H), BG_COLOR)
        canvas.paste(header.convert("RGB"), (0, 0))
        canvas.paste(left.convert("RGB"), (0, HEADER_HEIGHT))
        canvas.paste(right.convert("RGB"), (PANEL_W, HEADER_HEIGHT))
        canvas.paste(bottom.convert("RGB"), (0, HEADER_HEIGHT + PANEL_H))

        # Write frame
        ffmpeg_proc.stdin.write(canvas.tobytes())

        if (t + 1) % 100 == 0 or t == T - 1:
            elapsed_pct = (t + 1) / T * 100
            print(f"  [{t+1}/{T}] {elapsed_pct:.0f}%")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    print(f"\nDone! Output: {output_path}")
    if audio_tmp:
        Path(audio_tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olympic Observatory — breaking analysis renderer")
    parser.add_argument("--joints", required=True, help="Path to joints_3d.npy (T, 24, 3)")
    parser.add_argument("--video", default=None, help="Path to source video (left panel)")
    parser.add_argument("--audio", default=None, help="Path to audio file (for muxing)")
    parser.add_argument("--beats", default=None, help="Path to beats.json")
    parser.add_argument("--output", default="observatory_output.mp4", help="Output MP4 path")
    parser.add_argument("--bpm", type=float, default=125.0, help="BPM for display")
    args = parser.parse_args()

    render_observatory(
        joints_path=args.joints,
        video_path=args.video,
        audio_path=args.audio,
        beats_path=args.beats,
        output_path=args.output,
        bpm=args.bpm,
    )
