"""
Breakdown Renderer (v4) — composite analytics for Instagram and landscape.

Composes existing panels into vertical (1080x1920) and landscape (1920x1080)
layouts with musicality grading, beat dots, and auto-timestamped exports.

Version history:
  v1 — render_video.py / render_combined.py (basic overlays)
  v2 — render_skeleton / render_spatial / render_timelines (core suite)
  v3 — render_trails / render_pitch / render_worldstate (extended views)
  v4 — render_breakdown (this file: composite dashboard)

Usage:
    python experiments/render_breakdown.py \
        --joints experiments/results/joints_3d_REAL_seq4.npy \
        --video experiments/results/gvhmr_mesh_clean_seq4.mp4 \
        --beats experiments/results/beats.npy \
        --layout vertical
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from world_state import compute_world_state, print_summary
from components.base import RendererBase
from components.video_overlay import VideoOverlay
from components.energy_flow import EnergyPanel
from components.multi_view import MultiViewPanel
from components.move_bar import MoveBar
from components.musicality_grade import MusalityGradePanel
from components.panel import Panel, FontCache, BG, TXT, TXT_DIM


# ──────────────────────────────────────────────────────────────────
# StatsFooter — inline panel, too small to warrant its own file
# ──────────────────────────────────────────────────────────────────

class StatsFooter(Panel):
    """BPM, duration, beat stats, and watermark."""

    def __init__(self, width: int, height: int, state, bpm: float = 0.0):
        super().__init__(width, height, state)
        self.bpm = bpm

    def prerender(self) -> None:
        ws = self.state
        self._bg, d = self._blank()
        f = FontCache.get

        duration = ws.frames / ws.fps
        y = (self.h - 16) // 2

        # Left: BPM + Duration
        bpm_text = f"BPM: {self.bpm:.0f}" if self.bpm > 0 else "BPM: —"
        dur_text = f"Duration: {duration:.1f}s"
        d.text((16, y), bpm_text, fill=TXT, font=f(bold=True, size=14))
        d.text((140, y), dur_text, fill=TXT_DIM, font=f(size=13))

        # Center: Beat stats
        if ws.beat_hits:
            n_hits = sum(1 for b in ws.beat_hits if b["hit"])
            total = len(ws.beat_hits)
            stats = f"{total} Beats  |  {n_hits} Hits ({ws.beat_hit_pct:.0f}%)"
            bb = d.textbbox((0, 0), stats, font=f(bold=True, size=13))
            sw = bb[2] - bb[0]
            d.text(((self.w - sw) // 2, y), stats, fill=TXT, font=f(bold=True, size=13))

        # Right: watermark
        wm = "@bboy_analytics"
        wm_font = f(bold=True, size=12)
        bb2 = d.textbbox((0, 0), wm, font=wm_font)
        wm_w = bb2[2] - bb2[0]
        d.text((self.w - wm_w - 16, y), wm, fill=TXT_DIM, font=wm_font)

    def draw(self, frame_idx: int) -> Image.Image:
        return self._bg.copy()


# ──────────────────────────────────────────────────────────────────
# Layout builders
# ──────────────────────────────────────────────────────────────────

def load_segments(segments_path: str | None) -> list[dict]:
    """Load segment classifications from JSON."""
    if segments_path is None:
        return []
    p = Path(segments_path)
    if not p.exists():
        print(f"Warning: segments file not found: {segments_path}")
        return []
    with open(p) as f:
        return json.load(f)


def build_vertical(ws, segments: list[dict], bpm: float) -> RendererBase:
    """
    Instagram vertical 1080x1920.

    ┌──────────────────┐
    │   VIDEO          │  720px
    │   (1080x720)     │
    ├──────────────────┤
    │  COM TRACKER     │  300px
    ├──────────────────┤
    │  ENERGY          │  200px
    ├──────────────────┤
    │  MOVE BAR        │  150px
    ├──────────────────┤
    │  MUSICALITY      │  250px
    ├──────────────────┤
    │  STATS FOOTER    │  100px
    ├──────────────────┤
    │  (safe area)     │  200px
    └──────────────────┘
    """
    W, H = 1080, 1920
    r = RendererBase(W, H, fps=ws.fps)

    r.set_video_rect(0, 0, 1080, 720)
    r.set_video_overlay(VideoOverlay(1080, 720, ws))

    r.add_panel(MultiViewPanel(1080, 300, ws), 0, 720)
    r.add_panel(EnergyPanel(1080, 200, ws), 0, 1020)

    if segments:
        r.add_panel(MoveBar(1080, 150, ws, segments), 0, 1220)
    else:
        # Empty spacer if no segments
        r.add_panel(EnergyPanel(1080, 150, ws), 0, 1220)  # duplicate energy as filler

    r.add_panel(MusalityGradePanel(1080, 250, ws), 0, 1370)
    r.add_panel(StatsFooter(1080, 100, ws, bpm=bpm), 0, 1620)

    return r


def build_landscape(ws, segments: list[dict], bpm: float) -> RendererBase:
    """
    Landscape 1920x1080.

    ┌──────────────┬──────────────┐
    │ VIDEO 960×540│ COM 960×540  │
    ├──────────────┴──────────────┤
    │ ENERGY (1920×160)           │
    ├─────────────────────────────┤
    │ MOVE BAR (1920×100)         │
    ├─────────────────────────────┤
    │ MUSICALITY (1920×180)       │
    ├─────────────────────────────┤
    │ FOOTER (1920×100)           │
    └─────────────────────────────┘
    """
    W, H = 1920, 1080
    r = RendererBase(W, H, fps=ws.fps)

    r.set_video_rect(0, 0, 960, 540)
    r.set_video_overlay(VideoOverlay(960, 540, ws))

    r.add_panel(MultiViewPanel(960, 540, ws), 960, 0)
    r.add_panel(EnergyPanel(1920, 160, ws), 0, 540)

    if segments:
        r.add_panel(MoveBar(1920, 100, ws, segments), 0, 700)
    else:
        r.add_panel(EnergyPanel(1920, 100, ws), 0, 700)

    r.add_panel(MusalityGradePanel(1920, 180, ws), 0, 800)
    r.add_panel(StatsFooter(1920, 100, ws, bpm=bpm), 0, 980)

    return r


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def save_metadata(output_path: str, ws, layout: str, bpm: float, resolution: str):
    """Save metadata JSON sidecar alongside the MP4."""
    from components.musicality_grade import grade_mu

    letter, label, _, pct = grade_mu(ws.mu)
    meta = {
        "version": "v4.1",
        "layout": layout,
        "resolution": resolution,
        "frames": ws.frames,
        "fps": ws.fps,
        "duration_s": round(ws.frames / ws.fps, 2),
        "musicality": {
            "mu": round(ws.mu, 4),
            "grade": letter,
            "label": label,
            "pct": pct,
            "beat_hit_pct": round(ws.beat_hit_pct, 1),
        },
        "bpm": round(bpm, 1),
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
    }
    json_path = output_path.replace(".mp4", ".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {json_path}")


def render_breakdown(
    joints_path: str,
    video_path: str,
    layout: str = "vertical",
    beats_path: str | None = None,
    segments_path: str | None = None,
    audio_path: str | None = None,
    output_dir: str = "experiments/exports/breakdown",
):
    """Render breakdown video."""
    base = Path(__file__).parent

    # Resolve paths
    def resolve(p):
        if p and not Path(p).is_absolute():
            return str(base.parent / p)
        return p

    joints_path = resolve(joints_path)
    video_path = resolve(video_path)
    audio_path = resolve(audio_path)

    # Load joints
    print(f"Loading joints: {joints_path}")
    joints = np.load(joints_path)
    fps = 29.97
    print(f"  Shape: {joints.shape} ({joints.shape[0]/fps:.1f}s)")

    # Load beats
    beat_times = None
    if beats_path:
        bp = resolve(beats_path)
        if Path(bp).exists():
            beat_times = np.load(bp)
            print(f"  Beats: {len(beat_times)} times")

    # Load segments
    segments = load_segments(resolve(segments_path) if segments_path else None)
    if segments:
        print(f"  Segments: {len(segments)}")

    # Compute world state
    print("Computing world state...")
    ws = compute_world_state(joints, fps, beat_times)
    print_summary(ws)

    # Compute BPM from beat times
    bpm = 0.0
    if beat_times is not None and len(beat_times) > 1:
        bpm = 60.0 / float(np.median(np.diff(beat_times)))

    # Determine which layouts to render
    layouts_to_render = []
    if layout == "both":
        layouts_to_render = ["vertical", "landscape"]
    else:
        layouts_to_render = [layout]

    # Ensure output dir
    out_dir = Path(resolve(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for lay in layouts_to_render:
        print(f"\nBuilding {lay} layout...")
        if lay == "vertical":
            renderer = build_vertical(ws, segments, bpm)
            resolution = "1080x1920"
        else:
            renderer = build_landscape(ws, segments, bpm)
            resolution = "1920x1080"

        output_path = str(out_dir / f"breakdown_{lay}_{timestamp}.mp4")

        renderer.render(
            source_video=video_path,
            output_path=output_path,
            n_frames=joints.shape[0],
            audio_source=audio_path,
        )

        save_metadata(output_path, ws, lay, bpm, resolution)
        print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Breakdown Renderer (v4)")
    parser.add_argument("--joints", required=True, help="Path to joints_3d.npy")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--layout", choices=["vertical", "landscape", "both"],
                        default="vertical", help="Output layout")
    parser.add_argument("--beats", default=None, help="Path to beats.npy")
    parser.add_argument("--segments", default=None, help="Path to segments.json")
    parser.add_argument("--audio", default=None, help="Audio source for muxing")
    parser.add_argument("--output-dir", default="experiments/exports/breakdown")
    args = parser.parse_args()

    render_breakdown(
        joints_path=args.joints,
        video_path=args.video,
        layout=args.layout,
        beats_path=args.beats,
        segments_path=args.segments,
        audio_path=args.audio,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
