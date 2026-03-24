"""
World State Renderer — modular component-based visualization.

Assembles panels into layouts for different analysis views.

Usage:
    python experiments/render_worldstate.py [--layout full|minimal|analysis]
        [--joints path] [--video path] [--audio path] [--output path]

Layouts:
    full:     Video + scalar strip + energy + data points / COM / patterns
    minimal:  Video + scalar strip + energy only
    analysis: Split-screen video + pattern focus
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from world_state import compute_world_state, print_summary
from components.base import RendererBase
from components.video_overlay import VideoOverlay
from components.data_points import DataPointsPanel
from components.com_tracker import COMPanel
from components.energy_flow import EnergyPanel
from components.pattern_detect import PatternPanel
from components.scalar_strip import ScalarStrip


W, H = 1920, 1080


def build_full_layout(ws, audio_path=None) -> RendererBase:
    """
    Full layout: all panels.

    ┌─────────────────────────────────────────────────┐
    │ VIDEO + OVERLAY (1920 × 640)                     │
    ├─────────────────────────────────────────────────┤
    │ SCALAR STRIP (1920 × 30)                         │
    ├─────────────────────────────────────────────────┤
    │ ENERGY + dK/dt (1920 × 130)                      │
    ├───────────────┬──────────────┬──────────────────┤
    │ DATA POINTS   │ COM TRACKER  │ PATTERN DETECT   │
    │ (640 × 280)   │ (640 × 280)  │ (640 × 280)      │
    └───────────────┴──────────────┴──────────────────┘
    """
    r = RendererBase(W, H)
    r.set_video_rect(0, 0, W, 640)
    r.set_video_overlay(VideoOverlay(W, 640, ws))
    r.add_panel(ScalarStrip(W, 30, ws), 0, 640)
    r.add_panel(EnergyPanel(W, 130, ws), 0, 670)
    r.add_panel(DataPointsPanel(640, 280, ws), 0, 800)
    r.add_panel(COMPanel(640, 280, ws), 640, 800)
    r.add_panel(PatternPanel(640, 280, ws), 1280, 800)
    return r


def build_minimal_layout(ws) -> RendererBase:
    """
    Minimal: big video + scalar + energy.

    ┌─────────────────────────────────────────────────┐
    │ VIDEO + OVERLAY (1920 × 880)                     │
    ├─────────────────────────────────────────────────┤
    │ SCALAR STRIP (1920 × 30)                         │
    ├─────────────────────────────────────────────────┤
    │ ENERGY + dK/dt (1920 × 170)                      │
    └─────────────────────────────────────────────────┘
    """
    r = RendererBase(W, H)
    r.set_video_rect(0, 0, W, 880)
    r.set_video_overlay(VideoOverlay(W, 880, ws))
    r.add_panel(ScalarStrip(W, 30, ws), 0, 880)
    r.add_panel(EnergyPanel(W, 170, ws), 0, 910)
    return r


def build_analysis_layout(ws) -> RendererBase:
    """
    Analysis: split video + full pattern + data panels.

    ┌─────────────────────┬───────────────────────────┐
    │ VIDEO (960 × 540)   │ DATA POINTS (960 × 540)   │
    ├─────────────────────┴───────────────────────────┤
    │ SCALAR STRIP (1920 × 30)                         │
    ├─────────────────────────────────────────────────┤
    │ ENERGY (1920 × 140)                              │
    ├──────────────────────┬──────────────────────────┤
    │ COM TRACKER          │ PATTERN DETECT            │
    │ (960 × 370)          │ (960 × 370)               │
    └──────────────────────┴──────────────────────────┘
    """
    r = RendererBase(W, H)
    r.set_video_rect(0, 0, 960, 540)
    r.set_video_overlay(VideoOverlay(960, 540, ws))
    r.add_panel(DataPointsPanel(960, 540, ws), 960, 0)
    r.add_panel(ScalarStrip(W, 30, ws), 0, 540)
    r.add_panel(EnergyPanel(W, 140, ws), 0, 570)
    r.add_panel(COMPanel(960, 370, ws), 0, 710)
    r.add_panel(PatternPanel(960, 370, ws), 960, 710)
    return r


LAYOUTS = {
    "full": build_full_layout,
    "minimal": build_minimal_layout,
    "analysis": build_analysis_layout,
}


def main():
    parser = argparse.ArgumentParser(description="World State Renderer")
    parser.add_argument("--layout", choices=list(LAYOUTS.keys()), default="full")
    parser.add_argument("--joints", default="experiments/results/joints_3d_REAL_seq4.npy")
    parser.add_argument("--video", default="experiments/results/gvhmr_mesh_clean_seq4.mp4")
    parser.add_argument("--audio", default=None, help="Audio source for muxing")
    parser.add_argument("--beats", default=None, help=".npy of beat times (seconds)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--sg-window", type=int, default=15)
    args = parser.parse_args()

    base = Path(__file__).parent

    # Resolve paths relative to project root if needed
    joints_path = args.joints
    if not Path(joints_path).is_absolute():
        joints_path = str(base.parent / joints_path)

    video_path = args.video
    if not Path(video_path).is_absolute():
        video_path = str(base.parent / video_path)

    # Load data
    print(f"Loading joints: {joints_path}")
    joints = np.load(joints_path)
    print(f"  Shape: {joints.shape} ({joints.shape[0]/args.fps:.1f}s)")

    beat_times = None
    if args.beats:
        beats_path = args.beats if Path(args.beats).is_absolute() else str(base.parent / args.beats)
        if Path(beats_path).exists():
            beat_times = np.load(beats_path)
            print(f"  Beats: {len(beat_times)} times")

    # Compute world state
    print("Computing world state...")
    ws = compute_world_state(joints, args.fps, beat_times, sg_window=args.sg_window)
    print_summary(ws)

    # Build layout
    print(f"\nBuilding layout: {args.layout}")
    renderer = LAYOUTS[args.layout](ws)

    # Output path
    if args.output:
        output = args.output
    else:
        output = str(base / f"results/worldstate_{args.layout}.mp4")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Resolve audio
    audio = args.audio
    if audio and not Path(audio).is_absolute():
        audio = str(base.parent / audio)

    # Render
    renderer.render(
        source_video=video_path,
        output_path=output,
        n_frames=joints.shape[0],
        audio_source=audio,
    )

    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
