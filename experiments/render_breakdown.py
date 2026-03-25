"""
Breakdown Renderer (v6) — "Clear Split + Sliding Timeline"

Clean separation of original footage from computed analytics.
Design tokens + declarative grid layouts for vertical (1080x1920)
and landscape (1920x1080) exports.

Version history:
  v1–v3  — early renderers (see archive/)
  v4     — render_breakdown: composite dashboard
  v5     — multi-view skeleton + data points overlay
  v6     — clean split, sliding timeline, speed-colored skeleton, metrics sidebar

Usage:
    python experiments/render_breakdown.py \
        --joints experiments/results/joints_3d_REAL_seq4.npy \
        --video josh_input/bcone_seq4/video.mp4 \
        --beats experiments/results/beats.npy \
        --audio josh_input/bcone_seq4/audio.wav \
        --layout both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from world_state import compute_world_state, classify_phases, print_summary
from components.base import RendererBase
from components.video_overlay import VideoOverlay
from components.multi_view import MultiViewPanel
from components.move_bar import MoveBar
from components.sliding_timeline import SlidingTimeline
from components.metrics_sidebar import MetricsSidebar
from components.layout import TOKENS, Row, Col, compute_grid, apply_grid
from components.panel import Panel, FontCache, BG, TXT, TXT_DIM, C_BASS, C_PERC, C_HARM


# ──────────────────────────────────────────────────────────────────
# LegendBar — explains colors and symbols
# ──────────────────────────────────────────────────────────────────

class LegendBar(Panel):
    """Bottom bar: explains beat dots, audio band colors, watermark."""

    def prerender(self) -> None:
        self._bg, d = self._blank()
        f = FontCache.get
        cy = self.h // 2
        C_HIT = (80, 220, 100)
        C_MISS = (180, 40, 40)

        x = 16
        d.ellipse([x, cy - 4, x + 8, cy + 4], fill=C_HIT)
        d.text((x + 12, cy - 7), "hit", fill=TXT, font=f(size=11))
        x += 45
        d.ellipse([x, cy - 3, x + 6, cy + 3], fill=C_MISS)
        d.text((x + 10, cy - 7), "miss", fill=TXT_DIM, font=f(size=11))
        x += 50
        d.text((x, cy - 7), "|", fill=(60, 60, 70), font=f(size=12))
        x += 16

        for name, c in [("BASS", C_BASS), ("PERC", C_PERC), ("HARM", C_HARM)]:
            d.rounded_rectangle([x, cy - 6, x + 12, cy + 6], radius=3, fill=c)
            d.text((x + 16, cy - 7), name, fill=c, font=f(bold=True, size=11))
            x += 16 + int(f(bold=True, size=11).getlength(name)) + 12

        # Watermark (right-aligned)
        wm = "@bboy_analytics"
        wm_font = f(bold=True, size=11)
        wm_w = int(wm_font.getlength(wm))
        d.text((self.w - wm_w - 16, cy - 7), wm, fill=TXT_DIM, font=wm_font)

    def draw(self, frame_idx: int) -> Image.Image:
        return self._bg.copy()


# ──────────────────────────────────────────────────────────────────
# Data source detection
# ──────────────────────────────────────────────────────────────────

def detect_data_source(joints_path: str) -> str:
    """Auto-detect data source from path."""
    p = joints_path.lower()
    if "josh" in p:
        return "JOSH"
    return "GVHMR"


# ──────────────────────────────────────────────────────────────────
# Audio bass extraction (inline, from wav file)
# ──────────────────────────────────────────────────────────────────

def extract_bass(audio_path: str | None, n_frames: int, fps: float) -> np.ndarray | None:
    """Extract bass frequency band energy per frame from audio file."""
    if audio_path is None or not Path(audio_path).exists():
        return None
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        # Mel spectrogram, extract bass band (20-250 Hz)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bass_mask = (freqs >= 20) & (freqs <= 250)
        bass_energy = S[bass_mask, :].mean(axis=0)
        # Resample to video frame count
        indices = np.linspace(0, len(bass_energy) - 1, n_frames).astype(int)
        return bass_energy[indices]
    except Exception as e:
        print(f"  Warning: bass extraction failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Layout builders (declarative, grid-based)
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


def load_brace_segments(video_id: str, start_frame: int = 0,
                        end_frame: int = 999, fps: float = 29.97) -> list[dict]:
    """Load ground truth dance segments from BRACE dataset.

    Args:
        video_id: YouTube video ID (e.g. "RS0mFARO1x4")
        start_frame: Global BRACE frame number for clip start (inclusive)
        end_frame: Global BRACE frame number for clip end (exclusive)
        fps: Video frame rate for time conversion

    Returns:
        List of segment dicts with start_s, end_s, dance_type, dancer keys.
        Frames are mapped to local time (0-based) relative to start_frame.
    """
    csv_path = Path(__file__).parent.parent / "data" / "brace" / "annotations" / "segments.csv"
    if not csv_path.exists():
        print(f"  Warning: BRACE segments not found: {csv_path}")
        return []

    segments = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["video_id"] != video_id:
                continue
            sf = int(row["start_frame"])
            ef = int(row["end_frame"])
            # Skip segments entirely outside our clip range
            if sf >= end_frame or ef <= start_frame:
                continue
            # Clamp to local frame range
            local_sf = max(0, sf - start_frame)
            local_ef = min(end_frame - start_frame, ef - start_frame)
            segments.append({
                "start_s": round(local_sf / fps, 2),
                "end_s": round(local_ef / fps, 2),
                "dance_type": row["dance_type"],
                "dancer": row.get("dancer", ""),
            })
    return segments


def _build_renderer(W, H, rows, skeleton_views, ws, segments, bpm,
                     data_source, perc, harm, bass, dancer_name=""):
    """Shared layout wiring for all layouts."""
    rects = compute_grid(W, H, rows, gap=TOKENS["gap"])
    slot = {r.slot: r for r in rects}
    r = RendererBase(W, H, fps=ws.fps)

    panel_map = {
        "skeleton": MultiViewPanel(
            slot["skeleton"].w, slot["skeleton"].h, ws,
            views=skeleton_views, speed_colored=True, ghost_trails=True,
        ),
        "metrics": MetricsSidebar(
            slot["metrics"].w, slot["metrics"].h, ws,
            bpm=bpm, data_source=data_source,
        ),
        "timeline": SlidingTimeline(
            slot["timeline"].w, slot["timeline"].h, ws,
            perc=perc, harm=harm, bass=bass, window_seconds=8.0,
        ),
        "moves": MoveBar(slot["moves"].w, slot["moves"].h, ws, segments),
        "legend": LegendBar(slot["legend"].w, slot["legend"].h, ws),
    }
    overlay_map = {
        "video": VideoOverlay(slot["video"].w, slot["video"].h, ws,
                              segments=segments, dancer_name=dancer_name),
    }
    apply_grid(r, rects, panel_map, overlay_map)
    return r


def build_vertical(ws, segments, bpm, data_source="GVHMR",
                    perc=None, harm=None, bass=None, dancer_name=""):
    """Instagram vertical 1080x1920 (v6)."""
    rows = [
        Row(608, [Col("video", 1.0, is_video=True)]),
        Row(360, [Col("skeleton", 0.65), Col("metrics", 0.35)]),
        Row(260, [Col("timeline", 1.0)]),
        Row(80,  [Col("moves", 1.0)]),
        Row(40,  [Col("legend", 1.0)]),
    ]
    return _build_renderer(1080, 1920, rows, [("FRONT", 0, 1, False)],
                            ws, segments, bpm, data_source, perc, harm, bass,
                            dancer_name=dancer_name)


def build_landscape(ws, segments, bpm, data_source="GVHMR",
                     perc=None, harm=None, bass=None, dancer_name=""):
    """Landscape 1920x1080 (v6)."""
    rows = [
        Row(460, [Col("video", 0.45, is_video=True),
                  Col("skeleton", 0.35), Col("metrics", 0.20)]),
        Row(200, [Col("timeline", 1.0)]),
        Row(60,  [Col("moves", 0.55), Col("legend", 0.45)]),
    ]
    return _build_renderer(1920, 1080, rows,
                            [("FRONT", 0, 1, False), ("SIDE", 2, 1, False)],
                            ws, segments, bpm, data_source, perc, harm, bass,
                            dancer_name=dancer_name)


# ──────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────

def save_metadata(output_path: str, ws, layout: str, bpm: float,
                  resolution: str, data_source: str):
    """Save metadata JSON sidecar alongside the MP4."""
    from components.musicality_grade import grade_mu

    letter, label, _, pct = grade_mu(ws.beat_hit_pct)
    meta = {
        "version": "v6",
        "layout": layout,
        "resolution": resolution,
        "data_source": data_source,
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


# ──────────────────────────────────────────────────────────────────
# Main render pipeline
# ──────────────────────────────────────────────────────────────────

def render_breakdown(
    joints_path: str,
    video_path: str,
    layout: str = "vertical",
    beats_path: str | None = None,
    segments_path: str | None = None,
    audio_path: str | None = None,
    output_dir: str = "experiments/exports/breakdown",
    brace_video_id: str | None = None,
    brace_start_frame: int = 0,
):
    """Render v6 breakdown video."""
    base = Path(__file__).parent

    # Resolve paths
    def resolve(p):
        if p and not Path(p).is_absolute():
            return str(base.parent / p)
        return p

    joints_path = resolve(joints_path)
    video_path = resolve(video_path)
    audio_path = resolve(audio_path)

    # Auto-detect BRACE video ID from known clips
    _KNOWN_BRACE = {"bcone_seq4": "RS0mFARO1x4", "RS0mFARO1x4": "RS0mFARO1x4"}
    if brace_video_id is None:
        for key, vid in _KNOWN_BRACE.items():
            if key in (video_path or "") or key in (joints_path or ""):
                brace_video_id = vid
                print(f"  Auto-detected BRACE video: {brace_video_id}")
                break

    # Auto-detect data source
    data_source = detect_data_source(joints_path)
    print(f"Data source: {data_source}")

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

    # Load segments — BRACE ground truth takes priority over everything else
    segments = []
    dancer_name = ""
    if brace_video_id:
        end_frame = brace_start_frame + joints.shape[0]
        segments = load_brace_segments(brace_video_id, brace_start_frame,
                                       end_frame, fps)
        if segments:
            dancer_name = segments[0].get("dancer", "")
            print(f"  Segments (BRACE ground truth): {len(segments)}")
            if dancer_name:
                print(f"  Dancer: {dancer_name}")
            for seg in segments:
                print(f"    {seg['start_s']:.1f}s–{seg['end_s']:.1f}s  {seg['dance_type']}")
        else:
            print(f"  Warning: no BRACE segments found for {brace_video_id} "
                  f"frames {brace_start_frame}–{end_frame}")

    # Fall back to segments file if no BRACE segments
    if not segments:
        segments = load_segments(resolve(segments_path) if segments_path else None)
        if segments:
            print(f"  Segments (file): {len(segments)}")

    # Compute world state
    print("Computing world state...")
    ws = compute_world_state(joints, fps, beat_times)
    print_summary(ws)

    # Auto-classify phases if no segments provided
    if not segments:
        segments = classify_phases(ws)
        print(f"  Segments (auto): {len(segments)}")
        for seg in segments:
            print(f"    {seg['start_s']:.1f}s–{seg['end_s']:.1f}s  {seg['dance_type']}")

    # Load audio features
    results_dir = Path(resolve("experiments/results"))
    perc, harm, bass = None, None, None
    perc_path = results_dir / "audio_percussive.npy"
    harm_path = results_dir / "audio_harmonic.npy"
    if perc_path.exists():
        perc = np.load(str(perc_path))
        print(f"  Audio percussive: {perc.shape}")
    if harm_path.exists():
        harm = np.load(str(harm_path))
        print(f"  Audio harmonic: {harm.shape}")

    # Extract bass from audio file
    if audio_path:
        bass = extract_bass(audio_path, joints.shape[0], fps)
        if bass is not None:
            print(f"  Audio bass: {bass.shape}")

    # Compute BPM
    bpm = 0.0
    if beat_times is not None and len(beat_times) > 1:
        bpm = 60.0 / float(np.median(np.diff(beat_times)))

    # Determine layouts
    layouts_to_render = ["vertical", "landscape"] if layout == "both" else [layout]

    # Output dir
    out_dir = Path(resolve(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for lay in layouts_to_render:
        print(f"\nBuilding {lay} layout (v6)...")
        if lay == "vertical":
            renderer = build_vertical(ws, segments, bpm, data_source,
                                       perc=perc, harm=harm, bass=bass,
                                       dancer_name=dancer_name)
            resolution = "1080x1920"
        else:
            renderer = build_landscape(ws, segments, bpm, data_source,
                                        perc=perc, harm=harm, bass=bass,
                                        dancer_name=dancer_name)
            resolution = "1920x1080"

        output_path = str(out_dir / f"breakdown_{lay}_{timestamp}.mp4")

        renderer.render(
            source_video=video_path,
            output_path=output_path,
            n_frames=joints.shape[0],
            audio_source=audio_path,
        )

        save_metadata(output_path, ws, lay, bpm, resolution, data_source)
        print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Breakdown Renderer (v6)")
    parser.add_argument("--joints", required=True, help="Path to joints_3d.npy")
    parser.add_argument("--video", required=True,
                        help="Path to source video (original clean footage)")
    parser.add_argument("--layout", choices=["vertical", "landscape", "both"],
                        default="vertical", help="Output layout")
    parser.add_argument("--beats", default=None, help="Path to beats.npy")
    parser.add_argument("--segments", default=None, help="Path to segments.json")
    parser.add_argument("--audio", default=None,
                        help="Audio source for muxing + bass extraction")
    parser.add_argument("--output-dir", default="experiments/exports/breakdown")
    parser.add_argument("--brace-video-id", default=None,
                        help="BRACE dataset video ID (e.g. RS0mFARO1x4) for ground truth segments")
    parser.add_argument("--brace-start-frame", type=int, default=0,
                        help="Global BRACE start frame for sub-clips (default: 0)")
    args = parser.parse_args()

    render_breakdown(
        joints_path=args.joints,
        video_path=args.video,
        layout=args.layout,
        beats_path=args.beats,
        segments_path=args.segments,
        audio_path=args.audio,
        output_dir=args.output_dir,
        brace_video_id=args.brace_video_id,
        brace_start_frame=args.brace_start_frame,
    )


if __name__ == "__main__":
    main()
