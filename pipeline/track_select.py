"""Smart TRAM track analysis and selection for multi-person scenes.

Analyzes all TRAM track files, segments them at identity swap points
(large displacement discontinuities), scores segments deterministically,
and selects the best clean single-dancer segments.

Usage:
    python -m pipeline.track_select --tram-dir josh_input/bcone_seq4/tram --total-frames 999
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch

from .config import (
    IDENTITY_TRACKING_THRESHOLD_M,
    TRACK_SEGMENT_MIN_FRAMES,
    TRACK_STAGE_RADIUS_M,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor or ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def _segment_track(
    frames: np.ndarray,
    trans: np.ndarray,
    threshold_m: float,
    min_frames: int,
) -> list[dict]:
    """Split a single track at displacement discontinuities.

    Returns list of segment dicts with quality metrics.
    """
    n = len(frames)
    if n < min_frames:
        return []

    displacements = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
    switches = np.where(displacements > threshold_m)[0]

    # Build segment boundaries: [0, sw1+1, sw2+1, ..., n]
    boundaries = [0] + [sw + 1 for sw in switches] + [n]
    segments = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]  # exclusive
        if end - start < min_frames:
            continue

        seg_frames = frames[start:end]
        seg_trans = trans[start:end]
        seg_displacements = np.linalg.norm(np.diff(seg_trans, axis=0), axis=-1)
        frame_gaps = np.diff(seg_frames)

        segments.append({
            "start_idx": int(start),
            "end_idx": int(end - 1),
            "n_frames": int(end - start),
            "frame_start": int(seg_frames[0]),
            "frame_end": int(seg_frames[-1]),
            "mean_displacement_m": float(seg_displacements.mean()) if len(seg_displacements) > 0 else 0,
            "max_displacement_m": float(seg_displacements.max()) if len(seg_displacements) > 0 else 0,
            "x_range_m": float(seg_trans[:, 0].max() - seg_trans[:, 0].min()),
            "z_range_m": float(seg_trans[:, 2].max() - seg_trans[:, 2].min()),
            "consecutive_ratio": float((frame_gaps == 1).sum() / max(len(frame_gaps), 1)) if len(frame_gaps) > 0 else 1.0,
            "max_gap": int(frame_gaps.max()) if len(frame_gaps) > 0 else 0,
        })

    return segments


def analyze_tracks(
    tram_dir: str,
    threshold_m: float = IDENTITY_TRACKING_THRESHOLD_M,
    min_frames: int = TRACK_SEGMENT_MIN_FRAMES,
) -> list[dict]:
    """Analyze all TRAM tracks and segment them into clean pieces.

    Returns list of segment dicts, each annotated with track file and quality metrics.
    """
    all_segments = []

    for fname in sorted(os.listdir(tram_dir)):
        if not fname.endswith(".npy") or not fname.startswith("hps_track_"):
            continue

        # TRAM saves per-track dicts with allow_pickle=True (contains numpy arrays + torch tensors)
        data = np.load(os.path.join(tram_dir, fname), allow_pickle=True).item()
        frames = _to_numpy(data["frame"])
        trans = _to_numpy(data["pred_trans"])
        if trans.ndim == 3:
            trans = trans.squeeze(1)

        track_id = data.get("id", fname)
        segments = _segment_track(frames, trans, threshold_m, min_frames)

        for seg in segments:
            seg["track_file"] = fname
            seg["track_id"] = int(track_id) if isinstance(track_id, (int, np.integer)) else track_id
            all_segments.append(seg)

    return all_segments


def score_segment(seg: dict, total_frames: int) -> float:
    """Deterministic segment quality score. Higher = better.

    Components (all physical quantities or percentages):
    - 60%: frame count / total_frames
    - 20%: smoothness = 1 - mean_disp / threshold, clipped [0,1]
    - 10%: compactness = 1 if x_range AND z_range < stage radius, else 0
    - 10%: consecutiveness = ratio of consecutive frame pairs
    """
    frame_score = seg["n_frames"] / max(total_frames, 1)
    smoothness = max(0, 1 - seg["mean_displacement_m"] / IDENTITY_TRACKING_THRESHOLD_M)
    compact = 1.0 if (seg["x_range_m"] < TRACK_STAGE_RADIUS_M and seg["z_range_m"] < TRACK_STAGE_RADIUS_M) else 0.0
    consecutive = seg["consecutive_ratio"]

    return 0.6 * frame_score + 0.2 * smoothness + 0.1 * compact + 0.1 * consecutive


def select_best_segments(
    tram_dir: str,
    total_frames: int = 999,
    threshold_m: float = IDENTITY_TRACKING_THRESHOLD_M,
    min_frames: int = TRACK_SEGMENT_MIN_FRAMES,
) -> dict:
    """Analyze all tracks and select the best non-overlapping segments.

    Returns result dict with selected segments and coverage stats.
    """
    all_segments = analyze_tracks(tram_dir, threshold_m, min_frames)

    # Score and sort
    for seg in all_segments:
        seg["score"] = score_segment(seg, total_frames)
    all_segments.sort(key=lambda s: -s["score"])

    # Greedy non-overlapping selection
    covered: set[int] = set()
    selected = []
    for seg in all_segments:
        seg_frames = set(range(seg["frame_start"], seg["frame_end"] + 1))
        if seg_frames & covered:
            continue
        selected.append(seg)
        covered.update(seg_frames)

    # Find primary track (most selected frames)
    track_frame_counts: dict[str, int] = {}
    for seg in selected:
        tf = seg["track_file"]
        track_frame_counts[tf] = track_frame_counts.get(tf, 0) + seg["n_frames"]
    primary = max(track_frame_counts, key=track_frame_counts.get) if track_frame_counts else ""

    total_clean = sum(s["n_frames"] for s in selected)

    return {
        "selected_segments": selected,
        "total_clean_frames": total_clean,
        "coverage_pct": round(100 * total_clean / max(total_frames, 1), 1),
        "primary_track": primary,
        "primary_track_id": selected[0]["track_id"] if selected else None,
        "all_tracks_analyzed": len(set(s["track_file"] for s in all_segments)),
        "all_segments_found": len(all_segments),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAM track analysis and selection")
    parser.add_argument("--tram-dir", required=True, help="Directory with hps_track_*.npy files")
    parser.add_argument("--total-frames", type=int, default=999, help="Total video frames")
    parser.add_argument("--threshold", type=float, default=IDENTITY_TRACKING_THRESHOLD_M)
    parser.add_argument("--min-frames", type=int, default=TRACK_SEGMENT_MIN_FRAMES)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = select_best_segments(
        args.tram_dir,
        total_frames=args.total_frames,
        threshold_m=args.threshold,
        min_frames=args.min_frames,
    )

    print(f"Analyzed {result['all_tracks_analyzed']} tracks, found {result['all_segments_found']} clean segments")
    print(f"Selected {len(result['selected_segments'])} non-overlapping segments:")
    for i, seg in enumerate(result["selected_segments"]):
        print(f"  [{i}] {seg['track_file']} frames {seg['frame_start']}-{seg['frame_end']} "
              f"({seg['n_frames']} frames, score={seg['score']:.3f}, "
              f"maxDisp={seg['max_displacement_m']:.3f}m)")
    print(f"\nTotal clean frames: {result['total_clean_frames']}/{args.total_frames} "
          f"({result['coverage_pct']}%)")
    print(f"Primary track: {result['primary_track']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {args.output}")
