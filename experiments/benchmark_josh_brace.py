"""Run a BRACE-aligned structural benchmark for JOSH vs GVHMR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.brace_benchmark import (
    build_benchmark_report,
    infer_josh_valid_mask,
    load_brace_ground_truth_2d,
    load_brace_segments,
    load_brace_sequence,
    load_brace_shot_boundaries,
    load_projected_2d,
    write_benchmark_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JOSH vs GVHMR on BRACE segments")
    parser.add_argument("--josh-joints", required=True, help="Dense JOSH joints (.npy)")
    parser.add_argument("--josh-meta", required=True, help="Dense JOSH metadata (.json)")
    parser.add_argument("--josh-valid-mask", default=None, help="Optional JOSH valid mask (.npy)")
    parser.add_argument("--josh-2d", default=None, help="Optional JOSH projected 2D joints (.npy)")
    parser.add_argument("--gvhmr-joints", required=True, help="GVHMR joints (.npy)")
    parser.add_argument("--gvhmr-2d", default=None, help="Optional GVHMR/ViTPose 2D joints (.npy)")
    parser.add_argument("--brace-dir", default="data/brace", help="Path to BRACE dataset root")
    parser.add_argument("--video-id", required=True, help="BRACE YouTube video id")
    parser.add_argument("--seq-idx", type=int, required=True, help="BRACE sequence index")
    parser.add_argument("--sequence-name", default="", help="Optional local sequence label")
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--min-window-frames", type=int, default=45)
    parser.add_argument(
        "--output-dir",
        default="experiments/results/benchmarks",
        help="Directory where benchmark outputs will be written",
    )
    args = parser.parse_args()

    if args.josh_2d is None:
        auto_josh_2d = Path(args.josh_joints).with_name("joints_2d_josh_coco.npy")
        if auto_josh_2d.exists():
            args.josh_2d = str(auto_josh_2d)

    josh_joints = np.load(args.josh_joints)
    gvhmr_joints = np.load(args.gvhmr_joints)
    josh_valid_mask = infer_josh_valid_mask(josh_joints, args.josh_valid_mask)
    with open(args.josh_meta) as f:
        josh_meta = json.load(f)

    sequence = load_brace_sequence(args.brace_dir, args.video_id, args.seq_idx)
    segments = load_brace_segments(args.brace_dir, sequence)
    shot_boundaries, shot_boundaries_status = load_brace_shot_boundaries(args.brace_dir, sequence)
    gt_status, gt_frames = load_brace_ground_truth_2d(args.brace_dir, sequence, segments)

    report = build_benchmark_report(
        josh_joints=josh_joints,
        josh_meta=josh_meta,
        josh_valid_mask=josh_valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segments=segments,
        shot_boundaries=shot_boundaries,
        fps=args.fps,
        min_window_frames=args.min_window_frames,
        josh_2d=load_projected_2d(args.josh_2d),
        gvhmr_2d=load_projected_2d(args.gvhmr_2d),
        gt_status=gt_status,
        gt_frames=gt_frames,
        shot_boundaries_status=shot_boundaries_status,
    )
    if args.sequence_name:
        report["sequence"]["sequence_name"] = args.sequence_name

    out_dir = Path(args.output_dir)
    if args.sequence_name:
        out_dir = out_dir / args.sequence_name
    else:
        out_dir = out_dir / f"{args.video_id}_seq{args.seq_idx}"
    outputs = write_benchmark_outputs(report, out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
