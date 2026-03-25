"""Run numerical root-cause analysis on a failing powermove slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.powermove_root_cause import (  # noqa: E402
    build_powermove_root_cause_report,
    write_root_cause_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether a powermove failure is projection- or model-driven")
    parser.add_argument("--brace-dir", default="data/brace")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--seq-idx", type=int, required=True)
    parser.add_argument("--josh-joints", required=True)
    parser.add_argument("--josh-2d", required=True)
    parser.add_argument("--baseline-2d", required=True)
    parser.add_argument("--camera-k", required=True)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--target-start", type=int, required=True)
    parser.add_argument("--target-end", type=int, required=True)
    parser.add_argument("--control-start", type=int, required=True)
    parser.add_argument("--control-end", type=int, required=True)
    parser.add_argument("--output-dir", default="experiments/results/powermove_debug")
    parser.add_argument("--sequence-name", default="")
    args = parser.parse_args()

    report = build_powermove_root_cause_report(
        brace_dir=args.brace_dir,
        video_id=args.video_id,
        seq_idx=args.seq_idx,
        josh_3d=np.load(args.josh_joints),
        josh_2d=np.load(args.josh_2d),
        baseline_2d=np.load(args.baseline_2d)[:, :, :2],
        camera_K=np.load(args.camera_k),
        image_width=args.image_width,
        image_height=args.image_height,
        target_start=args.target_start,
        target_end_exclusive=args.target_end,
        control_start=args.control_start,
        control_end_exclusive=args.control_end,
    )
    out_dir = Path(args.output_dir)
    if args.sequence_name:
        out_dir = out_dir / args.sequence_name
    outputs = write_root_cause_outputs(report, out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
