"""Build the layered no-rerun powermove gates report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.brace_benchmark import (  # noqa: E402
    load_brace_ground_truth_2d,
    load_brace_segments,
    load_brace_sequence,
    load_brace_shot_boundaries,
    load_interpolated_brace_keypoints,
    load_manual_brace_keypoints,
    load_projected_2d,
)
from pipeline.powermove_diagnostics import (  # noqa: E402
    build_segment_diagnostics_report,
    select_target_segment,
)
from pipeline.powermove_gates import (  # noqa: E402
    PowermoveGateThresholds,
    build_powermove_gate_report,
    write_powermove_gate_outputs,
)
from pipeline.powermove_root_cause import (  # noqa: E402
    build_powermove_root_cause_report,
)


def _auto_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if Path(path).exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate layered JOSH powermove gates")
    parser.add_argument("--brace-dir", default="data/brace")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--seq-idx", type=int, required=True)
    parser.add_argument("--sequence-name", default="")
    parser.add_argument("--josh-joints", required=True)
    parser.add_argument("--josh-meta", required=True)
    parser.add_argument("--josh-valid-mask", default=None)
    parser.add_argument("--josh-source-track-ids", default=None)
    parser.add_argument("--josh-2d", default=None)
    parser.add_argument("--gvhmr-joints", required=True)
    parser.add_argument("--baseline-2d", required=True, help="Current baseline 2D path, typically ViTPose-backed")
    parser.add_argument("--camera-k", required=True)
    parser.add_argument("--segment-uid", default=None)
    parser.add_argument("--dance-type", default="powermove")
    parser.add_argument("--control-start", type=int, default=780)
    parser.add_argument("--control-end", type=int, default=825)
    parser.add_argument("--min-window-frames", type=int, default=45)
    parser.add_argument("--candidate-min-frames", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--output-dir", default="experiments/results/powermove_debug")
    args = parser.parse_args()

    sequence = load_brace_sequence(args.brace_dir, args.video_id, args.seq_idx)
    segments = load_brace_segments(args.brace_dir, sequence)
    target_segment = select_target_segment(
        segments,
        segment_uid=args.segment_uid,
        dance_type=args.dance_type,
    )
    shot_boundaries, _ = load_brace_shot_boundaries(args.brace_dir, sequence)

    josh_joints = np.load(args.josh_joints)
    gvhmr_joints = np.load(args.gvhmr_joints)

    josh_valid_mask_path = _auto_optional_path(args.josh_valid_mask)
    if josh_valid_mask_path is None:
        auto_valid_mask = Path(args.josh_joints).with_name("joints_3d_josh_valid_mask.npy")
        josh_valid_mask_path = str(auto_valid_mask) if auto_valid_mask.exists() else None
    if josh_valid_mask_path is None:
        josh_valid_mask = np.isfinite(josh_joints).all(axis=(1, 2))
    else:
        josh_valid_mask = np.load(josh_valid_mask_path).astype(bool)

    source_track_ids_path = _auto_optional_path(args.josh_source_track_ids)
    if source_track_ids_path is None:
        auto_track_ids = Path(args.josh_joints).with_name("joints_3d_josh_source_track_ids.npy")
        source_track_ids_path = str(auto_track_ids) if auto_track_ids.exists() else None
    source_track_ids = np.load(source_track_ids_path) if source_track_ids_path is not None else None

    josh_2d_path = _auto_optional_path(args.josh_2d)
    if josh_2d_path is None:
        auto_josh_2d = Path(args.josh_joints).with_name("joints_2d_josh_coco.npy")
        josh_2d_path = str(auto_josh_2d) if auto_josh_2d.exists() else None
    josh_2d = load_projected_2d(josh_2d_path)
    baseline_2d = np.load(args.baseline_2d)[:, :, :2]

    with open(args.josh_meta) as f:
        josh_meta = json.load(f)

    gt_status, gt_frames = load_brace_ground_truth_2d(args.brace_dir, sequence, [target_segment])
    manual_gt_frames = load_manual_brace_keypoints(args.brace_dir, sequence, [target_segment])
    interpolated_gt_frames = load_interpolated_brace_keypoints(args.brace_dir, sequence, [target_segment])

    diagnostics_report = build_segment_diagnostics_report(
        josh_joints=josh_joints,
        josh_meta=josh_meta,
        josh_valid_mask=josh_valid_mask,
        gvhmr_joints=gvhmr_joints,
        sequence=sequence,
        segment=target_segment,
        shot_boundaries=shot_boundaries,
        fps=float(josh_meta.get("fps", 29.97)),
        min_window_frames=args.min_window_frames,
        candidate_min_frames=args.candidate_min_frames,
        source_track_ids=source_track_ids,
        gt_status=gt_status,
        gt_frames=gt_frames,
        manual_gt_frames=manual_gt_frames,
        interpolated_gt_frames=interpolated_gt_frames,
        josh_2d=josh_2d,
        gvhmr_2d=baseline_2d,
    )
    best_candidate = diagnostics_report["candidate_windows"][0] if diagnostics_report["candidate_windows"] else None
    target_start = int(best_candidate["local_start_frame"]) if best_candidate is not None else int(target_segment.local_start_frame)
    target_end = int(best_candidate["local_end_frame_exclusive"]) if best_candidate is not None else int(target_segment.local_end_frame_exclusive)

    root_cause_report = build_powermove_root_cause_report(
        brace_dir=args.brace_dir,
        video_id=args.video_id,
        seq_idx=args.seq_idx,
        josh_3d=josh_joints,
        josh_2d=josh_2d,
        baseline_2d=baseline_2d,
        camera_K=np.load(args.camera_k),
        image_width=args.image_width,
        image_height=args.image_height,
        target_start=target_start,
        target_end_exclusive=target_end,
        control_start=args.control_start,
        control_end_exclusive=args.control_end,
    )

    gate_report = build_powermove_gate_report(
        diagnostics_report=diagnostics_report,
        root_cause_report=root_cause_report,
        thresholds=PowermoveGateThresholds(benchmark_min_frames=args.min_window_frames),
    )
    gate_report["artifacts"] = {
        "powermove_report": "powermove_report.md",
        "root_cause_report": "root_cause_report.md",
        "target_window_start_frame": target_start,
        "target_window_end_frame_exclusive": target_end,
        "control_window_start_frame": args.control_start,
        "control_window_end_frame_exclusive": args.control_end,
    }

    out_dir = Path(args.output_dir)
    if args.sequence_name:
        out_dir = out_dir / args.sequence_name
    else:
        out_dir = out_dir / f"{sequence.video_id}_seq{sequence.seq_idx}"
    out_dir = out_dir / target_segment.uid
    outputs = write_powermove_gate_outputs(gate_report, out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
