"""Generate a focused diagnostics report for a failing BRACE powermove segment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from pipeline.brace_benchmark import (
    load_brace_ground_truth_2d,
    load_brace_segments,
    load_brace_sequence,
    load_brace_shot_boundaries,
    load_interpolated_brace_keypoints,
    load_manual_brace_keypoints,
    load_projected_2d,
)
from pipeline.powermove_diagnostics import (
    build_segment_diagnostics_report,
    select_target_segment,
    write_diagnostics_outputs,
)


def _auto_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if Path(path).exists() else None


def _segment_interpolated_json(brace_dir: Path, year: int, video_id: str, global_start: int, global_end: int, dance_type: str) -> str | None:
    patterns = [
        f"dataset/{year}/{video_id}/{video_id}_{global_start}-{global_end}_{dance_type}.json",
        f"dataset/{year}/{video_id}/{video_id}_{global_start}-{global_end}_{dance_type}.json.json",
        f"{video_id}_{global_start}-{global_end}_{dance_type}.json",
        f"{video_id}_{global_start}-{global_end}_{dance_type}.json.json",
    ]
    for pattern in patterns:
        matches = list(brace_dir.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def _render_candidate(
    *,
    repo_root: Path,
    output_dir: Path,
    josh_joints: str,
    josh_meta: str,
    gvhmr_joints: str,
    video: str,
    beats: str | None,
    audio: str | None,
    brace_video_id: str,
    brace_start_frame: int,
    candidate: dict,
    layout: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "experiments/render_model_comparison.py",
        "--josh-joints",
        josh_joints,
        "--josh-meta",
        josh_meta,
        "--gvhmr-joints",
        gvhmr_joints,
        "--video",
        video,
        "--layout",
        layout,
        "--output-dir",
        str(output_dir),
        "--brace-video-id",
        brace_video_id,
        "--brace-start-frame",
        str(brace_start_frame),
        "--window-start-frame",
        str(candidate["local_start_frame"]),
        "--window-end-frame",
        str(candidate["local_end_frame_exclusive"]),
    ]
    if beats is not None:
        cmd.extend(["--beats", beats])
    if audio is not None:
        cmd.extend(["--audio", audio])
    subprocess.run(cmd, check=True, cwd=repo_root)
    return str(
        output_dir / f"comparison_{layout}_{candidate['local_start_frame']}_{candidate['local_end_frame_exclusive']}.mp4"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a focused powermove diagnostics report")
    parser.add_argument("--brace-dir", default="data/brace")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--seq-idx", type=int, required=True)
    parser.add_argument("--josh-joints", required=True)
    parser.add_argument("--josh-meta", required=True)
    parser.add_argument("--gvhmr-joints", required=True)
    parser.add_argument("--segment-uid", default=None)
    parser.add_argument("--dance-type", default="powermove")
    parser.add_argument("--josh-valid-mask", default=None)
    parser.add_argument("--josh-source-track-ids", default=None)
    parser.add_argument("--josh-2d", default=None)
    parser.add_argument("--gvhmr-2d", default=None)
    parser.add_argument("--min-window-frames", type=int, default=45)
    parser.add_argument("--candidate-min-frames", type=int, default=8)
    parser.add_argument("--output-dir", default="experiments/results/powermove_debug")
    parser.add_argument("--video", default=None)
    parser.add_argument("--beats", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--render-top-k", type=int, default=0)
    parser.add_argument("--render-layout", choices=["vertical", "landscape"], default="landscape")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
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
    gvhmr_2d = load_projected_2d(_auto_optional_path(args.gvhmr_2d))

    with open(args.josh_meta) as f:
        josh_meta = json.load(f)

    gt_status, gt_frames = load_brace_ground_truth_2d(args.brace_dir, sequence, [target_segment])
    manual_gt_frames = load_manual_brace_keypoints(args.brace_dir, sequence, [target_segment])
    interpolated_gt_frames = load_interpolated_brace_keypoints(args.brace_dir, sequence, [target_segment])

    report = build_segment_diagnostics_report(
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
        gvhmr_2d=gvhmr_2d,
    )
    report["artifacts"] = {
        "josh_joints": args.josh_joints,
        "josh_meta": args.josh_meta,
        "josh_valid_mask": josh_valid_mask_path,
        "josh_source_track_ids": source_track_ids_path,
        "josh_2d": josh_2d_path,
        "gvhmr_joints": args.gvhmr_joints,
        "gvhmr_2d": _auto_optional_path(args.gvhmr_2d),
        "video": _auto_optional_path(args.video),
        "beats": _auto_optional_path(args.beats),
        "audio": _auto_optional_path(args.audio),
        "manual_keypoints_root": str(Path(args.brace_dir) / "manual_keypoints" / str(sequence.year) / sequence.video_id),
        "interpolated_segment_json": _segment_interpolated_json(
            Path(args.brace_dir),
            sequence.year,
            sequence.video_id,
            target_segment.global_start_frame,
            target_segment.global_end_frame_exclusive,
            target_segment.dance_type,
        ),
    }

    if args.render_top_k > 0 and args.video is not None:
        render_root = Path(args.output_dir) / "renders"
        for candidate in report["candidate_windows"][: args.render_top_k]:
            render_path = _render_candidate(
                repo_root=repo_root,
                output_dir=render_root,
                josh_joints=args.josh_joints,
                josh_meta=args.josh_meta,
                gvhmr_joints=args.gvhmr_joints,
                video=args.video,
                beats=_auto_optional_path(args.beats),
                audio=_auto_optional_path(args.audio),
                brace_video_id=sequence.video_id,
                brace_start_frame=sequence.start_frame,
                candidate=candidate,
                layout=args.render_layout,
            )
            report["review_renders"].append(
                {
                    "local_start_frame": candidate["local_start_frame"],
                    "local_end_frame_exclusive": candidate["local_end_frame_exclusive"],
                    "path": render_path,
                }
            )

    outputs = write_diagnostics_outputs(report, args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
