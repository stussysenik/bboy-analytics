"""Generate a focused diagnostics report for a failing BRACE powermove segment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def _stack_videos(left: str, right: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            left,
            "-i",
            right,
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def _render_candidate(
    *,
    output_dir: Path,
    josh_joints: str,
    gvhmr_joints: str,
    video: str,
    beats: str | None,
    audio: str | None,
    brace_video_id: str,
    brace_start_frame: int,
    candidate: dict,
    layout: str,
) -> str:
    from render_breakdown import render_breakdown

    output_dir.mkdir(parents=True, exist_ok=True)
    start = int(candidate["local_start_frame"])
    end = int(candidate["local_end_frame_exclusive"])
    josh_dir = output_dir / "_josh"
    gvhmr_dir = output_dir / "_gvhmr"
    josh_dir.mkdir(parents=True, exist_ok=True)
    gvhmr_dir.mkdir(parents=True, exist_ok=True)

    josh_outputs = render_breakdown(
        joints_path=josh_joints,
        video_path=video,
        layout=layout,
        beats_path=beats,
        audio_path=audio,
        output_dir=str(josh_dir),
        brace_video_id=brace_video_id,
        brace_start_frame=brace_start_frame,
        window_start_frame=start,
        window_end_frame=end,
    )
    gvhmr_outputs = render_breakdown(
        joints_path=gvhmr_joints,
        video_path=video,
        layout=layout,
        beats_path=beats,
        audio_path=audio,
        output_dir=str(gvhmr_dir),
        brace_video_id=brace_video_id,
        brace_start_frame=brace_start_frame,
        window_start_frame=start,
        window_end_frame=end,
    )
    comparison_path = output_dir / f"comparison_{layout}_{start}_{end}.mp4"
    _stack_videos(josh_outputs[0], gvhmr_outputs[0], str(comparison_path))
    return str(comparison_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a focused powermove diagnostics report")
    parser.add_argument("--brace-dir", default="data/brace")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--seq-idx", type=int, required=True)
    parser.add_argument("--sequence-name", default="")
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

    out_dir = Path(args.output_dir)
    if args.sequence_name:
        out_dir = out_dir / args.sequence_name
    else:
        out_dir = out_dir / f"{sequence.video_id}_seq{sequence.seq_idx}"
    out_dir = out_dir / target_segment.uid

    if args.render_top_k > 0 and args.video is not None:
        render_root = out_dir / "renders"
        for candidate in report["candidate_windows"][: args.render_top_k]:
            render_path = _render_candidate(
                output_dir=render_root,
                josh_joints=args.josh_joints,
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

    outputs = write_diagnostics_outputs(report, out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
