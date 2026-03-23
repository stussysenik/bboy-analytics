"""CLI for Battle Analytics Recap pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path


def cmd_audio(args: argparse.Namespace) -> int:
    """Analyze audio: beats, BPM, sections."""
    from .audio import analyze_audio

    video_path = Path(args.video) if args.video else None
    brace_path = Path(args.brace_beats) if args.brace_beats else None

    print("▸ Analyzing audio...")
    result = analyze_audio(
        video_path=video_path,
        brace_beats_path=brace_path,
        video_id=args.video_id,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "audio_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  BPM: {result['bpm']}  |  Beats: {result['n_beats']}  |  Source: {result['source']}")
    print(f"  Saved: {out_path}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Compute all metrics from joints_3d.npy."""
    from .validate import validate_joints, validate_audio, ValidationError
    from .metrics import compute_all_metrics

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load joints
    joints_path = Path(args.joints) if args.joints else output_dir / "joints_3d.npy"
    if args.synthetic:
        print("▸ Generating synthetic joint data...")
        np.random.seed(42)
        n_frames = 900  # 30s at 30fps
        t = np.linspace(0, 30, n_frames)
        joints = np.zeros((n_frames, 22, 3))
        # Simulate walking + arm movement
        for j in range(22):
            joints[:, j, 0] = np.sin(t * 0.5 + j * 0.3) * 0.2  # X sway
            joints[:, j, 1] = 1.0 + j * 0.05 + np.sin(t * 2 + j) * 0.05  # Y height
            joints[:, j, 2] = t * 0.1 + np.sin(t * 0.8 + j * 0.5) * 0.1  # Z forward
        joints += np.random.randn(*joints.shape) * 0.01
        fps = 30.0
    else:
        try:
            joints = validate_joints(joints_path)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
        fps = args.fps

    # Load audio analysis (if available)
    audio_path = output_dir / "audio_analysis.json"
    beat_times = None
    if audio_path.exists():
        with open(audio_path) as f:
            audio = json.load(f)
        beat_times = np.array(audio.get("beat_times", []))
        print(f"  Using {len(beat_times)} beats from {audio.get('source', 'unknown')}")

    # Load segments (if available)
    segments = None
    if args.segments:
        import csv
        with open(args.segments) as f:
            reader = csv.DictReader(f)
            vid = args.video_id or ""
            segments = [
                {**row, "start_frame": int(row["start_frame"]), "end_frame": int(row["end_frame"])}
                for row in reader if row.get("video_id", "") == vid
            ]
        if segments:
            print(f"  Loaded {len(segments)} segments for {vid}")

    print("▸ Computing metrics...")
    metrics = compute_all_metrics(joints, fps, beat_times, segments)

    # Save
    # Remove large arrays from JSON (keep in separate files)
    metrics_json = json.loads(json.dumps(metrics, default=lambda x: None))
    if "energy_curve" in metrics_json.get("energy", {}):
        del metrics_json["energy"]["energy_curve"]
    if "com_trajectory" in metrics_json.get("space", {}):
        del metrics_json["space"]["com_trajectory"]
    if "com_y" in metrics_json.get("space", {}):
        del metrics_json["space"]["com_y"]

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Save freeze and timeline separately
    with open(output_dir / "freeze_events.json", "w") as f:
        json.dump(metrics["complexity"]["freeze_events"], f, indent=2)

    with open(output_dir / "timeline.json", "w") as f:
        json.dump({
            "energy_sections": metrics["energy"]["energy_sections"],
            "freeze_events": metrics["complexity"]["freeze_events"],
            "inversion_events": metrics["complexity"]["inversion_events"],
            "move_distribution": metrics["complexity"]["move_type_distribution_pct"],
        }, f, indent=2)

    # Print summary
    m = metrics["musicality"]
    print(f"\n  Musicality:  μ = {m['mu']:.3f} [{m['interpretation']['musicality']}]")
    print(f"  Timing:      τ* = {m['tau_star_ms']:.0f} ms [{m['interpretation']['timing']}]")
    print(f"  Flow:        {metrics['flow']['flow_score']:.0f}/100")
    print(f"  Space:       {metrics['space']['stage_coverage_m2']:.1f} m²")
    print(f"  Freezes:     {metrics['complexity']['freeze_count']}")
    print(f"  Inversions:  {metrics['complexity']['inversion_count']}")
    print(f"\n  Saved: metrics.json, timeline.json, freeze_events.json → {output_dir}/")

    # Render visualizations
    print("\n▸ Rendering visualizations...")
    from .render import render_all
    created = render_all(metrics, output_dir, fps)
    for f in created:
        print(f"  Saved: {f}")

    return 0


def cmd_package(args: argparse.Namespace) -> int:
    """Bundle recap into a zip."""
    from .package import generate_summary, create_zip
    from .validate import validate_metrics, validate_audio

    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"ERROR: {output_dir} not found")
        return 1

    metrics = validate_metrics(output_dir / "metrics.json")
    audio_path = output_dir / "audio_analysis.json"
    audio = json.load(open(audio_path)) if audio_path.exists() else {}

    print("▸ Generating summary...")
    generate_summary(metrics, audio, output_dir / "summary.txt")
    print(f"  Saved: summary.txt")

    print("▸ Creating zip...")
    zip_path = create_zip(output_dir)
    print(f"  Saved: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate outputs in a recap directory."""
    from .validate import validate_output_dir

    output_dir = Path(args.output)
    status = validate_output_dir(output_dir)

    print(f"▸ Validating {output_dir}/")
    all_ok = True
    for name, exists in status.items():
        icon = "✓" if exists else "✗"
        print(f"  {icon} {name}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n  All outputs present.")
    else:
        missing = sum(1 for v in status.values() if not v)
        print(f"\n  {missing} outputs missing.")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bboy-recap", description="Battle Analytics Recap")
    sub = parser.add_subparsers(dest="command", required=True)

    # audio
    p = sub.add_parser("audio", help="Analyze audio (beats, BPM, sections)")
    p.add_argument("--video", help="Path to video file")
    p.add_argument("--brace-beats", help="Path to BRACE audio_beats.json")
    p.add_argument("--video-id", help="Video ID for BRACE lookup")
    p.add_argument("-o", "--output", default="battle-recap", help="Output directory")

    # metrics
    p = sub.add_parser("metrics", help="Compute all metrics from joints data")
    p.add_argument("--joints", help="Path to joints_3d.npy")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    p.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    p.add_argument("--segments", help="Path to BRACE segments.csv")
    p.add_argument("--video-id", help="Video ID for segment filtering")
    p.add_argument("-o", "--output", default="battle-recap", help="Output directory")

    # package
    p = sub.add_parser("package", help="Bundle outputs into a zip")
    p.add_argument("-o", "--output", default="battle-recap", help="Recap directory to bundle")

    # validate
    p = sub.add_parser("validate", help="Check output completeness")
    p.add_argument("-o", "--output", default="battle-recap", help="Recap directory to validate")

    args = parser.parse_args(argv)
    commands = {"audio": cmd_audio, "metrics": cmd_metrics, "package": cmd_package, "validate": cmd_validate}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
