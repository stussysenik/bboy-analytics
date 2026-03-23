from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audio_motion import audio_motion_alignment, synthetic_alignment_payload
from .manifest import load_papers, load_scenarios
from .runner import execute_ladder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extreme-motion paper ladder experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute the paper ladder")
    run_parser.add_argument("--papers", default="papers.yaml", help="Paper manifest path")
    run_parser.add_argument("--scenarios", default="scenarios.json", help="Scenario bank path")
    run_parser.add_argument("--output-root", default="runs", help="Directory for generated outputs")
    run_parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Evaluate every paper even if a gate fails",
    )

    analyze_parser = subparsers.add_parser("score-audio-motion", help="Score an audio-motion pair")
    analyze_parser.add_argument("--synthetic", action="store_true", help="Use a built-in synthetic clip")
    analyze_parser.add_argument("--samples", type=int, default=256, help="Synthetic sample count")
    analyze_parser.add_argument("--input", help="Path to a JSON file with joints/audio/fps/sample_rate")

    return parser


def _score_audio_motion(args: argparse.Namespace) -> int:
    if args.synthetic:
        print(json.dumps(synthetic_alignment_payload(samples=args.samples), indent=2))
        return 0

    if not args.input:
        raise SystemExit("Provide --input or use --synthetic")

    payload = json.loads(Path(args.input).read_text())
    joints = np.asarray(payload["joints"], dtype=np.float64)
    audio = np.asarray(payload["audio"], dtype=np.float64)
    metrics = audio_motion_alignment(
        joints=joints,
        audio=audio,
        fps=float(payload["fps"]),
        sample_rate=int(payload["sample_rate"]),
    )
    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


def _run_ladder(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    papers = load_papers(base_dir / args.papers)
    scenarios = load_scenarios(base_dir / args.scenarios)
    summary = execute_ladder(
        papers=papers,
        scenarios=scenarios,
        base_dir=base_dir,
        output_root=base_dir / args.output_root,
        continue_on_fail=args.continue_on_fail,
    )
    print(json.dumps({"run_id": summary.run_id, "promoted_ids": summary.promoted_ids}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_ladder(args)
    if args.command == "score-audio-motion":
        return _score_audio_motion(args)

    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
