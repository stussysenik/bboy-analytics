"""Download and extract BRACE release assets needed for local benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.brace_assets import download_brace_artifact, extract_brace_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch BRACE release assets")
    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=["manual_keypoints", "interpolated_keypoints"],
        choices=["manual_keypoints", "interpolated_keypoints", "audio_features"],
        help="Artifacts to fetch from the BRACE releases",
    )
    parser.add_argument("--brace-dir", default="data/brace", help="Local BRACE root")
    parser.add_argument("--year", type=int, default=None, help="Optional BRACE video year filter")
    parser.add_argument("--video-id", default=None, help="Optional BRACE video-id filter")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and re-extract matching files")
    args = parser.parse_args()

    outputs: dict[str, dict[str, object]] = {}
    for artifact in args.artifacts:
        archive_path = download_brace_artifact(
            artifact,
            brace_dir=args.brace_dir,
            overwrite=args.overwrite,
        )
        extracted = extract_brace_artifact(
            artifact,
            brace_dir=args.brace_dir,
            archive_path=archive_path,
            year=args.year,
            video_id=args.video_id,
            overwrite=args.overwrite,
        )
        outputs[artifact] = {
            "archive": str(archive_path),
            "extracted_files": len(extracted),
            "sample_files": extracted[:5],
        }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

