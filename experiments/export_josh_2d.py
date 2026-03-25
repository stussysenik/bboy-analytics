"""Export dense clip-aligned JOSH joints into full-frame COCO-17 projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.josh_projection import export_josh_projected_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense JOSH 2D COCO-17 projections")
    parser.add_argument("--joints", required=True, help="Dense JOSH joints (.npy)")
    parser.add_argument("--video", required=True, help="Source video (.mp4)")
    parser.add_argument("--output", default=None, help="Output .npy path")
    args = parser.parse_args()

    outputs = export_josh_projected_2d(
        joints_path=args.joints,
        video_path=args.video,
        output_path=args.output,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

