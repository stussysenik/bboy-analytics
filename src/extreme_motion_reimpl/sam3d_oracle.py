"""SAM3D oracle baseline — near-perfect masks with minimal noise.

Upper-bound reference for the SAM3D scoring gate.  Uses the same three-stage
pipeline as the reimplementation but with oracle-grade parameters:

  - noise_level=0.03  (vs 0.10 in reimpl)
  - k=30              (vs 20 in reimpl)
  - All frames used for voting

All metrics should land comfortably above the reimpl gate thresholds.

Reference: Yang et al., "SAM3D: Segment Anything in 3D Scenes", 2023.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .sam3d_reimpl import run_sam3d_pipeline


def run_oracle(seed: int = 42) -> dict:
    """Run SAM3D pipeline with oracle-grade parameters."""
    return run_sam3d_pipeline(
        noise_level=0.03,
        k=30,
        n_point_samples=2000,
        seed=seed,
        n_frames=32,
        use_all_frames=True,
    )


def main() -> None:
    t0 = time.monotonic()
    metrics = run_oracle(seed=42)
    wall = time.monotonic() - t0

    loc = len(Path(__file__).read_text().splitlines())

    payload = {
        "paper_id": "sam3d",
        "mode": "oracle",
        "canonical_metrics": {
            "mask_projection_iou": metrics["mask_projection_iou"],
            "region_merge_stability": metrics["region_merge_stability"],
        },
        "applied_metrics": {
            "dancer_bleed_rate": metrics["dancer_bleed_rate"],
            "scene_coherence": metrics["scene_coherence"],
            "audio_motion_stability": metrics["audio_motion_stability"],
        },
        "runtime_cost": {"wall_clock_seconds": round(wall, 4), "gpu_hours": 0.0},
        "code_stats": {
            "implementation_loc": loc,
            "files_touched": 1,
            "shared_module_ratio": 0.90,
        },
        "artifacts": [],
        "open_questions": [
            "Oracle noise_level=0.03 may still be generous for real SAM outputs.",
        ],
        "notes": (
            "Oracle baseline: noise_level=0.03, k=30. Reuses reimpl pipeline "
            "with tighter parameters for upper-bound reference."
        ),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
