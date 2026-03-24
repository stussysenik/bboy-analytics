"""JOSH inference pipeline wrapper — runs SAM3 -> TRAM -> DECO -> JOSH."""
import os
import subprocess
import time

from .config import JOSH_DIR


def run_josh_pipeline(
    input_folder: str,
    chunk_size: int = 15,
    josh_dir: str | None = None,
    skip_sam3: bool = False,
    visualize: bool = False,
) -> dict:
    """Run the full JOSH preprocessing + inference pipeline.

    Args:
        input_folder: Path containing rgb/ folder with extracted frames
        chunk_size: SAM3 chunk size (15 fits in 24GB VRAM)
        josh_dir: JOSH repo root (default: config.JOSH_DIR)
        skip_sam3: Skip SAM3 if masks already exist
        visualize: Enable JOSH visualization output

    Returns:
        Dict with per-step timing and status
    """
    josh = josh_dir or str(JOSH_DIR)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    results = {}

    steps = []
    if not skip_sam3:
        steps.append(("sam3", [
            "python", "-m", "preprocess.run_sam3",
            "--input_folder", input_folder,
            "--chunk_size", str(chunk_size),
        ]))
    steps.append(("tram", [
        "python", "-m", "preprocess.run_tram", "--input_folder", input_folder,
    ]))
    steps.append(("deco", [
        "python", "-m", "preprocess.run_deco", "--input_folder", input_folder,
    ]))

    # Check frame count for inference mode
    import glob
    n_frames = len(glob.glob(os.path.join(input_folder, "rgb", "*.jpg")))
    if n_frames >= 200:
        steps.append(("josh_inference", [
            "python", "josh/inference_long_video.py", "--input_folder", input_folder,
        ]))
        agg_cmd = ["python", "josh/aggregate_results.py", "--input_folder", input_folder]
        if visualize:
            agg_cmd.append("--visualize")
        steps.append(("josh_aggregate", agg_cmd))
    else:
        inf_cmd = ["python", "josh/inference.py", "--input_folder", input_folder]
        if visualize:
            inf_cmd.append("--visualize")
        steps.append(("josh_inference", inf_cmd))

    for name, cmd in steps:
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=josh, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        results[name] = {
            "elapsed_s": round(elapsed, 1),
            "returncode": proc.returncode,
        }
        if proc.returncode != 0:
            results[name]["stderr"] = proc.stderr[-500:] if proc.stderr else ""
            raise RuntimeError(f"Step '{name}' failed (exit {proc.returncode}): {proc.stderr[-300:]}")

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run JOSH inference pipeline")
    parser.add_argument("--input", required=True, help="Input folder with rgb/ frames")
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--skip-sam3", action="store_true")
    args = parser.parse_args()

    result = run_josh_pipeline(args.input, chunk_size=args.chunk_size, skip_sam3=args.skip_sam3)
    print(json.dumps(result, indent=2))
