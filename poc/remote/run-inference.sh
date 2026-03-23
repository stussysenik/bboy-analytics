#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GVHMR Inference + Joint Extraction
# Usage: bash remote/run-inference.sh <video_path> [--static]
# Output: results/joints_3d.npy, results/metadata.json
# ============================================================

VIDEO_PATH="${1:?Usage: run-inference.sh <video_path> [--static]}"
if [ ! -f "${VIDEO_PATH}" ]; then
    echo "ERROR: Video file not found: ${VIDEO_PATH}"
    exit 1
fi
STATIC_FLAG=""
if [[ "${2:-}" == "--static" || "${2:-}" == "-s" ]]; then
    STATIC_FLAG="-s"
    echo "Static camera mode (skipping visual odometry)"
fi

GVHMR_DIR="${HOME}/gvhmr"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "╔══════════════════════════════════════════╗"
echo "║  GVHMR Inference                         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Video: ${VIDEO_PATH}"
echo "  Output: ${RESULTS_DIR}/"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate gvhmr

# ─── Step 1: Run GVHMR ────────────────────────────────────
echo ""
echo "▸ Step 1/3: Running GVHMR inference..."
cd "${GVHMR_DIR}"

VIDEO_BASENAME=$(basename "${VIDEO_PATH}" .mp4)
OUTPUT_DIR="${GVHMR_DIR}/outputs/demo/${VIDEO_BASENAME}"

START_TIME=$(date +%s)

python tools/demo/demo.py \
    --video="${VIDEO_PATH}" \
    ${STATIC_FLAG}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "  Inference completed in ${ELAPSED}s"
echo "  Output: ${OUTPUT_DIR}/"

# ─── Step 2: Extract joint positions ──────────────────────
echo ""
echo "▸ Step 2/3: Extracting 3D joint positions..."

# Validate GVHMR produced output
RESULTS_PT="${OUTPUT_DIR}/hmr4d_results.pt"
if [ ! -f "${RESULTS_PT}" ]; then
    echo "ERROR: GVHMR did not produce output: ${RESULTS_PT}"
    echo "  Check GPU memory and logs above."
    exit 1
fi

# Detect video FPS via ffprobe
VIDEO_FPS=$(ffprobe -v quiet -select_streams v -of csv=p=0 -show_entries stream=r_frame_rate "${VIDEO_PATH}" | head -1 | bc -l 2>/dev/null || echo "30")
VIDEO_FPS=$(printf "%.0f" "${VIDEO_FPS}")
echo "  Detected FPS: ${VIDEO_FPS}"

cd "${SCRIPT_DIR}"
VIDEO_BASENAME="${VIDEO_BASENAME}" RESULTS_DIR="${RESULTS_DIR}" GVHMR_DIR="${GVHMR_DIR}" \
    python remote/extract-joints.py "${VIDEO_BASENAME}" --fps "${VIDEO_FPS}"

# ─── Step 3: Copy rendered video ──────────────────────────
echo ""
echo "▸ Step 3/3: Copying rendered output..."
if [ -f "${OUTPUT_DIR}/incam_global.mp4" ]; then
    cp "${OUTPUT_DIR}/incam_global.mp4" "${RESULTS_DIR}/rendered_${VIDEO_BASENAME}.mp4"
    echo "  Saved rendered video: rendered_${VIDEO_BASENAME}.mp4"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Inference complete!                     ║"
echo "║                                          ║"
echo "║  Results in: ${RESULTS_DIR}/"
echo "║  Next: python analyze.py                 ║"
echo "╚══════════════════════════════════════════╝"
