#!/usr/bin/env bash
set -euo pipefail

# Full JOSH pipeline: SAM3 → TRAM → DECO → JOSH → Extract joints
# Run in tmux: tmux new -s josh 'bash run_josh_pipeline.sh'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INPUT=/teamspace/studios/this_studio/josh_input/bcone_seq4
JOSH_DIR=/teamspace/studios/this_studio/josh
cd "$JOSH_DIR"

echo "=========================================="
echo "  JOSH Pipeline — $(date)"
echo "  Input: $INPUT"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Step 1: SAM3 segmentation (15-frame chunks for 24GB VRAM)
echo ""
echo "[1/5] SAM3 segmentation..."
START=$(date +%s)
python -m preprocess.run_sam3 --input_folder "$INPUT" --chunk_size 15
echo "  SAM3 done in $(($(date +%s) - START))s"
echo "  Masks: $(ls "$INPUT/mask/"*.json 2>/dev/null | wc -l) JSONs"

# Step 2: TRAM (HMR pose estimation)
echo ""
echo "[2/5] TRAM pose estimation..."
START=$(date +%s)
python -m preprocess.run_tram --input_folder "$INPUT"
echo "  TRAM done in $(($(date +%s) - START))s"
echo "  Tracks: $(ls "$INPUT/tram/"*.npy 2>/dev/null | wc -l)"

# Step 3: DECO (contact estimation)
echo ""
echo "[3/5] DECO contact estimation..."
START=$(date +%s)
python -m preprocess.run_deco --input_folder "$INPUT"
echo "  DECO done in $(($(date +%s) - START))s"

# Step 4: JOSH inference (chunk-based for >=200 frames)
echo ""
echo "[4/5] JOSH inference (chunk processing)..."
START=$(date +%s)
python josh/inference_long_video.py --input_folder "$INPUT"
python josh/aggregate_results.py --input_folder "$INPUT"
echo "  JOSH done in $(($(date +%s) - START))s"

# Step 5: Extract joints
echo ""
echo "[5/5] Joint extraction..."
cd /teamspace/studios/this_studio
python poc/remote/extract-joints-josh.py \
  --josh-dir josh_input/bcone_seq4 \
  --body-model-path gvhmr_src/inputs/checkpoints/body_models \
  --fps 30

echo ""
echo "=========================================="
echo "  PIPELINE COMPLETE — $(date)"
echo "=========================================="
echo "  Joints: $(ls josh_input/bcone_seq4/joints_3d_josh.npy 2>/dev/null && echo 'OK' || echo 'MISSING')"
