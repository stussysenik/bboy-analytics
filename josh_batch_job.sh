#!/usr/bin/env bash
set -euo pipefail

# JOSH Batch Job — runs DECO → JOSH inference → Aggregation → Joint extraction
# Supports resumable stages: bash josh_batch_job.sh [all|inference|aggregate|extract]
#
# IMPORTANT: Use THIS script, NOT run_remaining.sh or run_remaining2.sh.
# This script has checkpoint/resume logic (skips DECO if done, resumes inference).
#
# Submit to Lightning L4:
#   lightning run job --name josh-inference-v6 --machine L4 --studio gvhmr \
#     --command "bash /teamspace/studios/this_studio/josh_batch_job.sh inference"
#
# Stages:
#   all       — run everything (DECO + inference + aggregate + extract)
#   inference — skip DECO, run inference + aggregate + extract
#   aggregate — skip inference, just aggregate + extract
#   extract   — skip everything, just extract joints from existing chunks

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INPUT=/teamspace/studios/this_studio/josh_input/bcone_seq4
JOSH_DIR=/teamspace/studios/this_studio/josh
STAGE=${1:-all}

echo "=========================================="
echo "  JOSH Batch Job v4 (bboy-tuned) — $(date)"
echo "  Stage: $STAGE"
echo "  Config: $(python3 -c 'from josh.josh.config import BBOY_PRESET; print(BBOY_PRESET)' 2>/dev/null || echo 'bboy-tuned')"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo "=========================================="

# Verify prerequisites exist
echo "Checking prerequisites..."
echo "  Masks: $(ls $INPUT/mask/*.json 2>/dev/null | wc -l)/999"
echo "  TRAM tracks: $(ls $INPUT/tram/*.npy 2>/dev/null | wc -l)"
echo "  DECO: $(ls $INPUT/deco/*.npy 2>/dev/null | wc -l)"
echo "  VIMO ckpt: $(ls -lh $JOSH_DIR/data/checkpoints/vimo_checkpoint.pth.tar 2>/dev/null | awk '{print $5}' || echo 'MISSING')"
echo "  DECO ckpt: $(ls -lh $JOSH_DIR/data/checkpoints/deco_best.pth 2>/dev/null | awk '{print $5}' || echo 'MISSING')"
echo "  SMPL: $(ls $JOSH_DIR/data/smpl/SMPL_NEUTRAL.pkl 2>/dev/null && echo 'OK' || echo 'MISSING')"
echo "  Existing chunks: $(ls -d $INPUT/josh_*/scene.pkl 2>/dev/null | wc -l) with scene.pkl"

cd "$JOSH_DIR"

# DECO (skip if already done)
if [ "$STAGE" = "all" ]; then
    DECO_COUNT=$(ls $INPUT/deco/*.npy 2>/dev/null | wc -l)
    if [ "$DECO_COUNT" -ge 1 ]; then
        echo ""
        echo "[SKIP] DECO already done ($DECO_COUNT files)"
    else
        echo ""
        echo "[3/5] DECO contact estimation..."
        python -m preprocess.run_deco --input_folder "$INPUT"
        echo "DECO done"
    fi
fi

# JOSH inference (skip-if-done per chunk is handled in inference_long_video.py)
if [ "$STAGE" = "inference" ]; then
    DECO_COUNT=$(ls $INPUT/deco/*.npy 2>/dev/null | wc -l)
    if [ "$DECO_COUNT" -lt 1 ]; then
        echo "ERROR: DECO outputs missing. Run 'all' stage first." >&2
        exit 1
    fi
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "inference" ]; then
    echo ""
    echo "[4/5] JOSH inference (chunk processing for $(ls $INPUT/rgb/*.jpg | wc -l) frames)..."
    START_T=$(date +%s)
    python josh/inference_long_video.py --input_folder "$INPUT"
    echo "JOSH inference done in $(($(date +%s) - START_T))s"
fi

# Aggregation (also runs after inference stage)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "inference" ] || [ "$STAGE" = "aggregate" ]; then
    echo ""
    echo "[4.5/5] JOSH aggregation..."
    python josh/aggregate_results.py --input_folder "$INPUT"
    echo "Aggregation done"
fi

# Joint extraction (also runs after inference stage)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "inference" ] || [ "$STAGE" = "aggregate" ] || [ "$STAGE" = "extract" ]; then
    echo ""
    echo "[5/5] Joint extraction..."
    cd /teamspace/studios/this_studio
    python poc/remote/extract-joints-josh.py \
      --josh-dir josh_input/bcone_seq4 \
      --body-model-path gvhmr_src/inputs/checkpoints/body_models \
      --fps 30
fi

echo ""
echo "=========================================="
echo "  BATCH JOB COMPLETE — $(date)"
echo "  Stage: $STAGE"
echo "=========================================="
ls -lh $INPUT/joints_3d_josh.npy 2>/dev/null || echo "JOINTS NOT YET GENERATED (run 'extract' stage)"
ls -d $INPUT/josh_*/scene.pkl 2>/dev/null | wc -l | xargs -I{} echo "  Chunks with scene.pkl: {}"
