#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INPUT=/teamspace/studios/this_studio/josh_input/bcone_seq4
cd /teamspace/studios/this_studio/josh

echo "=== Finishing SAM3 (remaining frames) ==="
python -m preprocess.run_sam3 --input_folder "$INPUT" --chunk_size 15
echo "Masks: $(ls "$INPUT/mask/"*.json 2>/dev/null | wc -l)/999"

echo ""
echo "=== [2/5] TRAM ==="
python -m preprocess.run_tram --input_folder "$INPUT"

echo ""
echo "=== [3/5] DECO ==="
python -m preprocess.run_deco --input_folder "$INPUT"

echo ""
echo "=== [4/5] JOSH inference ==="
python josh/inference_long_video.py --input_folder "$INPUT"
python josh/aggregate_results.py --input_folder "$INPUT"

echo ""
echo "=== [5/5] Joint extraction ==="
cd /teamspace/studios/this_studio
python poc/remote/extract-joints-josh.py \
  --josh-dir josh_input/bcone_seq4 \
  --body-model-path gvhmr_src/inputs/checkpoints/body_models \
  --fps 30

echo ""
echo "=== PIPELINE COMPLETE ==="
ls -lh josh_input/bcone_seq4/joints_3d_josh.npy 2>/dev/null || echo "JOINTS MISSING"
