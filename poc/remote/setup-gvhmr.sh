#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GVHMR Setup Script — Run on GPU instance
# Installs GVHMR + all dependencies + downloads checkpoints
# Expected time: ~15 minutes on first run
# ============================================================

GVHMR_DIR="${HOME}/gvhmr"
CHECKPOINT_DIR="${GVHMR_DIR}/inputs/checkpoints"

echo "╔══════════════════════════════════════════╗"
echo "║  GVHMR Setup for Bboy POC               ║"
echo "╚══════════════════════════════════════════╝"

# ─── Step 0: Check disk space ─────────────────────────────
echo ""
echo "▸ Pre-check: Disk space..."
AVAIL_KB=$(df "${HOME}" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
echo "  Available: ${AVAIL_GB} GB"
if [ "${AVAIL_GB}" -lt 20 ]; then
    echo "ERROR: Need at least 20GB free (have ${AVAIL_GB}GB). GVHMR models are ~15GB."
    exit 1
fi

# ─── Step 1: Check GPU ─────────────────────────────────────
echo ""
echo "▸ Step 1/6: Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires a CUDA GPU."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU: ${GPU_NAME} (${GPU_MEM} MB)"

if [ "${GPU_MEM}" -lt 15000 ]; then
    echo "  WARNING: ${GPU_MEM}MB VRAM is below recommended 16GB. May OOM on long clips."
fi

# Check CUDA version compatibility
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "  CUDA: ${CUDA_VER}"
    if [[ "${CUDA_VER}" != 12.* ]]; then
        echo "  WARNING: CUDA ${CUDA_VER} detected but PyTorch wheels target CUDA 12.1."
        echo "  This may cause runtime errors. See https://pytorch.org/get-started/locally/"
    fi
fi

# ─── Step 2: Create conda environment ──────────────────────
echo ""
echo "▸ Step 2/6: Setting up Python environment..."
if ! command -v conda &>/dev/null; then
    echo "  conda not found. Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    conda init bash
    source "${HOME}/.bashrc" 2>/dev/null || true
fi

if conda env list | grep -q "gvhmr"; then
    echo "  Environment 'gvhmr' already exists. Activating..."
else
    echo "  Creating conda environment 'gvhmr' (Python 3.10)..."
    conda create -y -n gvhmr python=3.10
fi

# Activate in a way that works in scripts
eval "$(conda shell.bash hook)"
conda activate gvhmr

echo "  Python: $(python --version)"
echo "  Location: $(which python)"

# ─── Step 3: Clone and install GVHMR ──────────────────────
echo ""
echo "▸ Step 3/6: Installing GVHMR..."
if [ -d "${GVHMR_DIR}" ]; then
    echo "  GVHMR directory exists. Pulling latest..."
    cd "${GVHMR_DIR}" && git pull 2>/dev/null || true
else
    echo "  Cloning GVHMR..."
    git clone https://github.com/zju3dv/GVHMR "${GVHMR_DIR}"
fi

cd "${GVHMR_DIR}"

# Install PyTorch first (CUDA 12.1)
echo "  Installing PyTorch..."
pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install pytorch3d (the notorious one)
echo "  Installing pytorch3d (this may take a few minutes)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git" || {
    echo "  pytorch3d from git failed. Trying conda..."
    conda install -y -c pytorch3d pytorch3d || {
        echo "  Trying pre-built wheel..."
        pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html || {
            echo "ERROR: pytorch3d installation failed. GVHMR cannot run without it."
            exit 1
        }
    }
}

# Install remaining requirements
echo "  Installing requirements..."
pip install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt
pip install -q -e .

# ─── Step 4: Download checkpoints ─────────────────────────
echo ""
echo "▸ Step 4/6: Downloading checkpoints..."
mkdir -p "${CHECKPOINT_DIR}"

# GVHMR model
if [ ! -f "${CHECKPOINT_DIR}/gvhmr/gvhmr_siga24_release.ckpt" ]; then
    echo "  Downloading GVHMR checkpoint..."
    mkdir -p "${CHECKPOINT_DIR}/gvhmr"
    # The actual download URL — check GVHMR README for current links
    echo "  NOTE: GVHMR checkpoints require manual download from Google Drive."
    echo "  See: https://github.com/zju3dv/GVHMR#getting-started"
    echo "  Place files in: ${CHECKPOINT_DIR}/"
    echo ""
    echo "  Required checkpoints:"
    echo "    - gvhmr/gvhmr_siga24_release.ckpt"
    echo "    - hmr2/epoch=10-step=25000.ckpt"
    echo "    - vitpose/vitpose-h-multi-coco.pth"
    echo "    - yolo/yolov8x.pt"
    echo ""
    echo "  If gdown is available, trying automatic download..."
    pip install -q gdown 2>/dev/null
    python -c "
import subprocess, sys
try:
    subprocess.run([sys.executable, 'tools/download_checkpoints.py'], check=True)
    print('  Checkpoints downloaded successfully.')
except Exception as e:
    print(f'  Auto-download failed: {e}')
    print('  Please download manually from the GVHMR README.')
" || echo "  Auto-download not available. Manual download needed."

    # Validate critical checkpoints exist
    if [ ! -f "${CHECKPOINT_DIR}/gvhmr/gvhmr_siga24_release.ckpt" ]; then
        echo ""
        echo "ERROR: GVHMR checkpoint not found after download attempt."
        echo "  Expected: ${CHECKPOINT_DIR}/gvhmr/gvhmr_siga24_release.ckpt"
        echo "  Download manually from: https://github.com/zju3dv/GVHMR#getting-started"
        exit 1
    fi
else
    echo "  Checkpoints already present."
fi

# ─── Step 5: Setup SMPL body model ────────────────────────
echo ""
echo "▸ Step 5/6: Setting up SMPL body model..."
SMPL_DIR="${CHECKPOINT_DIR}/body_models/smplx"
mkdir -p "${SMPL_DIR}"

if [ ! -f "${SMPL_DIR}/SMPLX_NEUTRAL.npz" ]; then
    echo "  SMPL model not found."
    echo "  You need to register at https://smplx.is.tue.mpg.de/"
    echo "  Then download SMPLX_NEUTRAL.npz and place it in:"
    echo "    ${SMPL_DIR}/"
    echo ""
    echo "  Alternatively, if you have it locally, scp it:"
    echo "    scp SMPLX_NEUTRAL.npz user@this-machine:${SMPL_DIR}/"
else
    echo "  SMPL model found."
fi

# ─── Step 6: Verify installation ──────────────────────────
echo ""
echo "▸ Step 6/6: Verifying installation..."
python -c "
import torch
import sys

print(f'  Python: {sys.version}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

try:
    import pytorch3d
    print(f'  pytorch3d: OK')
except ImportError:
    print(f'  pytorch3d: MISSING (critical)')

try:
    import smplx
    print(f'  smplx: OK')
except ImportError:
    print(f'  smplx: MISSING')
"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup complete!                         ║"
echo "║                                          ║"
echo "║  Next: bash remote/run-inference.sh <video>"
echo "╚══════════════════════════════════════════╝"
