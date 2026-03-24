#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# JOSH Installation Script
# Installs JOSH (ICLR 2026) in a separate environment.
# Requires: GPU with 24GB VRAM, CUDA 12.x
#
# Usage: bash poc/remote/setup-josh.sh
# ============================================================

echo "╔══════════════════════════════════════════╗"
echo "║  JOSH Setup                                ║"
echo "╚══════════════════════════════════════════╝"

# ─── Pre-checks ──────────────────────────────────────────
echo ""
echo "▸ Pre-flight checks..."

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: ${GPU_NAME} (${GPU_MEM})"
else
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
    echo "  JOSH requires a GPU with 24GB VRAM (RTX 4090 / A10G / A100)."
    echo "  Continuing anyway — will fail at inference if no GPU."
fi

# CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}')
    echo "  CUDA: ${CUDA_VER}"
else
    echo "  CUDA toolkit not found (nvcc). PyTorch will use its bundled CUDA."
fi

# Disk
DISK_FREE=$(df -h ~ | awk 'NR==2{print $4}')
echo "  Disk free: ${DISK_FREE}"

# ─── Environment setup ──────────────────────────────────
echo ""
echo "▸ Setting up Python environment..."

# Try conda first (preferred)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"

    # Check if josh env already exists
    if conda env list | grep -q "josh"; then
        echo "  Conda env 'josh' already exists. Activating..."
        conda activate josh
    else
        echo "  Creating conda env 'josh' (python 3.10)..."
        if conda create -y -n josh python=3.10 2>/dev/null; then
            conda activate josh
            echo "  Conda env 'josh' created and activated."
        else
            echo "  Conda env creation failed (Studio may limit envs)."
            echo "  Falling back to pip install in current env."
            echo "  WARNING: This may conflict with existing packages."
        fi
    fi
else
    echo "  Conda not available. Using current Python environment."
fi

python --version
echo "  Python: $(which python)"

# ─── Clone JOSH ──────────────────────────────────────────
JOSH_DIR="${HOME}/josh"

if [ -d "${JOSH_DIR}" ]; then
    echo ""
    echo "▸ JOSH already cloned at ${JOSH_DIR}"
    cd "${JOSH_DIR}"
    git pull --rebase 2>/dev/null || true
else
    echo ""
    echo "▸ Cloning JOSH..."
    git clone --recursive https://github.com/genforce/JOSH "${JOSH_DIR}"
    cd "${JOSH_DIR}"
fi

# Ensure submodules are initialized
echo "▸ Initializing submodules..."
git submodule update --init --recursive

# ─── Install PyTorch ─────────────────────────────────────
echo ""
echo "▸ Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ─── Install JOSH dependencies ──────────────────────────
echo ""
echo "▸ Installing JOSH dependencies..."

# Check for requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "  Installed requirements.txt"
fi

# Install smplx for body model
pip install smplx

# chumpy needs special handling
pip install --no-build-isolation git+https://github.com/mattloper/chumpy 2>/dev/null || \
    echo "  WARNING: chumpy install failed (may not be needed)"

# Install submodule dependencies
echo ""
echo "▸ Installing submodule dependencies..."

# MASt3R
if [ -d "third_party/mast3r" ] || [ -d "mast3r" ]; then
    MAST3R_DIR=$(find . -maxdepth 2 -name "mast3r" -type d | head -1)
    if [ -n "${MAST3R_DIR}" ] && [ -f "${MAST3R_DIR}/requirements.txt" ]; then
        pip install -r "${MAST3R_DIR}/requirements.txt" 2>/dev/null || true
        echo "  Installed MASt3R requirements"
    fi
fi

# TRAM
if [ -d "third_party/tram" ] || [ -d "tram" ]; then
    TRAM_DIR=$(find . -maxdepth 2 -name "tram" -type d | head -1)
    if [ -n "${TRAM_DIR}" ] && [ -f "${TRAM_DIR}/requirements.txt" ]; then
        pip install -r "${TRAM_DIR}/requirements.txt" 2>/dev/null || true
        echo "  Installed TRAM requirements"
    fi
fi

# DECO
if [ -d "third_party/deco" ] || [ -d "deco" ]; then
    DECO_DIR=$(find . -maxdepth 2 -name "deco" -type d | head -1)
    if [ -n "${DECO_DIR}" ] && [ -f "${DECO_DIR}/requirements.txt" ]; then
        pip install -r "${DECO_DIR}/requirements.txt" 2>/dev/null || true
        echo "  Installed DECO requirements"
    fi
fi

# ─── SMPL body model ────────────────────────────────────
echo ""
echo "▸ Setting up SMPL body model..."

GVHMR_MODELS="${HOME}/gvhmr/inputs/checkpoints/body_models"
JOSH_DATA="${JOSH_DIR}/data"

mkdir -p "${JOSH_DATA}"

if [ -d "${GVHMR_MODELS}" ]; then
    # Symlink from GVHMR
    if [ ! -e "${JOSH_DATA}/body_models" ]; then
        ln -s "${GVHMR_MODELS}" "${JOSH_DATA}/body_models"
        echo "  Symlinked body_models from GVHMR: ${GVHMR_MODELS}"
    else
        echo "  body_models already exists at ${JOSH_DATA}/body_models"
    fi

    # JOSH may expect data/smpl/ specifically
    if [ ! -e "${JOSH_DATA}/smpl" ]; then
        if [ -d "${GVHMR_MODELS}/smpl" ]; then
            ln -s "${GVHMR_MODELS}/smpl" "${JOSH_DATA}/smpl"
            echo "  Symlinked smpl/ from GVHMR"
        else
            echo "  WARNING: JOSH needs SMPL .pkl files at ${JOSH_DATA}/smpl/"
            echo "  GVHMR only has SMPLX. You may need to download SMPL from:"
            echo "    https://smpl.is.tue.mpg.de/"
            echo "  Place SMPL_NEUTRAL.pkl in ${JOSH_DATA}/smpl/"
        fi
    fi
else
    echo "  WARNING: GVHMR body models not found at ${GVHMR_MODELS}"
    echo "  Download SMPL model from https://smpl.is.tue.mpg.de/"
    echo "  Place files in ${JOSH_DATA}/smpl/"
fi

# ─── Download checkpoints ───────────────────────────────
echo ""
echo "▸ Downloading checkpoints..."
echo "  NOTE: JOSH may have its own checkpoint download script."
echo "  Check the README for download links."

# Look for download script
if [ -f "download_data.sh" ]; then
    echo "  Found download_data.sh — running it..."
    bash download_data.sh || echo "  WARNING: download_data.sh had errors"
elif [ -f "scripts/download.sh" ]; then
    bash scripts/download.sh || echo "  WARNING: download script had errors"
else
    echo "  No automatic download script found."
    echo "  Check ~/josh/README.md for checkpoint download instructions."
fi

# ─── Verify ──────────────────────────────────────────────
echo ""
echo "▸ Verification..."
python -c "
import torch
import smplx
print('  torch:', torch.__version__)
print('  smplx: OK')
print('  CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  JOSH setup complete!                      ║"
echo "║                                            ║"
echo "║  Next: bash poc/remote/run-josh-inference.sh║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  JOSH directory: ${JOSH_DIR}"
echo "  README: ${JOSH_DIR}/README.md"
ls -la "${JOSH_DIR}/"
