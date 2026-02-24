#!/bin/bash
# =============================================================================
# Canine Health AI Demo — Environment Setup
# GPU: Blackwell GB202 — requires CUDA 12.8+ and PyTorch 2.6+
# =============================================================================
set -e

echo "============================================================"
echo " Canine Health AI Demo — Install"
echo " PyTorch 2.6+ with CUDA 12.8"
echo "============================================================"

# --- Python / venv ---
VENV=/opt/video_image_health_detection/venv
if [ ! -d "$VENV" ]; then
    echo "[1/6] Creating virtual environment at $VENV"
    sudo mkdir -p /opt/video_image_health_detection
    sudo python3 -m venv $VENV
    sudo chown -R $USER:$USER /opt/video_image_health_detection
fi
source $VENV/bin/activate

# --- Detect CUDA version ---
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
echo "[2/6] Detected CUDA: $CUDA_VER"

# --- PyTorch 2.6 with CUDA 12.8 ---
echo "[3/6] Installing PyTorch 2.6 + CUDA 12.8"
pip install --upgrade pip wheel
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Verify GPU
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- Core ML packages ---
echo "[4/6] Installing core ML packages"
pip install \
    timm==1.0.3 \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.34.0

# --- Computer vision / detection ---
pip install \
    ultralytics==8.3.2 \
    opencv-python==4.10.0.84 \
    albumentations==1.4.18 \
    Pillow==10.4.0

# --- Signal processing (for rPPG heart rate) ---
pip install \
    scipy==1.14.1 \
    heartpy==1.2.7

# --- API / serving ---
pip install \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    python-multipart==0.0.9

# --- Utilities ---
pip install \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    tqdm==4.66.5 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    kaggle==1.6.17

echo "[5/6] All packages installed"

# --- Test quick smoke test ---
python3 -c "
from ultralytics import YOLO
import torch
import timm
from transformers import SegformerForSemanticSegmentation
print('  ultralytics OK')
print('  timm OK')
print('  transformers OK')
print()
print('ALL GOOD — ready to train!')
"

echo ""
echo "============================================================"
echo " Install complete. Activate env with:"
echo "   source $VENV/bin/activate"
echo "============================================================"
