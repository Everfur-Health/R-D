#!/bin/bash
# =============================================================================
# CANINE HEALTH AI — COMPLETE RUN ORDER  (v4 — CUDA 12.8)
# =============================================================================
#
# Six diagnostic modules — three need training, three need none:
#
#   Module          Model               Train time
#   Eye Segment     SegFormer-B2        25-40 min
#   Eye Detect      YOLOv8m             15-25 min
#   Skin Disease    EfficientNet-B4     15-20 min
#   Breed ID        EfficientNet-B4     40-55 min
#   Pain/Distress   EfficientNet-B4     10-15 min
#   Heart Rate      rPPG / POS algo     NO TRAINING NEEDED
#   Gait/Lameness   YOLOv8-pose         NO TRAINING NEEDED
#
# NOTE: MMPose and MediaPipe are NOT used and NOT required.
# The gait module uses ultralytics YOLOv8-pose only.
# mmcv has no working wheels for CUDA 12.8 — do not install it.
#
# DATASET PATHS:
#   ~/Documents/datasets/Pain/     Dog Emotion (angry/happy/relaxed/sad)
#   ~/Documents/datasets/Eye/      DogEyeSeg4 + dog-diseases-9class
#   ~/Documents/datasets/skin/     dogs-skin-disease-dataset (Kaggle)
#   ~/Documents/datasets/breed/    stanford-dogs-dataset (Kaggle)
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Canine Health AI — Complete Run Guide  (v4)             ║"
echo "║  CUDA 12.8 · PyTorch 2.6 · YOLOv8-pose       ║"
echo "╚══════════════════════════════════════════════════════════╝"

source /opt/video_image_health_detection/venv/bin/activate
cd ~/canine-demo

echo ""
echo "▸ Verifying GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  GPU  : {torch.cuda.get_device_name(0)}')
print(f'  CUDA : {torch.version.cuda}')
print(f'  Torch: {torch.__version__}')
"

echo ""
echo "▸ Installing required packages..."
pip install --quiet \
    transformers timm ultralytics scipy matplotlib \
    seaborn tqdm scikit-learn fastapi uvicorn \
    python-multipart kaggle albumentations

python -c "from ultralytics import YOLO; print('  ultralytics OK')"
python -c "from transformers import SegformerForSemanticSegmentation; print('  transformers OK')"
python -c "import timm; print('  timm OK')"
python -c "import scipy; print('  scipy OK')"

echo ""
echo "▸ Verifying dataset paths..."
python - << 'PYEOF'
from pathlib import Path
DATASETS = {
    "Pain":  "~/Documents/everfur/datasets/Pain",
    "Eye":   "~/Documents/everfur/datasets/Eye",
    "Skin":  "~/Documents/everfur/datasets/skin",
    "Breed": "~/Documents/everfur/datasets/breed",
}
all_ok = True
for label, path in DATASETS.items():
    p = Path(path).expanduser()
    if p.exists():
        imgs = list(p.rglob('*.jpg')) + list(p.rglob('*.png'))
        print(f"  OK  {label}: {len(imgs)} images")
    else:
        print(f"  MISSING  {label}: {p}")
        all_ok = False
if not all_ok:
    print("\nMissing datasets. Run:  bash setup/download_data.sh ~/Documents/datasets")
    import sys; sys.exit(1)
PYEOF

echo ""
echo "=================================================="
echo "  TRAINING"
echo "=================================================="

echo ""
echo "STEP 1 — Pain/Distress classifier  (~10-15 min)"
if [ -f "models/pain/best_model.pth" ]; then
    echo "  Skipping — already trained."
else
    python modules/pain/train.py \
        --data ~/Documents/everfur/datasets/Pain \
        --mode binary --epochs 40 --batch_size 32
fi

echo ""
echo "STEP 2 — Eye segmentation  (~25-40 min)"
if [ -d "models/eye_seg/best_model" ]; then
    echo "  Skipping — already trained."
else
    python modules/eye/train_seg.py \
        --data ~/Documents/everfur/datasets/Eye/DogEyeSeg4_dataset2 \
        --epochs 50 --batch_size 8
fi

echo ""
echo "STEP 3 — Eye YOLO detector  (~15-25 min)"
if [ -f "models/eye_yolo/best.pt" ]; then
    echo "  Skipping — already trained."
else
    python modules/eye/train_yolo.py \
        --data ~/Documents/everfur/datasets/Eye/eye_part2/dog-diseases-9class \
        --epochs 50
fi

echo ""
echo "STEP 4 — Skin disease classifier  (~15-20 min)"
echo "  NOTE: If you previously trained and see '100% confidence / dogs-skin-disease-dataset'"
echo "  as the diagnosis, that model was trained on the wrong nested path."
echo "  Delete models/skin/best_model.pth and this step will retrain correctly."
if [ -f "models/skin/best_model.pth" ]; then
    echo "  Skipping — already trained. Delete models/skin/best_model.pth to retrain."
else
    python modules/skin/train.py \
        --data ~/Documents/everfur/datasets/skin \
        --epochs 50 --batch_size 32
fi

echo ""
echo "STEP 5 — Breed identification  (~40-55 min)"
if [ -f "models/breed/best_model.pth" ]; then
    echo "  Skipping — already trained."
else
    python modules/breed/train.py \
        --data ~/Documents/everfur/datasets/breed \
        --epochs 60 --batch_size 48
fi

echo ""
echo "=================================================="
echo "  MODEL VERIFICATION"
echo "=================================================="
python - << 'PYEOF'
from pathlib import Path
models = {
    "Pain":    "models/pain/best_model.pth",
    "Eye Seg": "models/eye_seg/best_model",
    "Eye YOLO":"models/eye_yolo/best.pt",
    "Skin":    "models/skin/best_model.pth",
    "Breed":   "models/breed/best_model.pth",
}
missing = []
for name, path in models.items():
    p = Path(path)
    exists = p.exists()
    status = "OK  " if exists else "MISS"
    print(f"  [{status}] {name}: {path}")
    if not exists:
        missing.append(name)

print("  [OK  ] Heart Rate: no checkpoint (rPPG algorithm)")
print("  [OK  ] Gait:       no checkpoint (YOLOv8-pose auto-downloads)")

if missing:
    print(f"\n  {len(missing)} model(s) missing: {missing}")
else:
    print("\n  All models ready!")
PYEOF

echo ""
echo "▸ Quick sanity check..."
python - << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from serving.inference import get_skin_model, get_pain_model, get_breed_model
for name, fn, path in [
    ("Skin",  get_skin_model,  "models/skin/best_model.pth"),
    ("Pain",  get_pain_model,  "models/pain/best_model.pth"),
    ("Breed", get_breed_model, "models/breed/best_model.pth"),
]:
    if Path(path).exists():
        try:
            m = fn()
            cls = m['classes']
            n = len(cls)
            print(f"  OK  {name}: {n} classes: {cls[:4]}{'...' if n > 4 else ''}")
        except Exception as e:
            print(f"  WARN {name}: {e}")
PYEOF

echo ""
echo "=================================================="
echo "  LAUNCHING SERVER"
echo "=================================================="
echo ""
echo "  Open: http://localhost:8000"
echo ""
echo "  Tabs:"
echo "    Eye Disease    image -> segmentation overlay"
echo "    Skin Disease   image -> GradCAM heatmap"
echo "    Breed ID       image -> top-5 breeds"
echo "    Pain Detection image -> DogFACS expression analysis"
echo "    Heart Rate     video -> rPPG BPM (no training)"
echo "    Gait Analysis  video -> lameness index (no training)"
echo ""
echo "  Ctrl+C to stop."
echo ""

uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
