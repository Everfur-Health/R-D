
# Canine Health AI Demo — Complete Guide

## Architecture Overview

```
5 Diagnostic Modules:
┌──────────────────┬─────────────────────────────────────────────────┐
│ Module           │ Architecture + Dataset                          │
├──────────────────┼─────────────────────────────────────────────────┤
│ Eye Disease      │ SegFormer-B2 → DogEyeSeg4 (segmentation)        │
│                  │ YOLOv8m → dog-diseases-9class (detection)       │
├──────────────────┼─────────────────────────────────────────────────┤
│ Skin Disease     │ EfficientNet-B4 → Kaggle dog skin (4 classes)   │
├──────────────────┼─────────────────────────────────────────────────┤
│ Breed ID         │ EfficientNet-B4 → Stanford Dogs (120 breeds)    │
├──────────────────┼─────────────────────────────────────────────────┤
│ Pain Detection   │ EfficientNet-B4 + GradCAM → Dog Emotion (binary)│
├──────────────────┼─────────────────────────────────────────────────┤
│ Heart Rate       │ POS rPPG algorithm (no ML training needed)      │
└──────────────────┴─────────────────────────────────────────────────┘
```

---

## Prerequisites

### GPU: Blackwell GB202
 **CUDA 12.8+** and **PyTorch 2.6+**.
Make sure you have the correct CUDA toolkit:
```bash
nvcc --version   # should show 12.8+
nvidia-smi       
```

### Kaggle API Key (for dataset downloads)
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Place `kaggle.json` at `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`

---

## Step-by-Step Run Order

### STEP 0 — Install Environment
```bash
bash setup/install.sh
source /opt/video_image_health_detection/venv/bin/activate
```
Expected time: ~5 minutes

### STEP 1 — Download Datasets
```bash
bash setup/download_data.sh ~/Documents/everfur/datasets
```
Downloads:
- `dogs-skin-disease-dataset` (Kaggle, ~50MB)
- `stanford-dogs-dataset` (Kaggle, ~800MB)

Your existing datasets (Eye/, Pain/, Arthritis/) are auto-detected.

### STEP 2 — Verify Datasets
```bash
python setup/verify_datasets.py --root ~/Documents/everfur/datasets
```
Shows class counts, image totals, structure check.
Review `datasets_report.json` to understand class balance before training.

### STEP 3 — Train Eye Disease Segmentation
```bash
# SegFormer-B2 on DogEyeSeg4
python modules/eye/train_seg.py \
  --data ~/Documents/everfur/datasets/Eye \
  --epochs 50 \
  --batch_size 8

# YOLOv8m on dog-diseases-9class
python modules/eye/train_yolo.py \
  --data ~/Documents/everfur/datasets/Eye/eye_part2/dog-diseases-9class \
  --epochs 50
```
Expected time: 25-40 min (seg) + 15-25 min (YOLO)
Expected mIoU: 0.45-0.65 (disease classes), YOLO mAP50: 0.5-0.75

### STEP 4 — Train Skin Disease Classifier
```bash
python modules/skin/train.py \
  --data ~/Documents/everfur/datasets/skin \
  --epochs 50 \
  --batch_size 32
```
Expected time: 10-20 min
Expected val accuracy: 75-88%

### STEP 5 — Train Breed Classifier
```bash
python modules/breed/train.py \
  --data ~/Documents/everfur/datasets/breed \
  --epochs 60 \
  --batch_size 48
```
Expected time: 30-60 min (120 classes × ~180 images)
Expected val accuracy: 70-85% (fine-grained task)

### STEP 6 — Train Pain Detector
```bash
python modules/pain/train.py \
  --data ~/Documents/everfur/datasets/Pain \
  --mode binary \
  --epochs 40
```
Expected time: 10-20 min
Expected val accuracy: 75-88%

### STEP 7 — Heart Rate (no training needed)
```bash
# Test on a video:
python modules/heart_rate/analyze.py --video path/to/dog_video.mp4
```
Pure signal processing — works immediately, no training.

### STEP 8 — Launch Demo Server
```bash
cd ~/canine-demo
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```
Open browser: http://localhost:8000

---

## Model Outputs

All trained models are saved to:
```
models/
  eye_seg/best_model/     ← HuggingFace format (SegFormer)
  eye_yolo/best.pt        ← YOLO weights
  skin/best_model.pth     ← PyTorch checkpoint
  breed/best_model.pth    ← PyTorch checkpoint
  pain/best_model.pth     ← PyTorch checkpoint
```

---

## API Endpoints

| Method | Path | Input | Returns |
|--------|------|-------|---------|
| GET | /api/status | — | Model load status |
| POST | /api/eye/segment | JPG/PNG | Segmentation overlay + diseases |
| POST | /api/eye/detect | JPG/PNG | YOLO bounding boxes |
| POST | /api/skin/predict | JPG/PNG | Disease + GradCAM |
| POST | /api/breed/predict | JPG/PNG | Top-5 breeds |
| POST | /api/pain/predict | JPG/PNG | Calm/distress + GradCAM |
| POST | /api/heartrate/video | MP4/MOV | BPM + PPG plot |

Interactive API docs: http://localhost:8000/docs

---

## Scientific Background

### Eye Disease — SegFormer-B2
- **Why SegFormer over U-Net**: Hierarchical Mix-Transformer backbone captures local texture (corneal cloudiness, vessel patterns) AND global context (anatomical structure) simultaneously. No positional encoding → better generalization across eye sizes/breeds.
- **Why Dice + CE loss**: Background is ~85-90% of pixels. CrossEntropy alone → model predicts all background and gets 88% accuracy but detects nothing. Dice loss directly optimizes IoU regardless of class frequency.
- **Classes**: Corneal Edema (inflamed corneal stroma), Episcleral Congestion (engorged episcleral vessels), Epiphora (tear overflow), Cherry Eye (prolapsed nictitating membrane gland).

### Skin Disease — EfficientNet-B4
- **4 classes**: Hot Spot (acute moist dermatitis, bacterial), Ringworm (dermatophytosis, fungal — contagious to humans!), Flea Allergy Dermatitis, Mange (Sarcoptic/Demodex mite infestation).
- **GradCAM**: Shows which skin regions (texture, color, lesion boundary) drove the prediction. Clinically useful for confirming the model is focusing on the right area.

### Breed — Stanford Dogs 120 Classes
- Fine-grained visual classification: distinguishing Norwegian Elkhound from Siberian Husky requires subtle inter-class differences.
- Two-stage training: head-only warmup (5 epochs) → full fine-tuning with 10× lower backbone LR. Prevents catastrophic forgetting of pre-trained features.
- MixUp (α=0.3) is critical: creates smooth decision boundaries between visually similar breeds.

### Pain Detection — DogFACS
- **DogFACS** (Dog Facial Action Coding System): validated ethogram mapping AU43 (orbital tightening), AU101 (brow lowering) to pain/negative valence.
- **Binary mapping**: angry + sad → distress (low valence, high/low arousal), happy + relaxed → calm. More clinically defensible than 4-class.
- **Limitation**: Emotion proxy, not direct pain labels. Gold standard: pre/post-analgesia comparison with WSAVA/Glasgow pain scores.

### Heart Rate — rPPG / POS Algorithm
- **Principle**: Blood absorbs more green light than surrounding tissue. Each heartbeat → tiny periodic change in RGB color of skin/fur ROI.
- **POS (Plane-Orthogonal-to-Skin)**: Projects the RGB signal onto a plane orthogonal to the skin tone vector in color space. Separates the pulse signal from motion artifacts and lighting changes. More robust than ICA/CHROM for non-ideal conditions.
- **For dogs**: ROI = upper 45% of dog bounding box (face/head). Bandpass: 1.0–3.67 Hz (60–220 BPM). Dog resting HR: 60–120 BPM.
- **Reference**: Hu et al. 2024, *Frontiers in Veterinary Science* — validated on pet dogs and cats using this exact approach.
- **Limitation**: Hair/fur reduces skin pixel fraction vs. human faces. Works best: short fur, close-up, good lighting, dog at rest.

---

## Expected Performance

| Module | Metric | Expected Range |
|--------|--------|----------------|
| Eye Seg (SegFormer) | Disease mIoU | 0.40–0.65 |
| Eye YOLO | mAP50 | 0.50–0.75 |
| Skin | Val accuracy | 75–88% |
| Breed (120 classes) | Top-1 val acc | 65–82% |
| Breed (120 classes) | Top-5 val acc | 87–95% |
| Pain (binary) | Val accuracy | 75–88% |
| Heart Rate | MAE vs ground truth | ~15 BPM (typical) |

Performance depends heavily on dataset quality and size. The skin/emotion datasets
are small (~90-400 images per class), so expect higher variance.

---

## Scaling to Production

### From Demo → Clinical Tool
1. **Eye**: Add more DogEyeSeg4 data. Consider specialist annotation from veterinary ophthalmologists. Add glaucoma (elevated IOP → cupped disc) and uveitis classes.
2. **Skin**: Acquire dermoscopy-quality images. Add: seborrhea, folliculitis, vasculitis, autoimmune conditions.
3. **Breed**: Fine-tune on mixed-breed dogs (currently only pure breeds in Stanford Dogs).
4. **Pain**: Collect pre/post-analgesia video pairs with vet-assigned Glasgow CMPS scores. This eliminates the emotion proxy limitation entirely.
5. **Heart Rate**: Validate against ECG/stethoscope ground truth across breeds, fur colors, and lighting conditions.

### GPU Scaling (Two DGX Sparks)
```
Spark 1 (API + no GPU): uvicorn serving.api:app --workers 4
Spark 2 (GPU inference): CUDA_VISIBLE_DEVICES=0 celery worker

Connect via Redis:
  Redis → task queue → Celery GPU worker → result back to API
```

### Performance at Scale
- Skin/Breed/Pain: ~50ms/image
- Eye segmentation: ~150ms/image  
- Heart rate: ~3s for 15s video (5fps processing)
- YOLOv8n detection: ~10ms/image
