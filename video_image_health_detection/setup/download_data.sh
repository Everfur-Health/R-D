#!/bin/bash
# =============================================================================
# Canine Health AI Demo — Dataset Downloader
# Requires: kaggle API key at ~/.kaggle/kaggle.json
#   Get it from: https://www.kaggle.com/account → API → Create New Token
# =============================================================================
set -e

BASE_DIR="${1:-$HOME/Documents/datasets}"
mkdir -p "$BASE_DIR"

echo "============================================================"
echo " Downloading datasets to: $BASE_DIR"
echo "============================================================"
echo ""
echo "PREREQUISITE: Place kaggle.json at ~/.kaggle/kaggle.json"
echo "  chmod 600 ~/.kaggle/kaggle.json"
echo ""

# Check kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Move downloaded kaggle.json to ~/.kaggle/"
    echo "4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# ==========================================================
# 1. Dog Skin Disease (4 classes: hotspot, mange, ringworm, flea_allergy)
# ==========================================================
echo "[1/3] Dog Skin Disease Dataset (yashmotiani)"
mkdir -p "$BASE_DIR/skin"
cd "$BASE_DIR/skin"
kaggle datasets download -d yashmotiani/dogs-skin-disease-dataset --unzip
echo "  ✓ Skin disease data at: $BASE_DIR/skin/"
ls -la

# ==========================================================
# 2. Stanford Dogs (120 breeds, ~20k images)
# ==========================================================
echo ""
echo "[2/3] Stanford Dogs Dataset (120 breeds)"
mkdir -p "$BASE_DIR/breed"
cd "$BASE_DIR/breed"
kaggle datasets download -d jessicali9530/stanford-dogs-dataset --unzip
echo "  ✓ Breed data at: $BASE_DIR/breed/"

# ==========================================================
# 3. Print what we already have
# ==========================================================
echo ""
echo "[3/3] Existing datasets:"
echo "  Eye:   $BASE_DIR/Eye/           (DogEyeSeg4 segmentation)"
echo "  Eye:   $BASE_DIR/Eye/eye_part2/ (dog-diseases-9class YOLO)"
echo "  Pain:  $BASE_DIR/Pain/          (Dog Emotion 4-class)"
echo "  Gait:  $BASE_DIR/Arthritis/     (CAM videos — lower priority)"

echo ""
echo "============================================================"
echo " Downloads complete!"
echo " Now run: python setup/verify_datasets.py --root $BASE_DIR"
echo "============================================================"
