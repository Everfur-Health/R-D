#!/usr/bin/env python3
"""
00_explore_datasets.py
======================
Run this FIRST before anything else.
Produces a complete inventory of your three datasets so you know
exactly what you're working with before training a single model.

Usage:
    python 00_explore/explore_datasets.py --root ~/Documents/datasets

Output:
    Prints a formatted report to console.
    Writes datasets_report.json to current directory.
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict, Counter

# Optional: opencv for video metadata. Falls back gracefully if not installed.
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def human_size(n_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def count_images(directory: Path) -> dict:
    """Count image files by extension in a directory tree."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = Counter()
    total_bytes = 0
    for f in directory.rglob("*"):
        if f.suffix.lower() in exts:
            counts[f.suffix.lower()] += 1
            total_bytes += f.stat().st_size
    return {"count": sum(counts.values()), "by_ext": dict(counts), "total_bytes": total_bytes}


def explore_pain(pain_root: Path) -> dict:
    """
    Explore:
      Pain/Pain/Dog Emotion/
        angry/   happy/   relaxed/   sad/   labels.csv
    """
    print("\n" + "="*60)
    print("📁 PAIN / DOG EMOTION DATASET")
    print("="*60)

    emotion_dir = pain_root / "Pain" / "Dog Emotion"
    if not emotion_dir.exists():
        print(f"  ⚠️  Directory not found: {emotion_dir}")
        return {}

    classes = ["angry", "happy", "relaxed", "sad"]
    report  = {"classes": {}, "csv_rows": 0}

    for cls in classes:
        cls_dir = emotion_dir / cls
        if cls_dir.exists():
            imgs = count_images(cls_dir)
            report["classes"][cls] = imgs
            print(f"  {cls:>8}: {imgs['count']:>5} images  ({human_size(imgs['total_bytes'])})")
        else:
            print(f"  {cls:>8}: directory not found")

    # Read labels.csv
    csv_path = emotion_dir / "labels.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        label_counts = Counter(r["label"] for r in rows)
        report["csv_rows"] = len(rows)
        print(f"\n  labels.csv: {len(rows)} rows")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"    {lbl:>8}: {cnt} rows in CSV")
        report["csv_label_counts"] = dict(label_counts)
    else:
        print("  ⚠️  labels.csv not found")

    total = sum(v["count"] for v in report["classes"].values())
    print(f"\n  TOTAL: {total} images across {len(report['classes'])} classes")

    # Pain mapping recommendation
    print("\n  📌 PAIN MAPPING STRATEGY:")
    print("     angry  + sad   → 'distress/pain proxy' (negative valence)")
    print("     happy  + relaxed → 'calm/healthy proxy' (positive valence)")
    print("     This gives us a binary classification baseline.")
    print("     For 4-class training, keep all labels — richer supervision.")

    return report


def explore_eye(eye_root: Path) -> dict:
    """
    Explore:
      Eye/
        DogEyeSeg4_dataset 2/    (segmentation: Images + Masks/Gray + Masks/Color)
        eye_part2/
          dog-diseases-9class/   (YOLO format: data.yaml, train/valid/test)
          dogeye5/               (5-class)
          dogskin4/              (skin disease)
    """
    print("\n" + "="*60)
    print("📁 EYE / DISEASE DATASET")
    print("="*60)

    report = {}

    # ── DogEyeSeg4 ──
    seg_dir = None
    for candidate in eye_root.iterdir():
        if "DogEyeSeg4" in candidate.name:
            seg_dir = candidate
            break

    if seg_dir:
        images_dir = seg_dir / "Images"
        masks_gray = seg_dir / "Masks" / "Gray"
        masks_color = seg_dir / "Masks" / "Color"

        n_images = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        n_masks  = len(list(masks_gray.glob("*.png"))) if masks_gray.exists() else 0

        print(f"\n  DogEyeSeg4 (segmentation):")
        print(f"    Images: {n_images} (320×320 PNG)")
        print(f"    Masks:  {n_masks} (grayscale, 0–4 class values)")
        print(f"    Classes: 0=background, 1=corneal edema, 2=episcleral congestion,")
        print(f"             3=epiphora, 4=Cherry Eye")

        # Count pixels per class in a sample mask to understand class balance
        if masks_gray.exists() and HAS_CV2:
            import numpy as np
            sample_masks = list(masks_gray.glob("*.png"))[:20]
            class_pixels = Counter()
            for m in sample_masks:
                mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    vals, cnts = zip(*Counter(mask.flatten().tolist()).items())
                    for v, c in zip(vals, cnts):
                        class_pixels[v] += c
            total_px = sum(class_pixels.values())
            print(f"\n    Class pixel distribution (sample of {len(sample_masks)} masks):")
            cls_names = {0: "background", 1: "corneal edema",
                         2: "episcleral", 3: "epiphora", 4: "cherry eye"}
            for cls_id in sorted(class_pixels):
                pct = class_pixels[cls_id] / total_px * 100
                name = cls_names.get(cls_id, f"class_{cls_id}")
                print(f"      Class {cls_id} ({name:>20}): {pct:5.1f}%")

        report["dogeye_seg4"] = {"n_images": n_images, "n_masks": n_masks}
    else:
        print("\n  ⚠️  DogEyeSeg4 directory not found")

    # ── dog-diseases-9class (YOLO format) ──
    nine_class_dir = eye_root / "eye_part2" / "dog-diseases-9class"
    if nine_class_dir.exists():
        yaml_path = nine_class_dir / "data.yaml"
        print(f"\n  dog-diseases-9class (YOLO format):")

        if yaml_path.exists():
            # Parse YAML without pyyaml dependency
            with open(yaml_path) as f:
                yaml_content = f.read()
            print(f"    data.yaml contents:")
            for line in yaml_content.strip().split("\n"):
                print(f"      {line}")

        for split in ["train", "valid", "test"]:
            split_dir = nine_class_dir / split
            if split_dir.exists():
                # YOLO format: images/ and labels/ subdirs
                images = count_images(split_dir / "images") if (split_dir / "images").exists() \
                         else count_images(split_dir)
                print(f"    {split:>6}: {images['count']} images")

        report["dog_diseases_9class"] = {"path": str(nine_class_dir)}

    # ── dogeye5 ──
    dogeye5_dir = eye_root / "eye_part2" / "dogeye5"
    if dogeye5_dir.exists():
        imgs = count_images(dogeye5_dir)
        print(f"\n  dogeye5 (5-class eye): {imgs['count']} images")
        # Check subdirectory names
        subdirs = [d.name for d in dogeye5_dir.iterdir() if d.is_dir()]
        print(f"    Classes: {subdirs}")

    # ── dogskin4 ──
    dogskin_dir = eye_root / "eye_part2" / "dogskin4"
    if dogskin_dir.exists():
        imgs = count_images(dogskin_dir)
        print(f"\n  dogskin4 (4-class skin): {imgs['count']} images")
        subdirs = [d.name for d in dogskin_dir.iterdir() if d.is_dir()]
        print(f"    Classes: {subdirs}")
        report["dogskin4"] = {"n_images": imgs["count"], "classes": subdirs}

    return report


def explore_arthritis(arth_root: Path) -> dict:
    """
    Explore:
      Arthritis/
        CAM Video Uploads/
        CAM Video Uploads 2/ ... 5/
          Each contains breed-named folders with .MOV files
          Naming convention: Jagger 1 towards.MOV, Jagger 2 away.MOV, etc.
    """
    print("\n" + "="*60)
    print("📁 ARTHRITIS / GAIT VIDEO DATASET")
    print("="*60)

    report  = {"dogs": {}, "total_videos": 0, "gait_types": Counter()}
    cam_dirs = sorted(arth_root.glob("CAM Video Uploads*"))

    if not cam_dirs:
        print("  ⚠️  No 'CAM Video Uploads' directories found")
        return report

    video_exts  = {".mov", ".mp4", ".avi", ".MOV", ".MP4", ".AVI"}
    all_dogs    = {}

    for cam_dir in cam_dirs:
        # Walk down — videos are nested in breed subfolders
        for vid_file in cam_dir.rglob("*"):
            if vid_file.suffix in video_exts:
                # Parse naming convention: "DogName N gait_type.MOV"
                # E.g. "Jagger 1 towards.MOV" → dog=Jagger, gait=towards
                parts  = vid_file.stem.split()
                dog_name = parts[0] if parts else "unknown"
                gait_type = parts[-1].lower() if len(parts) > 1 else "unknown"

                # Get breed from parent folder name (e.g. "CAM-6016b1bd8dbe8 Great Dane")
                breed = "unknown"
                for parent in vid_file.parents:
                    if parent == arth_root:
                        break
                    name = parent.name
                    # Breed is often at the end after a space
                    if " " in name and "CAM-" in name:
                        breed = " ".join(name.split()[1:])
                        break

                key = f"{dog_name}_{breed}"
                if key not in all_dogs:
                    all_dogs[key] = {
                        "dog_name": dog_name,
                        "breed": breed,
                        "videos": []
                    }

                video_info = {
                    "path":       str(vid_file),
                    "gait_type":  gait_type,
                    "file_size":  human_size(vid_file.stat().st_size),
                }

                # Get video metadata with OpenCV
                if HAS_CV2:
                    cap = cv2.VideoCapture(str(vid_file))
                    if cap.isOpened():
                        video_info["fps"]      = round(cap.get(cv2.CAP_PROP_FPS), 1)
                        video_info["frames"]   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_info["width"]    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        video_info["height"]   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_info["duration"] = round(
                            video_info["frames"] / max(video_info["fps"], 1), 1
                        )
                    cap.release()

                all_dogs[key]["videos"].append(video_info)
                report["gait_types"][gait_type] += 1
                report["total_videos"] += 1

    report["dogs"] = all_dogs

    print(f"\n  Total dogs:  {len(all_dogs)}")
    print(f"  Total videos: {report['total_videos']}")

    print("\n  Dogs found:")
    for key, dog in sorted(all_dogs.items()):
        n = len(dog["videos"])
        gaits = [v["gait_type"] for v in dog["videos"]]
        first = dog["videos"][0]
        res = f"{first.get('width','?')}×{first.get('height','?')}" if HAS_CV2 else "?"
        fps = first.get("fps", "?")
        print(f"    {dog['dog_name']:>12} ({dog['breed']:<20}): "
              f"{n} videos, {res} @ {fps}fps")
        print(f"                    Gaits: {', '.join(gaits)}")

    print(f"\n  Gait types across all dogs:")
    for gait, cnt in sorted(report["gait_types"].items()):
        print(f"    {gait:>20}: {cnt} videos")

    print("\n  📌 GAIT ANALYSIS STRATEGY:")
    print("     'towards' + 'away': Best for detecting front/rear limb lameness")
    print("     'left to right' + 'right to left': Lateral view — best for stride symmetry")
    print("     'trot': Faster gait amplifies lameness signals")
    print("     'clockwise' + 'anti clockwise': Circle patterns test turning ability")
    print("     → Use lateral views for primary lameness scoring")
    print("     → Trot videos show most diagnostic value in veterinary practice")

    return report


def main():
    parser = argparse.ArgumentParser(description="Canine dataset explorer")
    parser.add_argument("--root", default=os.path.expanduser("~/Documents/datasets"),
                        help="Root of your datasets directory")
    parser.add_argument("--output", default="datasets_report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Dataset root not found: {root}")
        sys.exit(1)

    print(f"\n🔍 Exploring datasets in: {root}")
    if not HAS_CV2:
        print("  ⚠️  OpenCV not installed — video metadata will not be available.")
        print("     Install with: pip install opencv-python-headless")

    report = {
        "root":      str(root),
        "pain":      explore_pain(root / "Pain"),
        "eye":       explore_eye(root / "Eye"),
        "arthritis": explore_arthritis(root / "Arthritis"),
    }

    # Write JSON report
    # Convert Counter objects to dicts for JSON serialization
    report["arthritis"]["gait_types"] = dict(report["arthritis"]["gait_types"])

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n\n✅ Full report saved to: {args.output}")
    print("\n📌 RECOMMENDED DEMO PRIORITY:")
    print("   1. Eye Disease (richest labeled data — segmentation + 9-class YOLO format)")
    print("   2. Pain Emotion (clean 4-class images, great for GradCAM demo)")
    print("   3. Gait/Arthritis (video ML — most impactful but needs pose extraction first)")


if __name__ == "__main__":
    main()
