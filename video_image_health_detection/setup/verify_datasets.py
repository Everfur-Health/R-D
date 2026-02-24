#!/usr/bin/env python3
"""
Canine Health AI — Dataset Verification
Run FIRST to check all datasets are correctly structured before training.
Usage: python setup/verify_datasets.py --root ~/Documents/datasets
"""
import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

def count_images(path, exts={'.jpg','.jpeg','.png','.webp','.bmp'}):
    """Count images recursively."""
    return sum(1 for p in Path(path).rglob('*') if p.suffix.lower() in exts)

def get_class_dist(path):
    """Get image count per class (assumes class = subdirectory name)."""
    dist = {}
    for d in sorted(Path(path).iterdir()):
        if d.is_dir():
            n = count_images(d)
            if n > 0:
                dist[d.name] = n
    return dist

def check_segmentation_dataset(path):
    """Check segmentation dataset structure (images + masks)."""
    imgs = list(Path(path).rglob('*.png')) + list(Path(path).rglob('*.jpg'))
    # Look for masks directory
    masks_dir = Path(path) / 'Masks' / 'Grayscale'
    masks_ok = masks_dir.exists()
    mask_count = count_images(masks_dir) if masks_ok else 0
    return len(imgs), mask_count, masks_ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.path.expanduser('~/Documents/datasets'))
    args = parser.parse_args()
    root = Path(args.root)

    print("=" * 60)
    print(" CANINE HEALTH AI — Dataset Verification")
    print(f" Root: {root}")
    print("=" * 60)

    report = {}
    all_ok = True

    # ----------------------------------------------------------
    # 1. Skin Disease
    # ----------------------------------------------------------
    print("\n[1] Skin Disease Dataset")
    skin_path = root / 'skin'
    if not skin_path.exists():
        print(f"  ✗ NOT FOUND: {skin_path}")
        print("    Run: bash setup/download_data.sh")
        all_ok = False
        report['skin'] = {'status': 'missing'}
    else:
        dist = get_class_dist(skin_path)
        if not dist:
            # Try subdirectory search
            for sub in skin_path.rglob('*'):
                if sub.is_dir():
                    d = get_class_dist(sub)
                    if len(d) >= 3:
                        dist = d
                        skin_path = sub
                        break
        total = sum(dist.values())
        print(f"  ✓ Found at: {skin_path}")
        print(f"  Classes ({len(dist)}): {total} total images")
        for cls, n in dist.items():
            bar = '█' * (n // 10)
            print(f"    {cls:25s} {n:4d}  {bar}")
        report['skin'] = {'path': str(skin_path), 'classes': dist, 'total': total}
        if len(dist) < 2:
            print("  ⚠ WARNING: Expected ≥2 classes. Check directory structure.")

    # ----------------------------------------------------------
    # 2. Eye Disease Segmentation
    # ----------------------------------------------------------
    print("\n[2] Eye Disease Segmentation (DogEyeSeg4)")
    eye_seg_path = root / 'Eye' / 'DogEyeSeg4'
    if not eye_seg_path.exists():
        eye_seg_path = root / 'Eye'
        # Look for grayscale masks
    imgs, masks, masks_ok = check_segmentation_dataset(eye_seg_path)
    if masks_ok and masks > 0:
        print(f"  ✓ Found at: {eye_seg_path}")
        print(f"  Images: {imgs}  |  Masks: {masks}")
        print(f"  Classes: 0=background, 1=corneal edema, 2=episcleral congestion, 3=epiphora, 4=cherry eye")
        report['eye_seg'] = {'path': str(eye_seg_path), 'images': imgs, 'masks': masks}
    else:
        print(f"  ⚠ Masks not found at: {eye_seg_path}/Masks/Grayscale/")
        print(f"    (Found {imgs} images, {masks} masks)")
        report['eye_seg'] = {'path': str(eye_seg_path), 'images': imgs, 'masks': masks, 'warning': 'no masks'}

    # ----------------------------------------------------------
    # 3. Eye YOLO (dog-diseases-9class)
    # ----------------------------------------------------------
    print("\n[3] Eye Disease YOLO (dog-diseases-9class)")
    yolo_path = root / 'Eye' / 'eye_part2' / 'dog-diseases-9class'
    if not yolo_path.exists():
        # Search
        for p in (root / 'Eye').rglob('data.yaml'):
            yolo_path = p.parent
            break
    if yolo_path.exists():
        yaml_file = yolo_path / 'data.yaml'
        if yaml_file.exists():
            import yaml
            with open(yaml_file) as f:
                cfg = yaml.safe_load(f)
            print(f"  ✓ Found at: {yolo_path}")
            print(f"  Classes: {cfg.get('nc', '?')} → {cfg.get('names', [])}")
            train_n = count_images(yolo_path / 'train' / 'images') if (yolo_path / 'train' / 'images').exists() else 0
            val_n = count_images(yolo_path / 'valid' / 'images') if (yolo_path / 'valid' / 'images').exists() else 0
            print(f"  Train: {train_n}  |  Val: {val_n}")
            report['eye_yolo'] = {'path': str(yolo_path), 'train': train_n, 'val': val_n}
        else:
            print(f"  ⚠ data.yaml not found at {yolo_path}")
    else:
        print(f"  ✗ NOT FOUND: {yolo_path}")

    # ----------------------------------------------------------
    # 4. Breed (Stanford Dogs)
    # ----------------------------------------------------------
    print("\n[4] Dog Breed Dataset (Stanford Dogs 120 classes)")
    breed_path = root / 'breed'
    if not breed_path.exists():
        print(f"  ✗ NOT FOUND: {breed_path}")
        all_ok = False
        report['breed'] = {'status': 'missing'}
    else:
        # Stanford dogs has Images/ with subdirs like n02085782-Japanese_spaniel
        imgs_dir = breed_path / 'images' / 'Images'
        if not imgs_dir.exists():
            imgs_dir = breed_path / 'Images'
        if not imgs_dir.exists():
            for p in breed_path.rglob('*'):
                if p.is_dir() and len(list(p.iterdir())) > 50:
                    imgs_dir = p
                    break
        dist = get_class_dist(imgs_dir)
        total = sum(dist.values())
        print(f"  ✓ Found at: {imgs_dir}")
        print(f"  Classes: {len(dist)} breeds  |  Total images: {total}")
        if len(dist) >= 5:
            sample = list(dist.items())[:5]
            for cls, n in sample:
                print(f"    {cls[:40]:40s} {n:4d}")
            print(f"    ... ({len(dist)-5} more breeds)")
        report['breed'] = {'path': str(imgs_dir), 'classes': len(dist), 'total': total}

    # ----------------------------------------------------------
    # 5. Pain / Emotion
    # ----------------------------------------------------------
    print("\n[5] Pain / Emotion Dataset (Dog Emotion 4-class)")
    pain_path = root / 'Pain' / 'Dog Emotion'
    if not pain_path.exists():
        pain_path = root / 'Pain'
    dist = get_class_dist(pain_path)
    if not dist:
        # Try flattened
        for sub in pain_path.rglob('*'):
            if sub.is_dir():
                d = get_class_dist(sub)
                if d:
                    dist = d
                    break
    if dist:
        total = sum(dist.values())
        print(f"  ✓ Found at: {pain_path}")
        print(f"  Classes: {dist}")
        print(f"  Total: {total}")
        report['pain'] = {'path': str(pain_path), 'classes': dist, 'total': total}
    else:
        print(f"  ✗ NOT FOUND: {pain_path}")
        all_ok = False

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    report_path = Path('datasets_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f" Report saved: {report_path}")
    if all_ok:
        print(" ✓ All critical datasets found — ready to train!")
    else:
        print(" ⚠ Some datasets missing. See above.")
    print("=" * 60)

if __name__ == '__main__':
    main()
