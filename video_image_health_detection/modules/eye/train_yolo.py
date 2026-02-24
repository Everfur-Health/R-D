#!/usr/bin/env python3
"""
Canine Health AI — Eye Disease Detection (YOLOv8)
Fine-tunes YOLOv8m on dog-diseases-9class dataset.

Why YOLOv8 alongside SegFormer:
  - SegFormer does pixel-level segmentation (best for precise localization)
  - YOLO does fast bounding-box detection (better for multi-disease, quick screening)
  - Together: YOLO for fast triage, SegFormer for detailed analysis

Usage:
  python modules/eye/train_yolo.py --data ~/Documents/datasets/Eye/eye_part2/dog-diseases-9class
"""
import os, sys, argparse, shutil
from pathlib import Path
import yaml


def find_data_yaml(data_path: Path) -> Path:
    """Find data.yaml in the given path or subdirectories."""
    # Direct
    if (data_path / 'data.yaml').exists():
        return data_path / 'data.yaml'
    # Search
    for p in data_path.rglob('data.yaml'):
        # Prefer base dataset over augmented versions
        if 'augm' not in str(p) and 'cutmix' not in str(p):
            return p
    # Return first found
    for p in data_path.rglob('data.yaml'):
        return p
    return None


def patch_data_yaml(yaml_path: Path, out_dir: Path) -> Path:
    """
    Fix relative paths in data.yaml (YOLO needs absolute paths or
    paths relative to yaml location). Copies yaml to working dir.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    yaml_root = yaml_path.parent
    # Make paths absolute
    for key in ['train', 'val', 'test']:
        if key in cfg and cfg[key]:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(yaml_root / p)

    # Save patched yaml
    patched_path = out_dir / 'data_patched.yaml'
    #with open(patched_path, 'w') as f:
    #    yaml.dump(cfg, f)

    print(f"  Classes ({cfg.get('nc', '?')}): {cfg.get('names', [])}")
    return patched_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        default=os.path.expanduser(
                            '~/Documents/datasets/Eye/eye_part2/dog-diseases-9class'))
    parser.add_argument('--out', default='models/eye_yolo')
    parser.add_argument('--model', default='yolov8m.pt',
                        help='YOLOv8 variant: yolov8n/s/m/l/x.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" EYE DISEASE YOLO TRAINING")
    print("=" * 60)

    # --- Find dataset ---
    data_path = Path(args.data)
    yaml_file = find_data_yaml(data_path)
    if yaml_file is None:
        print(f"ERROR: No data.yaml found under {data_path}")
        sys.exit(1)

    print(f"\nDataset YAML: {yaml_file}")
    patched_yaml = patch_data_yaml(yaml_file, out_dir)

    # --- Check task type ---
    with open(patched_yaml) as f:
        cfg = yaml.safe_load(f)

    # Determine if detection or classification
    train_path = Path(cfg.get('train', ''))
    has_labels = (train_path.parent / 'labels').exists() or \
                 (train_path / '..').resolve().parent.name == 'labels'
    # Simple check: if 'images' dir and 'labels' dir coexist → detection
    task = 'detect'

    print(f"\nTask: {task}")
    print(f"Model: {args.model}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    results = model.train(
        data=str(patched_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        project=str(out_dir),
        name='train',
        exist_ok=True,

        # Augmentation (matched to medical imaging)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,         # No vertical flip (eyes have anatomical orientation)
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.2,

        # Training
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        patience=15,

        # Logging
        save=True,
        save_period=10,
        verbose=True,
    )

    # Copy best weights to standard location
    best_pt = out_dir / 'train' / 'weights' / 'best.pt'
    if best_pt.exists():
        dest = out_dir / 'best.pt'
        shutil.copy(best_pt, dest)
        print(f"\n✓ Best weights: {dest}")

    print("\n" + "=" * 60)
    print(" YOLO TRAINING COMPLETE")
    print(f" Best model: {out_dir / 'best.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
