#!/usr/bin/env python3
"""
Canine Health AI — Breed Classification
Model: EfficientNet-B4 (timm) on Stanford Dogs Dataset (120 breeds, ~20k images)

Why this dataset:
  - 120 breeds covering toy/small/medium/large/giant sizes
  - ~180 images/class → transfer learning essential
  - Fine-grained visual classification (very similar classes)

Training strategy:
  - Stage 1 (epochs 1-5):  Freeze backbone, train head only → fast feature alignment
  - Stage 2 (epochs 6+):   Unfreeze backbone with differential LR
  - Label smoothing 0.15 helps with inter-breed similarity

Usage:
  python modules/breed/train.py --data ~/Documents/datasets/breed --epochs 60
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.utils import (get_classification_transforms, EarlyStopping,
                           save_training_curves, mixup_data, mixup_criterion)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
import timm
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def find_stanford_dogs_root(base: Path) -> Path:
    """
    Stanford Dogs kaggle download structure:
    breed/
      images/
        Images/
          n02085782-Japanese_spaniel/
          n02085936-Maltese_dog/
          ...
    """
    # Try common patterns
    candidates = [
        base / 'images' / 'Images',
        base / 'Images',
        base,
    ]
    for c in candidates:
        if c.exists():
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if len(subdirs) > 10:
                return c
    # Recursive search
    for p in base.rglob('*'):
        if p.is_dir() and len(list(p.iterdir())) > 50:
            if any(d.is_dir() for d in p.iterdir()):
                return p
    return base


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs with albumentations transforms.
    Handles the 'n02085782-Japanese_spaniel' directory naming.
    """
    def __init__(self, root: Path, transform, max_per_class: int = None):
        self.transform = transform
        self.samples = []
        self.classes = []

        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        # Extract readable names: n02085782-Japanese_spaniel → Japanese Spaniel
        self.classes = [self._clean_name(d.name) for d in subdirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for i, d in enumerate(subdirs):
            imgs = (list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) +
                    list(d.glob('*.png')))
            if max_per_class:
                imgs = imgs[:max_per_class]
            for img_path in imgs:
                self.samples.append((str(img_path), i))

    @staticmethod
    def _clean_name(dirname: str) -> str:
        """n02085782-Japanese_spaniel → Japanese Spaniel"""
        name = dirname.split('-', 1)[-1] if '-' in dirname else dirname
        return name.replace('_', ' ').title()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Return dummy on error
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return aug['image'], label

    @property
    def targets(self):
        return [s[1] for s in self.samples]


def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B4 for fine-grained breed classification.
    For 120 classes with ~180 imgs/class, we need:
      - Strong pretrained backbone (ImageNet-21k)
      - Dropout to prevent memorization
    """
    model = timm.create_model(
        'efficientnet_b4',
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.4,       # Higher dropout for fine-grained
        drop_path_rate=0.2,
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.3)
        optimizer.zero_grad()
        out = model(images)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc='  Val  ', leave=False):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.expanduser('~/Documents/datasets/breed'))
    parser.add_argument('--out', default='models/breed')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Epochs to train only head before unfreezing backbone')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(" BREED CLASSIFIER TRAINING (120 breeds)")
    print(f" Device: {device}")
    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # --- Data ---
    dogs_root = find_stanford_dogs_root(Path(args.data))
    print(f"\nDataset root: {dogs_root}")

    train_tf = get_classification_transforms(args.img_size, train=True)
    val_tf = get_classification_transforms(args.img_size, train=False)

    full_ds = StanfordDogsDataset(dogs_root, transform=train_tf)
    num_classes = len(full_ds.classes)
    print(f"Classes: {num_classes} breeds  |  Samples: {len(full_ds)}")

    # 80/20 split
    from collections import defaultdict
    import random
    random.seed(42)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_ds.samples):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    for cls, indices in class_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * 0.8)
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    class SubsetTransform(torch.utils.data.Dataset):
        def __init__(self, base, indices, transform):
            self.base = base
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            import cv2
            path, label = self.base.samples[self.indices[i]]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img)
            return aug['image'], label

    train_ds = SubsetTransform(full_ds, train_idx, train_tf)
    val_ds = SubsetTransform(full_ds, val_idx, val_tf)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Weighted sampler
    train_targets = [full_ds.samples[i][1] for i in train_idx]
    counts = np.bincount(train_targets, minlength=num_classes)
    w = 1.0 / (counts + 1e-6)
    sample_weights = [w[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Save classes
    with open(out_dir / 'class_names.json', 'w') as f:
        json.dump(full_ds.classes, f, indent=2)

    # --- Model ---
    model = build_model(num_classes).to(device)
    print(f"Model: EfficientNet-B4  |  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    # Stage 1: head-only warmup
    print(f"\nStage 1: Head warmup ({args.warmup_epochs} epochs)...")
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'head' not in name:
            param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.warmup_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        print(f"  Epoch {epoch:3d}  train_acc={t_acc:.4f}  val_acc={v_acc:.4f}")

    # Stage 2: Full fine-tuning with differential LR
    print(f"\nStage 2: Full fine-tuning ({args.epochs - args.warmup_epochs} epochs)...")
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters()
                       if 'classifier' not in n and 'head' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if 'classifier' in n or 'head' in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr / 10},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6
    )
    early_stop = EarlyStopping(patience=12, mode='max')
    best_preds, best_labels = [], []

    for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_acc={t_acc:.4f}  val_acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_preds, best_labels = preds, labels
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'classes': full_ds.classes,
                'img_size': args.img_size,
            }, out_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

        if early_stop(v_acc):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save artifacts
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)
    save_training_curves(history, str(out_dir / 'training_curves.png'))

    # Top-5 accuracy
    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE — Best val_acc: {best_val_acc:.4f}")
    print(f" Model: {out_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
