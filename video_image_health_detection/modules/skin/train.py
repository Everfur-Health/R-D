#!/usr/bin/env python3
"""
Canine Health AI — Skin Disease Classification
Model: EfficientNet-B4 (timm) fine-tuned on dog skin disease dataset
Classes: hotspot, ringworm, flea_allergy, mange (+ healthy if present)

Dataset: Kaggle "dogs-skin-disease-dataset"
  4 classes × ~90 images per class training = ~360 train images
  Strategy: Heavy augmentation + MixUp + WeightedRandomSampler

Usage:
  python modules/skin/train.py --data ~/Documents/datasets/skin --epochs 50
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.utils import (get_classification_transforms, EarlyStopping,
                           save_training_curves, mixup_data, mixup_criterion)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def find_dataset_root(data_path: Path, max_depth: int = 5) -> Path:
    """
    Find the deepest directory that looks like a flat ImageFolder root.
    Handles Kaggle double-nesting, e.g.:
      skin/dogs-skin-disease-dataset/dogs-skin-disease-dataset/hotspot/...
    Uses BFS so it always finds the shallowest valid class root first.
    """
    from collections import deque

    IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

    def has_images(d: Path) -> bool:
        return any(f for ext in IMG_EXTS for f in d.glob(ext))

    def is_class_root(d: Path) -> bool:
        subs = [s for s in d.iterdir() if s.is_dir()]
        if len(subs) < 2:
            return False
        with_images = sum(1 for s in subs if has_images(s))
        return with_images >= max(2, len(subs) // 2)

    queue = deque([(data_path, 0)])
    while queue:
        current, depth = queue.popleft()
        if is_class_root(current):
            return current
        if depth < max_depth:
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    queue.append((child, depth + 1))

    return data_path


def build_split_datasets(root: Path, img_size: int = 224, val_split: float = 0.2):
    """
    Build train/val datasets. If no train/val split exists, create one.
    Supports:
      - Pre-split: root/train/, root/val/
      - Flat: root/class1/, root/class2/, ...
    """
    train_dir = root / 'train'
    val_dir = root / 'val'

    if train_dir.exists() and val_dir.exists():
        print(f"  Using pre-split: train={train_dir}  val={val_dir}")
        train_tf = get_classification_transforms(img_size, train=True)
        val_tf = get_classification_transforms(img_size, train=False)

        class AlbumentationsDataset(torch.utils.data.Dataset):
            def __init__(self, img_folder, transform):
                self.ds = ImageFolder(str(img_folder))
                self.transform = transform
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                img_pil, label = self.ds[idx]
                import numpy as np
                img_np = np.array(img_pil)
                aug = self.transform(image=img_np)
                return aug['image'], label
            @property
            def classes(self): return self.ds.classes

        train_ds = AlbumentationsDataset(train_dir, train_tf)
        val_ds = AlbumentationsDataset(val_dir, val_tf)
        return train_ds, val_ds, train_ds.classes

    # Flat structure → split
    print("  No pre-split found — creating 80/20 split in memory")
    from torch.utils.data import Subset
    full_tf_train = get_classification_transforms(img_size, train=True)
    full_tf_val = get_classification_transforms(img_size, train=False)

    class AlbDataset(torch.utils.data.Dataset):
        def __init__(self, folder, transform):
            self.ds = ImageFolder(str(folder))
            self.transform = transform
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            img_pil, label = self.ds[idx]
            img_np = np.array(img_pil)
            aug = self.transform(image=img_np)
            return aug['image'], label
        @property
        def classes(self): return self.ds.classes

    full_ds_base = ImageFolder(str(root))
    classes = full_ds_base.classes
    n = len(full_ds_base)

    # Stratified split
    from collections import defaultdict
    import random
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_ds_base.samples):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    random.seed(42)
    for cls, indices in class_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * (1 - val_split))
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img_pil, label = self.base[self.indices[idx]]
            img_np = np.array(img_pil)
            aug = self.transform(image=img_np)
            return aug['image'], label

    train_ds = SubsetWithTransform(full_ds_base, train_idx, full_tf_train)
    val_ds = SubsetWithTransform(full_ds_base, val_idx, full_tf_val)
    return train_ds, val_ds, classes


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B4 from timm with custom head.
    Why B4:
      - compound scaling (depth+width+resolution) gives best accuracy/compute tradeoff
      - pretrained on ImageNet-21k → rich veterinary-applicable features
      - ~19M params, fits in GPU memory with batch_size=32 easily
    """
    model = timm.create_model(
        'efficientnet_b4',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )
    return model


def train_epoch(model, loader, optimizer, criterion, scheduler, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0., 0, 0

    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)

        if use_mixup and np.random.rand() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
            optimizer.zero_grad()
            out = model(images)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    if scheduler is not None:
        scheduler.step()

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
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease Classifier')
    parser.add_argument('--data', default=os.path.expanduser('~/Documents/datasets/skin'))
    parser.add_argument('--out', default='models/skin')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(" SKIN DISEASE CLASSIFIER TRAINING")
    print(f" Device: {device}")
    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # --- Data ---
    data_root = find_dataset_root(Path(args.data))
    print(f"\nDataset root: {data_root}")
    train_ds, val_ds, classes = build_split_datasets(data_root, args.img_size)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Save class names
    with open(out_dir / 'class_names.json', 'w') as f:
        json.dump(classes, f, indent=2)

    # Weighted sampler to handle class imbalance
    # Count labels in training set
    if hasattr(train_ds, 'targets'):
        targets = train_ds.targets
    else:
        # Extract labels from wrapped dataset
        targets = []
        for i in range(len(train_ds)):
            _, label = train_ds[i]
            targets.append(label)

    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- Model ---
    print(f"\nBuilding EfficientNet-B4 ({num_classes} classes)...")
    model = build_model(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M  |  Trainable: {trainable_params/1e6:.1f}M")

    # --- Loss with class weights ---
    loss_weights = torch.tensor(class_weights / class_weights.sum() * num_classes,
                                 dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)

    # --- Differential learning rates ---
    # Backbone: lower LR (features already learned from ImageNet)
    # Head: higher LR (task-specific)
    backbone_params = [p for n, p in model.named_parameters()
                       if 'classifier' not in n and 'head' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if 'classifier' in n or 'head' in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr / 10},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=10, mode='max')

    # --- Training loop ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.
    best_preds, best_labels = [], []

    print(f"\nTraining for up to {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        v_loss, v_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_preds, best_labels = preds, labels
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'classes': classes,
                'img_size': args.img_size,
            }, out_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

        if early_stop(v_acc):
            print(f"  Early stopping at epoch {epoch}")
            break

    # --- Save artifacts ---
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)
    save_training_curves(history, str(out_dir / 'training_curves.png'))

    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Skin Disease Classifier — Best Val Acc: {best_val_acc:.4f}')
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE")
    print(f" Best val accuracy: {best_val_acc:.4f}")
    print(f" Model saved: {out_dir / 'best_model.pth'}")
    print("\n Per-class report:")
    print(classification_report(best_labels, best_preds, target_names=classes))
    print("=" * 60)


if __name__ == '__main__':
    main()
