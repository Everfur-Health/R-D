#!/usr/bin/env python3
"""
Canine Health AI — Pain Detection from Images
Model: EfficientNet-B4 with GradCAM visualization
Dataset: Dog Emotion (angry, happy, relaxed, sad)
Mode: binary — maps (angry+sad) → 'distress', (happy+relaxed) → 'calm'

Scientific grounding:
  - DogFACS (Dog Facial Action Coding System): AU43 orbital tightening,
    AU101 brow lowering are associated with pain and negative valence
  - Hager et al. 2019 (Nature): orthopedic pain dogs show suppressed,
    low-valence expressions (sad/withdrawn)
  - Binary mapping is more clinically defensible than 4-class pain label

Note: This uses emotion as a proxy for pain. For clinical deployment,
you'd want pre/post-analgesia labels or direct veterinary pain scores.

Usage:
  python modules/pain/train.py --data ~/Documents/datasets/Pain --mode binary --epochs 40
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.utils import (get_classification_transforms, EarlyStopping,
                           save_training_curves, mixup_data, mixup_criterion, GradCAM)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Emotion → pain mapping
BINARY_MAP = {
    'angry': 1,    # distress
    'sad': 1,      # distress
    'happy': 0,    # calm
    'relaxed': 0,  # calm
}
BINARY_NAMES = ['calm', 'distress']


class PainDataset(torch.utils.data.Dataset):
    """Dog emotion dataset with optional binary remapping."""

    def __init__(self, folder, transform, mode='binary'):
        self.base = ImageFolder(str(folder))
        self.transform = transform
        self.mode = mode

        if mode == 'binary':
            # Remap class indices
            original_classes = self.base.classes
            self.class_to_binary = {}
            for c in original_classes:
                c_lower = c.lower()
                for key, val in BINARY_MAP.items():
                    if key in c_lower:
                        self.class_to_binary[self.base.class_to_idx[c]] = val
                        break
                else:
                    # Unknown class → default to calm
                    self.class_to_binary[self.base.class_to_idx[c]] = 0
            self.classes = BINARY_NAMES
        else:
            self.classes = self.base.classes

        print(f"  Classes: {self.classes}")
        if mode == 'binary':
            print(f"  Mapping: {self.class_to_binary}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_pil, label = self.base[idx]
        img_np = np.array(img_pil)
        aug = self.transform(image=img_np)
        img = aug['image']

        if self.mode == 'binary':
            label = self.class_to_binary.get(label, 0)

        return img, label

    @property
    def targets(self):
        if self.mode == 'binary':
            return [self.class_to_binary.get(lbl, 0) for _, lbl in self.base.samples]
        return [lbl for _, lbl in self.base.samples]


def find_pain_dataset(base: Path):
    """Find the dataset root with class subdirectories."""
    # Try direct paths
    for candidate in [
        base / 'Dog Emotion',
        base / 'dog_emotion',
        base,
    ]:
        if candidate.exists():
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if len(subdirs) >= 2:
                return candidate
    # Recursive
    for p in base.rglob('*'):
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if len(subdirs) >= 2 and any('happy' in d.name.lower() or 'sad' in d.name.lower()
                                          for d in subdirs):
                return p
    return base


def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model(
        'efficientnet_b4',
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        if np.random.rand() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
            optimizer.zero_grad()
            out = model(images)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
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


def save_gradcam_samples(model, val_ds, device, out_dir: Path, n=6):
    """Save GradCAM visualizations on validation samples."""
    gradcam = GradCAM(model)
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    classes = val_ds.classes

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    indices = np.random.choice(len(val_ds), n, replace=False)
    model.eval()

    for i, idx in enumerate(indices):
        img_t, label = val_ds[idx]
        inp = img_t.unsqueeze(0).to(device)
        inp.requires_grad_(True)

        cam = gradcam.generate(inp)
        pred = model(inp).argmax(1).item()

        # Denormalize
        img_np = img_t.permute(1, 2, 0).numpy()
        img_disp = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

        overlay = gradcam.overlay(img_bgr, cam, alpha=0.45)

        axes[i][0].imshow(img_disp)
        axes[i][0].set_title(f'True: {classes[label]}')
        axes[i][0].axis('off')

        axes[i][1].imshow(cam, cmap='jet')
        axes[i][1].set_title('GradCAM Heatmap')
        axes[i][1].axis('off')

        axes[i][2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i][2].set_title(f'Pred: {classes[pred]}')
        axes[i][2].axis('off')

    gradcam.remove_hooks()
    fig.suptitle('Pain Detection — GradCAM Visualization', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / 'gradcam_samples.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  GradCAM samples saved: {out_dir / 'gradcam_samples.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.expanduser('~/Documents/datasets/Pain'))
    parser.add_argument('--out', default='models/pain')
    parser.add_argument('--mode', default='binary', choices=['binary', 'four_class'])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(f" PAIN DETECTION TRAINING (mode={args.mode})")
    print("=" * 60)

    # --- Data ---
    data_root = find_pain_dataset(Path(args.data))
    print(f"\nDataset: {data_root}")

    # Check for pre-split
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'

    train_tf = get_classification_transforms(args.img_size, train=True)
    val_tf = get_classification_transforms(args.img_size, train=False)

    if train_dir.exists() and val_dir.exists():
        train_ds = PainDataset(train_dir, train_tf, args.mode)
        val_ds = PainDataset(val_dir, val_tf, args.mode)
    else:
        # In-memory split
        from collections import defaultdict
        import random
        full_ds = PainDataset(data_root, train_tf, args.mode)
        targets = full_ds.targets

        class_indices = defaultdict(list)
        for i, t in enumerate(targets):
            class_indices[t].append(i)

        train_idx, val_idx = [], []
        random.seed(42)
        for cls, idxs in class_indices.items():
            random.shuffle(idxs)
            split = int(len(idxs) * 0.8)
            train_idx.extend(idxs[:split])
            val_idx.extend(idxs[split:])

        class SplitDataset(torch.utils.data.Dataset):
            def __init__(self, base, indices, transform):
                self.base = base
                self.indices = indices
                self.base.transform = transform
            def __len__(self): return len(self.indices)
            def __getitem__(self, i): return self.base[self.indices[i]]
            @property
            def classes(self): return self.base.classes

        train_ds = SplitDataset(full_ds, train_idx, train_tf)
        val_ds = SplitDataset(PainDataset(data_root, val_tf, args.mode), val_idx, val_tf)

    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Classes: {classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    with open(out_dir / 'class_names.json', 'w') as f:
        json.dump({'classes': classes, 'mode': args.mode}, f, indent=2)

    # Weighted sampler
    if hasattr(train_ds, 'targets'):
        targets_list = train_ds.targets
    else:
        targets_list = [train_ds[i][1] for i in range(len(train_ds))]
    counts = np.bincount(targets_list, minlength=num_classes)
    w = 1.0 / (counts + 1e-6)
    sample_weights = [w[t] for t in targets_list]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Model
    model = build_model(num_classes).to(device)

    # Loss
    loss_w = torch.tensor(w / w.sum() * num_classes, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=0.1)

    # Differential LR
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

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.
    best_preds, best_labels = [], []

    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
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
                'classes': classes,
                'mode': args.mode,
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

    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    ax.set_title(f'Pain Detection — Best Val Acc: {best_val_acc:.4f}')
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # GradCAM visualization
    try:
        save_gradcam_samples(model, val_ds, device, out_dir / 'viz')
    except Exception as e:
        print(f"  (GradCAM visualization skipped: {e})")

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE — Best val acc: {best_val_acc:.4f}")
    print(f" Model: {out_dir / 'best_model.pth'}")
    print("\n" + classification_report(best_labels, best_preds, target_names=classes))
    print("=" * 60)


if __name__ == '__main__':
    main()
