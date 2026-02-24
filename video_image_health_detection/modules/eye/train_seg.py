#!/usr/bin/env python3
"""
Canine Health AI — Eye Disease Segmentation
Model: SegFormer-B2 (HuggingFace) fine-tuned on DogEyeSeg4
Classes: 0=background, 1=corneal edema, 2=episcleral congestion, 3=epiphora, 4=cherry eye

Dataset: DogEyeSeg4
  - 320×320 PNG images + grayscale masks (pixel values 0-4)
  - Located at: ~/Documents/datasets/Eye/DogEyeSeg4/

Key challenge: Background is ~85% of pixels → Dice+CE combined loss essential

Usage:
  python modules/eye/train_seg.py --data ~/Documents/datasets/Eye --epochs 50
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.utils import EarlyStopping, save_training_curves, EYE_SEG_COLORS, EYE_SEG_NAMES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================================================================
# Dataset
# ===========================================================================

class DogEyeSegDataset(Dataset):
    """
    DogEyeSeg4 dataset loader.
    Expected structure:
      root/
        Images/  (or DogEyeSeg4/Images/)
          *.png
        Masks/
          Gray/
            *.png  (pixel values 0-4 = class ids)
    """
    NUM_CLASSES = 5  # 0=bg, 1-4 = diseases

    def __init__(self, root: Path, transform=None, split: str = 'train',
                 val_fraction: float = 0.2, seed: int = 42):
        self.transform = transform
        self.samples = []

        # Find images directory
        imgs_dir = self._find_dir(root, 'Images', 'images')
        masks_dir = self._find_dir(root, 'Masks/Gray', 'masks/gray')

        if imgs_dir is None or masks_dir is None:
            raise FileNotFoundError(
                f"Cannot find Images/ and Masks/Gray/ under {root}\n"
                f"  Found: {list(root.iterdir())}"
            )

        # Match image-mask pairs by stem
        mask_dict = {p.stem: p for p in masks_dir.glob('*.png')}
        all_samples = []
        for img_p in sorted(imgs_dir.glob('*.png')):
            if img_p.stem in mask_dict:
                all_samples.append((img_p, mask_dict[img_p.stem]))
        # Also try jpg
        for img_p in sorted(imgs_dir.glob('*.jpg')):
            if img_p.stem in mask_dict:
                all_samples.append((img_p, mask_dict[img_p.stem]))

        if not all_samples:
            raise FileNotFoundError(f"No image-mask pairs found under {root}")

        print(f"  Found {len(all_samples)} image-mask pairs")

        # Train/val split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        split_idx = int(len(all_samples) * (1 - val_fraction))

        if split == 'train':
            self.samples = [all_samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [all_samples[i] for i in indices[split_idx:]]

    def _find_dir(self, root: Path, *candidates) -> Path:
        for c in candidates:
            p = root / c
            if p.exists():
                return p
            # Recursive
            for sub in root.rglob(Path(c).name):
                if sub.is_dir() and str(sub).endswith(c.replace('/', os.sep)):
                    return sub
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].long()

        return img, mask


# ===========================================================================
# Loss
# ===========================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation.
    Critical for handling the severe background imbalance in DogEyeSeg4.
    Background ≈85% of pixels. CrossEntropy alone → model predicts all background.
    Dice directly optimizes IoU regardless of class frequencies.
    """
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        start_c = 1 if self.ignore_bg else 0
        dice_sum = 0.
        for c in range(start_c, self.num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            dice = (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_sum += 1 - dice

        return dice_sum / (self.num_classes - start_c)


class CombinedLoss(nn.Module):
    """Dice + CrossEntropy (weighted)."""
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.w = dice_weight

    def forward(self, logits, targets):
        return self.w * self.dice(logits, targets) + (1 - self.w) * self.ce(logits, targets)


# ===========================================================================
# Metrics
# ===========================================================================

@torch.no_grad()
def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int) -> dict:
    """Per-class IoU."""
    iou = {}
    for c in range(num_classes):
        pred_c = pred_mask == c
        true_c = true_mask == c
        intersection = (pred_c & true_c).sum()
        union = (pred_c | true_c).sum()
        iou[c] = float(intersection) / (float(union) + 1e-8) if union > 0 else float('nan')
    return iou


# ===========================================================================
# Transforms
# ===========================================================================

def get_seg_transforms(img_size=320, train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})


# ===========================================================================
# Model
# ===========================================================================

def build_segformer(num_classes: int = 5) -> nn.Module:
    """
    SegFormer-B2 from HuggingFace.
    Pre-trained on ADE20K (150 categories) → rich semantic features.
    Why SegFormer over U-Net:
      - Hierarchical Mix-Transformer backbone captures both local texture
        (corneal cloudiness) and global context (eye anatomy)
      - No positional encoding → better generalization to different image sizes
      - Lightweight all-MLP decoder → fast inference
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b2',
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label={i: EYE_SEG_NAMES[i] for i in range(num_classes)},
        label2id={v: k for k, v in EYE_SEG_NAMES.items()},
    )
    return model


# ===========================================================================
# Training
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.
    for imgs, masks in tqdm(loader, desc='  Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=imgs)
        logits = outputs.logits  # (B, C, H/4, W/4)
        # Upsample to mask resolution
        logits = F.interpolate(logits, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes: int = 5):
    model.eval()
    total_loss = 0.
    all_iou = {c: [] for c in range(num_classes)}

    for imgs, masks in tqdm(loader, desc='  Val  ', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(pixel_values=imgs)
        logits = F.interpolate(outputs.logits, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        trues = masks.cpu().numpy()
        for b in range(preds.shape[0]):
            iou = compute_iou(preds[b], trues[b], num_classes)
            for c, v in iou.items():
                if not np.isnan(v):
                    all_iou[c].append(v)

    mean_iou = {c: float(np.mean(v)) if v else 0.0 for c, v in all_iou.items()}
    disease_iou = [mean_iou[c] for c in range(1, num_classes)]  # exclude bg
    miou = float(np.mean(disease_iou))
    return total_loss / len(loader), miou, mean_iou


def save_sample_predictions(model, val_ds, device, out_dir: Path, n_samples: int = 6):
    """Save overlay visualization of model predictions."""
    from shared.utils import colorize_seg_mask, overlay_seg_mask
    model.eval()
    out_dir.mkdir(exist_ok=True)

    indices = np.random.choice(len(val_ds), min(n_samples, len(val_ds)), replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_t, mask_true = val_ds[idx]
            img_input = img_t.unsqueeze(0).to(device)
            out = model(pixel_values=img_input)
            logits = F.interpolate(out.logits, size=(320, 320), mode='bilinear', align_corners=False)
            pred_mask = logits.argmax(1).squeeze().cpu().numpy()
            true_mask = mask_true.numpy() if hasattr(mask_true, 'numpy') else np.array(mask_true)

            # Denormalize image for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            from shared.utils import colorize_seg_mask, overlay_seg_mask
            axes[i][0].imshow(img_np)
            axes[i][0].set_title('Image')
            axes[i][0].axis('off')

            true_color = colorize_seg_mask(true_mask.astype(int), EYE_SEG_COLORS)
            axes[i][1].imshow(cv2.cvtColor(overlay_seg_mask(img_bgr, true_mask.astype(int), EYE_SEG_COLORS), cv2.COLOR_BGR2RGB))
            axes[i][1].set_title('Ground Truth')
            axes[i][1].axis('off')

            axes[i][2].imshow(cv2.cvtColor(overlay_seg_mask(img_bgr, pred_mask, EYE_SEG_COLORS), cv2.COLOR_BGR2RGB))
            axes[i][2].set_title('Prediction')
            axes[i][2].axis('off')

    fig.suptitle('Eye Disease Segmentation — Sample Predictions', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / 'sample_predictions.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sample predictions saved: {out_dir / 'sample_predictions.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.expanduser('~/Documents/datasets/Eye'))
    parser.add_argument('--out', default='models/eye_seg')
    parser.add_argument('--img_size', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(" EYE DISEASE SEGMENTATION TRAINING")
    print(f" Device: {device}  |  Batch: {args.batch_size}  |  Size: {args.img_size}")
    print("=" * 60)

    # --- Data ---
    data_root = Path(args.data)
    # Try DogEyeSeg4 subdirectory first
    seg_root = data_root / 'DogEyeSeg4' if (data_root / 'DogEyeSeg4').exists() else data_root

    train_tf = get_seg_transforms(args.img_size, train=True)
    val_tf = get_seg_transforms(args.img_size, train=False)

    print("\nLoading train split...")
    train_ds = DogEyeSegDataset(seg_root, transform=train_tf, split='train')
    print("\nLoading val split...")
    val_ds = DogEyeSegDataset(seg_root, transform=val_tf, split='val')
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- Model ---
    NUM_CLASSES = 5
    print(f"\nBuilding SegFormer-B2 ({NUM_CLASSES} classes)...")
    model = build_segformer(NUM_CLASSES).to(device)

    # --- Loss & optimizer ---
    criterion = CombinedLoss(NUM_CLASSES, dice_weight=0.5)

    # SegFormer: all params same LR (small)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    early_stop = EarlyStopping(patience=10, mode='max')

    # --- Training loop ---
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    best_miou = 0.

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, miou, per_class_iou = eval_epoch(model, val_loader, criterion, device, NUM_CLASSES)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_miou'].append(miou)

        iou_str = '  '.join(f"{EYE_SEG_NAMES[c][:8]}={per_class_iou[c]:.3f}"
                            for c in range(1, NUM_CLASSES))
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"t_loss={t_loss:.4f}  v_loss={v_loss:.4f}  mIoU={miou:.4f}")
        print(f"    {iou_str}")

        if miou > best_miou:
            best_miou = miou
            model.save_pretrained(str(out_dir / 'best_model'))
            with open(out_dir / 'best_metrics.json', 'w') as f:
                json.dump({'epoch': epoch, 'miou': miou, 'per_class_iou': per_class_iou}, f, indent=2)
            print(f"  ✓ Saved best model (mIoU={best_miou:.4f})")

        if early_stop(miou):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save curves and samples
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)
    save_training_curves(history, str(out_dir / 'training_curves.png'))
    save_sample_predictions(model, val_ds, device, out_dir / 'viz')

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE — Best mIoU: {best_miou:.4f}")
    print(f" Model saved: {out_dir / 'best_model'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
