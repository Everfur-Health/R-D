"""
Canine Health AI — Shared Utilities
Common transforms, metrics, visualization and helpers used across all training modules.
"""
import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

# ===========================================================================
# Color palettes
# ===========================================================================

# Eye disease segmentation colors (RGBA)
EYE_SEG_COLORS = {
    0: (0,   0,   0,   0),    # background — transparent
    1: (255, 100, 100, 180),  # corneal edema — red
    2: (255, 200, 50,  180),  # episcleral congestion — orange
    3: (50,  200, 255, 180),  # epiphora — blue
    4: (150, 255, 100, 180),  # cherry eye — green
}

EYE_SEG_NAMES = {
    0: 'Background',
    1: 'Corneal Edema',
    2: 'Episcleral Congestion',
    3: 'Epiphora',
    4: 'Cherry Eye',
}

# Skin disease label mapping (will be overridden by class_names.json if available)
SKIN_DISPLAY_NAMES = {
    'hotspot': 'Hot Spot (Acute Moist Dermatitis)',
    'ringworm': 'Ringworm (Dermatophytosis)',
    'flea_allergy': 'Flea Allergy Dermatitis',
    'flea allergy': 'Flea Allergy Dermatitis',
    'mange': 'Mange (Mite Infestation)',
    'normal': 'Healthy Skin',
}

SKIN_SEVERITY = {
    'hotspot': 'Moderate — requires veterinary treatment',
    'ringworm': 'Moderate — contagious, antifungal needed',
    'flea_allergy': 'Mild-Moderate — flea control + antihistamine',
    'flea allergy': 'Mild-Moderate — flea control + antihistamine',
    'mange': 'Serious — requires prescription treatment',
    'normal': 'No treatment needed',
}

# ===========================================================================
# Standard image transforms for classification (using albumentations)
# ===========================================================================

def get_classification_transforms(img_size=224, train=True):
    """
    Standard augmentation pipeline for classification tasks.
    Uses albumentations for speed and flexibility.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.7),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=int(img_size * 1.1), width=int(img_size * 1.1)),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_segmentation_transforms(img_size=320, train=True):
    """
    Paired image+mask transforms for segmentation.
    Geometric transforms applied identically to image AND mask.
    Color transforms on image ONLY.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.4),
            # Color jitter image only — handled separately via additional_targets in calling code
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
# GradCAM
# ===========================================================================

class GradCAM:
    """
    GradCAM for EfficientNet (timm) models.

    Fix log:
      v2 — hook onto model.blocks[-1][-1] (last individual block, not the
           Sequential container) to get a proper (B,C,H,W) activation tensor.
           Use .reshape(-1) instead of .squeeze() so probs is always 1-D
           even when batch=1 and num_classes=1.
           Single forward-backward: generate() drives everything; callers
           must NOT do their own forward-backward before calling generate().
    """
    def __init__(self, model, target_layer_name: Optional[str] = None):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hook_handles = []

        if target_layer_name is not None:
            # Navigate dotted path: e.g. "blocks.6.2"
            target = model
            for part in target_layer_name.split('.'):
                target = getattr(target, part)
        else:
            # timm EfficientNet-B4: model.blocks is an nn.Sequential of stages;
            # each stage is an nn.Sequential of MBConv blocks.
            # We want the very last MBConv block — blocks[-1][-1].
            # Fall back gracefully for other architectures.
            target = None
            try:
                target = model.blocks[-1][-1]   # last MBConv block ✓
            except (AttributeError, IndexError, TypeError):
                pass
            if target is None:
                try:
                    target = model.blocks[-1]    # last stage
                except AttributeError:
                    pass
            if target is None:
                # Generic fallback: walk children, grab last conv-ish layer
                for mod in reversed(list(model.modules())):
                    if isinstance(mod, torch.nn.Conv2d):
                        target = mod
                        break
            if target is None:
                target = list(model.children())[-2]

        self._hook_handles.append(
            target.register_forward_hook(self._save_activation)
        )
        self._hook_handles.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        # output may be a tensor or a tuple (some blocks return tuples)
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0]: gradient w.r.t. the layer output
        if grad_out and grad_out[0] is not None:
            self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """
        Single forward+backward pass. Returns heatmap (H, W) in [0, 1].
        DO NOT call model() before this — generate() owns the forward pass.
        """
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]

        self.model.eval()
        with torch.enable_grad():
            # Detach from any previous graph, re-attach for this pass
            inp = input_tensor.detach().requires_grad_(True)
            output = self.model(inp)                    # forward — hooks fire

            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())

            self.model.zero_grad()
            output[0, class_idx].backward()             # backward — hooks fire

        if self.activations is None or self.gradients is None:
            # Hooks didn't fire (unusual architecture) → return blank map
            return np.zeros((h_in, w_in), dtype=np.float32)

        # activations: (1, C, H', W')  gradients: same shape
        acts = self.activations.float()
        grads = self.gradients.float()

        # Ensure 4-D (some blocks return 2-D or 3-D tensors)
        if acts.dim() == 2:
            acts = acts.unsqueeze(-1).unsqueeze(-1)
            grads = grads.unsqueeze(-1).unsqueeze(-1)
        elif acts.dim() == 3:
            acts = acts.unsqueeze(0)
            grads = grads.unsqueeze(0)

        weights = grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)         # (1, 1, H', W')
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(h_in, w_in),
                            mode='bilinear', align_corners=False)

        # Always produce a 2-D (H, W) array — never 0-d
        cam_np = cam.squeeze(0).squeeze(0).cpu().numpy()        # (H, W)
        cam_np = cam_np.astype(np.float32)
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np

    def overlay(self, image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image.
        image_np: (H, W, 3) uint8 BGR
        cam: (H, W) float [0,1]
        """
        # Ensure cam is 2-D and same spatial size as image
        if cam.ndim != 2:
            cam = cam.reshape(image_np.shape[:2])
        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


# ===========================================================================
# Segmentation visualization
# ===========================================================================

def colorize_seg_mask(mask: np.ndarray, colors: dict) -> np.ndarray:
    """
    Convert class mask (H, W) integer → color image (H, W, 3) uint8 BGR.
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgba in colors.items():
        if rgba[3] == 0:
            continue  # skip transparent (background)
        r, g, b = rgba[:3]
        color_img[mask == class_id] = [b, g, r]  # BGR
    return color_img


def overlay_seg_mask(image: np.ndarray, mask: np.ndarray, colors: dict, alpha: float = 0.5) -> np.ndarray:
    """Blend segmentation mask onto image."""
    color_mask = colorize_seg_mask(mask, colors)
    # Only blend where mask is non-zero
    mask_bool = (mask > 0)[..., np.newaxis]
    blended = image.copy()
    blended = np.where(mask_bool,
                       cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0),
                       image)
    return blended.astype(np.uint8)


def draw_seg_legend(classes_present: list, colors: dict, names: dict) -> np.ndarray:
    """Draw a legend bar for visible segmentation classes."""
    if not classes_present:
        return np.zeros((30, 200, 3), dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(4, 0.4 * len(classes_present)))
    patches = []
    for c in classes_present:
        if c == 0:
            continue
        rgba = colors[c]
        patches.append(mpatches.Patch(
            color=(rgba[0]/255, rgba[1]/255, rgba[2]/255),
            label=names.get(c, f'Class {c}')
        ))
    ax.legend(handles=patches, loc='center', frameon=False, fontsize=9)
    ax.axis('off')
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


# ===========================================================================
# Training helpers
# ===========================================================================

class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, metric: float) -> bool:
        improved = (self.mode == 'max' and metric > self.best + self.min_delta) or \
                   (self.mode == 'min' and metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def compute_class_weights(dataset_path: str, num_classes: int = None) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from dataset directory.
    Assumes class = subdirectory name.
    """
    from collections import Counter
    counts = Counter()
    for d in Path(dataset_path).iterdir():
        if d.is_dir():
            n = sum(1 for f in d.rglob('*')
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            counts[d.name] = n
    total = sum(counts.values())
    sorted_classes = sorted(counts.keys())
    weights = torch.tensor([total / (len(counts) * counts[c]) for c in sorted_classes],
                            dtype=torch.float32)
    return weights


def save_training_curves(history: dict, save_path: str):
    """Save train/val accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    elif 'val_miou' in history:
        ax2.plot(epochs, history['val_miou'], 'g-', label='mIoU')
        ax2.set_title('mIoU')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves saved: {save_path}")


def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation for small datasets."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===========================================================================
# Inference helpers
# ===========================================================================

def preprocess_image(image_path_or_array, img_size=224):
    """
    Load and preprocess image for inference.
    Returns: tensor (1, 3, H, W), original numpy (H, W, 3) BGR
    """
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
    else:
        img = image_path_or_array.copy()

    if img is None:
        raise ValueError(f"Could not load image")

    original = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original


def image_to_b64(image_np: np.ndarray, ext: str = '.jpg') -> str:
    """Convert numpy image (BGR) to base64 string for web API."""
    import base64
    _, buffer = cv2.imencode(ext, image_np)
    return base64.b64encode(buffer).decode('utf-8')


def b64_to_image(b64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image (BGR)."""
    import base64
    data = base64.b64decode(b64_str)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
