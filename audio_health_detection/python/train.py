#!/usr/bin/env python3
"""
Training Script for Canine Cough Classifier

This script trains the CNN14-based model to classify dog coughs into
health categories. It handles the complete training loop including
data loading, optimization, validation, and checkpointing.

TRAINING CONCEPTS EXPLAINED:
===========================

1. THE TRAINING LOOP
   
   For each epoch (full pass through data):
     For each batch of samples:
       1. Forward pass: Input → Model → Predictions
       2. Compute loss: How wrong are the predictions?
       3. Backward pass: Compute gradients (which way to adjust weights)
       4. Update weights: Move a small step in the right direction
     
     Validate on held-out data (no weight updates)
     Save checkpoint if validation improved

2. LOSS FUNCTION: Cross-Entropy
   
   Measures how "surprised" the model is by the correct answer.
   - Model predicts [0.1, 0.7, 0.1, 0.1] (thinks class 1)
   - Actual label is class 1
   - Loss is low (model was right!)
   
   - Model predicts [0.1, 0.7, 0.1, 0.1] 
   - Actual label is class 2
   - Loss is high (model was wrong!)
   
   Loss = -log(predicted_probability_of_correct_class)

3. OPTIMIZER: AdamW
   
   Adam = Adaptive Moment estimation
   - Adjusts learning rate per-parameter automatically
   - Maintains "momentum" to escape local minima
   
   AdamW adds "weight decay" (L2 regularization)
   - Penalizes large weights
   - Prevents overfitting
   - Like adding a "tax" on model complexity

4. LEARNING RATE SCHEDULE
   
   Start with learning rate (how big steps to take).
   If validation loss stops improving:
   - Reduce learning rate (take smaller steps)
   - Allows fine-tuning near optimal solution
   
   We use ReduceLROnPlateau:
   - Monitor validation loss
   - If no improvement for 5 epochs, cut LR by half
   - Minimum LR: 1e-7

5. EARLY STOPPING
   
   Problem: Training too long causes overfitting
   - Model memorizes training data
   - Performs poorly on new data
   
   Solution: Stop when validation loss stops improving
   - Track best validation loss
   - If no improvement for N epochs, stop
   - Use the checkpoint from best epoch

6. BATCH SIZE TRADEOFFS
   
   - Larger batches: More stable gradients, faster (uses GPU parallelism)
   - Smaller batches: More noise (can help escape local minima), less memory
   - Sweet spot depends on dataset size and GPU memory
   - We use 16-32 typically

USAGE:
======
    # Basic training (looks for data in data/augmented)
    python train.py
    
    # Custom data path
    python train.py --data-dir data/processed
    
    # Resume from checkpoint
    python train.py --resume models/checkpoints/latest/best_model.pth
    
    # Quick test with fewer epochs
    python train.py --epochs 5 --no-augmented
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import our modules (from same directory)
from preprocessing import AudioPreprocessor
from model import CoughClassifier

# For metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not found - install for detailed metrics: pip install scikit-learn")


# ============================================
# Configuration
# ============================================

class TrainingConfig:
    """
    All training hyperparameters in one place.
    
    Hyperparameters are settings that control the training process.
    They're not learned - we set them before training.
    """
    
    # Data paths
    DATA_DIR = Path("../data/augmented")  # Use augmented data by default
    CHECKPOINT_DIR = Path("../models/checkpoints")
    PRETRAINED_PATH = Path("../models/pretrained/Cnn14_mAP=0.431.pth")
    
    # Audio preprocessing (must match what we used in preprocessing.py)
    SAMPLE_RATE = 16000
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 320
    F_MIN = 50
    F_MAX = 8000
    TARGET_DURATION = 5.0  # seconds - all clips padded/trimmed to this
    
    # Model
    NUM_CLASSES = 4
    DROPOUT = 0.5
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4       # Initial learning rate
    WEIGHT_DECAY = 1e-5        # L2 regularization strength
    EPOCHS = 50                # Maximum number of epochs
    PATIENCE = 10              # Early stopping patience
    
    # Learning rate scheduler
    LR_FACTOR = 0.5            # Multiply LR by this when reducing
    LR_PATIENCE = 5            # Epochs to wait before reducing LR
    MIN_LR = 1e-7              # Minimum learning rate
    
    # Data loading
    NUM_WORKERS = 4            # Parallel data loading processes
    PIN_MEMORY = True          # Faster GPU transfer
    
    # Classes
    CLASSES = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']


# ============================================
# Dataset Class
# ============================================

class CoughDataset(Dataset):
    """
    PyTorch Dataset for loading cough audio files.
    
    A Dataset in PyTorch must implement:
    - __len__: Return number of samples
    - __getitem__: Return one sample (features, label)
    
    The DataLoader then handles batching, shuffling, and parallel loading.
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str,  # 'train', 'val', or 'test'
        config: TrainingConfig,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing train/val/test splits
            split: Which split to load
            config: Training configuration
        """
        self.split_dir = data_dir / split
        self.config = config
        
        # Create preprocessor for loading and feature extraction
        self.preprocessor = AudioPreprocessor(
            sample_rate=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )
        
        # Target length in samples
        self.target_samples = int(config.TARGET_DURATION * config.SAMPLE_RATE)
        
        # Find all audio files with their labels
        self.samples = []
        
        for class_idx, class_name in enumerate(config.CLASSES):
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            
            for audio_file in class_dir.glob("*.wav"):
                self.samples.append({
                    'path': audio_file,
                    'label': class_idx,
                    'class_name': class_name,
                })
        
        # Shuffle samples (important for training)
        random.shuffle(self.samples)
        
        print(f"  {split}: Loaded {len(self.samples)} samples")
        
        # Print class distribution
        class_counts = {}
        for sample in self.samples:
            name = sample['class_name']
            class_counts[name] = class_counts.get(name, 0) + 1
        
        for name, count in sorted(class_counts.items()):
            print(f"    {name}: {count}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess one sample.
        
        Args:
            idx: Index of sample to load
            
        Returns:
            Tuple of (mel_spectrogram_tensor, label)
            - mel_spectrogram: Shape (1, n_mels, time_frames)
            - label: Integer class label (0-3)
        """
        sample = self.samples[idx]
        
        # Load audio
        audio = self.preprocessor.load_audio(sample['path'])
        
        # Pad or trim to target length
        audio = self.preprocessor.pad_or_trim(audio, self.target_samples)
        
        # Extract mel spectrogram
        mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
        
        # Convert to tensor with channel dimension: (1, n_mels, time)
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        return mel_tensor, sample['label']


# ============================================
# Training Functions
# ============================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    An epoch is one complete pass through the training data.
    
    Args:
        model: The neural network
        dataloader: Provides batches of training data
        criterion: Loss function (cross-entropy)
        optimizer: Updates weights (AdamW)
        device: CPU or GPU
        epoch: Current epoch number (for display)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()  # Set to training mode (enables dropout, batchnorm updates)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Move data to device (GPU if available)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients from previous batch
        outputs = model(inputs)  # Get predictions
        loss = criterion(outputs['logits'], targets)  # Compute loss
        
        # Backward pass
        loss.backward()  # Compute gradients
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = outputs['logits'].max(1)  # Get class with highest score
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss / (batch_idx + 1):.4f}",
            'acc': f"{100. * correct / total:.1f}%"
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on validation/test data.
    
    No gradient computation or weight updates - just measure performance.
    
    Args:
        model: The neural network
        dataloader: Provides batches of validation data
        criterion: Loss function
        device: CPU or GPU
        epoch: Current epoch number
        
    Returns:
        Tuple of (loss, accuracy, predictions_array, targets_array)
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # No gradient computation needed for validation
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass only
            outputs = model(inputs)
            loss = criterion(outputs['logits'], targets)
            
            total_loss += loss.item()
            
            # Collect predictions
            _, predicted = outputs['logits'].max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = (all_predictions == all_targets).mean()
    
    return avg_loss, accuracy, all_predictions, all_targets


# ============================================
# Main Training Function
# ============================================

def train(
    config: TrainingConfig,
    resume_from: Optional[Path] = None,
) -> Path:
    """
    Main training function.
    
    This orchestrates the entire training process:
    1. Set up data loaders
    2. Initialize model, optimizer, scheduler
    3. Training loop with validation
    4. Save best checkpoint
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Path to the best model checkpoint
    """
    print("=" * 60)
    print("Canine Cough Classifier - Training")
    print("=" * 60)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\n⚠ Using CPU (training will be slow)")
        print("  For faster training, use a machine with NVIDIA GPU")
    
    print(f"\nClasses: {config.CLASSES}")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.CHECKPOINT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {run_dir}")
    
    # Save configuration
    config_dict = {k: str(v) if isinstance(v, Path) else v 
                   for k, v in vars(config).items() 
                   if not k.startswith('_')}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # ====================================
    # Create Datasets and DataLoaders
    # ====================================
    
    print("\n" + "-" * 40)
    print("Loading datasets...")
    print("-" * 40)
    
    train_dataset = CoughDataset(config.DATA_DIR, 'train', config)
    val_dataset = CoughDataset(config.DATA_DIR, 'val', config)
    
    if len(train_dataset) == 0:
        print(f"\n⚠️ No training data found in {config.DATA_DIR / 'train'}")
        print("   Run organize_data.py and augment_data.py first!")
        sys.exit(1)
    
    # DataLoaders handle batching and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle each epoch for better training
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,  # Drop incomplete final batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    
    print(f"\nBatches per epoch: {len(train_loader)} train, {len(val_loader)} val")
    
    # ====================================
    # Compute Class Weights for Imbalanced Data
    # ====================================
    # If some classes have fewer samples, we weight their loss higher
    # This prevents the model from just predicting the majority class
    
    class_counts = [0] * config.NUM_CLASSES
    for sample in train_dataset.samples:
        class_counts[sample['label']] += 1
    
    print(f"\nClass distribution in training data:")
    for i, (name, count) in enumerate(zip(config.CLASSES, class_counts)):
        pct = count / sum(class_counts) * 100 if sum(class_counts) > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Compute weights: inverse of frequency (rare classes get higher weight)
    # Adding small epsilon to avoid division by zero for missing classes
    total_samples = sum(class_counts)
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = total_samples / (config.NUM_CLASSES * count)
        else:
            weight = 0.0  # No samples = no contribution to loss
            print(f"  ⚠ Warning: Missing samples for a class!")
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights (higher = rarer): {class_weights.tolist()}")
    
    # ====================================
    # Initialize Model
    # ====================================
    
    print("\n" + "-" * 40)
    print("Initializing model...")
    print("-" * 40)
    
    # Check for pretrained weights
    pretrained_path = None
    if config.PRETRAINED_PATH.exists():
        pretrained_path = str(config.PRETRAINED_PATH)
        print(f"✓ Found pretrained weights: {pretrained_path}")
    else:
        print("⚠ No pretrained weights found - training from scratch")
    
    model = CoughClassifier(
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        pretrained_path=pretrained_path,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # ====================================
    # Loss, Optimizer, Scheduler
    # ====================================
    
    # Cross-entropy loss for classification
    # Using class weights to handle imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW optimizer (Adam with decoupled weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize validation loss
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
        # Note: 'verbose' parameter removed in PyTorch 2.x
        # We'll print LR changes manually in the training loop
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and resume_from.exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # ====================================
    # Training Loop
    # ====================================
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }
    
    patience_counter = 0
    best_checkpoint_path = run_dir / "best_model.pth"
    
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print('=' * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, predictions, targets = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model.save_checkpoint(
                str(best_checkpoint_path),
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                train_loss=train_loss,
                train_acc=train_acc,
            )
            print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{config.PATIENCE} epochs")
            
            if patience_counter >= config.PATIENCE:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save history after each epoch
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    # ====================================
    # Final Evaluation
    # ====================================
    
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    print(f"\nLoading best model from: {best_checkpoint_path}")
    model = CoughClassifier.load_checkpoint(str(best_checkpoint_path), device=str(device))
    
    # Load test dataset
    test_dataset = CoughDataset(config.DATA_DIR, 'test', config)
    
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        
        test_loss, test_acc, test_preds, test_targets = validate(
            model, test_loader, criterion, device, epoch=-1
        )
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        if HAS_SKLEARN:
            # Only include labels that actually appear in the data
            unique_labels = sorted(set(test_targets) | set(test_preds))
            label_names = [config.CLASSES[i] for i in unique_labels]
            
            print("\nClassification Report:")
            print(classification_report(
                test_targets, test_preds,
                labels=unique_labels,
                target_names=label_names,
                digits=3,
            ))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(test_targets, test_preds, labels=unique_labels)
            
            # Pretty print confusion matrix
            print("\n" + " " * 20 + "Predicted")
            print(" " * 15 + "  ".join(f"{c[:8]:>8s}" for c in label_names))
            for i, row in enumerate(cm):
                label = f"Actual {label_names[i][:8]:>8s}"
                print(f"{label} " + "  ".join(f"{v:8d}" for v in row))
            
            # Save test results
            with open(run_dir / "test_results.json", "w") as f:
                json.dump({
                    'test_loss': float(test_loss),
                    'test_accuracy': float(test_acc),
                    'confusion_matrix': cm.tolist(),
                    'class_names': label_names,
                }, f, indent=2)
    else:
        print("\n⚠ No test data found - skipping final evaluation")
    
    # ====================================
    # Create "latest" symlink
    # ====================================
    
    latest_link = config.CHECKPOINT_DIR / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        import shutil
        shutil.rmtree(latest_link)
    
    latest_link.symlink_to(run_dir.name)
    print(f"\n✓ Created symlink: {latest_link} → {run_dir.name}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved to: {best_checkpoint_path}")
    print(f"Training history: {run_dir / 'history.json'}")
    
    return best_checkpoint_path


# ============================================
# Command Line Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Train the canine cough classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                              # Train with defaults
  python train.py --data-dir data/processed   # Use non-augmented data
  python train.py --epochs 10 --batch-size 8  # Quick test run
  python train.py --resume path/to/checkpoint # Resume training
        """
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=TrainingConfig.DATA_DIR,
        help=f"Data directory (default: {TrainingConfig.DATA_DIR})"
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=TrainingConfig.EPOCHS,
        help=f"Number of epochs (default: {TrainingConfig.EPOCHS})"
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=TrainingConfig.BATCH_SIZE,
        help=f"Batch size (default: {TrainingConfig.BATCH_SIZE})"
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=TrainingConfig.LEARNING_RATE,
        help=f"Learning rate (default: {TrainingConfig.LEARNING_RATE})"
    )
    
    parser.add_argument(
        '--resume', '-r',
        type=Path,
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        '--no-augmented',
        action='store_true',
        help="Use data/processed instead of data/augmented"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create config with CLI overrides
    config = TrainingConfig()
    config.DATA_DIR = args.data_dir
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    if args.no_augmented:
        config.DATA_DIR = Path("data/processed")
    
    # Run training
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
