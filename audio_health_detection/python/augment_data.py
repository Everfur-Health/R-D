#!/usr/bin/env python3
"""
Data Augmentation Script for Canine Cough Detection

This script expands the training dataset by creating modified versions of
existing audio files. Augmentation is crucial for training robust models,
especially with small datasets.

DATA AUGMENTATION EXPLAINED:
===========================

1. WHY AUGMENT DATA?
   - Deep learning needs lots of data (thousands to millions of samples)
   - Real medical/veterinary audio datasets are small (hundreds)
   - Augmentation creates "new" training examples from existing ones
   - Helps model generalize (not memorize specific recordings)

2. HOW IT WORKS:
   Original cough recording → Apply random transformations → "New" sample
   
   The model sees the same cough with different:
   - Background noise levels
   - Recording volumes
   - Pitch (simulates different dog sizes)
   - Speed (natural variation in cough timing)

3. AUGMENTATIONS WE USE:
   
   a) NOISE ADDITION (most important!)
      - Real recordings have background noise
      - We add Gaussian noise at various levels (SNR 15-30 dB)
      - Makes model robust to recording quality variation
   
   b) PITCH SHIFTING (±3 semitones)
      - Different dogs have different vocal tract sizes
      - A Chihuahua cough sounds higher than a Great Dane
      - Shifting pitch simulates breed/size variation
   
   c) TIME STRETCHING (0.8x to 1.2x speed)
      - Coughs naturally vary in duration
      - Stretching/compressing makes model timing-invariant
      - 0.8x = 20% slower, 1.2x = 20% faster
   
   d) VOLUME SCALING (±6 dB)
      - Recording levels vary (distance to mic, sensitivity)
      - Scaling teaches model to ignore overall volume
   
   e) TIME SHIFTING (shift within ±30% of duration)
      - Cough might not start at exact beginning of clip
      - Shifting makes model position-invariant

4. IMPORTANT RULES:
   - ONLY augment training data (never val/test!)
   - Validation/test must represent real-world distribution
   - Keep original files too (augmented versions are additions)

USAGE:
======
    # Basic usage (augments data/processed/train → data/augmented/train)
    python augment_data.py
    
    # Custom augmentation factor (5x expansion)
    python augment_data.py --factor 5
    
    # Preview augmentations
    python augment_data.py --preview sample.wav
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Audio libraries
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Installing required libraries...")
    os.system("pip install librosa soundfile")
    import librosa
    import soundfile as sf


# ============================================
# Configuration
# ============================================

# Default paths
DEFAULT_INPUT_DIR = "../data/processed"
DEFAULT_OUTPUT_DIR = "../data/augmented"

# Audio settings
SAMPLE_RATE = 16000

# Number of augmented versions to create per original file
DEFAULT_AUGMENTATION_FACTOR = 5


# ============================================
# Augmentation Functions
# ============================================

def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add Gaussian noise at a specified Signal-to-Noise Ratio.
    
    SNR (Signal-to-Noise Ratio) in dB:
    - 30 dB = very clean (noise barely audible)
    - 20 dB = some noise (typical good recording)
    - 10 dB = noisy (challenging but still usable)
    
    Formula: SNR_dB = 10 * log10(signal_power / noise_power)
    
    Args:
        audio: Input audio array
        snr_db: Target SNR in decibels (higher = cleaner)
        
    Returns:
        Audio with added noise
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate required noise power for target SNR
    # SNR = 10 * log10(Ps/Pn) → Pn = Ps / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise with calculated power
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    
    # Add noise to signal
    noisy_audio = audio + noise
    
    # Clip to [-1, 1] range
    return np.clip(noisy_audio, -1.0, 1.0)


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift the pitch of audio by a number of semitones.
    
    Semitones are musical half-steps:
    - +12 semitones = one octave up (2x frequency)
    - -12 semitones = one octave down (0.5x frequency)
    - ±3 semitones is subtle but noticeable
    
    This uses librosa's pitch_shift which preserves duration
    (unlike simple resampling).
    
    Args:
        audio: Input audio array
        sr: Sample rate
        semitones: Number of semitones to shift (positive = higher pitch)
        
    Returns:
        Pitch-shifted audio
    """
    # librosa.effects.pitch_shift handles the complex math
    # It uses phase vocoder to shift pitch without changing duration
    shifted = librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=semitones,
    )
    return shifted


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Change the speed of audio without changing pitch.
    
    Rate interpretation:
    - rate = 1.0: no change
    - rate = 2.0: twice as fast (half duration)
    - rate = 0.5: half speed (double duration)
    
    Uses phase vocoder algorithm (librosa.effects.time_stretch).
    
    Args:
        audio: Input audio array
        rate: Speed multiplier (>1 = faster, <1 = slower)
        
    Returns:
        Time-stretched audio
    """
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched


def change_volume(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Change the volume of audio by a specified amount in decibels.
    
    Decibel interpretation:
    - +6 dB ≈ 2x amplitude (sounds notably louder)
    - -6 dB ≈ 0.5x amplitude (sounds notably quieter)
    - Human perception: ~1 dB is barely noticeable
    
    Args:
        audio: Input audio array
        gain_db: Volume change in decibels
        
    Returns:
        Volume-adjusted audio (clipped to [-1, 1])
    """
    # Convert dB to linear multiplier
    # dB = 20 * log10(multiplier) → multiplier = 10^(dB/20)
    multiplier = 10 ** (gain_db / 20)
    
    # Apply gain and clip
    return np.clip(audio * multiplier, -1.0, 1.0)


def time_shift(audio: np.ndarray, shift_fraction: float) -> np.ndarray:
    """
    Shift audio in time (circular shift).
    
    This moves the audio forward or backward in time, with
    the overflow wrapping around to the other end.
    
    Args:
        audio: Input audio array
        shift_fraction: Fraction of audio length to shift
                       Positive = shift right (add silence at start)
                       Negative = shift left
        
    Returns:
        Time-shifted audio
    """
    shift_samples = int(len(audio) * shift_fraction)
    return np.roll(audio, shift_samples)


def apply_random_augmentation(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, dict]:
    """
    Apply a random combination of augmentations.
    
    Each augmentation is applied with some probability, and
    parameters are randomly sampled within reasonable ranges.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        
    Returns:
        Tuple of (augmented_audio, dict_of_applied_augmentations)
    """
    augmented = audio.copy()
    applied = {}
    
    # 1. Noise addition (50% chance)
    if random.random() < 0.5:
        snr = random.uniform(15, 30)  # 15-30 dB SNR
        augmented = add_noise(augmented, snr_db=snr)
        applied['noise'] = f"SNR={snr:.1f}dB"
    
    # 2. Pitch shifting (50% chance)
    if random.random() < 0.5:
        semitones = random.uniform(-3, 3)
        augmented = pitch_shift(augmented, sr, semitones)
        applied['pitch'] = f"{semitones:+.1f} semitones"
    
    # 3. Time stretching (40% chance)
    if random.random() < 0.4:
        rate = random.uniform(0.85, 1.15)
        augmented = time_stretch(augmented, rate)
        applied['stretch'] = f"{rate:.2f}x"
    
    # 4. Volume change (50% chance)
    if random.random() < 0.5:
        gain = random.uniform(-6, 6)
        augmented = change_volume(augmented, gain)
        applied['volume'] = f"{gain:+.1f}dB"
    
    # 5. Time shift (30% chance)
    if random.random() < 0.3:
        shift = random.uniform(-0.2, 0.2)
        augmented = time_shift(augmented, shift)
        applied['shift'] = f"{shift*100:+.0f}%"
    
    # Ensure at least one augmentation was applied
    if not applied:
        # Apply noise as fallback
        snr = random.uniform(15, 30)
        augmented = add_noise(augmented, snr_db=snr)
        applied['noise'] = f"SNR={snr:.1f}dB"
    
    return augmented, applied


# ============================================
# Dataset Augmentation Function
# ============================================

def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentation_factor: int = DEFAULT_AUGMENTATION_FACTOR,
    verbose: bool = True,
) -> dict:
    """
    Augment the training dataset.
    
    This function:
    1. Copies all original files (train, val, test)
    2. Creates augmented versions ONLY for training data
    3. Maintains the same directory structure
    
    Args:
        input_dir: Directory with organized dataset (from organize_data.py)
        output_dir: Where to create augmented dataset
        augmentation_factor: Number of augmented versions per original
        verbose: Print detailed progress
        
    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Data Augmentation")
    print("=" * 60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentation factor: {augmentation_factor}x (training data only)")
    
    # Verify input directory
    train_dir = input_dir / "train"
    if not train_dir.exists():
        print(f"\n⚠️  Training directory not found: {train_dir}")
        print("   Run organize_data.py first!")
        return {}
    
    # Clear output directory
    if output_dir.exists():
        print(f"\nRemoving existing output directory...")
        shutil.rmtree(output_dir)
    
    stats = {
        'original_train': 0,
        'augmented': 0,
        'val_copied': 0,
        'test_copied': 0,
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_input = input_dir / split
        split_output = output_dir / split
        
        if not split_input.exists():
            continue
        
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split...")
        print('=' * 40)
        
        # Process each category
        for category_dir in sorted(split_input.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            audio_files = list(category_dir.glob("*.wav"))
            
            print(f"\n  {category}: {len(audio_files)} files")
            
            for i, audio_file in enumerate(audio_files):
                # Load audio
                audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE)
                
                # Create output directory
                out_category_dir = split_output / category
                out_category_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy original file
                original_name = f"{category}_{i:04d}_orig.wav"
                original_path = out_category_dir / original_name
                sf.write(str(original_path), audio, sr)
                
                if split == 'train':
                    stats['original_train'] += 1
                    
                    # Create augmented versions (training only!)
                    for aug_idx in range(augmentation_factor):
                        aug_audio, aug_info = apply_random_augmentation(audio, sr)
                        
                        # Ensure same length as original
                        if len(aug_audio) > len(audio):
                            aug_audio = aug_audio[:len(audio)]
                        elif len(aug_audio) < len(audio):
                            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))
                        
                        # Save augmented version
                        aug_name = f"{category}_{i:04d}_aug{aug_idx:02d}.wav"
                        aug_path = out_category_dir / aug_name
                        sf.write(str(aug_path), aug_audio, sr)
                        stats['augmented'] += 1
                        
                        if verbose and (i == 0 and aug_idx < 3):
                            # Show first few augmentations as examples
                            aug_str = ", ".join(f"{k}:{v}" for k, v in aug_info.items())
                            print(f"    └── {aug_name}: {aug_str}")
                
                elif split == 'val':
                    stats['val_copied'] += 1
                elif split == 'test':
                    stats['test_copied'] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Augmentation Complete!")
    print("=" * 60)
    
    print(f"\nTraining data:")
    print(f"  Original files: {stats['original_train']}")
    print(f"  Augmented files: {stats['augmented']}")
    print(f"  Total: {stats['original_train'] + stats['augmented']}")
    print(f"  Expansion: {(stats['original_train'] + stats['augmented']) / max(stats['original_train'], 1):.1f}x")
    
    print(f"\nValidation: {stats['val_copied']} files (copied without augmentation)")
    print(f"Test: {stats['test_copied']} files (copied without augmentation)")
    
    # Show final directory structure
    print(f"\nOutput structure:")
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            total = sum(1 for _ in split_dir.rglob("*.wav"))
            print(f"  {split_dir}: {total} files")
    
    return stats


def preview_augmentation(
    audio_file: Path,
    output_dir: Optional[Path] = None,
    num_versions: int = 5,
):
    """
    Preview augmentation by creating multiple versions of one file.
    
    Useful for understanding what augmentations do before running
    on the full dataset.
    """
    print(f"Previewing augmentation for: {audio_file}")
    
    # Load audio
    audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE)
    print(f"  Duration: {len(audio)/sr:.2f}s")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path("augmentation_preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    stem = audio_file.stem
    sf.write(str(output_dir / f"{stem}_original.wav"), audio, sr)
    print(f"  Saved: {stem}_original.wav")
    
    # Create augmented versions
    for i in range(num_versions):
        aug_audio, aug_info = apply_random_augmentation(audio, sr)
        
        # Match length
        if len(aug_audio) > len(audio):
            aug_audio = aug_audio[:len(audio)]
        elif len(aug_audio) < len(audio):
            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))
        
        aug_str = "_".join(f"{k}" for k in aug_info.keys())
        filename = f"{stem}_aug{i:02d}_{aug_str}.wav"
        sf.write(str(output_dir / filename), aug_audio, sr)
        
        aug_details = ", ".join(f"{k}: {v}" for k, v in aug_info.items())
        print(f"  Saved: {filename}")
        print(f"         [{aug_details}]")
    
    print(f"\n✓ Preview files saved to: {output_dir}")


# ============================================
# Command Line Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Augment audio dataset for cough classification training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_data.py                     # Use defaults (5x augmentation)
  python augment_data.py --factor 10         # 10x augmentation
  python augment_data.py --preview file.wav  # Preview augmentation
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Input directory with organized dataset (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for augmented dataset (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--factor', '-f',
        type=int,
        default=DEFAULT_AUGMENTATION_FACTOR,
        help=f"Number of augmented versions per original (default: {DEFAULT_AUGMENTATION_FACTOR})"
    )
    
    parser.add_argument(
        '--preview', '-p',
        type=Path,
        help="Preview augmentation on a single file"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Less verbose output"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.preview:
        # Preview mode
        preview_augmentation(args.preview)
    else:
        # Full augmentation
        augment_dataset(
            input_dir=args.input,
            output_dir=args.output,
            augmentation_factor=args.factor,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
