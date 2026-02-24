#!/usr/bin/env python3
"""
Dataset Organization Script for Canine Cough Detection

This script organizes raw audio files into a structured dataset suitable
for training machine learning models.

DATASET ORGANIZATION EXPLAINED:
==============================

1. WHY ORGANIZE DATA?
   - Machine learning needs consistent data format
   - Train/validation/test splits prevent overfitting evaluation
   - Labeled directories make loading easy

2. DIRECTORY STRUCTURE WE CREATE:
   
   data/processed/
   ├── train/           (70% of data - used for learning)
   │   ├── healthy/
   │   ├── kennel_cough/
   │   ├── heart_failure_cough/
   │   └── other_respiratory/
   ├── val/             (15% of data - used to tune hyperparameters)
   │   └── (same structure)
   └── test/            (15% of data - final evaluation, never seen during training)
       └── (same structure)

3. WHY THESE SPLIT RATIOS?
   - 70% train: Enough data to learn patterns
   - 15% val: Tune hyperparameters without biasing test score
   - 15% test: Honest evaluation on truly unseen data
   
   If dataset is small (<500 samples), consider 80/10/10 split.

4. STRATIFIED SPLITTING
   - Each split maintains the same class proportions
   - If 30% of data is "kennel_cough", train/val/test all have ~30%
   - Prevents bias from unbalanced splits

USAGE:
======
    # Basic usage (looks in data/raw, outputs to data/processed)
    python organize_data.py
    
    # Custom paths
    python organize_data.py --input path/to/raw --output path/to/processed
    
    # Dry run (preview without copying)
    python organize_data.py --dry-run
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Audio validation
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

# Default paths (relative to project root)
DEFAULT_INPUT_DIR = "../data/raw"
DEFAULT_OUTPUT_DIR = "../data/processed"

# Target sample rate for all audio
TARGET_SAMPLE_RATE = 16000

# Duration constraints (in seconds)
MIN_DURATION = 0.5   # Files shorter than this are skipped
MAX_DURATION = 30.0  # Files longer than this are truncated

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Category mapping: maps various folder/file names to our standard categories
# This handles inconsistent naming in source datasets
CATEGORY_MAPPING = {
    # Healthy / Normal sounds
    'healthy': 'healthy',
    'normal': 'healthy',
    'no_cough': 'healthy',
    'negative': 'healthy',
    
    # Kennel cough (infectious tracheobronchitis)
    'kennel_cough': 'kennel_cough',
    'kennel': 'kennel_cough',
    'tracheobronchitis': 'kennel_cough',
    'infectious_cough': 'kennel_cough',
    'dry_cough': 'kennel_cough',
    
    # Heart failure related cough
    'heart_failure_cough': 'heart_failure_cough',
    'heart_cough': 'heart_failure_cough',
    'cardiac_cough': 'heart_failure_cough',
    'heart_failure': 'heart_failure_cough',
    'cardiac': 'heart_failure_cough',
    'wet_cough': 'heart_failure_cough',
    'congestive': 'heart_failure_cough',
    
    # Other respiratory sounds
    'other': 'other_respiratory',
    'other_respiratory': 'other_respiratory',
    'respiratory': 'other_respiratory',
    'cough': 'other_respiratory',  # Generic cough goes here
    'wheeze': 'other_respiratory',
    'sneeze': 'other_respiratory',
    'breathing': 'other_respiratory',
    'bark': 'other_respiratory',
}

# Final category list
CATEGORIES = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']


# ============================================
# Utility Functions
# ============================================

def find_audio_files(directory: Path) -> List[Path]:
    """
    Recursively find all audio files in a directory.
    
    Supports common audio formats: WAV, MP3, OGG, FLAC, M4A
    """
    extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma'}
    files = []
    
    for ext in extensions:
        # Case-insensitive search
        files.extend(directory.rglob(f"*{ext}"))
        files.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(set(files))


def infer_category(filepath: Path) -> str:
    """
    Infer the category of an audio file from its path.
    
    Checks both directory names and filename for category keywords.
    Returns 'other_respiratory' if no match found.
    """
    # Convert path to lowercase for matching
    path_str = str(filepath).lower()
    
    # Check each keyword
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in path_str:
            return category
    
    # Default category
    return 'other_respiratory'


def validate_audio(filepath: Path) -> Tuple[bool, str, float]:
    """
    Validate an audio file can be loaded and meets duration requirements.
    
    Returns:
        Tuple of (is_valid, message, duration_seconds)
    """
    try:
        # Get duration without loading entire file
        duration = librosa.get_duration(path=str(filepath))
        
        if duration < MIN_DURATION:
            return False, f"Too short ({duration:.2f}s < {MIN_DURATION}s)", duration
        
        # Try loading a small portion to verify format
        y, sr = librosa.load(str(filepath), sr=TARGET_SAMPLE_RATE, duration=1.0)
        
        return True, "OK", duration
        
    except Exception as e:
        return False, f"Load error: {str(e)}", 0.0


def process_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> bool:
    """
    Process and copy an audio file: resample, normalize, convert to WAV.
    
    Args:
        input_path: Source audio file
        output_path: Destination path (will be .wav)
        target_sr: Target sample rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load and resample
        audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        
        # Truncate if too long
        max_samples = int(MAX_DURATION * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Normalize to [-1, 1]
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        sf.write(str(output_path), audio, target_sr)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")
        return False


def stratified_split(
    files_by_category: Dict[str, List[Path]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Split files into train/val/test sets, maintaining category proportions.
    
    Args:
        files_by_category: Dict mapping category name to list of file paths
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        
    Returns:
        Dict with keys 'train', 'val', 'test', each containing category→files dict
    """
    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list),
    }
    
    for category, files in files_by_category.items():
        # Shuffle files for this category
        files = files.copy()
        random.shuffle(files)
        
        # Calculate split points
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Split
        splits['train'][category] = files[:n_train]
        splits['val'][category] = files[n_train:n_train + n_val]
        splits['test'][category] = files[n_train + n_val:]
    
    return splits


# ============================================
# Main Organization Function
# ============================================

def organize_dataset(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Main function to organize the dataset.
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Where to create the organized dataset
        dry_run: If True, only preview actions without copying
        verbose: Print detailed progress
        
    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Dataset Organization")
    print("=" * 60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    if dry_run:
        print("\n[DRY RUN - No files will be copied]")
    
    # Find all audio files
    print("\n1. Scanning for audio files...")
    audio_files = find_audio_files(input_dir)
    print(f"   Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("\n⚠️  No audio files found!")
        print(f"   Make sure your audio files are in: {input_dir}")
        print("   Supported formats: WAV, MP3, OGG, FLAC, M4A")
        return {}
    
    # Categorize and validate files
    print("\n2. Validating and categorizing files...")
    files_by_category = defaultdict(list)
    invalid_files = []
    
    for filepath in audio_files:
        # Validate
        is_valid, message, duration = validate_audio(filepath)
        
        if not is_valid:
            invalid_files.append((filepath, message))
            if verbose:
                print(f"   ✗ {filepath.name}: {message}")
            continue
        
        # Categorize
        category = infer_category(filepath)
        files_by_category[category].append(filepath)
        
        if verbose:
            print(f"   ✓ {filepath.name} → {category} ({duration:.1f}s)")
    
    # Print category summary
    print(f"\n   Valid files by category:")
    total_valid = 0
    for category in CATEGORIES:
        count = len(files_by_category[category])
        total_valid += count
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"   {category:25s}: {count:4d} {bar}")
    
    print(f"\n   Total valid: {total_valid}")
    print(f"   Invalid/skipped: {len(invalid_files)}")
    
    if total_valid == 0:
        print("\n⚠️  No valid audio files to process!")
        return {}
    
    # Split into train/val/test
    print("\n3. Splitting into train/val/test sets...")
    splits = stratified_split(files_by_category)
    
    for split_name in ['train', 'val', 'test']:
        total = sum(len(files) for files in splits[split_name].values())
        print(f"   {split_name:5s}: {total:4d} files")
    
    # Process and copy files
    print("\n4. Processing and copying files...")
    stats = {'processed': 0, 'failed': 0}
    
    if not dry_run:
        # Clear output directory if it exists
        if output_dir.exists():
            print(f"   Removing existing output directory...")
            shutil.rmtree(output_dir)
    
    for split_name, categories in splits.items():
        for category, files in categories.items():
            for i, filepath in enumerate(files):
                # Generate output filename
                output_filename = f"{category}_{i:04d}.wav"
                output_path = output_dir / split_name / category / output_filename
                
                if dry_run:
                    if verbose:
                        print(f"   [DRY RUN] {filepath.name} → {output_path}")
                    stats['processed'] += 1
                else:
                    success = process_audio_file(filepath, output_path)
                    if success:
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("Organization Complete!")
    print("=" * 60)
    
    print(f"\nProcessed: {stats['processed']} files")
    if stats['failed'] > 0:
        print(f"Failed: {stats['failed']} files")
    
    if not dry_run:
        print(f"\nOutput structure:")
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            if split_dir.exists():
                total = sum(1 for _ in split_dir.rglob("*.wav"))
                print(f"  {split_dir}: {total} files")
                for cat_dir in sorted(split_dir.iterdir()):
                    if cat_dir.is_dir():
                        count = len(list(cat_dir.glob("*.wav")))
                        print(f"    └── {cat_dir.name}: {count}")
    
    return stats


# ============================================
# Command Line Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Organize audio dataset for cough classification training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_data.py                          # Use default paths
  python organize_data.py --input /path/to/audio   # Custom input path
  python organize_data.py --dry-run                # Preview without copying
  python organize_data.py --quiet                  # Less verbose output
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Input directory with raw audio files (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for organized dataset (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Preview actions without copying files"
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
        help="Random seed for reproducible splits (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Run organization
    organize_dataset(
        input_dir=args.input,
        output_dir=args.output,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
