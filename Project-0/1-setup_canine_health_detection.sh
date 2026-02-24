#!/usr/bin/env bash

mkdir -p ./canine-health-detection
cd ./canine-health-detection

# INITIAL SETUP FOR CANINE HEALTH DETECTION PROJECT

# Create virtual environment with Python 3
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# FOR GPU (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# FOR CPU ONLY
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Install audio processing libraries
pip install librosa soundfile audioread resampy

# Install API framework
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings

# Install ML utilities
pip install numpy scipy pandas scikit-learn

# Install visualization
pip install matplotlib pillow

# Install utilities
pip install loguru tqdm rich python-dotenv pyyaml aiofiles httpx

# Install audio augmentation (for training)
pip install audiomentations

# Install PANNs for pre-trained weights
pip install panns-inference



# DATASET SETUP

# Create data directory
mkdir -p ./data/{raw,processed,augmented}
cd ./data/raw

# Dataset 1: Dog Disease Sound Dataset (Kaggle)
# You need a Kaggle account and API key
pip install kaggle
# Put your kaggle.json in ~/.kaggle/
kaggle datasets download -d ziadelhussein/dog-disease-sound-dataset
unzip dog-disease-sound-dataset.zip -d dog_disease_sounds

# Dataset 2: Dogs vs Cats Audio (FigShare) - for general dog sounds
# https://figshare.com/articles/dataset/Dogs_versus_Cats_Audio_Dataset/20219408?file=36137315
wget --user-agent="Mozilla/5.0" "https://ndownloader.figshare.com/files/36137315"  -O dogs_cats_audio.zip
unzip dogs_cats_audio.zip -d dogs_cats_audio

deactivate

# SETUP SCRIPTS
mkdir -p ../../python
cd ../../python

cat > ./organize_data.py << 'EOF'
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
EOF


cat > ./augment_data.py << 'EOF'
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
EOF


# DOWNLOAD PRE-TRAINED MODEL WEIGHTS

cd ..
mkdir -p ./models/pretrained
cd ./models/pretrained

# Download CNN14 weights from PANNs (trained on AudioSet)
wget -O Cnn14_mAP=0.431.pth "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

echo "CNN14 weights downloaded successfully!"
ls -lh Cnn14_mAP=0.431.pth


cd ../../python

cat > ./model.py << 'EOF'
#!/usr/bin/env python3
"""
Cough Classification Model for Canine Health Detection

This module implements the CNN14 neural network architecture for classifying
dog coughs into health categories (healthy, kennel cough, heart failure cough).

ARCHITECTURE NOTE:
==================
This model EXACTLY matches the CNN14 architecture from PANNs (Pre-trained Audio
Neural Networks) so we can load the pretrained weights correctly.

PANNs CNN14 structure:
- bn0: BatchNorm2d on input (64 features for 64 mel bins)
- conv_block1: 1 → 64 channels (takes raw spectrogram)
- conv_block2: 64 → 128
- conv_block3: 128 → 256  
- conv_block4: 256 → 512
- conv_block5: 512 → 1024
- conv_block6: 1024 → 2048
- fc1: 2048 → 2048 (with ReLU)
- fc_audioset: 2048 → 527 (original AudioSet classes - we replace this)

We load all layers EXCEPT fc_audioset, then add our own classification head.

DEEP LEARNING CONCEPTS EXPLAINED:
=================================

1. CONVOLUTIONAL NEURAL NETWORKS (CNNs)
   - Originally designed for images, work great on spectrograms too!
   - "Convolution" = sliding a small filter across the input, detecting patterns
   - Early layers detect edges/textures, later layers detect complex shapes
   - Spectrograms ARE images: height=frequency, width=time, brightness=intensity

2. WHY CNN14 SPECIFICALLY?
   - From PANNs (Pre-trained Audio Neural Networks) project
   - Pre-trained on AudioSet: 5000+ hours, 527 sound classes
   - Already knows "what sound looks like" - we just fine-tune for dog coughs
   - Good accuracy/speed tradeoff (80M parameters, runs on consumer GPUs)

3. TRANSFER LEARNING
   - Instead of training from scratch (needs millions of samples)
   - We take a model trained on related task (general audio)
   - Freeze early layers (keep general audio knowledge)
   - Train only final layers for our specific task (dog cough types)
   - Works with just hundreds of samples instead of millions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """
    A convolutional block matching PANNs CNN14 architecture.
    
    Structure:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → AvgPool
    
    This "double conv" pattern (from VGG networks) is very effective:
    - Two 3x3 convolutions have the same receptive field as one 5x5
    - But with fewer parameters and more non-linearity
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)):
        """Forward pass with configurable pooling size."""
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class CoughClassifier(nn.Module):
    """
    CNN14-based classifier for dog cough detection.
    
    This network takes a mel spectrogram as input and outputs probabilities
    for each cough category. Architecture exactly matches PANNs CNN14 for
    compatibility with pretrained weights.
    
    Example usage:
        model = CoughClassifier(num_classes=4)
        mel_spec = torch.randn(1, 1, 64, 251)  # (batch, channel, freq, time)
        output = model(mel_spec)
        # output['logits'] shape: (1, 4)
        # output['probabilities'] shape: (1, 4)
    """
    
    # Class labels for reference
    LABELS = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']
    
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.5,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize the cough classifier.
        
        Args:
            num_classes: Number of output classes (4 for our task)
            dropout: Dropout probability for regularization
            pretrained_path: Path to pretrained CNN14 weights (optional)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # ====================================
        # CNN14 Backbone (matches PANNs exactly)
        # ====================================
        
        # Initial batch normalization on the mel spectrogram input
        # Applied across the frequency dimension (64 mel bins)
        self.bn0 = nn.BatchNorm2d(64)
        
        # Convolutional blocks - note conv_block1 takes 1 input channel
        # This is the raw spectrogram (single "grayscale" channel)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Fully connected layer from PANNs (we keep this for transfer learning)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        # ====================================
        # Our Custom Classification Head
        # ====================================
        # Replace PANNs' 527-class AudioSet head with our 4-class head
        self.fc_cough = nn.Linear(2048, num_classes, bias=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weight()
        
        # Load pretrained weights if provided
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_weights(pretrained_path)
    
    def init_weight(self):
        """Initialize all weights."""
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_cough)
    
    def _load_pretrained_weights(self, path: str) -> None:
        """
        Load pretrained CNN14 weights from PANNs.
        
        The pretrained model was trained on AudioSet (527 classes).
        We load all layers EXCEPT fc_audioset, which we replace with
        our own classification head (fc_cough).
        """
        print(f"Loading pretrained weights from {path}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Get our model's state dict
            model_dict = self.state_dict()
            
            # Filter checkpoint: only load layers that exist in our model
            # Skip fc_audioset (PANNs output layer) - we use fc_cough instead
            pretrained_dict = {}
            skipped = []
            
            for key, value in state_dict.items():
                # Skip the AudioSet classification layer
                if 'fc_audioset' in key:
                    skipped.append(key)
                    continue
                
                # Check if this key exists in our model
                if key in model_dict:
                    # Check shape matches
                    if value.shape == model_dict[key].shape:
                        pretrained_dict[key] = value
                    else:
                        skipped.append(f"{key} (shape mismatch: {value.shape} vs {model_dict[key].shape})")
                else:
                    skipped.append(f"{key} (not in model)")
            
            # Load the filtered weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
            print(f"  ✓ Loaded {len(pretrained_dict)} weight tensors")
            print(f"  ✓ Skipped {len(skipped)} tensors (AudioSet head + mismatches)")
            
            if len(pretrained_dict) > 0:
                print("  ✓ Pretrained backbone loaded successfully!")
            
        except Exception as e:
            print(f"  Warning: Could not load pretrained weights: {e}")
            print("  Continuing with random initialization")
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)
               Typical shape: (batch, 1, 64, 251) for 5 seconds of audio
            return_embedding: If True, also return the 2048-dim embedding
            
        Returns:
            Dictionary containing:
            - 'logits': Raw scores for each class (batch, num_classes)
            - 'probabilities': Softmax probabilities (batch, num_classes)
            - 'embedding': (optional) 2048-dim feature vector (batch, 2048)
        """
        # Input shape: (batch, 1, freq, time) e.g., (16, 1, 64, 251)
        
        # Transpose for batch norm: (batch, 1, freq, time) → (batch, freq, 1, time)
        # Then apply BN across freq dimension, then transpose back
        x = x.transpose(1, 2)  # (batch, freq, 1, time)
        x = self.bn0(x)
        x = x.transpose(1, 2)  # (batch, 1, freq, time)
        
        # Pass through conv blocks with pooling
        # Each block halves spatial dimensions via 2x2 avg pooling
        x = self.conv_block1(x, pool_size=(2, 2))  # → (batch, 64, freq/2, time/2)
        x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2))  # → (batch, 128, freq/4, time/4)
        x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2))  # → (batch, 256, freq/8, time/8)
        x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2))  # → (batch, 512, freq/16, time/16)
        x = self.dropout(x)
        x = self.conv_block5(x, pool_size=(2, 2))  # → (batch, 1024, freq/32, time/32)
        x = self.dropout(x)
        x = self.conv_block6(x, pool_size=(1, 1))  # → (batch, 2048, freq/32, time/32) no pooling
        x = self.dropout(x)
        
        # Global pooling: (batch, 2048, H, W) → (batch, 2048)
        # Combine mean and max pooling for richer features
        x_mean = torch.mean(x, dim=(2, 3))  # Global average pooling
        x_max, _ = torch.max(x, dim=2)       # Max over frequency
        x_max, _ = torch.max(x_max, dim=2)   # Max over time
        x = x_mean + x_max  # Combine both
        
        # Fully connected layer (from pretrained)
        x = self.dropout(x)
        x = F.relu_(self.fc1(x))
        
        embedding = x
        
        # Our classification head
        x = self.dropout(x)
        logits = self.fc_cough(x)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
        }
        
        if return_embedding:
            result['embedding'] = embedding
        
        return result
    
    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Make a prediction with human-readable output.
        
        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = output['probabilities']
            
            # Get top prediction
            top_prob, top_idx = probs.max(dim=1)
            
            # Build results for each sample in batch
            results = []
            for i in range(probs.size(0)):
                result = {
                    'label': self.LABELS[top_idx[i].item()],
                    'confidence': top_prob[i].item(),
                    'all_probabilities': {
                        label: probs[i, j].item()
                        for j, label in enumerate(self.LABELS)
                    }
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        **kwargs,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'epoch': epoch,
            'labels': self.LABELS,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = 'cpu',
    ) -> 'CoughClassifier':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        num_classes = checkpoint.get('num_classes', 4)
        model = cls(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return model


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Cough Classifier Model - Test")
    print("=" * 60)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nCreating model...")
    model = CoughClassifier(num_classes=4)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    n_mels = 64
    time_frames = 251  # ~5 seconds at 16kHz with hop=320
    
    x = torch.randn(batch_size, 1, n_mels, time_frames).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x, return_embedding=True)
    
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probabilities shape: {output['probabilities'].shape}")
    print(f"Output embedding shape: {output['embedding'].shape}")
    
    # Test prediction
    print("\nTesting prediction...")
    prediction = model.predict(x[:1])
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print("All probabilities:")
    for label, prob in prediction['all_probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    
    # Test with pretrained weights if available
    pretrained_path = Path("../models/pretrained/Cnn14_mAP=0.431.pth")
    if pretrained_path.exists():
        print("\n" + "-" * 40)
        print("Testing pretrained weight loading...")
        print("-" * 40)
        model_pretrained = CoughClassifier(
            num_classes=4,
            pretrained_path=str(pretrained_path)
        )
        model_pretrained.to(device)
        
        # Test that it still works
        output = model_pretrained(x)
        print(f"Output with pretrained: {output['logits'].shape}")
    
    print("\n✓ Model working correctly!")
EOF


cat > ./preprocessing.py << 'EOF'
#!/usr/bin/env python3
"""
Audio Preprocessing Module for Canine Cough Detection

This module handles all audio loading and feature extraction. It converts
raw audio files into the format needed by neural networks: mel spectrograms.

CONCEPTS EXPLAINED:
==================

1. SAMPLING RATE (e.g., 16000 Hz)
   - Audio is continuous, but computers need discrete samples
   - 16000 Hz means 16000 measurements per second
   - Nyquist theorem: can capture frequencies up to HALF the sample rate
   - At 16kHz, we capture frequencies 0-8000 Hz (enough for dog sounds)

2. MEL SPECTROGRAM
   - A "picture" of sound showing frequency content over time
   - X-axis: time, Y-axis: frequency (Mel scale), Color: intensity
   - "Mel" scale mimics how ears perceive pitch (logarithmic)
   - Low frequencies get more resolution (where speech/cough info is)

3. FFT (Fast Fourier Transform)
   - Converts time-domain signal to frequency-domain
   - n_fft=1024: analyze 1024 samples at a time (64ms at 16kHz)
   - hop_length=320: slide the window by 320 samples (20ms) each step
   - This creates overlapping windows for smooth time resolution

4. WHY THESE SPECIFIC VALUES?
   - 16kHz: Standard for speech/respiratory analysis, captures dog sounds
   - 64 Mel bins: Good balance of frequency detail vs computation
   - 1024 FFT: Long enough to capture one cough "pulse" (~60ms)
   - 320 hop: 20ms resolution lets us see cough timing precisely
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings

# Suppress librosa warnings about audioread
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install librosa soundfile")
    raise


class AudioPreprocessor:
    """
    Handles all audio loading and feature extraction for cough detection.
    
    This class encapsulates the entire audio preprocessing pipeline:
    1. Load audio from various formats (WAV, MP3, etc.)
    2. Resample to target sample rate
    3. Normalize amplitude
    4. Extract mel spectrogram features
    
    Example usage:
        preprocessor = AudioPreprocessor(sample_rate=16000)
        audio = preprocessor.load_audio("dog_cough.wav")
        mel_spec = preprocessor.extract_mel_spectrogram(audio)
        # mel_spec shape: (n_mels, time_frames) e.g., (64, 251) for 5 seconds
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 320,
        f_min: int = 50,
        f_max: int = 8000,
    ):
        """
        Initialize the preprocessor with audio parameters.
        
        Args:
            sample_rate: Target sample rate in Hz. We use 16000 because:
                        - Captures frequencies up to 8kHz (Nyquist)
                        - Dog coughs have most energy below 4kHz
                        - Standard for respiratory sound analysis
                        
            n_mels: Number of Mel filter banks (frequency bins). 64 gives good
                   resolution while keeping computation manageable.
                   
            n_fft: FFT window size in samples. 1024 samples = 64ms at 16kHz.
                  This is long enough to capture the spectral content of one
                  cough "burst" while still having temporal precision.
                  
            hop_length: Samples between consecutive FFT windows. 320 = 20ms,
                       giving us 50 frames per second of audio.
                       
            f_min: Minimum frequency (Hz) to include. 50Hz removes rumble/noise.
            
            f_max: Maximum frequency (Hz). 8000Hz captures all dog sounds
                  (most energy is below 4kHz anyway).
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        
        # Pre-compute the Mel filterbank matrix for efficiency
        # This matrix converts FFT bins to Mel bins
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
    
    def load_audio(
        self,
        filepath: Union[str, Path],
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> np.ndarray:
        """
        Load an audio file and convert to mono at target sample rate.
        
        This function handles:
        - Multiple formats (WAV, MP3, OGG, FLAC, M4A)
        - Automatic resampling to target sample rate
        - Stereo to mono conversion (by averaging channels)
        - Optional duration limiting
        
        Args:
            filepath: Path to the audio file
            duration: Maximum duration in seconds (None = load entire file)
            offset: Start time in seconds to begin loading
            
        Returns:
            1D numpy array of audio samples, normalized to [-1, 1]
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        # librosa.load handles format detection and resampling automatically
        # sr=self.sample_rate forces resampling to our target rate
        # mono=True converts stereo to mono by averaging channels
        audio, sr = librosa.load(
            str(filepath),
            sr=self.sample_rate,
            mono=True,
            duration=duration,
            offset=offset,
        )
        
        # Normalize to [-1, 1] range
        # This ensures consistent amplitude regardless of recording levels
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        to_db: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert audio waveform to mel spectrogram.
        
        The mel spectrogram is the primary input feature for our CNN model.
        It represents the audio as a 2D "image" where:
        - X-axis = time (frames)
        - Y-axis = frequency (Mel bins)
        - Pixel value = energy/intensity at that time-frequency point
        
        Args:
            audio: 1D numpy array of audio samples
            to_db: Convert power to decibels (log scale). This is important
                  because human perception of loudness is logarithmic.
            normalize: Normalize to 0-1 range for neural network input
            
        Returns:
            2D numpy array of shape (n_mels, time_frames)
            For 5 seconds of audio at our settings: (64, 251)
        """
        # Compute the Short-Time Fourier Transform (STFT)
        # This breaks audio into overlapping windows and computes FFT on each
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',  # Hann window reduces spectral leakage
        )
        
        # Convert complex STFT to power spectrogram (magnitude squared)
        # We discard phase information - it's not useful for classification
        power_spec = np.abs(stft) ** 2
        
        # Apply Mel filterbank to convert to Mel scale
        # This reduces dimensionality and mimics human perception
        mel_spec = np.dot(self.mel_basis, power_spec)
        
        if to_db:
            # Convert to decibels (log scale)
            # ref=np.max normalizes so max value is 0 dB
            # This gives us values in range [-80, 0] typically
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if normalize:
            # Normalize to [0, 1] range for neural network
            # Typical dB range is [-80, 0], so we shift and scale
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        return mel_spec
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        include_deltas: bool = True,
    ) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs).
        
        MFCCs are a compact representation of the spectral envelope.
        They're widely used in speech recognition and can complement
        mel spectrograms for audio classification.
        
        The "cepstrum" is the spectrum of the log spectrum - it separates
        the source (vocal cords/cough) from the filter (throat shape).
        
        Args:
            audio: 1D numpy array of audio samples
            n_mfcc: Number of MFCC coefficients (typically 13)
            include_deltas: If True, also compute delta (velocity) and
                          delta-delta (acceleration) features
                          
        Returns:
            2D array of shape (n_features, time_frames)
            With deltas: n_features = 3 * n_mfcc = 39
        """
        # Compute MFCCs from audio
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        if include_deltas:
            # Delta: first derivative (rate of change)
            # Captures how features are changing over time
            delta = librosa.feature.delta(mfccs)
            
            # Delta-delta: second derivative (acceleration)
            # Captures how the rate of change is changing
            delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Stack all features together
            mfccs = np.vstack([mfccs, delta, delta2])
        
        return mfccs
    
    def detect_events(
        self,
        audio: np.ndarray,
        threshold_factor: float = 1.5,
        min_duration: float = 0.05,
        merge_gap: float = 0.1,
    ) -> list:
        """
        Detect acoustic events (coughs, breaths) using energy analysis.
        
        This is a simple rule-based detector that finds regions where
        the audio energy exceeds a threshold. It's useful for:
        - Segmenting recordings into individual coughs
        - Finding active regions for more detailed analysis
        - Providing interpretable output to users
        
        Args:
            audio: 1D numpy array of audio samples
            threshold_factor: Events are detected where energy > mean + factor * std
            min_duration: Minimum event duration in seconds
            merge_gap: Merge events closer than this (seconds)
            
        Returns:
            List of dicts with 'start', 'end', 'duration' in seconds
        """
        # Compute frame-level energy (RMS = Root Mean Square)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )[0]
        
        # Compute adaptive threshold
        threshold = np.mean(rms) + threshold_factor * np.std(rms)
        
        # Find frames above threshold
        above_threshold = rms > threshold
        
        # Convert to time and find contiguous regions
        frame_times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        
        events = []
        in_event = False
        event_start = 0
        
        for i, (is_above, t) in enumerate(zip(above_threshold, frame_times)):
            if is_above and not in_event:
                # Event starts
                in_event = True
                event_start = t
            elif not is_above and in_event:
                # Event ends
                in_event = False
                duration = t - event_start
                if duration >= min_duration:
                    events.append({
                        'start': event_start,
                        'end': t,
                        'duration': duration,
                    })
        
        # Handle event at end of audio
        if in_event:
            duration = frame_times[-1] - event_start
            if duration >= min_duration:
                events.append({
                    'start': event_start,
                    'end': frame_times[-1],
                    'duration': duration,
                })
        
        # Merge close events
        if merge_gap > 0 and len(events) > 1:
            merged = [events[0]]
            for event in events[1:]:
                if event['start'] - merged[-1]['end'] < merge_gap:
                    # Merge with previous
                    merged[-1]['end'] = event['end']
                    merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
                else:
                    merged.append(event)
            events = merged
        
        return events
    
    def pad_or_trim(
        self,
        audio: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Pad or trim audio to exact length.
        
        Neural networks need fixed-size inputs. This ensures all audio
        clips are exactly the same length by:
        - Padding with zeros if too short
        - Trimming from the end if too long
        
        Args:
            audio: 1D numpy array
            target_length: Desired length in samples
            
        Returns:
            Audio array of exactly target_length samples
        """
        current_length = len(audio)
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            # Trim from end
            return audio[:target_length]
        else:
            # Pad with zeros at end
            padding = target_length - current_length
            return np.pad(audio, (0, padding), mode='constant')
    
    def process_file(
        self,
        filepath: Union[str, Path],
        target_duration: float = 5.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete processing pipeline for a single audio file.
        
        This is a convenience method that runs the full pipeline:
        1. Load audio
        2. Pad/trim to target duration
        3. Extract mel spectrogram
        4. Detect events
        
        Args:
            filepath: Path to audio file
            target_duration: Target duration in seconds
            
        Returns:
            Tuple of (mel_spectrogram, metadata_dict)
        """
        # Load audio
        audio = self.load_audio(filepath)
        
        # Store original duration
        original_duration = len(audio) / self.sample_rate
        
        # Pad or trim to target length
        target_samples = int(target_duration * self.sample_rate)
        audio = self.pad_or_trim(audio, target_samples)
        
        # Extract mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Detect events
        events = self.detect_events(audio)
        
        metadata = {
            'filepath': str(filepath),
            'original_duration': original_duration,
            'target_duration': target_duration,
            'sample_rate': self.sample_rate,
            'mel_shape': mel_spec.shape,
            'num_events': len(events),
            'events': events,
        }
        
        return mel_spec, metadata


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Audio Preprocessor - Test")
    print("=" * 60)
    
    # Create preprocessor with default settings
    preprocessor = AudioPreprocessor()
    
    print(f"\nSettings:")
    print(f"  Sample rate: {preprocessor.sample_rate} Hz")
    print(f"  Mel bins: {preprocessor.n_mels}")
    print(f"  FFT size: {preprocessor.n_fft} samples ({preprocessor.n_fft/preprocessor.sample_rate*1000:.1f}ms)")
    print(f"  Hop length: {preprocessor.hop_length} samples ({preprocessor.hop_length/preprocessor.sample_rate*1000:.1f}ms)")
    print(f"  Frequency range: {preprocessor.f_min}-{preprocessor.f_max} Hz")
    
    # Test with synthetic audio
    print("\nGenerating synthetic test audio (3 second cough-like sound)...")
    
    duration = 3.0
    t = np.linspace(0, duration, int(preprocessor.sample_rate * duration))
    
    # Create cough-like bursts
    audio = np.zeros_like(t)
    burst_times = [0.3, 0.6, 1.2, 2.0]
    
    for bt in burst_times:
        burst_start = int(bt * preprocessor.sample_rate)
        burst_len = int(0.15 * preprocessor.sample_rate)
        if burst_start + burst_len < len(audio):
            burst_t = np.linspace(0, 0.15, burst_len)
            # Cough: broadband noise burst with envelope
            burst = np.random.randn(burst_len) * 0.5
            burst += 0.3 * np.sin(2 * np.pi * 400 * burst_t)  # Add tonal component
            envelope = np.exp(-burst_t * 20) * (1 - np.exp(-burst_t * 100))
            audio[burst_start:burst_start + burst_len] += burst * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {len(audio)/preprocessor.sample_rate:.2f}s")
    
    # Extract features
    print("\nExtracting mel spectrogram...")
    mel_spec = preprocessor.extract_mel_spectrogram(audio)
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    print(f"  Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
    
    # Detect events
    print("\nDetecting events...")
    events = preprocessor.detect_events(audio)
    print(f"  Found {len(events)} events:")
    for i, event in enumerate(events):
        print(f"    Event {i+1}: {event['start']:.2f}s - {event['end']:.2f}s ({event['duration']:.2f}s)")
    
    print("\n✓ Preprocessor working correctly!")
EOF


cat > ./train.py << 'EOF'
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
EOF



# Inference Setup

cat > ./inference.py << 'EOF'
#!/usr/bin/env python3
"""
Inference Module for Canine Cough Detection

This module provides easy-to-use functions for running inference on audio files.
It can be used as a library (import and call functions) or from the command line.

USAGE AS LIBRARY:
=================
    from inference import CoughAnalyzer
    
    analyzer = CoughAnalyzer('../models/checkpoints/latest/best_model.pth')
    result = analyzer.analyze('dog_cough.wav')
    
    print(result['label'])        # 'kennel_cough'
    print(result['confidence'])   # 0.92
    print(result['segments'])     # List of time-localized predictions

USAGE FROM COMMAND LINE:
========================
    python inference.py audio_file.wav
    python inference.py audio_file.wav --output results.json
    python inference.py --batch audio_dir/ --output batch_results.json

API DESIGN:
===========
This module is designed to be easily integrated into other applications:
- CoughAnalyzer class encapsulates all functionality
- Simple dictionary outputs (easy to serialize to JSON)
- No global state - multiple analyzers can run independently
- Thread-safe (each analyzer has its own model instance)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import torch

# Import our modules
from preprocessing import AudioPreprocessor
from model import CoughClassifier


# ============================================
# Data Classes for Clean Outputs
# ============================================

@dataclass
class SegmentResult:
    """Result for a single time segment."""
    start_time: float
    end_time: float
    label: str
    confidence: float
    all_probabilities: Dict[str, float]


@dataclass 
class AnalysisResult:
    """Complete analysis result for an audio file."""
    filepath: str
    duration: float
    
    # Overall classification
    label: str
    confidence: float
    all_probabilities: Dict[str, float]
    
    # Time segments
    segments: List[SegmentResult]
    
    # Metadata
    sample_rate: int
    device: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'filepath': self.filepath,
            'duration': self.duration,
            'label': self.label,
            'confidence': self.confidence,
            'all_probabilities': self.all_probabilities,
            'segments': [asdict(s) for s in self.segments],
            'sample_rate': self.sample_rate,
            'device': self.device,
        }


# ============================================
# Main Analyzer Class
# ============================================

class CoughAnalyzer:
    """
    High-level interface for cough audio analysis.
    
    This class encapsulates model loading, preprocessing, and inference
    into a simple API. Create one instance and reuse it for multiple
    audio files.
    
    Example:
        analyzer = CoughAnalyzer('../models/checkpoints/latest/best_model.pth')
        
        # Analyze single file
        result = analyzer.analyze('audio.wav')
        print(f"Classification: {result.label} ({result.confidence:.1%})")
        
        # Analyze with sliding window for timeline
        result = analyzer.analyze('audio.wav', return_segments=True)
        for seg in result.segments:
            print(f"{seg.start_time:.2f}s - {seg.end_time:.2f}s: {seg.label}")
    """
    
    # Default configuration
    SAMPLE_RATE = 16000
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 320
    TARGET_DURATION = 5.0
    
    # Sliding window settings
    WINDOW_SIZE = 1.0      # seconds
    WINDOW_STRIDE = 0.25   # seconds
    
    CLASSES = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            model_path: Path to the model checkpoint (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_path = Path(model_path)
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
        )
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = CoughClassifier.load_checkpoint(
            str(self.model_path),
            device=str(self.device)
        )
        self.model.eval()
        
        print(f"Analyzer initialized on {self.device}")
    
    def analyze(
        self,
        audio_path: Union[str, Path],
        return_segments: bool = True,
    ) -> AnalysisResult:
        """
        Analyze an audio file for cough classification.
        
        Args:
            audio_path: Path to the audio file
            return_segments: If True, also compute sliding window segments
            
        Returns:
            AnalysisResult with classification and optional segments
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio = self.preprocessor.load_audio(str(audio_path))
        duration = len(audio) / self.SAMPLE_RATE
        
        # ====================================
        # Overall Classification
        # ====================================
        target_samples = int(self.TARGET_DURATION * self.SAMPLE_RATE)
        audio_padded = self.preprocessor.pad_or_trim(audio, target_samples)
        mel_spec = self.preprocessor.extract_mel_spectrogram(audio_padded)
        
        # Run inference
        input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = output['probabilities'][0].cpu().numpy()
        
        pred_idx = probs.argmax()
        
        # ====================================
        # Sliding Window Segments
        # ====================================
        segments = []
        
        if return_segments:
            window_samples = int(self.WINDOW_SIZE * self.SAMPLE_RATE)
            stride_samples = int(self.WINDOW_STRIDE * self.SAMPLE_RATE)
            
            position = 0
            while position + window_samples <= len(audio):
                # Process window
                window_audio = audio[position:position + window_samples]
                window_padded = self.preprocessor.pad_or_trim(window_audio, target_samples)
                window_mel = self.preprocessor.extract_mel_spectrogram(window_padded)
                
                window_tensor = torch.FloatTensor(window_mel).unsqueeze(0).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    window_output = self.model(window_tensor)
                    window_probs = window_output['probabilities'][0].cpu().numpy()
                
                window_pred_idx = window_probs.argmax()
                
                segments.append(SegmentResult(
                    start_time=position / self.SAMPLE_RATE,
                    end_time=(position + window_samples) / self.SAMPLE_RATE,
                    label=self.CLASSES[window_pred_idx],
                    confidence=float(window_probs[window_pred_idx]),
                    all_probabilities={
                        self.CLASSES[i]: float(window_probs[i])
                        for i in range(len(self.CLASSES))
                    }
                ))
                
                position += stride_samples
        
        return AnalysisResult(
            filepath=str(audio_path),
            duration=duration,
            label=self.CLASSES[pred_idx],
            confidence=float(probs[pred_idx]),
            all_probabilities={
                self.CLASSES[i]: float(probs[i])
                for i in range(len(self.CLASSES))
            },
            segments=segments,
            sample_rate=self.SAMPLE_RATE,
            device=str(self.device),
        )
    
    def analyze_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_segments: bool = False,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            return_segments: If True, compute segments for each file
            
        Returns:
            List of AnalysisResult objects
        """
        results = []
        
        for path in audio_paths:
            try:
                result = self.analyze(path, return_segments=return_segments)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {path}: {e}")
        
        return results


# ============================================
# Convenience Functions
# ============================================

def analyze_file(
    audio_path: str,
    model_path: str = "../models/checkpoints/latest/best_model.pth",
) -> Dict:
    """
    Quick analysis of a single audio file.
    
    This creates an analyzer, runs inference, and returns a dictionary.
    For repeated analysis, use CoughAnalyzer directly to avoid reloading
    the model each time.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model checkpoint
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = CoughAnalyzer(model_path)
    result = analyzer.analyze(audio_path)
    return result.to_dict()


# ============================================
# Command Line Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio files for canine cough classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py audio.wav
  python inference.py audio.wav --model path/to/model.pth
  python inference.py --batch audio_dir/ --output results.json
  python inference.py audio.wav --no-segments  # Faster, overall classification only
        """
    )
    
    parser.add_argument(
        'audio',
        nargs='?',
        help="Path to audio file"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=Path("../models/checkpoints/latest/best_model.pth"),
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=Path,
        help="Directory containing audio files for batch processing"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help="Output JSON file for results"
    )
    
    parser.add_argument(
        '--no-segments',
        action='store_true',
        help="Skip segment analysis (faster)"
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help="Force device (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.audio and not args.batch:
        parser.print_help()
        return
    
    # Initialize analyzer
    print("=" * 60)
    print("🐕 Canine Cough Analyzer")
    print("=" * 60)
    
    try:
        analyzer = CoughAnalyzer(args.model, device=args.device)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nMake sure you've trained a model first:")
        print("  cd python && python train.py")
        return
    
    # Process files
    results = []
    
    if args.batch:
        # Batch processing
        audio_dir = args.batch
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
        print(f"\nProcessing {len(audio_files)} files from {audio_dir}")
        
        for audio_path in audio_files:
            try:
                result = analyzer.analyze(audio_path, return_segments=not args.no_segments)
                results.append(result)
                print(f"  ✓ {audio_path.name}: {result.label} ({result.confidence:.1%})")
            except Exception as e:
                print(f"  ✗ {audio_path.name}: {e}")
    
    else:
        # Single file
        audio_path = Path(args.audio)
        print(f"\nAnalyzing: {audio_path}")
        
        result = analyzer.analyze(audio_path, return_segments=not args.no_segments)
        results.append(result)
        
        # Print detailed results
        print("\n" + "-" * 40)
        print("RESULTS")
        print("-" * 40)
        print(f"\nClassification: {result.label.upper().replace('_', ' ')}")
        print(f"Confidence: {result.confidence:.1%}")
        
        print("\nAll Probabilities:")
        for label, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {label:25s} {prob*100:5.1f}% {bar}")
        
        if result.segments:
            print(f"\nSegment Timeline ({len(result.segments)} segments):")
            for seg in result.segments[:10]:  # Show first 10
                print(f"  {seg.start_time:5.2f}s - {seg.end_time:5.2f}s: {seg.label:20s} ({seg.confidence:.1%})")
            if len(result.segments) > 10:
                print(f"  ... and {len(result.segments) - 10} more segments")
    
    # Save to JSON if requested
    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
EOF

mkdir -p web/static
cd web

cat > ./static/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐕 Canine Cough Analyzer</title>
    
    <!-- Tailwind CSS for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- WaveSurfer.js for audio visualization -->
    <script src="https://unpkg.com/wavesurfer.js@7"></script>
    
    <style>
        /* Custom styles for the heatmap overlay */
        .heatmap-container {
            position: relative;
            width: 100%;
        }
        
        .heatmap-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            pointer-events: none;
        }
        
        .segment-marker {
            position: absolute;
            height: 100%;
            opacity: 0.3;
            transition: opacity 0.2s;
        }
        
        .segment-marker:hover {
            opacity: 0.5;
        }
        
        /* Loading spinner */
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Probability bars */
        .prob-bar {
            transition: width 0.3s ease;
        }
        
        /* Timeline */
        .timeline-container {
            position: relative;
            height: 40px;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .timeline-segment {
            position: absolute;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            color: white;
            font-weight: 500;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            cursor: pointer;
            transition: filter 0.2s;
        }
        
        .timeline-segment:hover {
            filter: brightness(1.1);
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">
                🐕 Canine Cough Analyzer
            </h1>
            <p class="text-gray-600">
                Upload an audio recording to analyze for respiratory health indicators
            </p>
        </div>
        
        <!-- Upload Section -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4 text-gray-700">Upload Audio</h2>
            
            <div class="flex flex-col sm:flex-row gap-4 items-center">
                <div class="flex-1 w-full">
                    <label class="block">
                        <span class="sr-only">Choose audio file</span>
                        <input type="file" 
                               id="audioInput" 
                               accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
                               class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer">
                    </label>
                </div>
                
                <button id="analyzeBtn" 
                        onclick="analyzeAudio()"
                        disabled
                        class="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2">
                    <span id="btnText">Analyze</span>
                    <div id="btnSpinner" class="spinner hidden"></div>
                </button>
            </div>
            
            <p class="text-sm text-gray-500 mt-2">
                Supported formats: WAV, MP3, OGG, FLAC, M4A (max 50 MB)
            </p>
        </div>
        
        <!-- Results Section (hidden until analysis) -->
        <div id="resultsSection" class="hidden">
            
            <!-- Audio Player with Waveform -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Audio Visualization</h2>
                
                <!-- Waveform container -->
                <div class="heatmap-container bg-gray-900 rounded-lg p-2 mb-4">
                    <div id="waveform"></div>
                    <div id="heatmapOverlay" class="heatmap-overlay"></div>
                </div>
                
                <!-- Playback controls -->
                <div class="flex items-center gap-4 mb-4">
                    <button id="playPauseBtn" 
                            onclick="togglePlayPause()"
                            class="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2">
                        <span id="playIcon">▶</span>
                        <span id="playText">Play</span>
                    </button>
                    
                    <button onclick="wavesurfer && wavesurfer.stop()"
                            class="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
                        ⏹ Stop
                    </button>
                    
                    <div class="flex-1 text-right text-gray-600">
                        <span id="currentTime">0:00</span> / <span id="totalTime">0:00</span>
                    </div>
                </div>
                
                <!-- Classification Timeline -->
                <div class="mb-2">
                    <h3 class="text-sm font-medium text-gray-600 mb-2">Classification Timeline</h3>
                    <div id="timeline" class="timeline-container"></div>
                </div>
                
                <!-- Legend -->
                <div class="flex flex-wrap gap-4 mt-4 text-sm">
                    <div class="flex items-center gap-2">
                        <div class="w-4 h-4 rounded" style="background: #4CAF50"></div>
                        <span>Healthy</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="w-4 h-4 rounded" style="background: #F44336"></div>
                        <span>Kennel Cough</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="w-4 h-4 rounded" style="background: #9C27B0"></div>
                        <span>Heart Failure Cough</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="w-4 h-4 rounded" style="background: #FF9800"></div>
                        <span>Other Respiratory</span>
                    </div>
                </div>
            </div>
            
            <!-- Classification Results -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Overall Classification</h2>
                
                <div class="flex flex-col sm:flex-row gap-6">
                    <!-- Primary Result -->
                    <div class="sm:w-1/3">
                        <div id="primaryResult" class="text-center p-4 rounded-lg bg-gray-50">
                            <div class="text-4xl mb-2" id="resultEmoji">🔍</div>
                            <div class="text-xl font-bold" id="resultLabel">-</div>
                            <div class="text-2xl font-bold" id="resultConfidence">-</div>
                        </div>
                    </div>
                    
                    <!-- All Probabilities -->
                    <div class="sm:w-2/3">
                        <h3 class="text-sm font-medium text-gray-600 mb-3">All Probabilities</h3>
                        <div id="probabilitiesContainer" class="space-y-3">
                            <!-- Filled by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Spectrogram -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Mel Spectrogram</h2>
                <div id="spectrogramContainer" class="text-center">
                    <p class="text-gray-500">Spectrogram will appear here</p>
                </div>
            </div>
            
            <!-- Analysis Details -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Analysis Details</h2>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
                    <div class="p-3 bg-gray-50 rounded-lg">
                        <div class="text-sm text-gray-500">Duration</div>
                        <div class="font-semibold" id="detailDuration">-</div>
                    </div>
                    <div class="p-3 bg-gray-50 rounded-lg">
                        <div class="text-sm text-gray-500">Sample Rate</div>
                        <div class="font-semibold" id="detailSampleRate">-</div>
                    </div>
                    <div class="p-3 bg-gray-50 rounded-lg">
                        <div class="text-sm text-gray-500">Segments</div>
                        <div class="font-semibold" id="detailSegments">-</div>
                    </div>
                    <div class="p-3 bg-gray-50 rounded-lg">
                        <div class="text-sm text-gray-500">Processing Time</div>
                        <div class="font-semibold" id="detailProcessing">-</div>
                    </div>
                </div>
            </div>
            
        </div>
        
        <!-- Error Display -->
        <div id="errorDisplay" class="hidden bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-6">
            <strong>Error:</strong> <span id="errorMessage"></span>
        </div>
        
        <!-- Footer -->
        <div class="text-center text-gray-500 text-sm mt-8">
            <p>⚠️ This tool is for screening purposes only. Always consult a veterinarian for health concerns.</p>
        </div>
        
    </div>
    
    <script>
        // ============================================
        // Global State
        // ============================================
        
        let wavesurfer = null;
        let currentFile = null;
        let analysisResult = null;
        
        const CLASS_COLORS = {
            'healthy': '#4CAF50',
            'kennel_cough': '#F44336',
            'heart_failure_cough': '#9C27B0',
            'other_respiratory': '#FF9800',
        };
        
        const CLASS_EMOJIS = {
            'healthy': '✅',
            'kennel_cough': '🔴',
            'heart_failure_cough': '💜',
            'other_respiratory': '🟠',
        };
        
        // ============================================
        // File Input Handler
        // ============================================
        
        document.getElementById('audioInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                currentFile = file;
                document.getElementById('analyzeBtn').disabled = false;
                
                // Reset previous results
                document.getElementById('resultsSection').classList.add('hidden');
                document.getElementById('errorDisplay').classList.add('hidden');
            }
        });
        
        // ============================================
        // Analysis Function
        // ============================================
        
        async function analyzeAudio() {
            if (!currentFile) return;
            
            // UI: Show loading state
            const btn = document.getElementById('analyzeBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            
            btn.disabled = true;
            btnText.textContent = 'Analyzing...';
            btnSpinner.classList.remove('hidden');
            document.getElementById('errorDisplay').classList.add('hidden');
            
            try {
                // Create form data
                const formData = new FormData();
                formData.append('file', currentFile);
                
                // Send to API
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData,
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }
                
                // Parse result
                analysisResult = await response.json();
                
                // Display results
                displayResults(analysisResult);
                
            } catch (error) {
                console.error('Analysis error:', error);
                document.getElementById('errorMessage').textContent = error.message;
                document.getElementById('errorDisplay').classList.remove('hidden');
            } finally {
                // Reset button
                btn.disabled = false;
                btnText.textContent = 'Analyze';
                btnSpinner.classList.add('hidden');
            }
        }
        
        // ============================================
        // Display Results
        // ============================================
        
        function displayResults(result) {
            // Show results section
            document.getElementById('resultsSection').classList.remove('hidden');
            
            // Initialize WaveSurfer
            initWaveSurfer();
            
            // Display classification
            displayClassification(result.classification);
            
            // Display timeline and heatmap
            displayTimeline(result.segments, result.duration);
            
            // Display spectrogram
            if (result.spectrogram_base64) {
                document.getElementById('spectrogramContainer').innerHTML = 
                    `<img src="${result.spectrogram_base64}" class="w-full rounded-lg" alt="Spectrogram">`;
            }
            
            // Display details
            document.getElementById('detailDuration').textContent = `${result.duration.toFixed(2)}s`;
            document.getElementById('detailSampleRate').textContent = `${result.sample_rate} Hz`;
            document.getElementById('detailSegments').textContent = result.segments.length;
            document.getElementById('detailProcessing').textContent = `${result.processing_time_ms.toFixed(0)}ms`;
        }
        
        // ============================================
        // WaveSurfer Initialization
        // ============================================
        
        function initWaveSurfer() {
            // Destroy existing instance
            if (wavesurfer) {
                wavesurfer.destroy();
            }
            
            // Create new instance
            wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: '#4a90d9',
                progressColor: '#1a56db',
                cursorColor: '#fff',
                height: 100,
                normalize: true,
                backend: 'WebAudio',
            });
            
            // Load the file
            wavesurfer.loadBlob(currentFile);
            
            // Event handlers
            wavesurfer.on('ready', function() {
                document.getElementById('totalTime').textContent = formatTime(wavesurfer.getDuration());
                
                // Draw heatmap overlay after waveform is ready
                if (analysisResult) {
                    drawHeatmapOverlay(analysisResult.segments, analysisResult.duration);
                }
            });
            
            wavesurfer.on('audioprocess', function() {
                document.getElementById('currentTime').textContent = formatTime(wavesurfer.getCurrentTime());
            });
            
            wavesurfer.on('play', function() {
                document.getElementById('playIcon').textContent = '⏸';
                document.getElementById('playText').textContent = 'Pause';
            });
            
            wavesurfer.on('pause', function() {
                document.getElementById('playIcon').textContent = '▶';
                document.getElementById('playText').textContent = 'Play';
            });
            
            wavesurfer.on('finish', function() {
                document.getElementById('playIcon').textContent = '▶';
                document.getElementById('playText').textContent = 'Play';
            });
        }
        
        function togglePlayPause() {
            if (wavesurfer) {
                wavesurfer.playPause();
            }
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        
        // ============================================
        // Classification Display
        // ============================================
        
        function displayClassification(classification) {
            const label = classification.label;
            const confidence = classification.confidence;
            const probs = classification.all_probabilities;
            
            // Primary result
            document.getElementById('resultEmoji').textContent = CLASS_EMOJIS[label] || '🔍';
            document.getElementById('resultLabel').textContent = label.replace(/_/g, ' ').toUpperCase();
            document.getElementById('resultConfidence').textContent = `${(confidence * 100).toFixed(1)}%`;
            document.getElementById('resultConfidence').style.color = CLASS_COLORS[label];
            
            // All probabilities
            const container = document.getElementById('probabilitiesContainer');
            container.innerHTML = '';
            
            // Sort by probability
            const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
            
            for (const [cls, prob] of sorted) {
                const percentage = (prob * 100).toFixed(1);
                const color = CLASS_COLORS[cls] || '#888';
                
                container.innerHTML += `
                    <div class="flex items-center gap-3">
                        <div class="w-32 text-sm capitalize">${cls.replace(/_/g, ' ')}</div>
                        <div class="flex-1 bg-gray-200 rounded-full h-4">
                            <div class="prob-bar h-4 rounded-full" 
                                 style="width: ${percentage}%; background: ${color}"></div>
                        </div>
                        <div class="w-16 text-right text-sm font-medium">${percentage}%</div>
                    </div>
                `;
            }
        }
        
        // ============================================
        // Timeline and Heatmap
        // ============================================
        
        function displayTimeline(segments, duration) {
            const container = document.getElementById('timeline');
            container.innerHTML = '';
            
            for (const segment of segments) {
                const left = (segment.start_time / duration) * 100;
                const width = ((segment.end_time - segment.start_time) / duration) * 100;
                
                const div = document.createElement('div');
                div.className = 'timeline-segment';
                div.style.left = `${left}%`;
                div.style.width = `${width}%`;
                div.style.background = segment.color;
                div.textContent = segment.confidence > 0.7 ? segment.label.split('_')[0] : '';
                div.title = `${segment.label}: ${(segment.confidence * 100).toFixed(1)}% (${segment.start_time.toFixed(2)}s - ${segment.end_time.toFixed(2)}s)`;
                
                // Click to seek
                div.onclick = () => {
                    if (wavesurfer) {
                        wavesurfer.seekTo(segment.start_time / duration);
                    }
                };
                
                container.appendChild(div);
            }
        }
        
        function drawHeatmapOverlay(segments, duration) {
            const overlay = document.getElementById('heatmapOverlay');
            overlay.innerHTML = '';
            
            for (const segment of segments) {
                const left = (segment.start_time / duration) * 100;
                const width = ((segment.end_time - segment.start_time) / duration) * 100;
                
                const div = document.createElement('div');
                div.className = 'segment-marker';
                div.style.left = `${left}%`;
                div.style.width = `${width}%`;
                div.style.background = segment.color;
                
                overlay.appendChild(div);
            }
        }
        
        // ============================================
        // Initialize on page load
        // ============================================
        
        document.addEventListener('DOMContentLoaded', async function() {
            // Check API health
            try {
                const response = await fetch('/api/health');
                const health = await response.json();
                
                if (!health.model_loaded) {
                    document.getElementById('errorMessage').textContent = 
                        'Model not loaded. Please ensure the model is trained and the path is correct.';
                    document.getElementById('errorDisplay').classList.remove('hidden');
                }
            } catch (error) {
                document.getElementById('errorMessage').textContent = 
                    'Cannot connect to API server. Make sure the server is running.';
                document.getElementById('errorDisplay').classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
EOF


cat > ./server.py << 'EOF'
#!/usr/bin/env python3
"""
Canine Cough Detection - Web API Server

This FastAPI server provides:
1. Audio file upload and analysis
2. Frame-level predictions (where in the audio events occur)
3. Visualization data (waveform, spectrogram, heatmap)
4. Static file serving for the frontend

USAGE:
======
    # Start the server (from the python/ directory)
    cd canine-health-detection/python
    python -m web.server
    
    # Or run directly
    cd canine-health-detection/python/web
    python server.py
    
    # Then open http://localhost:8000 in your browser
    # API documentation at http://localhost:8000/docs
"""

import io
import sys
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# Fix Python Path for Imports
# ============================================
# Add the parent directory (python/) to the path so we can import our modules
SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON_DIR = SCRIPT_DIR.parent.resolve()
PROJECT_DIR = PYTHON_DIR.parent.resolve()

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# Now we can import our modules
from preprocessing import AudioPreprocessor
from model import CoughClassifier

# Optional: matplotlib for spectrogram images
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not found - spectrogram images disabled")


# ============================================
# Configuration
# ============================================

class Config:
    """Server configuration."""
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Model path (relative to project root)
    MODEL_PATH = PROJECT_DIR / "models" / "checkpoints" / "latest" / "best_model.pth"
    
    # Audio settings (must match training)
    SAMPLE_RATE = 16000
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 320
    TARGET_DURATION = 5.0  # seconds
    
    # Analysis settings
    WINDOW_SIZE = 1.0      # Analysis window in seconds
    WINDOW_STRIDE = 0.25   # Stride between windows in seconds
    
    # Class info
    CLASSES = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']
    CLASS_COLORS = {
        'healthy': '#4CAF50',           # Green
        'kennel_cough': '#F44336',      # Red
        'heart_failure_cough': '#9C27B0',  # Purple
        'other_respiratory': '#FF9800',  # Orange
    }

config = Config()


# ============================================
# Response Models (Pydantic)
# ============================================

class SegmentPrediction(BaseModel):
    """A prediction for a segment of audio."""
    start_time: float
    end_time: float
    label: str
    confidence: float
    color: str
    all_probabilities: Dict[str, float]


class ClassificationResult(BaseModel):
    """Classification result for the entire audio."""
    label: str
    confidence: float
    all_probabilities: Dict[str, float]


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    success: bool
    filename: str
    duration: float
    sample_rate: int
    
    # Overall classification (entire clip)
    classification: ClassificationResult
    
    # Timeline of predictions (for heatmap)
    segments: List[SegmentPrediction]
    
    # Visualization data
    waveform: List[float]
    waveform_times: List[float]
    spectrogram_base64: Optional[str] = None
    
    # Processing info
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model metadata."""
    model_loaded: bool
    model_path: str
    classes: List[str]
    device: str
    sample_rate: int


# ============================================
# Global State (loaded on startup)
# ============================================

model: Optional[CoughClassifier] = None
preprocessor: Optional[AudioPreprocessor] = None
device: torch.device = torch.device('cpu')


def load_model():
    """Load the trained model."""
    global model, preprocessor, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    
    # Load model
    print(f"Looking for model at: {config.MODEL_PATH}")
    if config.MODEL_PATH.exists():
        print(f"Loading model from {config.MODEL_PATH}")
        model = CoughClassifier.load_checkpoint(str(config.MODEL_PATH), device=str(device))
        model.eval()
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model not found at {config.MODEL_PATH}")
        print("  Run training first: cd python && python train.py")
        model = None


# ============================================
# Analysis Functions
# ============================================

def analyze_audio_file(audio_data: bytes, filename: str) -> AnalysisResponse:
    """
    Analyze an uploaded audio file.
    
    This function:
    1. Loads and preprocesses the audio
    2. Runs overall classification on the entire clip
    3. Runs sliding window analysis to get time-localized predictions
    4. Generates visualization data
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save to temp file (librosa needs a file path)
    suffix = Path(filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        # Load audio
        audio = preprocessor.load_audio(tmp_path)
        duration = len(audio) / config.SAMPLE_RATE
        
        # ====================================
        # Overall Classification
        # ====================================
        target_samples = int(config.TARGET_DURATION * config.SAMPLE_RATE)
        audio_padded = preprocessor.pad_or_trim(audio, target_samples)
        mel_spec = preprocessor.extract_mel_spectrogram(audio_padded)
        
        # Run inference
        input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
        overall_prediction = model.predict(input_tensor)
        
        # ====================================
        # Sliding Window Analysis (for heatmap)
        # ====================================
        segments = []
        window_samples = int(config.WINDOW_SIZE * config.SAMPLE_RATE)
        stride_samples = int(config.WINDOW_STRIDE * config.SAMPLE_RATE)
        
        position = 0
        while position + window_samples <= len(audio):
            # Extract window
            window_audio = audio[position:position + window_samples]
            
            # Pad to target duration for model
            window_padded = preprocessor.pad_or_trim(window_audio, target_samples)
            window_mel = preprocessor.extract_mel_spectrogram(window_padded)
            
            # Run inference
            window_tensor = torch.FloatTensor(window_mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(window_tensor)
                probs = output['probabilities'][0].cpu().numpy()
            
            # Get prediction for this window
            pred_idx = int(probs.argmax())
            pred_label = config.CLASSES[pred_idx]
            pred_conf = float(probs[pred_idx])
            
            start_sec = position / config.SAMPLE_RATE
            end_sec = (position + window_samples) / config.SAMPLE_RATE
            
            segments.append(SegmentPrediction(
                start_time=start_sec,
                end_time=end_sec,
                label=pred_label,
                confidence=pred_conf,
                color=config.CLASS_COLORS.get(pred_label, '#888888'),
                all_probabilities={
                    config.CLASSES[i]: float(probs[i])
                    for i in range(len(config.CLASSES))
                }
            ))
            
            position += stride_samples
        
        # Handle final partial window
        if position < len(audio) and len(audio) - position > stride_samples:
            window_audio = audio[position:]
            window_padded = preprocessor.pad_or_trim(window_audio, target_samples)
            window_mel = preprocessor.extract_mel_spectrogram(window_padded)
            
            window_tensor = torch.FloatTensor(window_mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(window_tensor)
                probs = output['probabilities'][0].cpu().numpy()
            
            pred_idx = int(probs.argmax())
            pred_label = config.CLASSES[pred_idx]
            
            segments.append(SegmentPrediction(
                start_time=position / config.SAMPLE_RATE,
                end_time=len(audio) / config.SAMPLE_RATE,
                label=pred_label,
                confidence=float(probs[pred_idx]),
                color=config.CLASS_COLORS.get(pred_label, '#888888'),
                all_probabilities={
                    config.CLASSES[i]: float(probs[i])
                    for i in range(len(config.CLASSES))
                }
            ))
        
        # ====================================
        # Generate Visualization Data
        # ====================================
        
        # Downsample waveform for efficient transfer
        target_points = 1000
        step = max(1, len(audio) // target_points)
        waveform_downsampled = audio[::step].tolist()
        waveform_times = [i * step / config.SAMPLE_RATE for i in range(len(waveform_downsampled))]
        
        # Generate spectrogram image
        spectrogram_base64 = None
        if HAS_MATPLOTLIB:
            spectrogram_base64 = generate_spectrogram_image(audio)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            success=True,
            filename=filename,
            duration=duration,
            sample_rate=config.SAMPLE_RATE,
            classification=ClassificationResult(
                label=overall_prediction['label'],
                confidence=overall_prediction['confidence'],
                all_probabilities=overall_prediction['all_probabilities'],
            ),
            segments=segments,
            waveform=waveform_downsampled,
            waveform_times=waveform_times,
            spectrogram_base64=spectrogram_base64,
            processing_time_ms=processing_time,
        )
        
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def generate_spectrogram_image(audio: np.ndarray) -> str:
    """Generate a spectrogram image as base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Compute mel spectrogram
    mel_spec = preprocessor.extract_mel_spectrogram(audio, normalize=False)
    
    # Plot
    img = ax.imshow(
        mel_spec,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, len(audio) / config.SAMPLE_RATE, 0, config.N_MELS]
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Bin')
    ax.set_title('Mel Spectrogram')
    plt.colorbar(img, ax=ax, label='Magnitude')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Canine Cough Detection API",
    description="AI-powered analysis of dog respiratory sounds",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    index_path = SCRIPT_DIR / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>Canine Cough Detection</title></head>
        <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
            <h1>🐕 Canine Cough Detection API</h1>
            <p>Frontend not found at static/index.html</p>
            <p>API documentation: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    return ModelInfo(
        model_loaded=model is not None,
        model_path=str(config.MODEL_PATH),
        classes=config.CLASSES,
        device=str(device),
        sample_rate=config.SAMPLE_RATE,
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file.
    
    Accepts WAV, MP3, OGG, FLAC, M4A formats.
    
    Returns:
    - Overall classification of the audio
    - Time-segmented predictions (for visualization)
    - Waveform data
    - Spectrogram image (base64)
    """
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}"
        )
    
    # Read file
    audio_data = await file.read()
    
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    if len(audio_data) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")
    
    # Analyze
    try:
        result = analyze_audio_file(audio_data, file.filename)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Mount static files
static_dir = SCRIPT_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🐕 Canine Cough Detection - Web Server")
    print("=" * 60)
    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Python directory: {PYTHON_DIR}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"\nStarting server at http://{config.HOST}:{config.PORT}")
    print(f"API docs at http://{config.HOST}:{config.PORT}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,  # Pass the app object directly instead of string
        host=config.HOST,
        port=config.PORT,
    )
EOF

cd ../..

cat > ./run_python.sh << 'EOF'
#!/usr/bin/env bash

source ./venv/bin/activate
cd ./python

python organize_data.py --input ../data/raw --output ../data/processed
python augment_data.py --input ../data/processed --output ../data/augmented
python train.py
python web/server.py

EOF

chmod +x ./run_python.sh
echo "Present Directory: $(pwd)"
echo "Setup complete. Use ./run_python.sh to start the training and server process."
