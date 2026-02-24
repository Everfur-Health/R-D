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
