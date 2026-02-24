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
