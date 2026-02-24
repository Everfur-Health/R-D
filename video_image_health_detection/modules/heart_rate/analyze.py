#!/usr/bin/env python3
"""
Canine Health AI — Remote Heart Rate Detection (rPPG)
Method: Plane-Orthogonal-to-Skin (POS) algorithm
Source: Wang et al. 2017 + Hu et al. 2024 (Frontiers Vet Sci) adapted for dogs

HOW IT WORKS:
  Traditional PPG uses a contact sensor to measure blood volume changes.
  rPPG detects the same signal remotely from skin color changes in video:
    1. Blood absorbs more green light than surrounding tissue
    2. Each heartbeat → tiny periodic change in skin color (RGB channels)
    3. Bandpass filter to isolate pulse frequencies
    4. POS algorithm separates pulse from motion/lighting artifacts

FOR DOGS specifically (Hu et al. 2024):
  - ROI = face/muzzle region (visible skin, sparse fur)
  - Bandpass: 1.0–2.3 Hz (60–140 BPM — canine resting heart rate)
  - POS works better than ICA/CHROM for fur-covered ROIs
  - Nose/inner ear/gum areas have highest signal-to-noise ratio

This module:
  - Can process a video file → returns BPM estimate + PPG waveform plot
  - OR process a live camera stream (demo mode)
  - Fallback: if no dog detected, uses center crop of frame

Usage:
  python modules/heart_rate/analyze.py --video path/to/video.mp4
  python modules/heart_rate/analyze.py --live        (webcam demo)
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dog_detector import detect_dog, crop_face, crop_dog

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ===========================================================================
# POS Algorithm (Plane-Orthogonal-to-Skin)
# Reference: Wang et al. 2017 "Algorithmic Principles of Remote PPG"
# ===========================================================================

def pos_rppg(rgb_signals: np.ndarray, fps: float) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin) rPPG extraction.

    Args:
        rgb_signals: (T, 3) array of mean RGB values per frame
        fps: video frame rate

    Returns:
        pulse: (T,) normalized pulse signal

    How POS works:
      1. Normalize each frame's RGB by its mean (temporal normalization)
      2. Project onto a plane orthogonal to the skin tone vector
         The skin tone in RGB space follows a specific direction;
         the pulse signal lies perpendicular to it
      3. Two orthogonal directions S1, S2 are combined:
         pulse = S1 + (std(S1)/std(S2)) * S2
         This ratio ensures equal contribution from both axes
    """
    n = len(rgb_signals)
    if n < 10:
        return np.zeros(n)

    # Temporal normalization: divide each channel by its mean in a sliding window
    window_size = int(fps * 1.6)  # ~1.6 second window
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(window_size, 3)

    # Simple temporal mean normalization
    r = rgb_signals[:, 0].astype(np.float64)
    g = rgb_signals[:, 1].astype(np.float64)
    b = rgb_signals[:, 2].astype(np.float64)

    # Normalize by local mean (handle zero-mean frames)
    r_norm = r / (np.mean(r) + 1e-8)
    g_norm = g / (np.mean(g) + 1e-8)
    b_norm = b / (np.mean(b) + 1e-8)

    # POS projection
    # S1 = R - G (chrominance channel 1)
    # S2 = R + G - 2B (chrominance channel 2)
    S1 = r_norm - g_norm
    S2 = r_norm + g_norm - 2 * b_norm

    # Weight by standard deviation ratio
    std1 = np.std(S1) + 1e-8
    std2 = np.std(S2) + 1e-8

    pulse = S1 + (std1 / std2) * S2

    # Detrend (remove slow drift)
    pulse = pulse - np.polyval(np.polyfit(np.arange(n), pulse, 1), np.arange(n))

    return pulse


def bandpass_filter(signal: np.ndarray, fps: float,
                     low_hz: float = 1.0, high_hz: float = 2.3) -> np.ndarray:
    """
    Butterworth bandpass filter for pulse frequencies.
    For dogs at rest: 60–140 BPM = 1.0–2.3 Hz
    For dogs during exercise: can go up to 200+ BPM (3.3+ Hz)
    """
    nyq = fps / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)

    if low >= high:
        return signal

    b, a = butter(3, [low, high], btype='band')
    try:
        filtered = filtfilt(b, a, signal)
    except Exception:
        filtered = signal
    return filtered


def estimate_heart_rate(pulse: np.ndarray, fps: float,
                         low_bpm: float = 60.0, high_bpm: float = 220.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate heart rate from pulse signal using FFT.

    Returns:
        bpm: estimated heart rate in beats per minute
        freqs: frequency array (Hz)
        psd: power spectral density
    """
    n = len(pulse)
    if n < int(fps * 5):
        return 0.0, np.array([]), np.array([])

    # FFT
    freqs = fftfreq(n, d=1.0/fps)
    psd = np.abs(fft(pulse)) ** 2

    # Only positive frequencies in the valid BPM range
    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(mask):
        return 0.0, freqs, psd

    valid_psd = psd.copy()
    valid_psd[~mask] = 0

    # Find dominant frequency
    peak_idx = np.argmax(valid_psd)
    bpm = abs(freqs[peak_idx]) * 60.0

    return bpm, freqs[mask], psd[mask]


# ===========================================================================
# Video processing
# ===========================================================================

class HeartRateAnalyzer:
    """
    Process a video clip to estimate dog heart rate.
    """
    # Normal canine heart rate ranges
    NORMAL_RANGES = {
        'resting': (60, 120),    # 60-120 BPM
        'excited': (120, 180),
        'exercise': (150, 220),
    }

    def __init__(self, fps_target: float = 30.0,
                 window_sec: float = 10.0,
                 dog_conf: float = 0.3):
        """
        fps_target: target FPS to process (higher = more accurate, slower)
        window_sec: analysis window in seconds
        dog_conf: YOLOv8 confidence threshold for dog detection
        """
        self.fps_target = fps_target
        self.window_sec = window_sec
        self.dog_conf = dog_conf

    def extract_roi_signals(self, cap: cv2.VideoCapture,
                             max_frames: int = None) -> Tuple[np.ndarray, float, list]:
        """
        Extract mean RGB from dog face ROI for each frame.
        Returns: rgb_signals (T, 3), fps, annotated_frames
        """
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, int(fps / self.fps_target))

        if max_frames is None:
            max_frames = int(fps * self.window_sec)

        rgb_signals = []
        annotated_frames = []
        frame_count = 0
        processed = 0

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            # Detect dog and get face ROI
            bbox = detect_dog(frame, conf=self.dog_conf)
            roi = None
            if bbox is not None:
                roi = crop_face(frame, bbox, face_fraction=0.45)
                annotated = frame.copy()
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                               (0, 255, 0), 2)
                # Mark face ROI
                x1, y1, x2, y2 = bbox
                face_y2 = y1 + int((y2 - y1) * 0.45)
                cv2.rectangle(annotated, (x1, y1), (x2, face_y2), (0, 200, 255), 2)
                cv2.putText(annotated, "HR ROI", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                annotated_frames.append(annotated)
            else:
                # Fallback: center crop
                h, w = frame.shape[:2]
                m = 10
                roi = frame[h//4:3*h//4, w//4:3*w//4]

            if roi is not None and roi.size > 0:
                # Extract mean RGB from non-dark pixels
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mask = roi_rgb.mean(axis=2) > 30  # exclude very dark pixels (fur)
                if mask.sum() > 100:
                    mean_rgb = roi_rgb[mask].mean(axis=0)
                else:
                    mean_rgb = roi_rgb.mean(axis=(0, 1))
                rgb_signals.append(mean_rgb)
                processed += 1

        effective_fps = fps / frame_skip
        return np.array(rgb_signals), effective_fps, annotated_frames

    def analyze_video(self, video_path: str) -> dict:
        """
        Full pipeline: video → heart rate estimate.
        Returns dict with bpm, confidence, signal plot (base64).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}

        try:
            rgb_signals, eff_fps, annotated_frames = self.extract_roi_signals(cap)
        finally:
            cap.release()

        if len(rgb_signals) < 30:
            return {
                'error': 'Not enough frames for analysis (need ≥30 processed frames)',
                'frames_processed': len(rgb_signals),
            }

        # POS algorithm
        pulse_raw = pos_rppg(rgb_signals, eff_fps)

        # Bandpass filter (60-220 BPM for dogs)
        pulse_filtered = bandpass_filter(pulse_raw, eff_fps, low_hz=1.0, high_hz=3.67)

        # Heart rate estimate
        bpm, freqs, psd = estimate_heart_rate(pulse_filtered, eff_fps)

        # Confidence: peak prominence in PSD
        confidence = self._compute_confidence(freqs, psd)

        # Generate plot
        plot_b64 = self._generate_plot(pulse_filtered, rgb_signals, eff_fps, bpm, freqs, psd)

        # Annotated frame (first one with detection)
        keyframe_b64 = None
        if annotated_frames:
            from shared.utils import image_to_b64
            keyframe_b64 = image_to_b64(annotated_frames[0])

        # Clinical assessment
        assessment = self._assess_hr(bpm)

        return {
            'bpm': round(bpm, 1),
            'confidence': round(confidence, 3),
            'frames_analyzed': len(rgb_signals),
            'effective_fps': round(eff_fps, 1),
            'assessment': assessment,
            'plot_b64': plot_b64,
            'keyframe_b64': keyframe_b64,
            'note': 'For clinical use, validate with contact ECG/stethoscope',
        }

    def _compute_confidence(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Signal-to-noise ratio as confidence proxy."""
        if len(psd) < 3:
            return 0.0
        peak_idx = np.argmax(psd)
        peak_power = psd[peak_idx]
        noise_power = np.median(psd)
        snr = peak_power / (noise_power + 1e-8)
        # Normalize to [0, 1] with saturation at SNR=10
        return float(min(snr / 10.0, 1.0))

    def _assess_hr(self, bpm: float) -> str:
        if bpm == 0:
            return "Unable to detect pulse signal"
        if bpm < 40:
            return "Very low — possible bradycardia or poor signal quality"
        if bpm < 60:
            return "Below normal resting range (60–120 BPM)"
        if bpm <= 120:
            return f"Normal resting range ({bpm:.0f} BPM — typical: 60–120)"
        if bpm <= 160:
            return f"Elevated — excited/active state ({bpm:.0f} BPM)"
        return f"High — exercise or stress ({bpm:.0f} BPM)"

    def _generate_plot(self, pulse: np.ndarray, rgb: np.ndarray,
                        fps: float, bpm: float,
                        freqs: np.ndarray, psd: np.ndarray) -> str:
        """Generate 3-panel analysis plot, return as base64 PNG."""
        import base64
        from io import BytesIO

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        t = np.arange(len(pulse)) / fps

        # Panel 1: Raw RGB signals
        axes[0].plot(t, rgb[:, 0], 'r-', alpha=0.7, label='R', linewidth=1)
        axes[0].plot(t, rgb[:, 1], 'g-', alpha=0.7, label='G', linewidth=1)
        axes[0].plot(t, rgb[:, 2], 'b-', alpha=0.7, label='B', linewidth=1)
        axes[0].set_ylabel('Mean pixel value')
        axes[0].set_title('ROI Color Channels (blood volume changes)')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Filtered pulse signal
        axes[1].plot(t, pulse, 'k-', linewidth=1.5, label='Pulse (POS)')
        peaks, _ = find_peaks(pulse, distance=int(fps * 0.4), prominence=0.1 * np.std(pulse))
        if len(peaks) > 0:
            axes[1].plot(t[peaks], pulse[peaks], 'r^', markersize=8, label=f'Peaks ({len(peaks)})')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Bandpass-Filtered rPPG Signal (1.0–3.67 Hz)')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Power spectrum
        if len(freqs) > 0:
            bpm_axis = freqs * 60
            axes[2].plot(bpm_axis, psd, 'purple', linewidth=1.5)
            peak_bpm_idx = np.argmax(psd)
            axes[2].axvline(bpm_axis[peak_bpm_idx], color='red', linestyle='--',
                            label=f'HR = {bpm:.1f} BPM')
            axes[2].fill_between(bpm_axis, psd, alpha=0.2, color='purple')
            axes[2].set_xlabel('Heart Rate (BPM)')
            axes[2].set_ylabel('Power')
            axes[2].set_title(f'Power Spectrum — Estimated HR: {bpm:.1f} BPM')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)

        fig.suptitle('Canine Remote Heart Rate Analysis (rPPG / POS method)', fontsize=12, y=1.01)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Canine rPPG Heart Rate Detection')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--out', default='heart_rate_results',
                        help='Output directory')
    parser.add_argument('--window', type=float, default=15.0,
                        help='Analysis window in seconds (default: 15)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.video:
        print("Usage: python analyze.py --video path/to/video.mp4")
        print("\nAlgorithm: POS (Plane-Orthogonal-to-Skin)")
        print("Reference: Wang et al. 2017 + Hu et al. 2024 (Frontiers Vet Sci)")
        print("\nFor demos without a dog video, record a ~15 second clip of")
        print("the dog's face under good lighting (avoid deep shadows).")
        return

    print(f"Analyzing: {args.video}")
    analyzer = HeartRateAnalyzer(window_sec=args.window)
    result = analyzer.analyze_video(args.video)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nResults:")
    print(f"  Heart Rate:   {result['bpm']} BPM")
    print(f"  Confidence:   {result['confidence']:.1%}")
    print(f"  Assessment:   {result['assessment']}")
    print(f"  Frames used:  {result['frames_analyzed']} @ {result['effective_fps']} fps")

    # Save plot
    if result.get('plot_b64'):
        import base64
        plot_path = out_dir / 'analysis_plot.png'
        with open(plot_path, 'wb') as f:
            f.write(base64.b64decode(result['plot_b64']))
        print(f"\nPlot saved: {plot_path}")

    # Save JSON
    result_clean = {k: v for k, v in result.items() if k not in ('plot_b64', 'keyframe_b64')}
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result_clean, f, indent=2)
    print(f"Results: {out_dir / 'result.json'}")


if __name__ == '__main__':
    main()
