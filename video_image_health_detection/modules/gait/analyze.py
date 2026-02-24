#!/usr/bin/env python3
"""
Canine Health AI — Gait & Lameness Analysis
==========================================
Replaces the previous MMPose-based approach (which had incompatible
dependencies: chumpy, mmcv, numpy conflicts) with YOLOv8-pose, which
is already installed as part of the `ultralytics` package.

HOW IT WORKS
------------
Step 1 — Dog detection
  YOLOv8n detects the dog in each frame (COCO class 16).
  We use the largest detected dog and crop a padded bounding box.

Step 2 — Pose keypoint extraction (YOLOv8-pose)
  `yolov8m-pose.pt` predicts 17 COCO keypoints on the cropped image.
  COCO 17-point skeleton:
    0  = nose           (head)
    1,2 = eyes          (head)
    3,4 = ears          (head)
    5,6 = shoulders     (front torso)
    7,8 = elbows        (front legs)
    9,10 = wrists       (front paws — 'forepaws')
   11,12 = hips         (rear torso)
   13,14 = knees        (rear legs)
   15,16 = ankles       (rear paws — 'hindpaws')

  For dogs, this mapping isn't anatomically perfect (dogs have different
  proportions than humans) but the spatial relationships are sufficient
  for detecting left/right asymmetry.

Step 3 — Biomechanical metrics
  All metrics are computed over the full sequence of extracted keypoints:

  SYMMETRY RATIO (SR) — most clinically validated metric
    SR = |left_amplitude - right_amplitude| / mean_amplitude
    Computed separately for forepaws (wrists 9,10) and hindpaws (ankles 15,16)
    using the vertical (Y) oscillation range as a proxy for step length.
    SR > 0.1 is clinically significant (Weishaupt et al. 2010, VCOT).

  HEAD BOB INDEX (HBI)
    HBI = std(nose_y_velocity) normalized by fps
    Healthy dogs have a rhythmic, low-variance head bob. When a forelimb
    is painful, the head dips asymmetrically when weight lands on the
    affected limb (Keegan 2007, VCOT). High HBI → front-limb suspicion.

  HIP DROP INDEX (HDI)
    HDI = |mean(hip_L_y) - mean(hip_R_y)| / image_height
    The pelvis drops toward the healthy side when a hindlimb is painful.
    Best visible in the "away" (caudal) gait view (Rhodin et al. 2017).

  VELOCITY CONSISTENCY (VC)
    VC = std(forward_velocity) / mean(forward_velocity)  [coefficient of variation]
    Guarded, pain-avoidance movement → irregular speed → high VC.

  STRIDE REGULARITY
    Cross-correlation between left and right limb vertical trajectories.
    Perfect symmetry → correlation ≈ 1.0 at lag=0.
    Asymmetric gait → correlation < 1 or peak at non-zero lag.

  COMPOSITE LAMENESS INDEX (CLI)
    CLI = 0.35·SR_fore + 0.35·SR_hind + 0.15·HBI + 0.10·HDI + 0.05·VC
    Normalized to [0, 1]. CLI > 0.25 → notable asymmetry.
    Weights are literature-derived (SR is most validated).

Step 4 — Annotated output
  Each processed frame is annotated with:
  - Dog bounding box (green)
  - Skeleton lines (coloured by limb: blue=left, red=right, white=spine)
  - Keypoint dots with confidence scores
  - Live CLI score overlay

Usage
-----
  # From CLI:
  python modules/gait/analyze.py --video path/to/dog_walking.mp4

  # From the API (via inference.py):
  from modules.gait.analyze import GaitAnalyzer
  result = GaitAnalyzer().analyze_video(path)
"""

import os, sys, json, argparse, tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from scipy.signal import savgol_filter, correlate
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dog_detector import get_detector as get_yolo_detector, detect_dog


# ============================================================
# COCO keypoint indices for gait analysis
# ============================================================

KP = {
    'nose': 0,
    'left_eye': 1,   'right_eye': 2,
    'left_ear': 3,   'right_ear': 4,
    'left_shoulder': 5,  'right_shoulder': 6,
    'left_elbow': 7,     'right_elbow': 8,
    'left_wrist': 9,     'right_wrist': 10,   # ≈ forepaws
    'left_hip': 11,      'right_hip': 12,
    'left_knee': 13,     'right_knee': 14,
    'left_ankle': 15,    'right_ankle': 16,   # ≈ hindpaws
}

CONF_THRESHOLD = 0.3   # minimum keypoint confidence to use

# ============================================================
# Dog-anatomy skeleton drawing
# ============================================================

def _draw_dog_skeleton(
    out: np.ndarray,
    kps: np.ndarray,
    conf_thr: float,
) -> None:
    """
    Draw a quadruped-appropriate skeleton overlay IN-PLACE on `out`.

    Why the COCO human skeleton fails on dogs
    -----------------------------------------
    YOLOv8-pose is trained on people. The COCO 17-point layout assumes a
    VERTICAL body: head at top → shoulders → elbows → wrists hanging down,
    then hips → knees → ankles.

    A dog in lateral (side) view is HORIZONTAL. The model has to warp
    this layout sideways. What typically happens:
      • Nose / eyes / ears  → land near the dog's actual head ✓
      • Shoulders (5,6)     → land near the front body / withers area ✓±
      • Hips (11,12)        → land near the hindquarters ✓±
      • Elbows (7,8)        → unreliable — often on the dog's belly or back ✗
      • Wrists (9,10)       → unreliable — often belly / under-torso ✗
      • Knees (13,14)       → rear leg area ±
      • Ankles (15,16)      → lower rear ±

    Solution
    --------
    1. HEAD cluster  — draw nose ↔ ear connections only when the distance
                        is short (head is compact).
    2. SPINE         — draw a single line: nose → shoulder_mid → hip_mid.
                        These three regions are reliably detected and define
                        the dog's body axis.
    3. LIMB chains   — only draw a joint-to-joint segment when:
          (a) Both keypoints are above the confidence threshold.
          (b) The segment length is < 40 % of the frame diagonal
              (prevents wildly wrong long lines crossing the body).
          (c) The distal joint (wrist/ankle) is NOT significantly ABOVE
              the proximal joint (shoulder/hip) — quadruped legs go DOWN,
              not up. We allow up to 12 % of frame height upward tolerance
              to accommodate slight camera angles.
    4. DOTS          — draw every confident keypoint as a colour-coded circle.
                        Colour scheme: head=cyan, near-side limbs=orange,
                        far-side limbs=blue/purple, torso anchor=white.
    """
    h, w = out.shape[:2]
    diag = np.hypot(h, w)
    max_seg = diag * 0.40          # max believable segment length
    up_tol  = h * 0.12             # how far up a "distal" joint can still be

    def pt(idx):
        """Return (int x, int y) if confident, else None."""
        if kps[idx, 2] >= conf_thr:
            return (int(kps[idx, 0]), int(kps[idx, 1]))
        return None

    def seg(a, b, color, thickness=2, max_d=None):
        """Draw a line between two confident keypoints if distance is ok."""
        pa, pb = pt(a), pt(b)
        if pa is None or pb is None:
            return
        d = np.hypot(pa[0]-pb[0], pa[1]-pb[1])
        limit = max_d if max_d is not None else max_seg
        if d <= limit:
            cv2.line(out, pa, pb, color, thickness, cv2.LINE_AA)

    def limb_seg(prox, dist, color):
        """Draw a limb segment with downward-direction filter."""
        pa, pb = pt(prox), pt(dist)
        if pa is None or pb is None:
            return
        d  = np.hypot(pa[0]-pb[0], pa[1]-pb[1])
        dy = pb[1] - pa[1]            # positive = distal is lower (correct)
        if d > max_seg:
            return
        if dy < -up_tol:              # distal significantly above proximal → skip
            return
        cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)

    # ── 1. Head cluster ──────────────────────────────────────────────────────
    HEAD = (0, 220, 255)            # bright cyan
    head_max = diag * 0.15          # head connections must be short
    seg(0, 3, HEAD, max_d=head_max) # nose → left ear
    seg(0, 4, HEAD, max_d=head_max) # nose → right ear
    seg(1, 3, HEAD, max_d=head_max) # left eye → left ear
    seg(2, 4, HEAD, max_d=head_max) # right eye → right ear
    seg(3, 4, HEAD, max_d=head_max) # ear to ear (top of head)

    # ── 2. Spine ─────────────────────────────────────────────────────────────
    SPINE = (220, 220, 220)
    # Compute shoulder midpoint and hip midpoint for a more stable spine line
    s_pts = [pt(5), pt(6)]
    h_pts = [pt(11), pt(12)]
    s_valid = [p for p in s_pts if p is not None]
    h_valid = [p for p in h_pts if p is not None]

    shoulder_mid = None
    hip_mid      = None

    if s_valid:
        shoulder_mid = (
            int(sum(p[0] for p in s_valid) / len(s_valid)),
            int(sum(p[1] for p in s_valid) / len(s_valid)),
        )
    if h_valid:
        hip_mid = (
            int(sum(p[0] for p in h_valid) / len(h_valid)),
            int(sum(p[1] for p in h_valid) / len(h_valid)),
        )

    # Spine: shoulder_mid ↔ hip_mid
    if shoulder_mid and hip_mid:
        d = np.hypot(shoulder_mid[0]-hip_mid[0], shoulder_mid[1]-hip_mid[1])
        if d <= max_seg:
            cv2.line(out, shoulder_mid, hip_mid, SPINE, 3, cv2.LINE_AA)

    # Neck: nose → shoulder_mid
    nose = pt(0)
    if nose and shoulder_mid:
        d = np.hypot(nose[0]-shoulder_mid[0], nose[1]-shoulder_mid[1])
        if d <= max_seg:
            cv2.line(out, nose, shoulder_mid, SPINE, 2, cv2.LINE_AA)

    # ── 3. Limb chains (with geometric validation) ───────────────────────────
    # Colors: near-side (visible side in lateral view) = warmer orange/red
    #         far-side  (occluded side)                = cooler blue/purple
    FORE_L = (80,  160, 255)   # "left"  front limb — blue-ish
    FORE_R = (255, 140,  60)   # "right" front limb — orange
    HIND_L = (60,  100, 220)   # "left"  hind  limb — deeper blue
    HIND_R = (220,  80,  40)   # "right" hind  limb — deeper orange

    # Front limb chains: shoulder → elbow → wrist
    limb_seg(5,  7,  FORE_L);  limb_seg(7,  9,  FORE_L)   # left front
    limb_seg(6,  8,  FORE_R);  limb_seg(8,  10, FORE_R)   # right front

    # Hind limb chains: hip → knee → ankle
    limb_seg(11, 13, HIND_L);  limb_seg(13, 15, HIND_L)   # left hind
    limb_seg(12, 14, HIND_R);  limb_seg(14, 16, HIND_R)   # right hind

    # ── 4. Keypoint dots ─────────────────────────────────────────────────────
    DOT_COLORS = {
        # Head — cyan
        0: (0, 220, 255), 1: (0, 200, 220), 2: (0, 200, 220),
        3: (0, 180, 200), 4: (0, 180, 200),
        # Left side limbs — orange
        5: (80, 160, 255), 7: (80, 160, 255), 9:  (80, 160, 255),
        11:(60, 100, 220), 13:(60, 100, 220), 15: (60, 100, 220),
        # Right side limbs — blue
        6: (255, 140, 60), 8: (255, 140, 60), 10: (255, 140, 60),
        12:(220,  80, 40), 14:(220,  80, 40), 16: (220,  80, 40),
    }
    for i, (x, y, c) in enumerate(kps):
        if c >= conf_thr:
            col = DOT_COLORS.get(i, (200, 200, 200))
            cv2.circle(out, (int(x), int(y)), 5, col, -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 5, (0,0,0), 1, cv2.LINE_AA)  # outline


# ============================================================
# Frame annotation
# ============================================================

def annotate_frame(
    frame: np.ndarray,
    kps: Optional[np.ndarray],
    cli: float,
    dog_bbox: Optional[Tuple[int, int, int, int]] = None,
    conf_thr: float = CONF_THRESHOLD,
) -> np.ndarray:
    """
    Draw skeleton, dog bounding box, and CLI score on a video frame.

    dog_bbox: (x1, y1, x2, y2) in full-frame pixel coordinates.
              Drawn as a green box to show the user which animal is being tracked.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Dog bounding box ────────────────────────────────────────────────────
    if dog_bbox is not None:
        dx1, dy1, dx2, dy2 = dog_bbox
        cv2.rectangle(out, (dx1, dy1), (dx2, dy2), (0, 230, 0), 2, cv2.LINE_AA)
        cv2.putText(out, "DOG", (dx1 + 4, dy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2, cv2.LINE_AA)

    # ── Skeleton ────────────────────────────────────────────────────────────
    if kps is not None:
        _draw_dog_skeleton(out, kps, conf_thr)

    # ── CLI overlay ─────────────────────────────────────────────────────────
    if cli < 0.10:
        cli_col = (0, 220, 0)
    elif cli < 0.25:
        cli_col = (0, 200, 255)
    elif cli < 0.45:
        cli_col = (0, 140, 255)
    else:
        cli_col = (0, 0, 255)

    cv2.rectangle(out, (w - 186, 8), (w - 8, 52), (0, 0, 0), -1)
    cv2.putText(out, f"Lameness: {cli:.3f}", (w - 182, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cli_col, 2, cv2.LINE_AA)

    return out


# ============================================================
# Pose model (singleton)
# ============================================================

_pose_model = None

def get_pose_model():
    """Lazy-load YOLOv8m-pose (downloads ~50 MB once, cached by ultralytics)."""
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO('yolov8m-pose.pt')
        print("[GaitAnalyzer] YOLOv8m-pose loaded")
    return _pose_model


# ============================================================
# Keypoint extraction
# ============================================================

def extract_keypoints_from_frame(
    frame: np.ndarray,
    pose_model,
    conf_thr: float = CONF_THRESHOLD,
    target_center: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    """
    Run YOLOv8-pose on a frame and return the best matching keypoints.

    Selection strategy — two modes depending on whether a target_center is given:

    A) target_center provided (normal case — frame is already cropped to dog):
       Pick the detection whose center-of-mass is CLOSEST to the given point.
       This avoids picking up the human handler when the crop isn't perfectly tight.

    B) target_center=None (full-frame fallback):
       Pick the detection with the most high-confidence keypoints — NOT the
       largest area, since the human is always larger than the dog.

    Parameters
    ----------
    frame         : BGR image (can be a dog crop or the full frame)
    pose_model    : loaded YOLOv8-pose model
    conf_thr      : minimum keypoint confidence to count as valid
    target_center : (x, y) in pixel coords that the chosen detection should
                    be closest to. Pass the crop center when running on a dog crop.
    """
    h_frame, w_frame = frame.shape[:2]
    results = pose_model(frame, verbose=False)

    best_kps   = None
    best_score = -1.0

    for r in results:
        if r.keypoints is None:
            continue
        kps_data = r.keypoints.data          # (N, 17, 3)
        boxes    = r.boxes
        n_kps    = len(kps_data)
        n_box    = len(boxes.xyxy) if boxes is not None else 0
        n_det    = min(n_kps, n_box)
        if n_det == 0:
            continue

        for i in range(n_det):
            kps  = kps_data[i].cpu().numpy()  # (17, 3)
            bbox = boxes.xyxy[i].cpu().numpy() # x1,y1,x2,y2

            # Count valid (high-confidence) keypoints
            valid_count = int(np.sum(kps[:, 2] >= conf_thr))
            if valid_count == 0:
                continue

            if target_center is not None:
                # Mode A: proximity scoring — pick the detection closest to the dog
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                diag = np.hypot(w_frame, h_frame) + 1e-6
                dist = np.hypot(cx - target_center[0], cy - target_center[1])
                # Blend: 80% proximity, 20% keypoint quality
                score = (1.0 - dist / diag) * 0.8 + (valid_count / 17.0) * 0.2
            else:
                # Mode B: keypoint quality only — more robust than area on full frame
                score = valid_count / 17.0

            if score > best_score:
                best_score = score
                best_kps   = kps

    return best_kps  # (17, 3) or None


# ============================================================
# Biomechanical metrics
# ============================================================

def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing, falls back to moving average if too short."""
    if len(arr) < window + 2:
        return arr
    try:
        return savgol_filter(arr, window_length=min(window, len(arr) - 1 if len(arr) % 2 == 0 else len(arr)),
                             polyorder=2)
    except Exception:
        return arr


def get_trajectory(keypoints_seq: List[np.ndarray], kp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (x_arr, y_arr) trajectory for one keypoint across all frames.
    Frames where confidence < CONF_THRESHOLD are marked NaN.
    """
    xs, ys = [], []
    for kps in keypoints_seq:
        if kps is not None and kps[kp_idx, 2] >= CONF_THRESHOLD:
            xs.append(float(kps[kp_idx, 0]))
            ys.append(float(kps[kp_idx, 1]))
        else:
            xs.append(np.nan)
            ys.append(np.nan)
    return np.array(xs), np.array(ys)


def oscillation_amplitude(y_arr: np.ndarray) -> float:
    """
    Compute peak-to-trough oscillation amplitude of a trajectory.
    NaN values are ignored. Returns 0 if insufficient data.
    """
    valid = y_arr[~np.isnan(y_arr)]
    if len(valid) < 5:
        return 0.0
    smoothed = smooth(valid, window=7)
    return float(np.percentile(smoothed, 90) - np.percentile(smoothed, 10))


def symmetry_ratio(amp_left: float, amp_right: float) -> float:
    """
    SR = |left - right| / mean(left, right)
    Range [0, 2]. SR=0 is perfect symmetry. SR>0.1 is clinically notable.
    """
    mean_amp = (amp_left + amp_right) / 2.0
    if mean_amp < 1e-6:
        return 0.0
    return abs(amp_left - amp_right) / mean_amp


def head_bob_index(nose_y: np.ndarray, fps: float) -> float:
    """
    Std of the nose vertical velocity (pixels/second).
    Normalised by image height (passed separately).
    High HBI → irregular head movement → possible front-limb lameness.
    """
    valid = nose_y[~np.isnan(nose_y)]
    if len(valid) < 6:
        return 0.0
    smoothed = smooth(valid, window=7)
    velocity = np.diff(smoothed) * fps   # dy/dt in pixels/s
    return float(np.std(velocity))


def hip_drop_index(left_hip_y: np.ndarray, right_hip_y: np.ndarray,
                   img_height: float) -> float:
    """
    HDI = |mean(left_hip_y) - mean(right_hip_y)| / image_height
    Pelvis drops toward the healthy side in hindlimb lameness.
    Normalised to image height so it's resolution-independent.
    """
    lv = left_hip_y[~np.isnan(left_hip_y)]
    rv = right_hip_y[~np.isnan(right_hip_y)]
    if len(lv) < 3 or len(rv) < 3:
        return 0.0
    return abs(float(np.mean(lv)) - float(np.mean(rv))) / (img_height + 1e-6)


def velocity_consistency(left_hip_x: np.ndarray, right_hip_x: np.ndarray,
                          fps: float) -> float:
    """
    Coefficient of variation of forward (x-axis) velocity.
    Uses midpoint of hips as the body centre of mass proxy.
    High CV → irregular movement → guarded gait.
    """
    # Combine both hips to estimate pelvis x
    lx = left_hip_x.copy()
    rx = right_hip_x.copy()
    # Where one is NaN, use the other
    com_x = np.where(~np.isnan(lx) & ~np.isnan(rx), (lx + rx) / 2,
                     np.where(~np.isnan(lx), lx, rx))
    valid = com_x[~np.isnan(com_x)]
    if len(valid) < 6:
        return 0.0
    smoothed = smooth(valid, window=9)
    vel = np.abs(np.diff(smoothed)) * fps
    mean_v = np.mean(vel)
    if mean_v < 1.0:
        return 0.0   # dog is essentially stationary
    return float(np.std(vel) / mean_v)


def stride_regularity(left_y: np.ndarray, right_y: np.ndarray) -> float:
    """
    Normalized cross-correlation between left and right limb vertical trajectories.
    Perfect bilateral symmetry → score ≈ 1.0.
    Limb asymmetry → score < 1.0.
    """
    lv = left_y[~np.isnan(left_y)]
    rv = right_y[~np.isnan(right_y)]
    n = min(len(lv), len(rv))
    if n < 10:
        return 1.0   # not enough data — assume symmetric
    lv, rv = lv[:n] - np.mean(lv[:n]), rv[:n] - np.mean(rv[:n])
    corr = correlate(lv, rv, mode='full')
    norm = (np.std(lv) * np.std(rv) * n + 1e-8)
    corr_norm = corr / norm
    return float(np.max(np.abs(corr_norm)))


def composite_lameness_index(
    sr_fore: float, sr_hind: float,
    hbi: float, hdi: float, vc: float,
    img_height: float, fps: float
) -> Tuple[float, dict]:
    """
    Weighted composite lameness index normalised to [0, 1].

    Normalisation denominators from veterinary gait literature:
      SR: 0.6 = extreme asymmetry
      HBI: fps * 20 pixels/frame = 20fps-normalised HBI (~50 px/s at 30fps)
      HDI: 0.15 (15% of image height)
      VC: 1.5 (very irregular movement)

    Weights sum to 1.0; SR components have highest weight as most validated.
    """
    sr_fore_n  = min(sr_fore / 0.6, 1.0)
    sr_hind_n  = min(sr_hind / 0.6, 1.0)
    hbi_n      = min(hbi / (fps * 20), 1.0)
    hdi_n      = min(hdi / 0.15, 1.0)
    vc_n       = min(vc / 1.5, 1.0)

    cli = (0.35 * sr_fore_n +
           0.35 * sr_hind_n +
           0.15 * hbi_n +
           0.10 * hdi_n +
           0.05 * vc_n)

    return cli, {
        'sr_fore':  round(sr_fore, 4),
        'sr_hind':  round(sr_hind, 4),
        'sr_fore_normalised':  round(sr_fore_n, 4),
        'sr_hind_normalised':  round(sr_hind_n, 4),
        'head_bob_index':      round(hbi, 2),
        'hip_drop_index':      round(hdi, 4),
        'velocity_cv':         round(vc, 4),
    }


def assess_lameness(cli: float, sr_fore: float, sr_hind: float,
                    hbi: float, hdi: float) -> dict:
    """Clinical interpretation of lameness metrics."""
    if cli < 0.10:
        grade = 0
        label = "No lameness detected"
        desc  = "Gait appears symmetrical across all measured metrics."
    elif cli < 0.20:
        grade = 1
        label = "Mild asymmetry"
        desc  = "Subtle gait irregularity. May be within normal variation or early-stage lameness."
    elif cli < 0.35:
        grade = 2
        label = "Moderate asymmetry — veterinary evaluation recommended"
        desc  = "Noticeable asymmetry suggesting compensatory movement. Likely lameness present."
    elif cli < 0.55:
        grade = 3
        label = "Significant lameness"
        desc  = "Clear weight-shifting and asymmetric movement. Prompt veterinary assessment advised."
    else:
        grade = 4
        label = "Severe lameness"
        desc  = "Marked non-weight-bearing pattern. Urgent veterinary assessment required."

    # Localise which limb pair is most affected
    suspects = []
    if sr_fore > 0.15 or hbi > 40:
        suspects.append("forelimbs (front legs)")
    if sr_hind > 0.15 or hdi > 0.08:
        suspects.append("hindlimbs (back legs)")

    return {
        'lameness_grade': grade,
        'label': label,
        'description': desc,
        'suspected_limbs': suspects if suspects else ['insufficient signal'],
        'note': 'For definitive diagnosis, veterinary orthopaedic examination with force plate is required.',
    }


# ============================================================
# Analysis pipeline
# ============================================================

class GaitAnalyzer:
    """Full gait analysis pipeline for dog walking videos."""

    def __init__(self,
                 fps_target: float = 15.0,
                 conf_thr: float = CONF_THRESHOLD,
                 max_frames: int = 300):
        self.fps_target = fps_target
        self.conf_thr   = conf_thr
        self.max_frames = max_frames

    def analyze_video(self, video_path: str) -> dict:
        """
        Main entry point.  Returns dict compatible with the FastAPI response.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # How many original frames to skip between processed frames
        frame_skip = max(1, round(fps_orig / self.fps_target))
        eff_fps = fps_orig / frame_skip

        pose_model = get_pose_model()

        keypoints_seq: List[Optional[np.ndarray]] = []
        annotated_frames: List[np.ndarray]         = []
        frame_n = 0
        processed = 0

        try:
            while processed < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_n += 1
                if frame_n % frame_skip != 0:
                    continue

                # ── Detect dog at aggressive (low) confidence ────────────────
                # Using 0.15 instead of 0.30 so we pick up partially-visible
                # dogs, dogs at an angle, and small breeds.  A false dog detection
                # is much less harmful than falling back to the full frame where
                # the human handler is almost always detected instead.
                bbox = detect_dog(frame, conf=0.15)

                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    # 25 % padding — gives the pose model body context while
                    # still keeping the human mostly out of frame
                    pad_x = int((bx2 - bx1) * 0.25)
                    pad_y = int((by2 - by1) * 0.25)
                    cx1 = max(0, bx1 - pad_x);  cy1 = max(0, by1 - pad_y)
                    cx2 = min(frame_w, bx2 + pad_x); cy2 = min(frame_h, by2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    # Center of the crop in crop-local coordinates — used by the
                    # pose selector to pick the detection closest to the dog.
                    crop_h, crop_w = crop.shape[:2]
                    crop_center = (crop_w / 2.0, crop_h / 2.0)
                    offset_x, offset_y = cx1, cy1
                else:
                    # No dog found this frame — skip pose entirely.
                    # DO NOT fall back to the full frame: that almost always
                    # tracks the human handler instead of the dog, producing
                    # completely wrong gait metrics.
                    crop        = None
                    crop_center = None
                    offset_x = offset_y = 0

                if crop is None:
                    keypoints_seq.append(None)
                    processed += 1
                    continue

                kps = extract_keypoints_from_frame(crop, pose_model, self.conf_thr,
                                                   target_center=crop_center)

                # Remap keypoints from crop-local → full-frame coordinates
                if kps is not None and bbox is not None:
                    kps_remapped      = kps.copy()
                    kps_remapped[:, 0] += offset_x
                    kps_remapped[:, 1] += offset_y
                    kps = kps_remapped

                keypoints_seq.append(kps)

                # Annotate a sample of frames (memory-efficient)
                if len(annotated_frames) < 30:
                    running_cli = self._quick_cli(keypoints_seq, frame_h, eff_fps)
                    annotated   = annotate_frame(frame, kps, running_cli,
                                                 dog_bbox=bbox,
                                                 conf_thr=self.conf_thr)
                    annotated_frames.append(annotated)

                processed += 1

        finally:
            cap.release()

        if processed < 8:
            return {
                'error': f'Too few frames processed ({processed}). '
                         'Video must be ≥ 2 seconds with dog clearly visible.',
                'frames_processed': processed,
            }

        # ---- Compute all metrics ----
        metrics, cli, assessment = self._compute_metrics(
            keypoints_seq, frame_h, eff_fps
        )

        # ---- Generate result plots ----
        plot_b64 = self._generate_plot(keypoints_seq, eff_fps, cli, metrics)

        # ---- Pick representative annotated frame (peak lameness visible) ----
        keyframe_b64 = None
        if annotated_frames:
            import base64
            mid = len(annotated_frames) // 2
            _, buf = cv2.imencode('.jpg', annotated_frames[mid],
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframe_b64 = base64.b64encode(buf).decode('utf-8')

        return {
            'lameness_index':    round(float(cli), 4),
            'lameness_grade':    assessment['lameness_grade'],
            'label':             assessment['label'],
            'description':       assessment['description'],
            'suspected_limbs':   assessment['suspected_limbs'],
            'metrics':           metrics,
            'frames_analyzed':   processed,
            'effective_fps':     round(eff_fps, 1),
            'duration_s':        round(processed / eff_fps, 1),
            'plot_b64':          plot_b64,
            'keyframe_b64':      keyframe_b64,
            'note':              assessment['note'],
        }

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _quick_cli(self, seq: list, img_height: float, fps: float) -> float:
        """Fast approximate CLI from a partial sequence for live annotation."""
        if len(seq) < 8:
            return 0.0
        _, ly = get_trajectory(seq, KP['left_wrist'])
        _, ry = get_trajectory(seq, KP['right_wrist'])
        sr = symmetry_ratio(oscillation_amplitude(ly), oscillation_amplitude(ry))
        return min(sr / 0.6, 1.0) * 0.5   # rough estimate

    def _compute_metrics(
        self,
        seq: List[Optional[np.ndarray]],
        img_height: float,
        fps: float
    ) -> Tuple[dict, float, dict]:

        # Extract all needed trajectories
        _, nose_y    = get_trajectory(seq, KP['nose'])
        _, lw_y      = get_trajectory(seq, KP['left_wrist'])
        _, rw_y      = get_trajectory(seq, KP['right_wrist'])
        _, la_y      = get_trajectory(seq, KP['left_ankle'])
        _, ra_y      = get_trajectory(seq, KP['right_ankle'])
        lh_x, lh_y  = get_trajectory(seq, KP['left_hip'])
        rh_x, rh_y  = get_trajectory(seq, KP['right_hip'])
        ls_x, _     = get_trajectory(seq, KP['left_shoulder'])
        rs_x, _     = get_trajectory(seq, KP['right_shoulder'])

        # Forelimb symmetry (wrists = forepaws)
        amp_lw   = oscillation_amplitude(lw_y)
        amp_rw   = oscillation_amplitude(rw_y)
        sr_fore  = symmetry_ratio(amp_lw, amp_rw)

        # Hindlimb symmetry (ankles = hindpaws)
        amp_la   = oscillation_amplitude(la_y)
        amp_ra   = oscillation_amplitude(ra_y)
        sr_hind  = symmetry_ratio(amp_la, amp_ra)

        # Head bob
        hbi = head_bob_index(nose_y, fps)

        # Hip drop
        hdi = hip_drop_index(lh_y, rh_y, img_height)

        # Velocity consistency (hips give best horizontal movement signal)
        vc = velocity_consistency(lh_x, rh_x, fps)

        # Stride regularity (cross-correlation)
        fore_reg = stride_regularity(lw_y, rw_y)
        hind_reg = stride_regularity(la_y, ra_y)

        cli, raw_metrics = composite_lameness_index(
            sr_fore, sr_hind, hbi, hdi, vc, img_height, fps
        )
        raw_metrics['forelimb_stride_regularity'] = round(fore_reg, 4)
        raw_metrics['hindlimb_stride_regularity'] = round(hind_reg, 4)
        raw_metrics['forepaw_left_amplitude_px']  = round(amp_lw, 2)
        raw_metrics['forepaw_right_amplitude_px'] = round(amp_rw, 2)
        raw_metrics['hindpaw_left_amplitude_px']  = round(amp_la, 2)
        raw_metrics['hindpaw_right_amplitude_px'] = round(amp_ra, 2)

        assessment = assess_lameness(cli, sr_fore, sr_hind, hbi, hdi)

        return raw_metrics, cli, assessment

    def _generate_plot(
        self,
        seq: List[Optional[np.ndarray]],
        fps: float,
        cli: float,
        metrics: dict
    ) -> str:
        """4-panel gait analysis plot → base64 PNG."""
        import base64
        from io import BytesIO

        t = np.arange(len(seq)) / fps

        _, nose_y = get_trajectory(seq, KP['nose'])
        _, lw_y   = get_trajectory(seq, KP['left_wrist'])
        _, rw_y   = get_trajectory(seq, KP['right_wrist'])
        _, la_y   = get_trajectory(seq, KP['left_ankle'])
        _, ra_y   = get_trajectory(seq, KP['right_ankle'])
        _, lh_y   = get_trajectory(seq, KP['left_hip'])
        _, rh_y   = get_trajectory(seq, KP['right_hip'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#111827')
        for ax in axes.flatten():
            ax.set_facecolor('#1a2235')
            ax.tick_params(colors='#8892a4')
            ax.spines['bottom'].set_color('#253047')
            ax.spines['left'].set_color('#253047')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Panel 1 — Forelimb trajectories
        ax = axes[0, 0]
        ax.plot(t, lw_y, color='#ff6464', linewidth=1.5, label='Left forepaw', alpha=0.9)
        ax.plot(t, rw_y, color='#6496ff', linewidth=1.5, label='Right forepaw', alpha=0.9)
        ax.set_title(f'Forelimb Vertical Position  SR={metrics["sr_fore"]:.3f}',
                     color='#e2e8f0', fontsize=11)
        ax.set_xlabel('Time (s)', color='#8892a4')
        ax.set_ylabel('Y position (px)', color='#8892a4')
        ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
        ax.invert_yaxis()   # image coords: y=0 is top
        ax.grid(True, alpha=0.15)

        # Panel 2 — Hindlimb trajectories
        ax = axes[0, 1]
        ax.plot(t, la_y, color='#ff6464', linewidth=1.5, label='Left hindpaw', alpha=0.9)
        ax.plot(t, ra_y, color='#6496ff', linewidth=1.5, label='Right hindpaw', alpha=0.9)
        ax.set_title(f'Hindlimb Vertical Position  SR={metrics["sr_hind"]:.3f}',
                     color='#e2e8f0', fontsize=11)
        ax.set_xlabel('Time (s)', color='#8892a4')
        ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15)

        # Panel 3 — Head + hip tracking
        ax = axes[1, 0]
        ax.plot(t, nose_y, color='#ffd700', linewidth=1.5, label='Nose (head)', alpha=0.9)
        ax.plot(t, lh_y, color='#ff8c00', linewidth=1, label='Left hip', alpha=0.7, linestyle='--')
        ax.plot(t, rh_y, color='#98fb98', linewidth=1, label='Right hip', alpha=0.7, linestyle='--')
        ax.set_title(f'Head & Hip Trajectories  HBI={metrics["head_bob_index"]:.1f}',
                     color='#e2e8f0', fontsize=11)
        ax.set_xlabel('Time (s)', color='#8892a4')
        ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15)

        # Panel 4 — CLI bar chart + metric summary
        ax = axes[1, 1]
        metric_labels = ['Fore SR', 'Hind SR', 'Head Bob\n(norm)', 'Hip Drop\n(norm)', 'Velocity\nCV']
        metric_values = [
            metrics['sr_fore_normalised'],
            metrics['sr_hind_normalised'],
            min(metrics['head_bob_index'] / 40, 1.0),
            min(metrics['hip_drop_index'] / 0.15, 1.0),
            min(metrics['velocity_cv'] / 1.5, 1.0),
        ]
        colours = ['#ff6464', '#6496ff', '#ffd700', '#ff8c00', '#98fb98']
        bars = ax.bar(metric_labels, metric_values, color=colours, alpha=0.8, edgecolor='none')
        ax.axhline(0.25, color='#ef4444', linestyle='--', linewidth=1.5,
                   label='Concern threshold (0.25)')
        ax.set_ylim(0, 1.0)
        ax.set_title(f'Metric Breakdown  CLI = {cli:.3f}', color='#e2e8f0', fontsize=11)
        ax.set_ylabel('Normalised Score', color='#8892a4')
        ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
        ax.grid(True, alpha=0.15, axis='y')
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='white')

        fig.suptitle(f'Canine Gait Analysis — Composite Lameness Index: {cli:.3f}',
                     color='#e2e8f0', fontsize=13, y=1.01)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        import base64
        return base64.b64encode(buf.read()).decode('utf-8')


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Canine Gait Lameness Analysis')
    parser.add_argument('--video', required=True, help='Path to dog walking video')
    parser.add_argument('--out',   default='gait_results', help='Output directory')
    parser.add_argument('--fps',   type=float, default=15.0, help='Processing FPS')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {args.video}")
    print("Loading YOLOv8m-pose...")

    analyzer = GaitAnalyzer(fps_target=args.fps)
    result   = analyzer.analyze_video(args.video)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'='*55}")
    print(f" Composite Lameness Index : {result['lameness_index']:.4f}")
    print(f" Grade                    : {result['lameness_grade']}/4")
    print(f" Assessment               : {result['label']}")
    print(f" Suspected limbs          : {', '.join(result['suspected_limbs'])}")
    print(f" Frames analysed          : {result['frames_analyzed']} @ {result['effective_fps']} fps")
    print(f"\n Metrics:")
    for k, v in result['metrics'].items():
        print(f"   {k:35s}: {v}")
    print(f"{'='*55}")

    # Save plot
    if result.get('plot_b64'):
        import base64
        with open(out_dir / 'gait_plot.png', 'wb') as f:
            f.write(base64.b64decode(result['plot_b64']))
        print(f"\nPlot saved: {out_dir / 'gait_plot.png'}")

    # Save JSON
    clean = {k: v for k, v in result.items() if k not in ('plot_b64', 'keyframe_b64')}
    with open(out_dir / 'gait_result.json', 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Results: {out_dir / 'gait_result.json'}")


if __name__ == '__main__':
    main()
