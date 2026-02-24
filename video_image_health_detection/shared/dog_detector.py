"""
Canine Health AI — Dog Detector
Shared YOLOv8-based dog detection used by all inference modules.
Provides:
  - detect_dog(frame) → bbox (x1, y1, x2, y2) or None
  - crop_dog(frame, bbox, margin) → cropped image
  - crop_face(frame, bbox) → upper 45% of dog bbox (approximate face/head region)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

_detector = None  # singleton

def get_detector():
    """Lazy-load YOLOv8n (nano, fastest) for dog detection."""
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO('yolov8n.pt')  # auto-downloads ~6MB
        print("[DogDetector] YOLOv8n loaded")
    return _detector


def detect_dog(
    frame: np.ndarray,
    conf: float = 0.35,
    return_all: bool = False
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the most prominent dog in the frame.
    Returns (x1, y1, x2, y2) in pixel coords, or None if no dog found.

    COCO class 16 = dog
    """
    model = get_detector()
    results = model(frame, classes=[16], conf=conf, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            boxes.append((area, x1, y1, x2, y2))

    if not boxes:
        return None

    if return_all:
        return [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in sorted(boxes, reverse=True)]

    # Return largest dog
    _, x1, y1, x2, y2 = max(boxes)
    return (x1, y1, x2, y2)


def crop_dog(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.05
) -> np.ndarray:
    """
    Crop dog with optional margin padding.
    margin: fraction of bbox size to add as padding
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    dw = int((x2 - x1) * margin)
    dh = int((y2 - y1) * margin)
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw)
    y2 = min(h, y2 + dh)
    return frame[y1:y2, x1:x2]


def crop_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    face_fraction: float = 0.45
) -> Optional[np.ndarray]:
    """
    Crop approximate head/face region = upper portion of dog bbox.
    face_fraction: fraction of bbox height to take (default 45%)
    Used by heart rate module for rPPG ROI.
    """
    x1, y1, x2, y2 = bbox
    box_h = y2 - y1
    face_y2 = y1 + int(box_h * face_fraction)
    face_y2 = min(face_y2, y2)

    face_crop = frame[y1:face_y2, x1:x2]
    if face_crop.size == 0:
        return None
    return face_crop


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "Dog",
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw bounding box and label on frame."""
    out = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


def detect_and_crop(
    image: np.ndarray,
    target_size: int = 224,
    margin: float = 0.1
) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    Convenience: detect dog, crop + resize to target_size.
    Returns (cropped_resized, bbox) or (original_resized, None) if no dog found.
    """
    bbox = detect_dog(image)
    if bbox is not None:
        crop = crop_dog(image, bbox, margin=margin)
    else:
        crop = image  # fall back to whole image
    resized = cv2.resize(crop, (target_size, target_size))
    return resized, bbox
