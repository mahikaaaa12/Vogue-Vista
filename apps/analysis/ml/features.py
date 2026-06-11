"""
Geometric feature extraction.

We compute proportional measurements from MediaPipe landmarks. All values
are scale-invariant ratios — no absolute pixels, no hardcoded class
thresholds. These features are the input to the ML classifier.

Features:
    shoulder_width, waist_width (estimated), hip_width, torso_height,
    shoulder_to_hip_ratio, waist_to_hip_ratio, shoulder_to_waist_ratio,
    symmetry_score, vertical_balance, midline_offset.

Waist landmark approximation:
    MediaPipe does not output a waist landmark. We estimate the waist
    line as the narrowest horizontal silhouette slice between the
    shoulder line and the hip line, using the segmentation mask when
    available, otherwise interpolating at 55% of the torso height
    (this is a *geometric* anchor, NOT a body-shape decision rule).
"""
from __future__ import annotations
import numpy as np

def _pt(lm, key):
    p = lm[key]
    return np.array([p["x"], p["y"]])

def _dist(a, b):
    return float(np.linalg.norm(a - b))

def derive(landmarks: dict, segmentation_mask=None) -> dict:
    ls, rs = _pt(landmarks, "left_shoulder"), _pt(landmarks, "right_shoulder")
    lh, rh = _pt(landmarks, "left_hip"), _pt(landmarks, "right_hip")

    shoulder_mid = (ls + rs) / 2.0
    hip_mid      = (lh + rh) / 2.0

    joint_shoulder_width = _dist(ls, rs)
    joint_hip_width      = _dist(lh, rh)
    torso_height   = _dist(shoulder_mid, hip_mid)
    center_x = float((shoulder_mid[0] + hip_mid[0]) / 2.0)

    shoulder_y = float(shoulder_mid[1])
    hip_y = float(hip_mid[1])
    chest_y = shoulder_y + (hip_y - shoulder_y) * 0.18
    waist_y = shoulder_y + (hip_y - shoulder_y) * 0.55
    low_hip_y = hip_y + (hip_y - shoulder_y) * 0.16

    shoulder_width = _mask_width(segmentation_mask, shoulder_y, center_x) or joint_shoulder_width
    chest_width = _mask_width(segmentation_mask, chest_y, center_x) or shoulder_width
    waist_width = _mask_width(segmentation_mask, waist_y, center_x)
    hip_width = (
        _mask_width(segmentation_mask, low_hip_y, center_x)
        or _mask_width(segmentation_mask, hip_y, center_x)
        or joint_hip_width
    )

    # Waist estimation: search the horizontal slice with the smallest
    # silhouette width between the shoulder and hip rows.
    if not waist_width:
        waist_width = _estimate_waist(
            segmentation_mask,
            shoulder_mid,
            hip_mid,
            fallback=(shoulder_width + hip_width) / 2.0 * 0.85,
        )

    # Symmetry: deviation of left/right limb lengths from each other.
    sym_shoulder = abs(_dist(ls, shoulder_mid) - _dist(rs, shoulder_mid))
    sym_hip      = abs(_dist(lh, hip_mid)      - _dist(rh, hip_mid))
    symmetry_score = 1.0 - min(1.0, (sym_shoulder + sym_hip) / max(shoulder_width + hip_width, 1e-6))

    # Vertical balance: midline offset (shoulder vs hip center x).
    midline_offset = abs(shoulder_mid[0] - hip_mid[0])

    eps = 1e-6
    shoulder_to_hip = shoulder_width / (hip_width + eps)
    waist_to_hip = waist_width / (hip_width + eps)
    shoulder_to_waist = shoulder_width / (waist_width + eps)
    avg_shoulder_hip = (shoulder_width + hip_width) / 2.0
    waist_definition = (
        max(0.0, (avg_shoulder_hip - waist_width) / avg_shoulder_hip)
        if avg_shoulder_hip > eps
        else 0.0
    )
    avg_visibility = _avg_visibility(
        landmarks,
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    )

    measurements = {
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "hip_width": hip_width,
        "torso_height": torso_height,
    }
    features = {
        "shoulder_width":    shoulder_width,
        "chest_width":       chest_width,
        "hip_width":         hip_width,
        "waist_width":       waist_width,
        "shoulder_to_hip":   shoulder_to_hip,
        "chest_to_hip":      chest_width / (hip_width + eps),
        "waist_to_shoulder": waist_width / (shoulder_width + eps),
        "waist_to_hip":      waist_to_hip,
        "shoulder_to_waist": shoulder_to_waist,
        "waist_definition":  waist_definition,
        "body_balance":      shoulder_to_hip,
        "torso_height":      torso_height,
        "avg_visibility":    avg_visibility,
        "torso_aspect":      torso_height / (max(shoulder_width, hip_width) + eps),
        "symmetry":          symmetry_score,
        "midline_offset":    float(midline_offset),
    }
    return {"measurements": measurements, "features": features}

def feature_vector(features: dict) -> np.ndarray:
    """Stable ordering for ML input."""
    keys = ["shoulder_to_hip", "waist_to_hip", "shoulder_to_waist",
            "torso_aspect", "symmetry", "midline_offset"]
    return np.array([features[k] for k in keys], dtype=np.float32)

FEATURE_KEYS = ["shoulder_to_hip", "waist_to_hip", "shoulder_to_waist",
                "torso_aspect", "symmetry", "midline_offset"]

def _avg_visibility(landmarks: dict, keys: tuple[str, ...]) -> float:
    values = []
    for key in keys:
        point = landmarks.get(key, {})
        values.append(float(point.get("visibility", point.get("v", 0.0))))
    return float(sum(values) / len(values)) if values else 0.0

def _mask_width(mask, y_norm: float, center_x_norm: float, threshold: float = 0.5) -> float:
    """Width of the person-mask segment closest to the torso center."""
    if mask is None:
        return 0.0
    height, width = mask.shape[:2]
    y = max(0, min(height - 1, int(round(y_norm * height))))
    center_x = max(0, min(width - 1, int(round(center_x_norm * width))))
    xs = np.where(mask[y] > threshold)[0]
    if xs.size == 0:
        return 0.0

    segments = []
    start = int(xs[0])
    prev = int(xs[0])
    for x_raw in xs[1:]:
        x = int(x_raw)
        if x == prev + 1:
            prev = x
            continue
        segments.append((start, prev))
        start = prev = x
    segments.append((start, prev))

    def distance_to_center(segment):
        left, right = segment
        if left <= center_x <= right:
            return 0
        return min(abs(center_x - left), abs(center_x - right))

    left, right = min(segments, key=distance_to_center)
    return float((right - left + 1) / width)

def _estimate_waist(mask, shoulder_mid, hip_mid, fallback: float) -> float:
    """Find the narrowest silhouette slice between shoulder and hip rows."""
    if mask is None:
        return fallback
    import numpy as np
    h, w = mask.shape[:2]
    y0 = int(min(shoulder_mid[1], hip_mid[1]) * h)
    y1 = int(max(shoulder_mid[1], hip_mid[1]) * h)
    if y1 - y0 < 4:
        return fallback
    # Restrict to torso vertical band; measure silhouette width per row.
    band = (mask[y0:y1] > 0.5).astype(np.uint8)
    widths = []
    for row in band:
        xs = np.where(row > 0)[0]
        if xs.size:
            widths.append((xs.max() - xs.min()) / w)
    if not widths:
        return fallback
    return float(min(widths))

def build_features(measurements):

    return {

        "shoulder_width":
            measurements.get("shoulder_width", 0),

        "hip_width":
            measurements.get("hip_width", 0),

        "waist_width":
            measurements.get("waist_width", 0),

        "shoulder_to_hip":
            measurements.get("shoulder_to_hip", 0),
    }