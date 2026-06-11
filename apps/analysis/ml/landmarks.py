"""
Pose landmark extraction using MediaPipe Pose.

Returns:
    landmarks: dict[name] = {x, y, z, visibility} in normalized [0,1] coords.
    raw:       list of 33 dicts (full MediaPipe topology) for storage.
    mask:      person segmentation mask, when MediaPipe provides one.

We do NOT hardcode body-shape thresholds here — this module only produces
geometric landmarks. Downstream `features.py` derives proportional
measurements, and `classifier.py` does the ML-based shape prediction.
"""
from __future__ import annotations
import numpy as np

try:
    import mediapipe as mp
    _POSE = mp.solutions.pose
except Exception:  # pragma: no cover - mediapipe missing
    mp = None
    _POSE = None

# MediaPipe Pose index → semantic name (subset we care about).
_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
    0:  "nose",
}

class PoseExtractionError(Exception):
    pass

def extract(img_rgb: np.ndarray) -> tuple[dict, list[dict], np.ndarray | None]:
    if _POSE is None:
        raise PoseExtractionError("MediaPipe not installed.")

    with _POSE.Pose(static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5) as pose:
        result = pose.process(img_rgb)

    if not result.pose_landmarks:
        raise PoseExtractionError("No person detected. Ensure full front-view body is visible.")

    raw = []
    named = {}
    for idx, lm in enumerate(result.pose_landmarks.landmark):
        item = {"x": float(lm.x), "y": float(lm.y),
                "z": float(lm.z), "v": float(lm.visibility)}
        raw.append(item)
        if idx in _NAMES:
            named[_NAMES[idx]] = item

    # Sanity: require both shoulders and both hips to be visible enough.
    for key in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        if named.get(key, {}).get("v", 0) < 0.4:
            raise PoseExtractionError(
                f"Low landmark visibility for {key}. "
                "Re-take photo with better lighting and a full front-view pose.")

    return named, raw, result.segmentation_mask