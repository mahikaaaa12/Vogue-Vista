"""
Image preprocessing pipeline.

Responsibilities:
- Decode the uploaded file to a normalized RGB numpy array.
- EXIF-orient correction (selfies on mobile are commonly rotated).
- Resize to a consistent max edge to keep MediaPipe inference fast and
  scale-invariant (we work in normalized 0..1 coordinates downstream).
- Light denoise + contrast normalization to stabilize landmark detection
  without distorting silhouette geometry.
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageOps
import cv2

MAX_EDGE = 1024  # px

def load_image(file_obj) -> np.ndarray:
    """Read a Django UploadedFile / path / bytes into an RGB ndarray."""
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        file_obj.seek(0)
        img = Image.open(io.BytesIO(data))
    elif isinstance(file_obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(file_obj))
    else:
        img = Image.open(file_obj)

    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)

def normalize(img: np.ndarray) -> np.ndarray:
    """Resize (keep aspect), denoise, normalize contrast."""
    h, w = img.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # Mild denoise — preserves edges (important for silhouette extraction).
    img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    # CLAHE on the L channel for robust contrast across lighting conditions.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return img

def preprocess(file_obj) -> np.ndarray:
    return normalize(load_image(file_obj))
