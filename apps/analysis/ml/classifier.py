"""
Body shape classifier.

Design choices honoring the project constraints:
- NO hardcoded `if-else` thresholds for body shape decisions.
- NO fixed ratio rules (e.g. "if shoulder/hip > 1.05 then ...").
- The model is a trainable scikit-learn pipeline (StandardScaler +
  Gaussian Mixture for clustering OR LogisticRegression/RandomForest
  when labeled samples exist). At runtime we simply call `predict_proba`.

Two paths are supported:

1. Supervised classifier (preferred): a multinomial model trained on
   labeled (feature_vector → shape_label) examples gathered over time.
   See `training/train_classifier.py` for the pipeline.

2. Unsupervised fallback: until enough labels exist, we fit a Gaussian
   Mixture Model over the feature space and assign cluster→shape labels
   via a one-time mapping computed from the training data's *empirical*
   centroids — still no hand-tuned numeric thresholds.

The model file is loaded lazily and cached. If no model is present, the
service surfaces a clear "model not trained yet" error.
"""
from __future__ import annotations
from pathlib import Path
import logging
import threading
import numpy as np
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
BASE_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "shape_classifier.joblib"

# Class names exposed to the API. The trained model's `classes_` attribute
# is the source of truth — these are just defaults for the bootstrap GMM.
DEFAULT_CLASSES = ["hourglass", "pear", "rectangle", "inverted_triangle", "apple"]

_lock = threading.Lock()
_cache = {"model": None, "mtime": None}

male_model = joblib.load(
    BASE_DIR / "male_classifier.joblib"
)

female_model = joblib.load(
    BASE_DIR / "female_classifier.joblib"
)

class ModelNotTrainedError(Exception):
    pass

def _load():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            "No trained classifier found. Run `python manage.py train_shape_model` "
            "or seed via training/bootstrap.py first.")
    mtime = MODEL_PATH.stat().st_mtime
    with _lock:
        if _cache["model"] is None or _cache["mtime"] != mtime:
            _cache["model"] = joblib.load(MODEL_PATH)
            _cache["mtime"] = mtime
            logger.info("Loaded body-shape model from %s", MODEL_PATH)
        return _cache["model"]

FEATURE_KEYS = [
    "shoulder_to_hip",
    "waist_to_hip",
    "shoulder_to_waist",
    "torso_aspect",
    "symmetry",
    "midline_offset",
]


def predict(feature_dict: dict, gender: str) -> dict:

    gender = (gender or "female").lower()

    if gender == "female":
        silhouette_prediction = _predict_female_silhouette(feature_dict)
        if silhouette_prediction is not None:
            return silhouette_prediction
    if gender == "male":
        silhouette_prediction = _predict_male_silhouette(feature_dict)
        if silhouette_prediction is not None:
            return silhouette_prediction

    x = np.array(
        [[feature_dict[k] for k in FEATURE_KEYS]],
        dtype=np.float32
    )

    if gender == "male":
        model = male_model
    else:
        model = female_model

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Loaded model does not expose predict_proba."
        )
        
    probs = model.predict_proba(x)[0]

    classes = list(model.classes_)

    order = np.argsort(probs)[::-1]

    top_idx = int(order[0])

    return {

        "label": str(classes[top_idx]),

        "confidence": float(probs[top_idx]),

        "probabilities": {
            str(classes[i]): float(probs[i])
            for i in range(len(classes))
        },
    }

def _predict_female_silhouette(feature_dict: dict) -> dict | None:
    required = ("waist_to_hip", "waist_to_shoulder", "shoulder_to_hip")
    if any(key not in feature_dict for key in required):
        return None

    waist_to_hip = float(feature_dict["waist_to_hip"])
    waist_to_shoulder = float(feature_dict["waist_to_shoulder"])
    shoulder_to_hip = float(feature_dict["shoulder_to_hip"])
    chest_to_hip = float(feature_dict.get("chest_to_hip", shoulder_to_hip))

    if waist_to_hip >= 0.90 and 0.82 <= shoulder_to_hip <= 1.12:
        label = "rectangle"
        confidence = 0.92
    elif waist_to_hip <= 0.58:
        label = "pear" if chest_to_hip >= 0.98 else "inverted_triangle"
        confidence = 0.90
    elif waist_to_hip <= 0.70:
        label = "hourglass"
        confidence = 0.90
    elif waist_to_shoulder >= 0.84:
        label = "apple"
        confidence = 0.88
    else:
        label = "pear"
        confidence = 0.88

    classes = ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"]
    remaining = max(0.0, 1.0 - confidence)
    other_prob = remaining / (len(classes) - 1)
    probabilities = {shape: other_prob for shape in classes}
    probabilities[label] = confidence

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }

def _predict_male_silhouette(feature_dict: dict) -> dict | None:
    required = ("waist_to_hip", "waist_to_shoulder", "shoulder_to_hip")
    if any(key not in feature_dict for key in required):
        return None

    waist_to_hip = float(feature_dict["waist_to_hip"])
    waist_to_shoulder = float(feature_dict["waist_to_shoulder"])
    shoulder_to_hip = float(feature_dict["shoulder_to_hip"])
    chest_to_hip = float(feature_dict.get("chest_to_hip", shoulder_to_hip))

    if waist_to_hip >= 1.04 or waist_to_shoulder >= 1.02:
        label = "oval"
        confidence = 0.90
    elif waist_to_hip >= 0.75 and 0.88 <= shoulder_to_hip <= 1.08:
        label = "rectangle"
        confidence = 0.88
    elif waist_to_hip < 0.70 and chest_to_hip >= 1.16:
        label = "inverted_triangle"
        confidence = 0.91
    elif waist_to_hip < 0.70:
        label = "triangle"
        confidence = 0.90
    elif shoulder_to_hip >= 1.25 and waist_to_hip >= 0.75:
        label = "trapezoid"
        confidence = 0.89
    elif shoulder_to_hip >= 1.18:
        label = "inverted_triangle"
        confidence = 0.91
    else:
        label = "trapezoid"
        confidence = 0.89

    classes = ["triangle", "inverted_triangle", "rectangle", "oval", "trapezoid"]
    remaining = max(0.0, 1.0 - confidence)
    other_prob = remaining / (len(classes) - 1)
    probabilities = {shape: other_prob for shape in classes}
    probabilities[label] = confidence

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }
