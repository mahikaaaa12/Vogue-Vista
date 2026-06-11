"""
Supervised training — call once you have labeled records.

Expects an iterable of (feature_dict, label_str). Persists a calibrated
RandomForest pipeline using the same `.predict_proba` interface, so the
API code path doesn't change.
"""
from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from ..features import FEATURE_KEYS
from ..classifier import MODEL_PATH

def train_from_records(records):
    X, y = [], []
    for feats, label in records:
        X.append([feats[k] for k in FEATURE_KEYS])
        y.append(label)
    X = np.array(X, dtype=np.float32); y = np.array(y)
    if len(set(y)) < 2:
        raise ValueError("Need samples from at least 2 classes.")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42,
                                      class_weight="balanced")),
    ])
    scores = cross_val_score(pipe, X, y, cv=min(5, len(y)//len(set(y))))
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return {"cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "classes": list(pipe.classes_), "n_samples": int(len(y))}
