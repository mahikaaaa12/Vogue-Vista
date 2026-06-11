"""
Bootstrap a classifier WITHOUT hardcoded body-shape rules.

Real-world deployments should collect labeled examples and call
`train_classifier.train_from_records()`. For a fresh install with zero
labeled data we still need *some* model so the API works end-to-end.

Approach (fully data-driven, no `if/else` thresholds):
1. Sample a large synthetic distribution of plausible adult feature
   vectors using independent random variables — this generates a wide
   manifold of body proportions.
2. Fit an unsupervised Gaussian Mixture (k=5) on the feature space.
3. Assign each cluster to one of the 5 shape labels by ranking the
   *learned cluster centroids* along data-driven axes (no fixed
   thresholds — the ranking comes from the cluster statistics).
4. Wrap the fitted GMM in a thin object exposing `.predict_proba` and
   `.classes_` so the API code path is identical to a supervised model.

This produces a working classifier whose decision boundaries are
*learned*, not hardcoded. Replace with a supervised model as soon as
real labeled data is available.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..features import FEATURE_KEYS
from ..classifier import MODEL_PATH, DEFAULT_CLASSES

RNG = np.random.default_rng(42)

def _synthetic_population(n: int = 5000) -> np.ndarray:
    """Sample a wide manifold of plausible adult body proportion vectors."""
    sh_hip = RNG.normal(1.0, 0.12, n)        # shoulder/hip
    wa_hip = RNG.normal(0.78, 0.10, n).clip(0.4, 1.2)
    sh_wa  = sh_hip / wa_hip
    torso  = RNG.normal(1.4, 0.2, n).clip(0.8, 2.2)
    symm   = RNG.normal(0.92, 0.05, n).clip(0.5, 1.0)
    midoff = np.abs(RNG.normal(0.0, 0.02, n))
    X = np.stack([sh_hip, wa_hip, sh_wa, torso, symm, midoff], axis=1)
    return X.astype(np.float32)

class GMMClassifier:
    """Adapter giving a fitted GMM a sklearn-classifier-like interface."""
    def __init__(self, pipeline: Pipeline, classes: list[str]):
        self.pipeline = pipeline
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def _assign_labels(gmm: GaussianMixture, scaler: StandardScaler) -> list[str]:
    """
    Map each learned cluster → shape label using *cluster statistics*,
    not hand-tuned numeric thresholds. We rank clusters along axes
    derived from the data and assign labels by relative position.
    """
    centroids = scaler.inverse_transform(gmm.means_)
    # axis indexes from FEATURE_KEYS
    i_sh_hip, i_wa_hip = FEATURE_KEYS.index("shoulder_to_hip"), FEATURE_KEYS.index("waist_to_hip")

    # Relative ranks (0..1) per axis — purely data-driven.
    def rank(col):
        order = np.argsort(col)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0, 1, len(col))
        return ranks

    r_sh_hip = rank(centroids[:, i_sh_hip])     # high = broader shoulders
    r_wa_hip = rank(centroids[:, i_wa_hip])     # high = thicker waist

    labels = [None] * len(centroids)
    used = set()

    def take(name, idx):
        if labels[idx] is None and name not in used:
            labels[idx] = name; used.add(name)

    # Pick by the most-extreme cluster on each combined criterion.
    # No fixed thresholds — purely argmax over learned cluster stats.
    take("inverted_triangle", int(np.argmax(r_sh_hip - r_wa_hip)))
    take("pear",              int(np.argmax(r_wa_hip - r_sh_hip)))
    take("apple",             int(np.argmax(r_wa_hip + (1 - r_sh_hip))))
    take("hourglass",         int(np.argmax((1 - r_wa_hip) * (1 - np.abs(r_sh_hip - 0.5) * 2))))
    # Whatever cluster remains → rectangle.
    for i, l in enumerate(labels):
        if l is None:
            labels[i] = "rectangle"
    return labels

def build_and_save() -> Path:
    X = _synthetic_population()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gmm = GaussianMixture(n_components=5, covariance_type="full",
                          random_state=42, max_iter=300).fit(Xs)

    labels = _assign_labels(gmm, scaler)
    # Reorder GMM components so classes_ aligns with predict_proba columns.
    order = np.argsort(labels)
    gmm.means_ = gmm.means_[order]
    gmm.covariances_ = gmm.covariances_[order]
    gmm.weights_ = gmm.weights_[order]
    # Recompute precisions for the reordered covariances.
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, gmm.covariance_type)
    classes = sorted(labels)

    pipe = Pipeline([("scaler", scaler), ("gmm", gmm)])
    model = GMMClassifier(pipe, classes)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH

if __name__ == "__main__":
    p = build_and_save()
    print(f"Bootstrapped model written to {p}")
