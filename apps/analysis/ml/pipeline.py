"""
End-to-end inference pipeline.

    bytes/file → preprocess → landmarks → features → classifier → result

Each stage is independently testable and replaceable (e.g. swap
MediaPipe for Detectron2 by editing `landmarks.py` only).
"""
from __future__ import annotations
import time
import logging

from . import classifier, features as feat_mod, landmarks as lm_mod
from . import preprocessing
from .measurements import MeasurementExtractor
from .features import build_features

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    pass

def run(file_obj, gender="female") -> dict:
    t0 = time.perf_counter()
    try:
        img = preprocessing.preprocess(file_obj)
        extracted = lm_mod.extract(img)
        if len(extracted) == 3:
            named, raw, segmentation_mask = extracted
        else:
            named, raw = extracted
            segmentation_mask = None
        derived = feat_mod.derive(named, segmentation_mask=segmentation_mask)
        prediction = classifier.predict(derived["features"],gender)
    except lm_mod.PoseExtractionError as e:
        raise PipelineError(str(e))
    except classifier.ModelNotTrainedError as e:
        raise PipelineError(str(e))
    except Exception as e:
        logger.exception("Unexpected pipeline failure")
        raise PipelineError(f"Analysis failed: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "landmarks": raw,
        "measurements": derived["measurements"],
        "features": derived["features"],
        "predicted_shape": prediction["label"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "processing_ms": elapsed_ms,
    }