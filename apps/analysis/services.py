"""
Service layer — keeps views thin and pipeline reusable from Celery tasks.
"""
from __future__ import annotations
import logging
from django.db import transaction
from .models import BodyAnalysis
from .ml.pipeline import run as run_pipeline, PipelineError

logger = logging.getLogger(__name__)

def analyze_for_user(user, image_file) -> BodyAnalysis:
    """Synchronous analysis: persist record, run pipeline, store results."""
    record = BodyAnalysis.objects.create(user=user, image=image_file, status="pending")
    try:
        result = run_pipeline(record.image.path)
        with transaction.atomic():
            record.landmarks       = result["landmarks"]
            record.measurements    = result["measurements"]
            record.features        = result["features"]
            record.predicted_shape = result["predicted_shape"]
            record.confidence      = result["confidence"]
            record.probabilities   = result["probabilities"]
            record.processing_ms   = result["processing_ms"]
            record.status          = "done"
            record.save()
    except PipelineError as e:
        record.status = "failed"
        record.error  = str(e)
        record.save(update_fields=["status", "error"])
        logger.warning("Analysis %s failed: %s", record.id, e)
    return record
