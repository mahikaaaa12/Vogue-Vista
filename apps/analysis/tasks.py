"""Celery task wrapper — optional async path."""
try:
    from celery import shared_task
except Exception:  # pragma: no cover
    shared_task = None

if shared_task:
    @shared_task
    def analyze_async(analysis_id: int):
        from .models import BodyAnalysis
        from .ml.pipeline import run as run_pipeline, PipelineError
        record = BodyAnalysis.objects.get(pk=analysis_id)
        try:
            r = run_pipeline(record.image.path)
            for k in ("landmarks", "measurements", "features",
                      "predicted_shape", "confidence", "probabilities", "processing_ms"):
                setattr(record, k, r[k])
            record.status = "done"
        except PipelineError as e:
            record.status, record.error = "failed", str(e)
        record.save()
