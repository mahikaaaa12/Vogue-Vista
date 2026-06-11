"""
Analysis domain models.

We store:
- the uploaded image,
- the raw landmark vector (JSON) returned by pose estimation,
- the engineered feature vector fed to the ML model,
- the prediction with confidence distribution,
- timing + status for observability.
"""
from django.conf import settings
from django.db import models

def upload_path(instance, filename):
    return f"uploads/{instance.user_id}/{instance.id or 'tmp'}/{filename}"

class BodyAnalysis(models.Model):
    STATUS = (("pending", "pending"), ("done", "done"), ("failed", "failed"))

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="analyses")
    image = models.ImageField(upload_to=upload_path)

    landmarks = models.JSONField(null=True, blank=True,
        help_text="Raw pose landmarks from MediaPipe/Detectron2.")
    features = models.JSONField(null=True, blank=True,
        help_text="Engineered proportional features fed to the classifier.")
    measurements = models.JSONField(null=True, blank=True,
        help_text="Derived shoulder/waist/hip/symmetry metrics (pixel-normalized).")

    predicted_shape = models.CharField(max_length=32, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    probabilities = models.JSONField(null=True, blank=True,
        help_text="Per-class probability distribution.")

    status = models.CharField(max_length=16, choices=STATUS, default="pending")
    error = models.TextField(blank=True)
    processing_ms = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self):
        return f"Analysis<{self.id} user={self.user_id} shape={self.predicted_shape}>"
