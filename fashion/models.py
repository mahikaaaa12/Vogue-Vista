from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User


class ColorAnalysis(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='color_analyses'
    )

    season = models.CharField(max_length=50)
    undertone = models.CharField(max_length=50)
    skin_tone = models.CharField(max_length=50)

    best_colors = models.JSONField(default=list)
    avoid_colors = models.JSONField(default=list)

    created_at = models.DateTimeField(auto_now_add=True)


class BodyAnalysis(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='body_analyses'
    )

    body_type = models.CharField(max_length=50)

    bust = models.FloatField(null=True, blank=True)
    waist = models.FloatField(null=True, blank=True)
    hips = models.FloatField(null=True, blank=True)

    recommended_fits = models.JSONField(default=list)
    recommended_necklines = models.JSONField(default=list)
    recommended_fabrics = models.JSONField(default=list)

    created_at = models.DateTimeField(auto_now_add=True)