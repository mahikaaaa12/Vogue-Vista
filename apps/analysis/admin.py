from django.contrib import admin
from .models import BodyAnalysis

@admin.register(BodyAnalysis)
class BodyAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "predicted_shape", "confidence", "status", "created_at")
    list_filter  = ("status", "predicted_shape")
    search_fields = ("user__username", "user__email")
    readonly_fields = ("landmarks", "features", "measurements", "probabilities",
                       "processing_ms", "created_at")
