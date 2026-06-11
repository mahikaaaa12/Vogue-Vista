from rest_framework import serializers
from .models import BodyAnalysis

class BodyAnalysisSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = BodyAnalysis
        fields = ("id", "image", "image_url", "measurements", "features",
                  "predicted_shape", "confidence", "probabilities",
                  "status", "error", "processing_ms", "created_at")
        read_only_fields = ("measurements", "features", "predicted_shape",
                            "confidence", "probabilities", "status", "error",
                            "processing_ms", "created_at", "image_url")

    def get_image_url(self, obj):
        request = self.context.get("request")
        if obj.image and request:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url if obj.image else None

class AnalyzeUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

    def validate_image(self, f):
        # Hard cap to prevent abuse / memory issues.
        if f.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image must be ≤ 10MB.")
        return f
