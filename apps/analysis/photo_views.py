"""
Public photo-analysis endpoint matching the VOGUE VISTA frontend contract.

POST /api/analysis/photo/
Body (JSON): { "gender": "female"|"male", "mimeType": "image/jpeg", "base64Data": "<b64>" }
Response:    { emoji, shape, confidence, description, proportions{shoulders,bust,waist,hips},
               traits[], styleTips[{tip}], wear[], avoid[] }

The endpoint is intentionally AllowAny so the static HTML frontend can call it
directly. The classification itself comes from the ML pipeline — only the
human-readable presentation text is templated from the predicted label.
"""
from __future__ import annotations
import base64
import binascii
import logging
from pyexpat import features
import tempfile
from pathlib import Path

from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from .ml.pipeline import run as run_pipeline, PipelineError

from apps.analysis.recommendations.recommendation_engine import (
    generate_recommendations
)

from apps.analysis.intelligence.body_traits import (
    extract_traits
)

from apps.analysis.intelligence.proportion_engine import (
    compute_scores
)

logger = logging.getLogger(__name__)

_ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

# Presentation copy keyed by the *label* the classifier returns. These are
# UI strings, not classification rules — the prediction itself is fully
# data-driven by the trained model.
_PRESENTATION = {
    "hourglass":   {"emoji": "⧖", "description": "Balanced shoulder and hip line with a defined waist — the classic symmetrical silhouette."},
    "pear":        {"emoji": "◐", "description": "Hips read wider than the shoulder line, with a softly defined waist."},
    "apple":       {"emoji": "◉", "description": "Fullness through the midsection with a softer waist definition and slimmer lower body."},
    "rectangle":   {"emoji": "▭", "description": "Shoulders, waist and hips run on a similar vertical line — a long, athletic frame."},
    "inverted_triangle": {"emoji": "▽", "description": "Strong shoulder line tapering down through a narrower hip."},
    "triangle":    {"emoji": "△", "description": "Lower body anchors the silhouette, with a lighter upper frame."},
}

# Generic fallback so an unfamiliar label still produces a complete UI payload.
_DEFAULT_PRESENTATION = {"emoji": "◈", "description": "A distinctive silhouette read directly from your proportions."}
_DEFAULT_STYLE = {
    "wear":  ["Tailored mid-rise trousers", "Wrap and surplice tops", "Soft-shouldered blazers"],
    "avoid": ["Stiff boxy outerwear", "Garments that fight the natural waist"],
    "tips":  ["Dress to the proportions the camera actually measured, not the trend cycle.",
              "One defined line — shoulder, waist or hem — sets the whole outfit.",
              "Tonal layering reads more refined than high-contrast blocking."],
}

def _proportion_bars(measurements: dict, features: dict) -> dict:
    """Translate normalized metrics into 0-100 bar percentages for the UI."""
    def pct(v, lo, hi):
        if v is None:
            return 50
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 50
        return max(8, min(95, int(round((f - lo) / (hi - lo) * 100))))

    shoulder_w = (measurements or {}).get("shoulder_width") or (features or {}).get("shoulder_width")
    bust_w     = (measurements or {}).get("bust_width")     or (features or {}).get("bust_width")
    waist_w    = (measurements or {}).get("waist_width")    or (features or {}).get("waist_width")
    hip_w      = (measurements or {}).get("hip_width")      or (features or {}).get("hip_width")

    def label_for(v, lo, hi):
        if v is None:
            return "—"
        p = pct(v, lo, hi)
        if p < 35:  return "Narrow"
        if p < 65:  return "Balanced"
        return "Defined"

    return {
        "shoulders": {"label": label_for(shoulder_w, 0.10, 0.35), "pct": pct(shoulder_w, 0.10, 0.35)},
        "bust":      {"label": label_for(bust_w,     0.10, 0.35), "pct": pct(bust_w,     0.10, 0.35)},
        "waist":     {"label": label_for(waist_w,    0.08, 0.30), "pct": pct(waist_w,    0.08, 0.30)},
        "hips":      {"label": label_for(hip_w,      0.10, 0.36), "pct": pct(hip_w,      0.10, 0.36)},
    }

class AnalysisPhotoView(APIView):
    """Public JSON endpoint consumed by the static HTML frontend."""
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def get(self, request):
        return Response({
            "message": "Photo Analysis API Running"
        })

    def post(self, request):
        payload = request.data or {}
        gender = (payload.get("gender") or "female").lower()
        mime   = (payload.get("mimeType") or "").lower()
        b64    = payload.get("base64Data") or ""

        if mime not in _ALLOWED_MIME:
            return Response({"detail": f"Unsupported mimeType '{mime}'."},
                            status=status.HTTP_400_BAD_REQUEST)
        if not b64:
            return Response({"detail": "base64Data is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return Response({"detail": "base64Data is not valid base64."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Persist to a temp file so the pipeline can read it like any upload.
        ext = _EXT.get(mime, ".jpg")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        try:
            gender = request.data.get("gender", "female")

            result = run_pipeline(
                tmp.name,
                gender=gender
            )
        except PipelineError as e:
            return Response({"detail": str(e)},
                            status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as e:                              # noqa: BLE001
            logger.exception("Photo analysis failed")
            return Response({"detail": f"Unexpected error: {e}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            try:    tmp_path.unlink(missing_ok=True)
            except Exception:   pass

        label = (result.get("predicted_shape") or "").lower()
        features = result.get("features", {})

        traits = extract_traits(features)

        scores = compute_scores(features)

        recommendations = generate_recommendations(
            gender,
            label,
            features
        )
        pres  = _PRESENTATION.get(label, _DEFAULT_PRESENTATION)
        
        confidence_pct = int(round((result.get("confidence") or 0.0) * 100))

        return Response({
            "gender":      gender,
            "shape":       label.replace("_", " ").title() or "Undetermined",
            "emoji":       pres["emoji"],
            "confidence":  confidence_pct,
            "description": pres["description"],
            "proportions": _proportion_bars(result.get("measurements"), result.get("features")),
            "traits":     traits,
            "scores":      scores,
            "recommendations": recommendations,
            "processing_ms": result.get("processing_ms"),
        }, status=status.HTTP_200_OK)