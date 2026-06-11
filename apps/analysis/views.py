from rest_framework import generics, permissions, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import BodyAnalysis
from .serializers import BodyAnalysisSerializer, AnalyzeUploadSerializer
from .services import analyze_for_user

class AnalyzeBodyView(APIView):
    """
    POST /api/analyze-body/
    multipart/form-data: image=<file>
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes     = [MultiPartParser, FormParser]

    def post(self, request):
        ser = AnalyzeUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        record = analyze_for_user(request.user, ser.validated_data["image"])
        data = BodyAnalysisSerializer(record, context={"request": request}).data
        code = status.HTTP_201_CREATED if record.status == "done" else status.HTTP_422_UNPROCESSABLE_ENTITY
        return Response(data, status=code)

class HistoryView(generics.ListAPIView):
    """GET /api/history/  — list authenticated user's analyses."""
    serializer_class   = BodyAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return BodyAnalysis.objects.filter(user=self.request.user)

class AnalysisDetailView(generics.RetrieveDestroyAPIView):
    """GET / DELETE /api/history/<id>/"""
    serializer_class   = BodyAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return BodyAnalysis.objects.filter(user=self.request.user)
