from django.urls import path
from .views import AnalyzeBodyView, HistoryView, AnalysisDetailView
from .photo_views import AnalysisPhotoView

urlpatterns = [
    # Public endpoint consumed by the static HTML frontend (photo.html).
    path("analysis/photo/",    AnalysisPhotoView.as_view(),  name="analysis-photo"),

    # Authenticated REST surface (mobile / JWT clients).
    path("analyze-body/",      AnalyzeBodyView.as_view(),    name="analyze-body"),
    path("history/",           HistoryView.as_view(),        name="history"),
    path("history/<int:pk>/",  AnalysisDetailView.as_view(), name="history-detail"),
]
