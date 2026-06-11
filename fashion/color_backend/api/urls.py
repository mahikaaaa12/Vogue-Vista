from django import views
from django.urls import path
from .views import analyze_color_api

urlpatterns = [
    # Connects http://127.0.0.1:8000/api/v1/analyze-color/ to our OpenCV view
    path('api/v1/analyze-color/', analyze_color_api, name='analyze_color_api'),
    path('pic/', views.pic, name='pic'),
    path('manual/', views.manual, name='manual'),
    path('color-analysis/', views.color_analysis,name='color_analysis'),
]