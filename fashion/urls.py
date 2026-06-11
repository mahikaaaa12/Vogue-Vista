from django.urls import path
from apps.analysis.photo_views import AnalysisPhotoView
from fashion.color_backend.api.views import analyze_color_api
from . import views

urlpatterns = [
    path('', views.login_page, name='login'),
    path('signup/', views.signup_page, name='signup'),
    path('home/', views.home, name='home'),
    path('color_analysis.html', views.color, name='color'),
    path('Body-Type-Analysis.html', views.body, name='body'),
    path('photo.html', views.photo, name='photo'),
    path('measurements/', views.measurements, name='measurements'),
    path('main.html', views.main, name='main'),
    path('analyse.html', views.analyse, name='analyse'),
    path('analyse.htm', views.analyse),
    path('api/v1/analyze-color/',analyze_color_api,name='analyze_color_api'),
    path("manual/", views.manual, name="manual"),
    path("pic/", views.pic, name="pic"),
    path("profile/", views.profile, name="profile"),
    path("reset/", views.reset, name="reset"),
    path("api/analysis/photo/",AnalysisPhotoView.as_view(),name="analysis-photo"),
]