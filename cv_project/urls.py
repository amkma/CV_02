"""CV_02 URL configuration."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detector import views

urlpatterns = [
    path("",            views.home,       name="home"),
    path("api/upload/", views.api_upload,  name="api_upload"),
    path("api/hough/",  views.api_hough,   name="api_hough"),
    path("api/snake/",  views.api_snake,   name="api_snake"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
