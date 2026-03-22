"""CV_02 URL configuration — single file, all routes."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detector import views

urlpatterns = [
    path("",              views.upload,        name="upload"),
    path("hough/",        views.hough_detect,  name="hough_detect"),
    path("snake/",        views.snake_detect,  name="snake_detect"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
