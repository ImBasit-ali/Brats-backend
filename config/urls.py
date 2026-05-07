"""
URL configuration for BraTS AI backend.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.views.static import serve
from django.urls import re_path
from config.views import health_check

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('segmentation.urls')),
    path("health/", health_check),

]

# Serve uploaded and generated media files for the viewer in all environments.
urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]
