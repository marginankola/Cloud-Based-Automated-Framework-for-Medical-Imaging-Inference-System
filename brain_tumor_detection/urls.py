from django.contrib import admin
from django.urls import path, include, re_path # Added re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve # Added serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
]

# This is the crucial part for AWS/Production when DEBUG is False
if not settings.DEBUG:
    urlpatterns += [
        re_path(r'^media/(?P<path>.*)$', serve, {
            'document_root': settings.MEDIA_ROOT,
        }),
    ]
else:
    # This covers your local development environment
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
