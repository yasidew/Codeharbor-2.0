from django.urls import path
from .views import (
    ai_code_analysis_view,
    upload_code,
    analyze_code_with_ai,
    get_guidelines
)
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('ai-analysis/', ai_code_analysis_view, name='ai_code_analysis.html'),
    path('upload-code/', upload_code, name='upload_code'),
    path('analyze-code-ai/', analyze_code_with_ai, name='analyze_code_with_ai'),
    # path('get-guidelines/', get_guidelines, name='get_guidelines'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
