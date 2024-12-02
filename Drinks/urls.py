from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

import model
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('drinks/', views.drink_list),
    path('drinks/<int:id>', views.drink_detail),
    path('calculate-complexity-line-by-line/', views.calculate_complexity_multiple_java_files,
         name='calculate_complexity_line_by_line'),
    path('calculate-complexity/', views.calculate_complexity_line_by_line,
         name='calculate_complexity'),
    path('calculate-complexity-line-by-line-csharp/', views.calculate_complexity_line_by_line_csharp,
         name='calculate_complexity_line_by_line_csharp'),
path('calculate-complexity-line-by-line-csharp-files/', views.calculate_complexity_multiple_csharp_files,
         name='calculate_complexity_line_by_line_csharp'),

    path('calculate-complexity-excel/', views.calculate_complexity, name='calculate_complexity-excel'),

    path('python_code_analysis/', views.python_code_analysis, name='python_code_analysis'),

    path('java_code_analysis/', views.java_code_analysis, name='java_code_analysis'),

    path('js_code_analyser/', views.js_code_analyser, name='js_code_analyser'),

    path('php_code_analyser/', views.php_code_analyser, name='php_code_analyser'),

    path("detect-defects/", views.detect_defects_view, name="detect_defects"),

    path('analyze-code/', views.analyze_code_view, name='analyze_code'),

    # path("api/detect-defects/", model.detect_defects, name="api_detect_defects"),

    # path('upload/python/', views.upload_python_files, name='upload_python_files'),

    # path('upload/java/', views.upload_java_files, name='upload_java_files'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



urlpatterns = format_suffix_patterns(urlpatterns)
