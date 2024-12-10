from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import render
from django.urls import path, include
from rest_framework.urlpatterns import format_suffix_patterns

from . import views

# wasana


urlpatterns = [
    path('admin/', admin.site.urls),
    path('drinks/', views.drink_list),
    path('challenges/', include('challenges.urls')),
    path('', include("user.urls")),
    path('checker/', include('accessibility_checker.urls')),


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

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = format_suffix_patterns(urlpatterns)
