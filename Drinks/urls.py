from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render
from rest_framework.urlpatterns import format_suffix_patterns
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from games.views import editor_view
from user.views import UserAPI, logout, logout_all, user_profile_view
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
                  path('calculate-complexity-line-by-line-csharp-files/',
                       views.calculate_complexity_multiple_csharp_files,
                       name='calculate_complexity_line_by_line_csharp'),

                  path('calculate-complexity-excel/', views.calculate_complexity, name='calculate_complexity-excel'),
path('guidelines/', views.guidelines_view, name='guidelines'),

    # Code Analysis URLs
    path('python_code_analysis/', views.python_code_analysis, name='python_code_analysis'),
    path('java_code_analysis/', views.java_code_analysis, name='java_code_analysis'),

                  path("detect-defects/", views.detect_defects_view, name="detect_defects"),

    # Authentication and User Management
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/user/', UserAPI.as_view()),
    path('api/logout/', logout),
    path('api/logout-all/', logout_all),
    path("profile/<str:username>/", user_profile_view, name="user-profile"),

    # Web Pages
    path('login/', lambda request: render(request, 'login.html'), name='login'),
    path('dashboard/', lambda request: render(request, 'dashboard.html'), name='dashboard'),
    path('profile/', lambda request: render(request, 'update_profile.html'), name='profile'),

    # Other Apps
    path('checker/', include('accessibility_checker.urls')),
    path('games/', include('games.urls')),
    path('challenges/', include('challenges.urls')),
    path('editor/', editor_view, name='editor'),

    # Code Refactoring
    path('code-formatter/', include('code_formatter.urls')),

    # Root URL
    path('', views.home, name='home'),
]
# Add static file handling
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# Apply format suffix patterns
urlpatterns = format_suffix_patterns(urlpatterns)
