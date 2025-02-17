from django.urls import path
from . import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('check-accessibility/', views.check_accessibility, name='check_accessibility'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
]
