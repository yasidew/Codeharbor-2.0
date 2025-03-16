from django.urls import path
from . import views

from django.contrib.auth import views as auth_views

from .views import submit_score

urlpatterns = [
    path('', views.index, name='index'),
    path('check-accessibility/', views.check_accessibility, name='check_accessibility'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('submit_score/', submit_score, name='submit_score'),
]
