from django.shortcuts import render
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import UserAPI, logout, logout_all




urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/user', UserAPI.as_view()),
    path('api/logout', logout),
    path('api/logout-all', logout_all),
    path('login/', lambda request: render(request, 'login.html'), name='login'),
    path('dashboard/', lambda request: render(request, 'dashboard.html'), name='login'),
]