from django.urls import path
from . import views

urlpatterns = [
    path('refactor/', views.refactor_view, name='refactor_view'),
    path('upload-code/', views.upload_code, name='upload_code'),
    path('refactor-code/', views.refactor_code, name='refactor_code'),
]
