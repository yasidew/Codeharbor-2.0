from django.urls import path
from . import views
from .views import define_guidelines, edit_guideline, delete_guideline, get_guidelines, generate_guideline

urlpatterns = [
    path('refactor/', views.refactor_view, name='refactor_view'),
    path('upload-code/', views.upload_code, name='upload_code'),
    path('refactor-code/', views.refactor_code, name='refactor_code'),
    path('api/guidelines/<str:company_name>/', get_guidelines, name='get_guidelines'),
    path('define-guidelines/', define_guidelines, name='define_guidelines'),
    path("edit-guideline/<int:id>/", edit_guideline, name="edit_guideline"),
    path("delete-guideline/<int:guideline_id>/", delete_guideline, name="delete_guideline"),
    path("generate-guideline/", generate_guideline, name="generate_guideline"),
]
