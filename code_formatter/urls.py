from django.urls import path
from . import views
from .views import define_guidelines, edit_guideline, delete_guideline, get_guidelines, generate_guideline, \
    github_import_modal, get_github_token
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('refactor/', views.refactor_view, name='refactor_view'),
    path('upload-code/', views.upload_code, name='upload_code'),
    path('refactor-code/', views.refactor_code, name='refactor_code'),
    path('api/guidelines/<str:company_name>/', get_guidelines, name='get_guidelines'),
    path('define-guidelines/', define_guidelines, name='define_guidelines'),
    path("edit-guideline/<int:id>/", edit_guideline, name="edit_guideline"),
    path("delete-guideline/<int:guideline_id>/", delete_guideline, name="delete_guideline"),
    path("generate-guideline/", generate_guideline, name="generate_guideline"),
    path('github-import-modal/', github_import_modal, name='github_import_modal'),
    path('get-github-token/', get_github_token, name='get_github_token'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
