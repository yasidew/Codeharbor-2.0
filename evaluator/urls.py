from django.urls import path
from . import views
from .views import submit_html_view

urlpatterns = [
    path('evaluate/', views.evaluate_html_view, name='evaluate_html'),
    path("submit/", submit_html_view, name="submit_html"),
]