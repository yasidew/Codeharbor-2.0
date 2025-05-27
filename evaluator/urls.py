from django.urls import path
from . import views
from .views import submit_html_view, fetch_html_from_url

urlpatterns = [
    path('evaluate/', views.evaluate_html_view, name='evaluate_html'),
    path("submit/", submit_html_view, name="submit_html"),
    path("fetch-html/", fetch_html_from_url, name='fetch_html_from_url'),
    path("load-from-url/", views.load_from_url_modal, name='load_from_url_modal'),
]