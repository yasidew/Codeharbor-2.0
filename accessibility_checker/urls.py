from django.urls import path
from . import views
# from .views import AccessibilityScoresView

urlpatterns = [
    path('', views.index, name='index'),
    path('check-accessibility/', views.check_accessibility, name='check_accessibility'),

]
