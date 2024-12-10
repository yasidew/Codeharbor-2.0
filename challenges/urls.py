from django.urls import path

from . import views
from .views import get_all_challenges, get_challenges_for_current_month, get_challenges_by_month, create_challenge

urlpatterns = [
    path('', get_all_challenges, name='get_all_challenges'),
    path('current/', get_challenges_for_current_month, name='get_challenges_for_current_month'),
    path('<int:month>/', get_challenges_by_month, name='get_challenges_by_month'),  # URL for challenges by month
    path('create/', create_challenge, name='create_challenge'),
]
