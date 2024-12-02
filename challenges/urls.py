from django.urls import path
from .views import get_all_challenges, get_challenges_for_current_month, get_challenges_by_month

urlpatterns = [
    path('', get_all_challenges, name='get_all_challenges'),
    path('current/', get_challenges_for_current_month, name='get_challenges_for_current_month'),
    path('<int:month>/', get_challenges_by_month, name='get_challenges_by_month'),  # URL for challenges by month

]
