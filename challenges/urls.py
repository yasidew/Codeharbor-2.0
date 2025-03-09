from django.shortcuts import render
from django.urls import path

from . import views
from .views import get_all_challenges, get_challenges_for_current_month, get_challenges_by_month, create_challenge, \
    challenges_page, challenge_details_page

urlpatterns = [
    path('current/', get_challenges_for_current_month, name='get_challenges_for_current_month'),
    #path('current/', get_challenges_for_current_month, name='get_challenges_for_current_month'),
    path('<int:month>/', get_challenges_by_month, name='get_challenges_by_month'),  # URL for challenges by month
    path('create/', create_challenge, name='create_challenge'),
    path('cid/<int:challenge_id>/', views.get_challenge_by_id, name='get_challenge_by_id'),



    path('all/', challenges_page, name='challenges_page'),
    path('challenge-details/', challenge_details_page, name='challenge_details_page'),
]
