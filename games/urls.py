from django.urls import path
from .views import fetch_github_code_view, get_github_challenges, admin_github_scraper_view, user_severity_chart, \
    get_user_id, leaderboard_view, assign_badges_view, user_badges_view, total_critical_score_view, \
    user_assigned_badges, store_user_badge

urlpatterns = [
    path("fetch-github-code/", fetch_github_code_view, name="fetch_github_code"),
    path("github-challenges/", get_github_challenges, name="github_challenges"),

    path("github-scraper/", admin_github_scraper_view, name="admin_github_scraper"),
    path("severity-chart/<int:user_id>/<int:challenge_id>/", user_severity_chart, name="personalized_chart"),
    path("leaderboard/", leaderboard_view, name="leaderboard"),


#################### old ###########################
    path('assign-badges/<int:user_id>/', assign_badges_view, name="assign_badges"),
    # Get all earned badges (Supports GET request)
    path('user-badges/<int:user_id>/', user_badges_view, name="user_badges"),
#################### old ###########################


    #To get the total critical score
    path('total-critical-score/<int:user_id>/', total_critical_score_view, name="total_critical_score"),
    path('user-badges/<int:user_id>/', user_assigned_badges, name="user_assigned_badges"),
    path('store-badge/', store_user_badge, name="store_user_badge"),

]
