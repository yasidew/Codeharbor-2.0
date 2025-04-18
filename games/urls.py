from django.urls import path
from .views import fetch_github_code_view, get_github_challenges, admin_github_scraper_view, user_severity_chart, \
    get_user_id, leaderboard_view

urlpatterns = [
    path("fetch-github-code/", fetch_github_code_view, name="fetch_github_code"),
    path("github-challenges/", get_github_challenges, name="github_challenges"),

    path("github-scraper/", admin_github_scraper_view, name="admin_github_scraper"),
    path("severity-chart/<int:user_id>/<int:challenge_id>/", user_severity_chart, name="personalized_chart"),
    path("leaderboard/", leaderboard_view, name="leaderboard"),

]
