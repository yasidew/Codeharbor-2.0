
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from games.github_scraper import fetch_bad_code_from_github  # Import function
from games.models import GitHubChallenge, GitHubScraperGame, GitGameScore

from django.shortcuts import render

# def admin_github_scraper_view(request):
#     """Render the admin UI for GitHub scraper."""
#     return render(request, "admin_scraper.html")

from django.shortcuts import render
from .models import GitHubChallenge

def admin_github_scraper_view(request):
    """Render the admin UI for GitHub scraper with game details."""
    challenges = GitHubChallenge.objects.all()  # Fetch all GitHub challenges
    game = GitHubScraperGame.objects.first()  # Fetch the first game

    return render(request, "admin_scraper.html", {
        "challenges": challenges,
        "game": game,  # Pass game details to template
    })

def editor_view(request):
    """Render the Code Editor page."""
    return render(request, 'editor.html')

@csrf_exempt
def fetch_github_code_view(request):
    """Triggers GitHub scraper and returns a JSON response."""
    if request.method == "POST":
        fetch_bad_code_from_github()  # Run the scraper function
        return JsonResponse({"message": "✅ GitHub data fetched successfully!"}, status=200)
    else:
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=400)


def get_github_challenges(request):
    """Returns a JSON response with all GitHub accessibility challenges"""
    challenges = GitHubChallenge.objects.all().values("title", "repo_url", "file_url", "difficulty", "created_at")
    return JsonResponse(list(challenges), safe=False)

# @api_view(['GET'])
# @permission_classes([IsAuthenticated])
def user_severity_chart(request, user_id, challenge_id):
    """Fetch user severity scores for a specific challenge and render the pie chart."""
    try:
        user_scores = GitGameScore.objects.get(user_id=user_id, github_challenge_id=challenge_id)
    except GitGameScore.DoesNotExist:
        user_scores = None

    context = {
        "critical_score": user_scores.critical_score if user_scores else 0,
        "serious_score": user_scores.serious_score if hasattr(user_scores, "serious_score") else 0,
        "moderate_score": user_scores.moderate_score if user_scores else 0,
        "minor_score": user_scores.minor_score if user_scores else 0,
    }

    return render(request, "personalized_chart.html", context)

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

@login_required
def get_user_id(request):
    return JsonResponse({"success": True, "user_id": request.user.id})





