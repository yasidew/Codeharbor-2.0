# from django.shortcuts import render
#
# # Create your views here.
# from django.http import JsonResponse
# from rest_framework.decorators import api_view
# from .models import GitHubChallenge
#
# @api_view(['GET'])
# def get_challenges_by_difficulty(request, difficulty):
#     """ Fetch challenges based on difficulty level without using serializers """
#     if difficulty not in ["easy", "medium", "hard"]:
#         return JsonResponse({"error": "Invalid difficulty level"}, status=400)
#
#     challenges = GitHubChallenge.objects.filter(difficulty=difficulty)
#
#     if not challenges.exists():
#         return JsonResponse({"message": f"No {difficulty} challenges available"}, status=404)
#
#     # Manually convert QuerySet to JSON
#     challenges_data = list(challenges.values("id", "title", "repo_url", "file_url", "html_code", "difficulty", "created_at"))
#
#     return JsonResponse({"challenges": challenges_data}, safe=False)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from games.github_scraper import fetch_bad_code_from_github  # Import function
from games.models import GitHubChallenge, GitHubScraperGame

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