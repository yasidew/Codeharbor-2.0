from django.contrib.auth.models import User
from django.db.models import Avg, Sum, F
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from games.github_scraper import fetch_bad_code_from_github  # Import function
from games.models import GitHubChallenge, GitHubScraperGame, GitGameScore, UserBadge, Badge

from django.shortcuts import render

# def admin_github_scraper_view(request):
#     """Render the admin UI for GitHub scraper."""
#     return render(request, "admin_scraper.html")

from django.shortcuts import render

from .badge_assignment import award_badges
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


def leaderboard_view(request):
    leaderboard = (
        GitGameScore.objects.values("user__id", "user__username")
        .annotate(avg_score=Avg("score"))
        .order_by("-avg_score")  # Sort from highest to lowest
    )
    return render(request, "leaderboard.html", {"leaderboard": leaderboard})


@api_view(["POST"])
def assign_badges_view(request, user_id):
    """
    Assign badges to a user based on their critical score.
    Requires a POST request.
    """
    user = get_object_or_404(User, id=user_id)

    # Assign badges
    award_badges(user)

    # Fetch updated badges
    badges = UserBadge.objects.filter(user=user).select_related('badge')
    badge_list = [{"name": badge.badge.name, "description": badge.badge.description} for badge in badges]

    return Response(
        {"message": "Badges assigned successfully!", "user": user.username, "assigned_badges": badge_list}
    )

@api_view(["GET"])
def user_badges_view(request, user_id):
    """
    Returns a JSON response with all badges earned by a user.
    Supports GET requests.
    """
    user = get_object_or_404(User, id=user_id)
    badges = UserBadge.objects.filter(user=user).select_related('badge')

    badge_list = [{"name": badge.badge.name, "description": badge.badge.description} for badge in badges]

    return Response({"user": user.username, "badges": badge_list})





########################################### new #########################################
#Calculate the total critical score

@api_view(["GET"])
def total_critical_score_view(request, user_id):
    """
    Calculates and returns the total critical score for a user.
    Supports GET requests.
    """
    user = get_object_or_404(User, id=user_id)

    # Calculate total critical score
    total_critical_score = (
        GitGameScore.objects.filter(user=user)
        .aggregate(total_score=Sum(F('critical_score')))['total_score']
    )

    # Ensure no null values
    total_critical_score = total_critical_score or 0

    return Response({"user": user.username, "total_critical_score": total_critical_score})


# View to store badges

@api_view(["GET"])
def user_assigned_badges(request, user_id):
    """
    Retrieves all badges assigned to a user.
    Supports GET requests.
    """
    user = get_object_or_404(User, id=user_id)

    # Get all badges assigned to the user
    badges = UserBadge.objects.filter(user=user).select_related('badge')

    # Format badge data
    badge_list = [
        {"badge_name": badge.badge.name, "awarded_on": badge.awarded_on.strftime("%Y-%m-%d %H:%M:%S")}
        for badge in badges
    ]

    return Response({"user": user.username, "badges": badge_list})



# @api_view(["POST"])
# def store_user_badge(request):
#     """
#     Stores (assigns) a specific badge to a user.
#     Requires a POST request with 'user_id' and 'badge_name'.
#     """
#     # Extract data from request
#     user_id = request.data.get("user_id")
#     badge_name = request.data.get("badge_name")
#
#     # Validate required fields
#     if not user_id or not badge_name:
#         return Response({"error": "Missing required fields: user_id, badge_name"}, status=400)
#
#     # Get user and badge
#     user = get_object_or_404(User, id=user_id)
#     badge = get_object_or_404(Badge, name=badge_name)
#
#     # Check if user already has the badge
#     user_badge, created = UserBadge.objects.get_or_create(user=user, badge=badge)
#
#     if created:
#         message = f"Badge '{badge_name}' assigned to {user.username} successfully."
#     else:
#         message = f"User '{user.username}' already has the badge '{badge_name}'."
#
#     return Response({"message": message, "user": user.username, "badge": badge_name})



@api_view(["POST"])
def store_user_badge(request):
    """
    Stores (assigns) a single critical-score-based badge to a user.
    A user can only have ONE badge from this category at a time.
    """
    # Extract data from request
    user_id = request.data.get("user_id")

    if not user_id:
        return Response({"error": "Missing required field: user_id"}, status=400)

    # Get user
    user = get_object_or_404(User, id=user_id)

    # ✅ Fetch the total critical score
    total_critical_score = (
        GitGameScore.objects.filter(user=user)
        .aggregate(Sum('critical_score'))['critical_score__sum']
    ) or 0  # Default to 0 if None

    print(f"🔹 Total Critical Score for {user.username}: {total_critical_score}")  # Debugging

    # ✅ Define badge criteria based on critical score
    badge_criteria = [
        ("Platinum", 0),
        ("Gold", 10),
        ("Silver", 20),
        ("Bronze", 40),
        ("Participant", 100)
    ]

    # ✅ Determine which badge the user should receive
    assigned_badge = None
    for badge_name, max_critical_score in badge_criteria:
        if total_critical_score <= max_critical_score:
            assigned_badge, _ = Badge.objects.get_or_create(
                name=badge_name,
                defaults={"description": f"Awarded for having ≤ {max_critical_score} total critical issues."}
            )
            break  # Stop after assigning the first matching badge

    if not assigned_badge:
        return Response({"error": "No appropriate badge found for the given score."}, status=400)

    # ✅ Remove any existing critical-score-related badge
    critical_badges = ["Platinum", "Gold", "Silver", "Bronze", "Participant"]
    UserBadge.objects.filter(user=user, badge__name__in=critical_badges).delete()

    # ✅ Assign the new badge
    UserBadge.objects.create(user=user, badge=assigned_badge)

    return Response({
        "message": f"Badge '{assigned_badge.name}' assigned to {user.username}.",
        "user": user.username,
        "total_critical_score": total_critical_score,
        "assigned_badge": assigned_badge.name
    })