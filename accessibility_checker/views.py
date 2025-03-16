from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
import subprocess
import json

from rest_framework.decorators import api_view

from games.models import GitHubScraperGame, GitGameScore, GitHubChallenge


# Serve the index page
def index(request):
    return render(request, 'accessibility_checker.html')


@csrf_exempt  # Allows bypassing CSRF for API calls
@api_view(["POST"])
def check_accessibility(request):
    if request.method != 'POST' or 'html_file' not in request.FILES:
        return JsonResponse({'error': 'Invalid request method or no file provided'}, status=400)

    uploaded_file = request.FILES['html_file']

    try:
        # Decode the uploaded HTML file
        html_content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        return JsonResponse({'error': 'Failed to decode HTML file. Ensure it is a valid UTF-8 file.'}, status=400)

    try:
        # Run the accessibility checker using a Node.js script
        output = subprocess.run(
            ['node', 'axe-check.js'],
            input=html_content,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10  # ✅ Prevents hanging processes
        )

        # Attempt to parse JSON output
        try:
            results = json.loads(output.stdout)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Failed to analyze accessibility - invalid JSON output"}, status=500)

        # Prepare response with accessibility results
        result_data = {
            "passes": results.get("passes", []),
            "violations": results.get("violations", []),
            "incomplete": results.get("incomplete", []),
            "score": results.get("score", 0),
            "html_code": html_content  # ✅ Send HTML content for editor
        }

        return JsonResponse(result_data, status=200)

    except subprocess.TimeoutExpired:
        return JsonResponse({'error': 'Accessibility check timed out. Please try again.'}, status=500)


@csrf_exempt
@api_view(["POST"])
def submit_score(request):
    """Handles score submission from frontend, including severity-level scores."""
    try:
        data = json.loads(request.body)

        # Extract the user
        # ✅ Extract the authenticated user
        user = request.user
        if not user.is_authenticated:
            return JsonResponse({"error": "User is not authenticated"}, status=401)

        print("🔹 Received Data:", data)  # ✅ Debugging: Print received data
        print(f"🔹 Submitting Score for User: {user.id} ({user.username})")  # ✅ Ensure user ID is captured

        # ✅ Extract data from request
        game_name = data.get('game_name')
        challenge_id = data.get('challenge_id')
        score = data.get('score')

        # ✅ Extract severity-level scores
        critical_count = data.get('critical_score', 0)
        serious_count = data.get('serious_score', 0)
        moderate_count = data.get('moderate_score', 0)
        minor_count = data.get('minor_score', 0)

        print(
            f"🔹 Parsed Scores - Critical: {critical_count}, Serious: {serious_count}, Moderate: {moderate_count}, Minor: {minor_count}")

        # ✅ Ensure all required fields are present
        if not all([game_name, challenge_id, score]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        score = float(score)  # Convert score to float

        # ✅ Fetch game and challenge
        game = get_object_or_404(GitHubScraperGame, name=game_name)
        challenge = get_object_or_404(GitHubChallenge, id=challenge_id)

        # ✅ Update or create user's score entry
        user_score, created = GitGameScore.objects.get_or_create(
            user=request.user, game=game, github_challenge=challenge,
            defaults={
                'score': score,
                'critical_score': critical_count,
                'serious_score': serious_count,
                'moderate_score': moderate_count,
                'minor_score': minor_count,
                'attempts': 1
            }
        )

        if not created:
            print("🔹 Updating existing score entry")
            user_score.score = score
            user_score.critical_score = critical_count
            user_score.serious_score = serious_count
            user_score.moderate_score = moderate_count
            user_score.minor_score = minor_count
            user_score.attempts += 1  # ✅ Increment attempts
            user_score.save()

        return JsonResponse(
            {
                'message': 'Score updated successfully',
                'latest_score': user_score.score,
                'critical_score': user_score.critical_score,
                'serious_score': user_score.serious_score,
                'moderate_score': user_score.moderate_score,
                'minor_score': user_score.minor_score,
                'attempts': user_score.attempts,
                "user_id": user.id  # ✅ Include user ID in response
            },
            status=200
        )

    except Exception as e:
        print("❌ Error:", str(e))  # ✅ Log errors
        return JsonResponse({'error': str(e)}, status=400)
