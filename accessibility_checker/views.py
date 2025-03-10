# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from django.http import JsonResponse
# import subprocess
# import json
#
# from django.views.decorators.csrf import csrf_exempt
# from rest_framework.decorators import permission_classes
#
#
# # Serve the index page
# def index(request):
#     return render(request, 'accessibility_checker.html')
#
# from django.shortcuts import render
# from django.http import JsonResponse
# import subprocess
# import json
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework.permissions import IsAuthenticated
#
# # Serve the index page
# def index(request):
#     return render(request, 'accessibility_checker.html')
#
#
# @csrf_exempt
#
# def check_accessibility(request):
#     if request.method == 'POST' and request.FILES.get('html_file'):
#         uploaded_file = request.FILES['html_file']
#         try:
#             html_content = uploaded_file.read().decode('utf-8')
#             output = subprocess.run(
#                 ['node', 'axe-check.js'],
#                 input=html_content,
#                 text=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#             )
#             try:
#                 results = json.loads(output.stdout)
#             except json.JSONDecodeError:
#                 results = {"error": "Failed to analyze accessibility"}
#
#             # Prepare separate sections for passes, violations, and incomplete
#             result_data = {
#                 "passes": results.get("passes", []),
#                 "violations": results.get("violations", []),
#                 "incomplete": results.get("incomplete", []),
#                 "score": results.get("score", 0)
#             }
#
#             return JsonResponse(result_data)
#         except Exception as e:
#             return JsonResponse({'error': str(e)})
#     return JsonResponse({'error': 'Invalid request method or no file provided'})
#
#


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




@csrf_exempt  # For handling POST requests from JavaScript
@api_view(["POST"])
  # Ensure only logged-in users can submit scores
def submit_score(request):
    """Handles score submission from frontend"""
    try:
        data = json.loads(request.body)

        # Extract data from request
        game_name = data.get('game_name')
        challenge_id = data.get('challenge_id')
        score = data.get('score')

        if not all([game_name, challenge_id, score]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        score = float(score)  # Convert score to float

        # ✅ Fetch game and challenge
        game = get_object_or_404(GitHubScraperGame, name=game_name)
        challenge = get_object_or_404(GitHubChallenge, id=challenge_id)

        # ✅ Update or create user's score entry
        user_score, created = GitGameScore.objects.get_or_create(
            user=request.user, game=game, github_challenge=challenge,
            defaults={'score': score, 'attempts': 1}  # Set defaults for new records
        )

        if not created:
            user_score.score = score  # ✅ Always update the score with the latest one
            user_score.attempts += 1  # ✅ Increment attempts
            user_score.save()

        return JsonResponse(
            {'message': 'Score updated successfully', 'latest_score': user_score.score, 'attempts': user_score.attempts},
            status=200
        )

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

