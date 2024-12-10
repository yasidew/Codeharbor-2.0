from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
import subprocess
import json

from django.views.decorators.csrf import csrf_exempt


# Serve the index page
def index(request):
    return render(request, 'accessibility_checker.html')

from django.shortcuts import render
from django.http import JsonResponse
import subprocess
import json
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

# Serve the index page
def index(request):
    return render(request, 'accessibility_checker.html')


@csrf_exempt
def check_accessibility(request):
    if request.method == 'POST' and request.FILES.get('html_file'):
        uploaded_file = request.FILES['html_file']
        try:
            html_content = uploaded_file.read().decode('utf-8')
            output = subprocess.run(
                ['node', 'axe-check.js'],
                input=html_content,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            try:
                results = json.loads(output.stdout)
            except json.JSONDecodeError:
                results = {"error": "Failed to analyze accessibility"}

            # Prepare separate sections for passes, violations, and incomplete
            result_data = {
                "passes": results.get("passes", []),
                "violations": results.get("violations", []),
                "incomplete": results.get("incomplete", []),
                "score": results.get("score", 0)
            }

            return JsonResponse(result_data)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid request method or no file provided'})


