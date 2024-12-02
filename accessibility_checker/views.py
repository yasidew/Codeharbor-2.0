from django.shortcuts import render
from django.http import JsonResponse
import subprocess
import json

# Serve the index page
def index(request):
    return render(request, 'accessibility_checker.html')

# Handle accessibility check
def check_accessibility(request):
    if request.method == 'POST' and request.FILES.get('html_file'):
        uploaded_file = request.FILES['html_file']
        # Read the content of the uploaded HTML file
        try:
            html_content = uploaded_file.read().decode('utf-8')  # Read the file content as a string
            # Run axe-core with Puppeteer or similar tools
            output = subprocess.run(
                ['node', 'axe-check.js'],  # Run the Node.js script
                input=html_content,        # Pass the HTML content to the Node.js script
                text=True,                 # Ensure the input/output is treated as text (strings)
                stdout=subprocess.PIPE,    # Capture the output of the Node.js script
                stderr=subprocess.PIPE     # Capture errors if any
            )
            # Parse the output from axe-core
            try:
                results = json.loads(output.stdout)  # Parse the results from JSON output
            except json.JSONDecodeError:
                results = {"error": "Failed to analyze accessibility"}
            return JsonResponse(results)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid request method or no file provided'})
