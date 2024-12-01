import os
from django.http import JsonResponse
from django.shortcuts import render
from .utils import format_code_with_model

# Function to display the refactor page
def refactor_view(request):
    return render(request, 'code_formatter/refactor.html')

# Function to handle file upload and extract code
def upload_code(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read().decode('utf-8')  # Read the content of the file as text
        return JsonResponse({'code': file_content})  # Return the extracted code as JSON
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Function to handle code refactoring
# Function to calculate cyclomatic complexity (placeholder function)
def calculate_complexity(code):
    # Replace this with actual complexity calculation logic
    # For now, it returns a dummy value for demonstration
    return len(code.split('\n'))  # Example: number of lines in the code

# Function to handle code refactoring
def refactor_code(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        code = data.get('code')
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)

        # Calculate complexity before refactoring
        before_complexity = calculate_complexity(code)

        # Use the Singleton model for processing
        refactored_code = format_code_with_model(code)

        # Calculate complexity after refactoring
        after_complexity = calculate_complexity(refactored_code)

        # Return refactored code and complexity metrics
        return JsonResponse({
            'refactored_code': refactored_code,
            'before_metrics': before_complexity,
            'after_metrics': after_complexity,
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)

