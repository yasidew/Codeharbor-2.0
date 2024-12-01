import os
from django.http import JsonResponse
from django.shortcuts import render
from .complexity_calculator import calculate_complexity
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
def refactor_code(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        code = data.get('code')
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)

        # Calculate complexity of the original code
        original_complexity = calculate_complexity(code)

        # Refactor the code using your existing model
        refactored_code = format_code_with_model(code)

        # Calculate complexity of the refactored code
        refactored_complexity = calculate_complexity(refactored_code)

        # Return both complexities and the refactored code
        return JsonResponse({
            'refactored_code': refactored_code,
            'original_complexity': original_complexity,
            'refactored_complexity': refactored_complexity
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)

