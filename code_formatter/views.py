from django.http import JsonResponse
from django.shortcuts import render
from .utils import format_code_with_model
from .complexity_calculator import calculate_loc, calculate_readability

# Function to display the refactor page
def refactor_view(request):
    return render(request, 'code_formatter/refactor.html')

# Function to handle file upload and extract code
def upload_code(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read().decode('utf-8')
        return JsonResponse({'code': file_content})
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Function to handle code refactoring
def refactor_code(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        code = data.get('code')
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)

        # Calculate Lines of Code (LOC) and Readability for original code
        original_loc = calculate_loc(code)
        original_readability = calculate_readability(code)

        # Refactor the code
        refactored_code = format_code_with_model(code)

        # Calculate LOC and Readability for refactored code
        refactored_loc = calculate_loc(refactored_code)
        refactored_readability = calculate_readability(refactored_code)

        return JsonResponse({
            'refactored_code': refactored_code,
            'original_loc': original_loc,
            'refactored_loc': refactored_loc,
            'original_readability': original_readability,
            'refactored_readability': refactored_readability
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)
