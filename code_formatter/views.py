from django.http import JsonResponse
from django.shortcuts import render
from .complexity_calculator import calculate_loc, calculate_readability
import openai
import os
import json

# Create OpenAI Client with API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to display the refactor page
def refactor_view(request):
    return render(request, 'code_formatter/refactor.html')

# Function to handle file upload and extract code
def upload_code(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read().decode('utf-8')
        return JsonResponse({'code': file_content})

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Function to handle code refactoring using OpenAI API
def refactor_code(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            code = data.get('code', '').strip()

            if not code:
                return JsonResponse({'error': 'No code provided'}, status=400)

            # Calculate LOC and Readability for the original code
            original_loc = calculate_loc(code)
            original_readability = calculate_readability(code)

            # ✅ FIXED: Correct OpenAI API Call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # ✅ Use "gpt-4-turbo" or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are a professional code refactoring assistant."},
                    {"role": "user", "content": f"Refactor the following code to improve structure, readability, and efficiency:\n\n{code}"}
                ],
                temperature=0.2
            )

            # ✅ Extract the correct response format
            refactored_code = response.choices[0].message.content

            # Calculate LOC and Readability for the refactored code
            refactored_loc = calculate_loc(refactored_code)
            refactored_readability = calculate_readability(refactored_code)

            return JsonResponse({
                'refactored_code': refactored_code,
                'original_loc': original_loc,
                'refactored_loc': refactored_loc,
                'original_readability': original_readability,
                'refactored_readability': refactored_readability
            })

        except Exception as e:
            print("Error:", str(e))  # Debugging
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)
