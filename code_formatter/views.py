from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .complexity_calculator import calculate_loc, calculate_readability
import openai
import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from .models import Guideline
from .forms import GuidelineForm
from .models import RefactoringHistory
from .utils import analyze_code, refactor_code
from django.core.files.storage import FileSystemStorage

# Create OpenAI Client with API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GUIDELINE_PROMPTS = {
    "Factory": "What are the best practices for using the Factory Pattern in software design?",
    "Strategy": "What are the best practices for implementing the Strategy Pattern in code?",
    "Observer": "What are the best practices for using the Observer Pattern effectively?",
}

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

# API Endpoint for AI to Fetch Guidelines
def get_guidelines(request, company_name):
    guidelines = Guideline.objects.filter(company_name=company_name).values('pattern', 'rule')
    return JsonResponse({"guidelines": list(guidelines)})

def define_guidelines(request):
    if request.method == "POST":
        form = GuidelineForm(request.POST, request.FILES)  # Include request.FILES to handle file uploads
        if form.is_valid():
            guideline = form.save(commit=False)
            if 'company_logo' in request.FILES:
                guideline.company_logo = request.FILES['company_logo']
            guideline.save()
            return redirect('define_guidelines')

    guidelines = Guideline.objects.all()
    return render(request, "code_formatter/define_guidelines.html", {"guidelines": guidelines})

@csrf_exempt
def edit_guideline(request, id):
    if request.method == "POST":
        try:
            guideline = get_object_or_404(Guideline, id=id)

            if "company_name" in request.POST:
                guideline.company_name = request.POST["company_name"]

            if "pattern" in request.POST:
                guideline.pattern = request.POST["pattern"]

            if "rule" in request.POST:
                guideline.rule = request.POST["rule"]

            if "company_logo" in request.FILES:  # Handle logo update
                guideline.company_logo = request.FILES["company_logo"]

            guideline.save()

            return JsonResponse({
                "success": True,
                "company_name": guideline.company_name,
                "pattern": guideline.pattern,
                "rule": guideline.rule,
                "company_logo": guideline.company_logo.url if guideline.company_logo else None
            })
        except Guideline.DoesNotExist:
            return JsonResponse({"success": False, "error": "Guideline not found"})
    return JsonResponse({"success": False, "error": "Invalid request"})

def delete_guideline(request, guideline_id):
    if request.method == "POST":
        guideline = get_object_or_404(Guideline, id=guideline_id)
        guideline.delete()
        return JsonResponse({"success": True})  # ✅ Return a JSON response

    return JsonResponse({"success": False, "error": "Invalid request"}, status=400)


@csrf_exempt
def refactor_and_compare(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            original_code = data.get("code", "")

            print("Received code:", original_code)  # Debugging log

            # Analyze Original Code Complexity
            original_complexity, original_readability = analyze_code(original_code)

            # Refactor Code
            refactored_code = refactor_code(original_code)

            # Analyze Refactored Code Complexity
            refactored_complexity, refactored_readability = analyze_code(refactored_code)

            # Save to Database
            history = RefactoringHistory.objects.create(
                original_code=original_code,
                refactored_code=refactored_code,
                original_complexity=original_complexity,
                refactored_complexity=refactored_complexity,
                original_readability=original_readability,
                refactored_readability=refactored_readability
            )

            print("Saved to database:", history.id)  # Debugging log

            return JsonResponse({
                "success": True,
                "before_code": original_code,
                "after_code": refactored_code,
                "original_complexity": original_complexity,
                "refactored_complexity": refactored_complexity,
                "original_readability": original_readability,
                "refactored_readability": refactored_readability,
                "id": history.id
            })
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Invalid request"})


def refactor_code_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            original_code = data.get('code', '')

            # Simulate AI-based refactoring (Replace this with actual AI call)
            refactored_code = "Refactored: " + original_code  # Placeholder result
            original_complexity = 78  # Replace with actual analysis
            refactored_complexity = 92  # Replace with actual analysis
            original_readability = 117.33  # Replace with actual analysis
            refactored_readability = 116.73  # Replace with actual analysis

            # Save to database
            history = RefactoringHistory.objects.create(
                original_code=original_code,
                refactored_code=refactored_code,
                original_complexity=original_complexity,
                refactored_complexity=refactored_complexity,
                original_readability=original_readability,
                refactored_readability=refactored_readability,
            )

            return JsonResponse({
                "refactored_code": refactored_code,
                "original_loc": original_complexity,
                "refactored_loc": refactored_complexity,
                "original_readability": original_readability,
                "refactored_readability": refactored_readability,
                "id": history.id
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)

# ✅ AI-Powered Guideline Generation
@csrf_exempt
def generate_guideline(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            pattern = data.get("pattern", "").strip()

            if not pattern or pattern not in GUIDELINE_PROMPTS:
                return JsonResponse({"error": "Invalid pattern selected"}, status=400)

            prompt = GUIDELINE_PROMPTS[pattern]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional software architect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            # ✅ Correct way to extract the response
            guideline_text = response.choices[0].message.content.strip()

            return JsonResponse({"suggestion": guideline_text})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
