import os
import json
import requests
import openai
import base64
from django.http import JsonResponse
from django.shortcuts import render
from .utils import calculate_loc, calculate_readability, extract_code_from_github
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# OpenAI API Client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Trained Model
MODEL_PATH = "./models/custom_seq2seq_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# ✅ AI Code Analysis Page
def ai_code_analysis_view(request):
    return render(request, 'ai_code_analysis.html.html')

# ✅ File Upload & Extract Code
def upload_code(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read().decode('utf-8')
        return JsonResponse({'code': file_content})

    return JsonResponse({'error': 'Invalid request'}, status=400)

# ✅ AI-Driven Code Analysis (GPT-4o + Custom Model)
def analyze_code_with_ai(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            code = data.get('code', '').strip()
            github_url = data.get('github_url', '').strip()

            if not code and github_url:
                code = extract_code_from_github(github_url)
                if not code:
                    return JsonResponse({'error': 'No valid code extracted from GitHub'}, status=400)

            if not code:
                return JsonResponse({'error': 'No code provided'}, status=400)

            # Calculate LOC and Readability
            original_loc = calculate_loc(code)
            original_readability = calculate_readability(code)

            # ✅ Analyze Code Using GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that analyzes code and provides improvement suggestions."},
                    {"role": "user", "content": f"Analyze the following code and provide categorized issues:\n\n{code}"}
                ],
                temperature=0.2
            )

            gpt_analysis = response.choices[0].message.content

            # ✅ Analyze Code Using Trained Model
            inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            model_analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return JsonResponse({
                'gpt_analysis': gpt_analysis,
                'model_analysis': model_analysis,
                'original_loc': original_loc,
                'original_readability': original_readability,
            })

        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# ✅ AI Code Analysis Guidelines
def get_guidelines(request):
    guidelines = [
        {"category": "Security", "rule": "Avoid hardcoded secrets and use secure authentication."},
        {"category": "Performance", "rule": "Optimize loops and minimize redundant computations."},
        {"category": "Readability", "rule": "Follow naming conventions and reduce long functions."},
        {"category": "Best Practices", "rule": "Use design patterns and modularize code effectively."},
    ]
    return JsonResponse({'guidelines': guidelines})
