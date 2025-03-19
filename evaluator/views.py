from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from helper_functions import get_prompt_from_file, check_relevance, evaluate_guideline, simple_inference


@csrf_exempt
@require_http_methods(["POST"])
def evaluate_html_view(request):
    try:
        data = json.loads(request.body)
        selected_html_code = data.get('html_code', '')
        selected_guidelines = data.get('guidelines', [])

        # Ensure selected_html_code is not empty
        if not selected_html_code.strip():
            return JsonResponse({"error": "Please provide some HTML code."}, status=400)

        # Call the function to evaluate HTML and guidelines
        result = evaluate_html(selected_html_code, selected_guidelines)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def evaluate_html(selected_html_code, selected_guidelines):
    instructions_dir = "instructions"
    guideline_folders = [
        folder for folder in os.listdir(instructions_dir)
        if os.path.isdir(os.path.join(instructions_dir, folder))
    ]
    guideline_folders.sort(key=lambda x: int(x) if x.isdigit() else x)

    scores = {}

    # Categories of guidelines
    classification = [
        "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual",
        "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Visual", "Visual",
        "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual", "Visual",
        "Visual", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive",
        "Cognitive", "Cognitive", "Cognitive", "Visual", "Visual", "Visual", "Cognitive", "Cognitive",
        "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Visual", "Cognitive", "Cognitive", "Cognitive",
        "Visual", "Visual", "Visual", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Visual",
        "Cognitive", "Cognitive", "Visual", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive",
        "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive",
        "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive",
        "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Cognitive"
    ]

    cognitive_score = 0
    visual_score = 0
    cognitive_count = 0
    visual_count = 0

    for guideline in guideline_folders:
        if guideline not in selected_guidelines:
            continue

        folder_path = os.path.join(instructions_dir, guideline)
        check_file = os.path.join(folder_path, "check.txt")
        guidelines_file = os.path.join(folder_path, "guidelines.txt")

        if not os.path.exists(check_file) or not os.path.exists(guidelines_file):
            continue

        check_prompt = get_prompt_from_file(
            check_file) + " Output your answer in JSON format with only one key 'output'."
        eval_prompt = get_prompt_from_file(guidelines_file)

        if not check_relevance(selected_html_code, check_prompt, guideline):
            scores[guideline] = "N/A"
        else:
            score = evaluate_guideline(selected_html_code, eval_prompt, guideline)
            scores[guideline] = score
            if classification[int(guideline)] == "Cognitive":
                cognitive_score += int(score.split("/")[0])
                cognitive_count += 1
            else:
                visual_count += 1
                visual_score += int(score.split("/")[0])

    # Sort scores and fetch the lowest 3 scores
    lowest_scores = sorted(
        [(k, v) for k, v in scores.items() if v != "N/A"], key=lambda x: x[1]
    )[:3]

    check_files = [os.path.join(instructions_dir, guideline, "check.txt") for guideline, _ in lowest_scores]
    check_queries = []
    for file in check_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = f.read()
                data = data.replace(
                    "You are an accessibility compliance assistant. Your task is to analyze the given HTML to determine if it includes",
                    "")
                data = data.replace(
                    "If the HTML contains any such elements or meets the conditions, output 'Yes'. Otherwise, output 'No'.",
                    "")
                check_queries.append(data)

    times = []
    suggestions = []
    results = []
    for i, query in enumerate(check_queries):
        suggestion_input = f"Here is the HTML snippet:\n{selected_html_code}\nHow can I improve this to meet WCAG accessibility guidelines?"
        suggestion = simple_inference("qwen2.5-coder-1.5b-instruct", suggestion_input)

        if suggestion:
            suggestions.append({
                "guideline": lowest_scores[i][0],
                "suggestion": suggestion.strip()
            })

        results.append({
            "query": query.strip(),
            "suggestion": suggestion.strip() if suggestion else "No suggestion available"
        })

    return {
        "scores": scores,
        "results": results,
        "suggestions": suggestions,
        "cognitive_score": cognitive_score / cognitive_count if cognitive_count > 0 else "N/A",
        "visual_score": visual_score / visual_count if visual_count > 0 else "N/A"
    }


def submit_html_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            html_code = data.get("html_code", "")
            guidelines = data.get("guidelines", [])

            # Process the HTML code and guidelines here
            response_data = {
                "message": "Evaluation complete",
                "html_code": html_code,
                "selected_guidelines": guidelines
            }
            return JsonResponse(response_data)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return render(request, "evaluator/submit_html.html")