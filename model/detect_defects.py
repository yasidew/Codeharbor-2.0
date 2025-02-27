from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from django.http import JsonResponse
from django.shortcuts import render

def detect_defects(request):
    model_path = "./models/defect_detection_model"
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if request.method == "POST":
        try:
            code_snippet = request.POST.get("code_snippet")
            if not code_snippet:
                return JsonResponse({"error": "No code snippet provided."}, status=400)

            inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()

            return JsonResponse({"defect_detected": bool(prediction)})

        except Exception as e:
            return JsonResponse({"error": f"Detection failed: {e}"}, status=500)

    return render(request, "detect_defects.html")
