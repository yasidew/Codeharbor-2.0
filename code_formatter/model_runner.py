import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# model_path = os.path.join(os.path.dirname(__file__), "refactoring_model")
# model_path = "../code-refactoring/refactoring_model"
# Dynamically build absolute path to model directory
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(BASE_DIR, 'code-refactoring', 'refactoring_model')
# model_path = os.path.join(BASE_DIR, 'code-refactoring', 'refactoring_model', 'refactoring_model')

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, 'refactoring_model', 'refactoring_model')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'refactoring_model', 'refactoring_model')



# AWS PATH
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, 'refactoring_model')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_refactor_model(code: str, refactor_type: str, generation_config=None) -> str:
    if not generation_config:
        generation_config = {
            "num_beams": 4,
            "max_length": 512,
            "early_stopping": True
        }

    # Match training prompt structure: Type + Code
    prompt = f"Type: {refactor_type}\nCode:\n{code}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=generation_config["max_length"],
                       truncation=True, padding="max_length").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=generation_config["max_length"],
            num_beams=generation_config["num_beams"],
            # early_stopping=generation_config["early_stopping"]
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)