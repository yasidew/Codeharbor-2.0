import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the trained model
MODEL_PATH = os.path.join(BASE_DIR, "code-refactoring", "singleton_model")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def format_code_with_model(code):
    """
    Function to format code using the trained model.
    """
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    formatted_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return formatted_code
