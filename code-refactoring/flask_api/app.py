from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained model
model_path = "../refactoring_model"  # Adjust path as needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/refactor", methods=["POST"])
def refactor_code():
    data = request.get_json()
    input_code = data.get("code", "")

    # Allow optional fine-tuning parameters
    num_beams = data.get("num_beams", 5)  # Default 5
    max_length = data.get("max_length", 512)  # Default 512
    temperature = data.get("temperature", 1.0)  # Default 1.0
    top_k = data.get("top_k", 50)  # Default 50
    top_p = data.get("top_p", 0.95)  # Default 0.95

    if not input_code:
        return jsonify({"error": "No code provided"}), 400

    # Tokenize input
    inputs = tokenizer(input_code, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").to(device)

    # Generate refactored code with dynamic parameters
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True
    )

    refactored_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"refactored_code": refactored_code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
