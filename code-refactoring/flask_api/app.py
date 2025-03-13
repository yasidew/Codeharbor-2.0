from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)  # Allow Django to call this API

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

    if not input_code:
        return jsonify({"error": "No code provided"}), 400

    # Tokenize and generate
    inputs = tokenizer(input_code, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    refactored_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"refactored_code": refactored_code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
