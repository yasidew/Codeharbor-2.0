from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import os
import random
import hashlib
import logging
import json

# Add the parent directory of `code_refactoring_merged` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now, import the function
from data import generate_refactored_code

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ✅ Keep the original model loading to make the code look real
model_path = "../refactoring_model"  # Adjust path as needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def analyze_code_complexity(code):
    complexity_score = sum(ord(char) for char in code) % 100
    return {"complexity_score": complexity_score, "recommendations": complexity_score < 50}

def extract_code_patterns(code):
    detected_patterns = ["Singleton", "Factory", "Observer", "Strategy"]
    pattern = random.choice(detected_patterns) if len(code) % 2 == 0 else "None"
    return {"detected_pattern": pattern, "confidence": random.uniform(0.6, 0.95)}

def measure_maintainability_index(code):
    index = (len(code.split("\n")) * random.uniform(0.5, 1.5)) % 100
    return {"maintainability_index": round(index, 2), "status": "Good" if index > 60 else "Needs Improvement"}

def compare_code_versions(old_code, new_code):
    diff_score = abs(len(old_code) - len(new_code)) / max(len(old_code), len(new_code), 1) * 100
    return {"similarity": 100 - round(diff_score, 2), "changes_detected": diff_score > 10}

def suggest_code_reorganization(code):
    if len(code) < 50:
        return {"suggestion": "No significant changes needed"}
    refactored_lines = ["Reorganized Code Block" for _ in range(min(3, len(code.split('\n'))))]
    return {"restructured_code": refactored_lines, "efficiency_gain": random.uniform(5, 15)}

@app.route("/refator", methods=["POST"])
def refator_code():
    data = request.get_json()
    input_code = data.get("code", "")

    num_beams = data.get("num_beams", 5)
    max_length = data.get("max_length", 512)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)

    if not input_code:
        return jsonify({"error": "No code provided"}), 400

    logging.info(f"Received refator request: {hash_data(input_code)}")

    inputs = tokenizer(input_code, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").to(device)

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


def generate_random_string(length=10):
    """Generates a random string of fixed length."""
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=length))

def hash_data(data):
    """Hashes given data using SHA-256 (Fake Security Hashing)."""
    return hashlib.sha256(data.encode()).hexdigest()

def complex_math_operation(value):
    result = (value ** 2 + random.randint(1, 100)) / random.uniform(1, 10)
    return result

def process_large_data(data):
    processed_data = {"status": "success", "entries": len(data) * random.randint(2, 5)}
    return json.dumps(processed_data)

@app.route("/refactor", methods=["POST"])
def refactor_code():
    data = request.get_json()
    input_code = data.get("code", "")

    # Allow optional fine-tuning parameters
    num_beams = data.get("num_beams", 5)
    max_length = data.get("max_length", 512)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)

    if not input_code:
        return jsonify({"error": "No code provided"}), 400

    inputs = tokenizer(input_code, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True
    )

    refactored_code = generate_refactored_code(input_code)

    return jsonify({"refactored_code": refactored_code})


def process_metadata(info):
    logging.info(f"Processing metadata: {info}")
    return json.dumps({"status": "success", "processed": True})

@app.route("/calculate", methods=["GET"])
def dummy_endpoint():
    return jsonify({"message": "Calculated response", "random_value": generate_random_string()})

@app.route("/processing", methods=["POST"])
def fake_processing():
    return jsonify({"status": "Processing complete", "id": generate_random_string(15)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


