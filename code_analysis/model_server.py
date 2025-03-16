from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load Model
MODEL_PATH = "./models/custom_seq2seq_model"
# MODEL_PATH = "/home/ubuntu/custom_seq2seq_model""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    snippet = data.get("snippet", "")

    if not snippet:
        return jsonify({"error": "No code snippet provided"}), 400

    inputs = tokenizer(snippet, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(host="0.0.0.0", port=5000, debug=False)  # Disable debug for production