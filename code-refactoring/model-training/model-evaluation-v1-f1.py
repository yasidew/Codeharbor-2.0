from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from sklearn.metrics import precision_recall_fscore_support

# Step 1: Load the fine-tuned model and tokenizer
model_dir = "../factory_method_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Step 2: Load the test data
with open("dataset/processed_factory_method_data.json", "r") as file:
    data_splits = json.load(file)

test_inputs = data_splits["test"]["input"]
test_outputs = data_splits["test"]["output"]

# Additional attributes (optional, for debugging or analysis)
test_types = data_splits["test"]["type"]
test_complexities = data_splits["test"]["complexity"]
test_contexts = data_splits["test"]["context"]

# Step 3: Evaluate the model
predictions = []
references = []

for i, test_input in enumerate(test_inputs):
    # Tokenize the input
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Generate the model's output
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store predictions and references
    predictions.append(predicted_output.split())  # Split into tokens for F1 calculation
    references.append(test_outputs[i].split())  # Split into tokens for F1 calculation

# Step 4: Align Prediction and Reference Tokens
all_pred_tokens = []
all_ref_tokens = []

# Align lengths of predictions and references
for pred, ref in zip(predictions, references):
    min_len = min(len(pred), len(ref))  # Use the shorter length
    all_pred_tokens.extend(pred[:min_len])
    all_ref_tokens.extend(ref[:min_len])

# Debug: Check lengths of aligned tokens
print(f"Length of all_ref_tokens: {len(all_ref_tokens)}")
print(f"Length of all_pred_tokens: {len(all_pred_tokens)}")

# Step 5: Calculate Precision, Recall, and F1 Score (Token-Level)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_ref_tokens, all_pred_tokens, average="weighted", zero_division=0
)

# Step 6: Print Results
print(f"Precision: {precision * 100:.2f}")
print(f"Recall: {recall * 100:.2f}")
print(f"F1 Score: {f1 * 100:.2f}")

# Save metrics to a file (optional)
with open("f1_score_metrics.json", "w") as metrics_file:
    json.dump(
        {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        metrics_file,
        indent=4,
    )
print("F1 score metrics saved to f1_score_metrics.json")
