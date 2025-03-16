from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu

# Step 1: Load the fine-tuned model and tokenizer
model_dir = "../singleton_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Step 2: Load the test data
with open("dataset/processed_singleton_data.json", "r") as file:
    data_splits = json.load(file)

test_inputs = data_splits["test"]["input"]
test_outputs = data_splits["test"]["output"]

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
    predictions.append(predicted_output)
    references.append([test_outputs[i]])  # Wrap reference in a list for BLEU score

# Step 4: Calculate BLEU Score
bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(references, predictions)]
average_bleu = sum(bleu_scores) / len(bleu_scores)

# Step 5: Print Results
print(f"Average BLEU Score: {average_bleu * 100:.2f}")
for i, (pred, ref) in enumerate(zip(predictions, references)):
    print(f"\nTest Sample {i+1}:")
    print(f"Input:\n{test_inputs[i]}")
    print(f"Expected Output:\n{ref[0]}")
    print(f"Predicted Output:\n{pred}")

# Save predictions to a file (optional)
with open("test_predictions.json", "w") as pred_file:
    json.dump({"predictions": predictions, "references": references}, pred_file, indent=4)
