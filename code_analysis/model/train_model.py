from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.nn.utils.rnn import pad_sequence

# Initialize model and tokenizer
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Custom collation function for seq2seq tasks
def custom_collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

# Load and preprocess datasets
def load_and_preprocess_dataset(file_path, tokenizer):
    data = Dataset.from_json(file_path)

    def tokenize_function(example):
        inputs = tokenizer(
            example["func"], truncation=True, padding="max_length", max_length=512
        )
        targets = tokenizer(
            example["target"], truncation=True, padding="max_length", max_length=128
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    return data.map(tokenize_function, batched=True)

# Paths to the processed JSON files
train_file_path = "dataset/processed_data/custom_train.json"
test_file_path = "dataset/processed_data/custom_test.json"

# Preprocess datasets
train_dataset = load_and_preprocess_dataset(train_file_path, tokenizer)
test_dataset = load_and_preprocess_dataset(test_file_path, tokenizer)

# Set up DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(train_dataloader)}")

# Save the trained model
# model_save_path = "models/custom_seq2seq_model"
model_save_path = "./custom_seq2seq_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Evaluate the model
print("Evaluating model on test dataset...")
model.eval()

generated_targets = []
ground_truth_targets = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        outputs = model.generate(inputs["input_ids"], max_length=128)

        generated_targets.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ground_truth_targets.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

# Calculate F1 Score
binary_true = [1 if gt == pred else 0 for gt, pred in zip(ground_truth_targets, generated_targets)]
binary_pred = [1] * len(binary_true)  # Assume all generated are positive predictions

precision, recall, f1, _ = precision_recall_fscore_support(binary_true, binary_pred, average="binary")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display some examples
for i in range(5):
    print(f"Input Code: {test_dataset[i]['func']}")
    print(f"Generated Suggestion: {generated_targets[i]}")
    print(f"Ground Truth Suggestion: {ground_truth_targets[i]}")
    print()