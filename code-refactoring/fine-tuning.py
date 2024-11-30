from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import json
from datasets import Dataset

# Step 1: Load the processed dataset
with open("processed_singleton_data.json", "r") as file:
    data_splits = json.load(file)

# Add "type" to train and validation datasets
train_data = [{"input": inp, "output": out, "type": typ} for inp, out, typ in zip(data_splits["train"]["input"], data_splits["train"]["output"], data_splits["train"]["type"])]
val_data = [{"input": inp, "output": out, "type": typ} for inp, out, typ in zip(data_splits["validation"]["input"], data_splits["validation"]["output"], data_splits["validation"]["type"])]

# Step 2: Create datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Step 3: Tokenizer and model
model_name = "Salesforce/codet5-small"  # Use a smaller model if memory is an issue
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Step 4: Preprocess function
def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    outputs = tokenizer(
        examples["output"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Step 5: Define the Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length"
)

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,  # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Save the model
model.save_pretrained("./singleton_model")
tokenizer.save_pretrained("./singleton_model")
print("Model fine-tuning complete. Model saved as singleton_model")
