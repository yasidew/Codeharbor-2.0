from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import json
from datasets import Dataset

# Step 1: Load the processed dataset
with open("dataset/processed_singleton_data.json", "r") as file:
    data_splits = json.load(file)

# Add "type" to train and validation datasets
train_data = [{"input": inp, "output": out, "type": typ} for inp, out, typ in zip(data_splits["train"]["input"], data_splits["train"]["output"], data_splits["train"]["type"])]
val_data = [{"input": inp, "output": out, "type": typ} for inp, out, typ in zip(data_splits["validation"]["input"], data_splits["validation"]["output"], data_splits["validation"]["type"])]

# Step 2: Create datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Step 3: Tokenizer and model
model_name = "Salesforce/codet5-base"
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
    learning_rate=7e-5,  # Increased learning rate for faster convergence
    per_device_train_batch_size=8,  # Utilize Colab Pro GPU RAM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Accumulate gradients to mimic larger batch sizes
    num_train_epochs=20,  # Increased epochs to let the model learn better
    weight_decay=0.01,
    save_total_limit=3,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision for faster training
    dataloader_num_workers=4,  # Utilize CPU cores for loading data
    report_to="none"  # Suppress unnecessary logging
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
