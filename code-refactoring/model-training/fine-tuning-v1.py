from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import json
from datasets import Dataset

# Step 1: Load the processed dataset
with open("dataset/processed_factory_method_data.json", "r") as file:
    data_splits = json.load(file)

# Add all attributes for train and validation datasets
train_data = [
    {
        "input": inp,
        "output": out,
        "type": typ,
        "complexity": cmp,
        "language": lang,
        "context": ctx,
        "edge_cases": ec,
        "dependencies": deps,
        "performance_notes": pn,
        "real_world_usage": rwu,
        "testing_notes": tn,
        "comments": cm,
        "source": src,
    }
    for inp, out, typ, cmp, lang, ctx, ec, deps, pn, rwu, tn, cm, src in zip(
        data_splits["train"]["input"],
        data_splits["train"]["output"],
        data_splits["train"]["type"],
        data_splits["train"]["complexity"],
        data_splits["train"]["language"],
        data_splits["train"]["context"],
        data_splits["train"]["edge_cases"],
        data_splits["train"]["dependencies"],
        data_splits["train"]["performance_notes"],
        data_splits["train"]["real_world_usage"],
        data_splits["train"]["testing_notes"],
        data_splits["train"]["comments"],
        data_splits["train"]["source"],
    )
]

val_data = [
    {
        "input": inp,
        "output": out,
        "type": typ,
        "complexity": cmp,
        "language": lang,
        "context": ctx,
        "edge_cases": ec,
        "dependencies": deps,
        "performance_notes": pn,
        "real_world_usage": rwu,
        "testing_notes": tn,
        "comments": cm,
        "source": src,
    }
    for inp, out, typ, cmp, lang, ctx, ec, deps, pn, rwu, tn, cm, src in zip(
        data_splits["validation"]["input"],
        data_splits["validation"]["output"],
        data_splits["validation"]["type"],
        data_splits["validation"]["complexity"],
        data_splits["validation"]["language"],
        data_splits["validation"]["context"],
        data_splits["validation"]["edge_cases"],
        data_splits["validation"]["dependencies"],
        data_splits["validation"]["performance_notes"],
        data_splits["validation"]["real_world_usage"],
        data_splits["validation"]["testing_notes"],
        data_splits["validation"]["comments"],
        data_splits["validation"]["source"],
    )
]

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
    learning_rate=5e-5,  # Adjusted learning rate
    warmup_steps=500,  # Add warmup steps for smoother convergence
    lr_scheduler_type="cosine",  # Cosine schedule for decay
    per_device_train_batch_size=16,  # Larger batch size for stability
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Mimic larger batch size
    num_train_epochs=20,  # Increased epochs for thorough training
    weight_decay=0.01,
    save_total_limit=3,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision
    dataloader_num_workers=4,
    report_to="none",
    label_smoothing_factor=0.1  # Prevent overconfidence in predictions
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
model.save_pretrained("./factory_method_model")
tokenizer.save_pretrained("./factory_method_model")
print("Model fine-tuning complete. Model saved as factory_method_model")
