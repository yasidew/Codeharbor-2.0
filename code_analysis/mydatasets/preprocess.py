import os
import json
from datasets import Dataset
from transformers import AutoTokenizer
import torch

def preprocess_data_with_auto_tokenizer(data, tokenizer):
    """
    Tokenize the dataset using AutoTokenizer for both func and target fields.
    """
    def tokenize_function(example):
        inputs = tokenizer(example["func"], truncation=True, padding="max_length", max_length=512)
        targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=128)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],  # Use the tokenized target as labels for seq2seq training
        }

    # Debug: Print the first few samples before tokenization
    print("Data before tokenization:", data[:3])

    # Tokenize and format the dataset
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Debug: Print the first few samples after tokenization
    print("Data after tokenization:", tokenized_data[:3])

    return tokenized_data

def save_raw_dataset(data, file_path):
    """
    Save the raw dataset with func and target fields in JSON format.
    """
    with open(file_path, "w") as f:
        json.dump([{"func": example["func"], "target": example["target"]} for example in data], f, indent=4)

def preprocess_dataset(raw_data_path, model_name, output_dir="code_analysis/dataset/processed_data"):
    """
    Preprocess the raw dataset and save it.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    print(f"Loading dataset from: {raw_data_path}")
    with open(raw_data_path, 'r') as file:
        data = json.load(file)  # Adjust to handle full JSON format

    # Convert raw data to Hugging Face Dataset
    dataset = Dataset.from_dict({"func": [d["func"] for d in data], "target": [d["target"] for d in data]})

    # Split dataset
    split_data = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_data["train"]
    test_dataset = split_data["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Save raw datasets in JSON format
    train_raw_path = os.path.join(output_dir, "custom_train_raw.json")
    test_raw_path = os.path.join(output_dir, "custom_test_raw.json")
    save_raw_dataset(train_dataset, train_raw_path)
    save_raw_dataset(test_dataset, test_raw_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Preprocess data
    train_dataset = preprocess_data_with_auto_tokenizer(train_dataset, tokenizer)
    test_dataset = preprocess_data_with_auto_tokenizer(test_dataset, tokenizer)

    # Save tokenized datasets in JSON format
    train_tokenized_path = os.path.join(output_dir, "custom_train_tokenized.json")
    test_tokenized_path = os.path.join(output_dir, "custom_test_tokenized.json")
    train_dataset.to_json(train_tokenized_path)
    test_dataset.to_json(test_tokenized_path)

    print(f"Processed data saved at: {output_dir}")
    return train_dataset, test_dataset

# Preprocess the raw dataset
train_dataset, test_dataset = preprocess_dataset(
    raw_data_path="custom_dataset.json",
    model_name="Salesforce/codet5-base"
)