# import os
# import torch
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from datasets import load_dataset, Dataset
# from torch.utils.data import DataLoader
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
#
# def validate_dataset(dataset):
#     """
#     Validate that the dataset contains the 'func' field and all entries are valid strings.
#
#     Args:
#         dataset (Dataset): The loaded dataset.
#
#     Raises:
#         ValueError: If invalid entries are found in the dataset.
#     """
#     for example in dataset:
#         if "func" not in example or not isinstance(example["func"], str):
#             raise ValueError(f"Invalid entry found: {example}")
#
# def combine_datasets(default_path, custom_path):
#     """
#     Combine default and custom mydatasets into a unified dataset.
#
#     Args:
#         default_path (str): Path to the default dataset JSONL file.
#         custom_path (str): Path to the custom dataset JSONL file.
#
#     Returns:
#         Dataset: Combined dataset.
#     """
#     # Load default dataset
#     default_train = pd.read_json(default_path + "/train.jsonl", lines=True)
#     default_test = pd.read_json(default_path + "/test.jsonl", lines=True)
#
#     # Load custom dataset
#     custom_train = pd.read_json(custom_path + "/custom_train.jsonl", lines=True)
#     custom_test = pd.read_json(custom_path + "/custom_test.jsonl", lines=True)
#
#     # Combine mydatasets
#     train_data = pd.concat([default_train, custom_train], ignore_index=True)
#     test_data = pd.concat([default_test, custom_test], ignore_index=True)
#
#     # Shuffle the combined mydatasets
#     train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
#     test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     return train_data, test_data
#
# def preprocess_data(df, tokenizer):
#     """
#     Tokenize and preprocess the dataset for model training.
#
#     Args:
#         df (pd.DataFrame): The dataset as a pandas DataFrame.
#         tokenizer (RobertaTokenizer): The tokenizer.
#
#     Returns:
#         Dataset: Tokenized dataset.
#     """
#     # Convert DataFrame to Hugging Face Dataset
#     dataset = Dataset.from_pandas(df)
#
#     def tokenize_function(examples):
#         return tokenizer(
#             examples["func"],
#             truncation=True,
#             padding="max_length",
#             max_length=512
#         )
#
#     # Tokenize dataset
#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     return tokenized_dataset
#
# def train_model():
#     """
#     Train a defect detection model using the unified dataset.
#     """
#     try:
#         # Paths to mydatasets
#         default_path = "dataset/processed_data"
#         custom_path = "dataset/processed_data"
#
#         # Combine mydatasets
#         train_data, test_data = combine_datasets(default_path, custom_path)
#
#         # Initialize the tokenizer
#         tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#
#         # Tokenize mydatasets
#         train_dataset = preprocess_data(train_data, tokenizer)
#         test_dataset = preprocess_data(test_data, tokenizer)
#
#         # Prepare DataLoader with dynamic padding
#         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#         train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=8,
#             shuffle=True,
#             collate_fn=data_collator
#         )
#         test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=8,
#             shuffle=False,
#             collate_fn=data_collator
#         )
#
#         # Initialize the model
#         model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         # Define the optimizer
#         optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#
#         # Training loop
#         model.train()
#         for epoch in range(3):  # Adjust the number of epochs as needed
#             total_loss = 0
#             for batch in train_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 optimizer.zero_grad()
#                 outputs = model(**inputs, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#
#                 total_loss += loss.item()
#
#             print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
#
#         # Save the trained model
#         model_save_path = "./models/defect_detection_model"
#         model.save_pretrained(model_save_path)
#         tokenizer.save_pretrained(model_save_path)
#
#         print(f"Model training complete and saved to '{model_save_path}'.")
#
#         # Evaluate the model (Optional)
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch in test_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 outputs = model(**inputs)
#                 predictions = torch.argmax(outputs.logits, dim=-1)
#                 correct += (predictions == labels).sum().item()
#                 total += labels.size(0)
#
#         print(f"Test Accuracy: {correct / total:.2%}")
#
#     except ValueError as ve:
#         print(f"Validation Error: {ve}")
#     except Exception as e:
#         print(f"An error occurred during training: {e}")
#
# if __name__ == "__main__":
#     # Ensure the models directory exists
#     os.makedirs("./models", exist_ok=True)
#     train_model()



# import os
# import torch
# import pandas as pd
# from datasets import load_dataset, Dataset
# from torch.utils.data import DataLoader
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
#
# def validate_dataset(dataset):
#     """
#     Validate that the dataset contains the 'func' field and all entries are valid strings.
#
#     Args:
#         dataset (Dataset): The loaded dataset.
#
#     Raises:
#         ValueError: If invalid entries are found in the dataset.
#     """
#     for example in dataset:
#         if "func" not in example or not isinstance(example["func"], str):
#             raise ValueError(f"Invalid entry found: {example}")
#
# def combine_datasets(default_path, custom_path):
#     """
#     Combine default and custom datasets into a unified dataset.
#
#     Args:
#         default_path (str): Path to the default dataset JSONL file.
#         custom_path (str): Path to the custom dataset JSONL file.
#
#     Returns:
#         Dataset: Combined dataset.
#     """
#     # Load default dataset
#     default_train = pd.read_json(default_path + "/train.jsonl", lines=True)
#     default_test = pd.read_json(default_path + "/test.jsonl", lines=True)
#
#     # Load custom dataset
#     custom_train = pd.read_json(custom_path + "/custom_train.jsonl", lines=True)
#     custom_test = pd.read_json(custom_path + "/custom_test.jsonl", lines=True)
#
#     # Combine datasets
#     train_data = pd.concat([default_train, custom_train], ignore_index=True)
#     test_data = pd.concat([default_test, custom_test], ignore_index=True)
#
#     # Shuffle the combined datasets
#     train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
#     test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     return train_data, test_data
#
# def preprocess_data(df, tokenizer):
#     """
#     Tokenize and preprocess the dataset for model training.
#
#     Args:
#         df (pd.DataFrame): The dataset as a pandas DataFrame.
#         tokenizer (RobertaTokenizer): The tokenizer.
#
#     Returns:
#         Dataset: Tokenized dataset.
#     """
#     # Remove invalid rows
#     df = df[df["func"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)
#
#     # Convert DataFrame to Hugging Face Dataset
#     dataset = Dataset.from_pandas(df)
#
#     def tokenize_function(examples):
#         return tokenizer(
#             examples["func"],
#             truncation=True,
#             padding="max_length",
#             max_length=512
#         )
#
#     # Tokenize dataset
#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     return tokenized_dataset
#
# def train_model():
#     """
#     Train a defect detection model using the unified dataset.
#     """
#     try:
#         # Paths to datasets
#         default_path = "dataset/processed_data"
#         custom_path = "dataset/processed_data"
#
#         # Combine datasets
#         train_data, test_data = combine_datasets(default_path, custom_path)
#
#         # Initialize the tokenizer
#         tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#
#         # Tokenize datasets
#         train_dataset = preprocess_data(train_data, tokenizer)
#         test_dataset = preprocess_data(test_data, tokenizer)
#
#         # Prepare DataLoader with dynamic padding
#         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#         train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=8,
#             shuffle=True,
#             collate_fn=data_collator
#         )
#         test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=8,
#             shuffle=False,
#             collate_fn=data_collator
#         )
#
#         # Initialize the model
#         model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         # Define the optimizer
#         optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#
#         # Training loop
#         model.train()
#         for epoch in range(3):  # Adjust the number of epochs as needed
#             total_loss = 0
#             for batch in train_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 optimizer.zero_grad()
#                 outputs = model(**inputs, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#
#                 total_loss += loss.item()
#
#             print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
#
#         # Save the trained model
#         model_save_path = "./models/defect_detection_model"
#         model.save_pretrained(model_save_path)
#         tokenizer.save_pretrained(model_save_path)
#
#         print(f"Model training complete and saved to '{model_save_path}'.")
#
#         # Evaluate the model (Optional)
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch in test_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 outputs = model(**inputs)
#                 predictions = torch.argmax(outputs.logits, dim=-1)
#                 correct += (predictions == labels).sum().item()
#                 total += labels.size(0)
#
#         print(f"Test Accuracy: {correct / total:.2%}")
#
#     except ValueError as ve:
#         print(f"Validation Error: {ve}")
#     except Exception as e:
#         print(f"An error occurred during training: {e}")
#
# if __name__ == "__main__":
#     # Ensure the models directory exists
#     os.makedirs("./models", exist_ok=True)
#     train_model()





# import os
# import torch
# import pandas as pd
# from datasets import load_dataset, Dataset
# from torch.utils.data import DataLoader
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
#
# def validate_dataset(dataset):
#     """
#     Validate that the dataset contains the 'func' field and all entries are valid strings.
#
#     Args:
#         dataset (Dataset): The loaded dataset.
#
#     Raises:
#         ValueError: If invalid entries are found in the dataset.
#     """
#     for example in dataset:
#         if "func" not in example or not isinstance(example["func"], str):
#             raise ValueError(f"Invalid entry found: {example}")
#
# def combine_datasets(default_path, custom_path):
#     """
#     Combine default and custom datasets into a unified dataset.
#
#     Args:
#         default_path (str): Path to the default dataset JSONL file.
#         custom_path (str): Path to the custom dataset JSONL file.
#
#     Returns:
#         pd.DataFrame: Combined dataset as a pandas DataFrame.
#     """
#     # Load default dataset
#     default_train = pd.read_json(default_path + "/train.jsonl", lines=True)
#     default_test = pd.read_json(default_path + "/test.jsonl", lines=True)
#
#     # Load custom dataset
#     custom_train = pd.read_json(custom_path + "/custom_train.jsonl", lines=True)
#     custom_test = pd.read_json(custom_path + "/custom_test.jsonl", lines=True)
#
#     # Combine datasets
#     train_data = pd.concat([default_train, custom_train], ignore_index=True)
#     test_data = pd.concat([default_test, custom_test], ignore_index=True)
#
#     # Shuffle the combined datasets
#     train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
#     test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     return train_data, test_data
#
# def preprocess_data(df, tokenizer):
#     """
#     Tokenize and preprocess the dataset for model training.
#
#     Args:
#         df (pd.DataFrame): The dataset as a pandas DataFrame.
#         tokenizer (RobertaTokenizer): The tokenizer.
#
#     Returns:
#         Dataset: Tokenized dataset.
#     """
#     # Filter and clean invalid entries
#     df = df[df["func"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)
#
#     # Convert DataFrame to Hugging Face Dataset
#     dataset = Dataset.from_pandas(df)
#
#     def tokenize_function(examples):
#         return tokenizer(
#             examples["func"],
#             truncation=True,
#             padding="max_length",
#             max_length=512
#         )
#
#     # Tokenize dataset
#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     return tokenized_dataset
#
# def train_model():
#     """
#     Train a defect detection model using the unified dataset.
#     """
#     try:
#         # Paths to datasets
#         default_path = "dataset/processed_data"
#         custom_path = "dataset/processed_data"
#
#         # Combine datasets
#         train_data, test_data = combine_datasets(default_path, custom_path)
#
#         # Initialize the tokenizer
#         tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
#
#         # Tokenize datasets
#         train_dataset = preprocess_data(train_data, tokenizer)
#         test_dataset = preprocess_data(test_data, tokenizer)
#
#         # Prepare DataLoader with dynamic padding
#         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#         train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=8,
#             shuffle=True,
#             collate_fn=data_collator
#         )
#         test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=8,
#             shuffle=False,
#             collate_fn=data_collator
#         )
#
#         # Initialize the model for fine-tuning
#         model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         # Define the optimizer
#         optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#
#         # Training loop
#         model.train()
#         for epoch in range(3):  # Adjust the number of epochs as needed
#             total_loss = 0
#             for batch in train_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 optimizer.zero_grad()
#                 outputs = model(**inputs, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#
#                 total_loss += loss.item()
#
#             print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
#
#         # Save the fine-tuned model
#         model_save_path = "./models/defect_detection_model"
#         model.save_pretrained(model_save_path)
#         tokenizer.save_pretrained(model_save_path)
#
#         print(f"Model training complete and saved to '{model_save_path}'.")
#
#         # Evaluate the model (Optional)
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch in test_dataloader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 labels = batch["labels"].to(device)
#
#                 outputs = model(**inputs)
#                 predictions = torch.argmax(outputs.logits, dim=-1)
#                 correct += (predictions == labels).sum().item()
#                 total += labels.size(0)
#
#         print(f"Test Accuracy: {correct / total:.2%}")
#
#     except ValueError as ve:
#         print(f"Validation Error: {ve}")
#     except Exception as e:
#         print(f"An error occurred during training: {e}")
#
# if __name__ == "__main__":
#     # Ensure the models directory exists
#     os.makedirs("./models", exist_ok=True)
#     train_model()






import os
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding


def preprocess_data(df, tokenizer):
    """
    Tokenize and preprocess the dataset for model training.

    Args:
        df (pd.DataFrame): The dataset as a pandas DataFrame.
        tokenizer (RobertaTokenizer): The tokenizer.

    Returns:
        Dataset: Tokenized dataset.
    """
    # Filter valid strings in the "func" field
    df = df[df["func"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)].reset_index(drop=True)

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["func"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": examples["label"]
        }

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def inspect_batch(batch):
    """
    Inspect the batch for debugging purposes.

    Args:
        batch (dict): A batch of data from the DataLoader.
    """
    print("Inspecting batch...")
    print("Keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)


def train_custom_dataset_only():
    """
    Train a defect detection model using only the custom dataset.
    """
    try:
        # Path to custom dataset
        custom_path = "dataset/processed_data"

        # Load custom dataset
        train_data = pd.read_json(custom_path + "/custom_train.jsonl", lines=True)
        test_data = pd.read_json(custom_path + "/custom_test.jsonl", lines=True)

        # Initialize the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

        # Tokenize datasets
        print("Tokenizing training data...")
        train_dataset = preprocess_data(train_data, tokenizer)
        print("Training dataset tokenized successfully!")

        print("Tokenizing test data...")
        test_dataset = preprocess_data(test_data, tokenizer)
        print("Test dataset tokenized successfully!")

        # Prepare DataLoader with dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=data_collator
        )

        # Inspect one batch for debugging
        print("Inspecting one batch from DataLoader...")
        for batch in train_dataloader:
            inspect_batch(batch)
            break

        # Initialize the model
        model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # Training loop
        print("Starting training...")
        model.train()
        for epoch in range(3):  # Adjust the number of epochs as needed
            total_loss = 0
            for batch in train_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

        # Save the fine-tuned model
        model_save_path = "./models/custom_defect_detection_model"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        print(f"Model training complete and saved to '{model_save_path}'.")

        # Evaluate the model (Optional)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        print(f"Test Accuracy: {correct / total:.2%}")

    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except Exception as e:
        print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("./models", exist_ok=True)

    print("Training using only the custom dataset...")
    train_custom_dataset_only()

