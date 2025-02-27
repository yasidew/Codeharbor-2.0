# import os
# import json
# from mydatasets import load_dataset
#
# def preprocess_dataset(dataset_name="google/code_x_glue_cc_defect_detection", output_dir="dataset/processed_data"):
#     """
#     Preprocess a dataset and save it to disk in JSONL format.
#
#     Args:
#         dataset_name (str): The name of the dataset to preprocess.
#         output_dir (str): The directory where the processed data will be saved.
#     """
#     try:
#         # Load dataset
#         dataset = load_dataset(dataset_name)
#         train_data = dataset["train"]
#         test_data = dataset["test"]
#
#         # Ensure the output directory exists
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Save train and test splits as JSONL
#         train_file = os.path.join(output_dir, "train.jsonl")
#         test_file = os.path.join(output_dir, "test.jsonl")
#
#         with open(train_file, "w") as f:
#             for example in train_data:
#                 f.write(json.dumps(example) + "\n")  # Save each example as a JSON line
#
#         with open(test_file, "w") as f:
#             for example in test_data:
#                 f.write(json.dumps(example) + "\n")
#
#         print(f"Dataset '{dataset_name}' preprocessing complete!")
#         print(f"Train data saved at: {train_file}")
#         print(f"Test data saved at: {test_file}")
#
#     except Exception as e:
#         print(f"Failed to preprocess dataset {dataset_name}: {e}")
#
# if __name__ == "__main__":
#     preprocess_dataset()


import os
import json
import pandas as pd
# from mydatasets import load_dataset
from datasets import load_dataset
from io import StringIO


def preprocess_dataset(dataset_name="google/code_x_glue_cc_defect_detection", custom_dataset_path=None,
                       output_dir="dataset/processed_data"):
    """
    Preprocess a dataset and save it to disk in JSONL format.

    Args:
        dataset_name (str): The name of the dataset to preprocess.
        custom_dataset_path (str): Path to the custom dataset in JSONL format.
        output_dir (str): The directory where the processed data will be saved.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess default dataset
        if dataset_name:
            dataset = load_dataset(dataset_name)
            train_data = dataset["train"]
            test_data = dataset["test"]

            # Save train and test splits as JSONL
            with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
                for example in train_data:
                    f.write(json.dumps(example) + "\n")

            with open(os.path.join(output_dir, "test.jsonl"), "w") as f:
                for example in test_data:
                    f.write(json.dumps(example) + "\n")

            print(f"Default dataset '{dataset_name}' preprocessing complete!")

        # Preprocess custom dataset
        if custom_dataset_path:
            with open(custom_dataset_path, 'r') as file:
                data = file.read()
            df = pd.read_json(StringIO(data), lines=True)
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)

            train_df.to_json(os.path.join(output_dir, "custom_train.jsonl"), orient="records", lines=True)
            test_df.to_json(os.path.join(output_dir, "custom_test.jsonl"), orient="records", lines=True)

            print(f"Custom dataset '{custom_dataset_path}' preprocessing complete!")

    except Exception as e:
        print(f"Failed to preprocess datasets: {e}")

if __name__ == "__main__":
    preprocess_dataset(custom_dataset_path="dataset/raw_data/custom_dataset.jsonl")
