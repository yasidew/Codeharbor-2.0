import json
import random
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
with open("dataset/datasets/singleton_data.json", "r") as file:
    dataset = json.load(file)

# Step 2: Shuffle the dataset for randomness
random.shuffle(dataset)

# Step 3: Split dataset into input/output and type
inputs = [data["input"] for data in dataset]
outputs = [data["output"] for data in dataset]
types = [data["type"] for data in dataset]

# Step 4: Train-test-validation split
X_train, X_temp, y_train, y_temp, t_train, t_temp = train_test_split(
    inputs, outputs, types, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test, t_val, t_test = train_test_split(
    X_temp, y_temp, t_temp, test_size=0.5, random_state=42
)

# Step 5: Save the splits into separate files
data_splits = {
    "train": {"input": X_train, "output": y_train, "type": t_train},
    "validation": {"input": X_val, "output": y_val, "type": t_val},
    "test": {"input": X_test, "output": y_test, "type": t_test},
}

with open("dataset/processed_singleton_data.json", "w") as out_file:
    json.dump(data_splits, out_file, indent=4)

print("Dataset preprocessing complete. File saved as processed_singleton_data.json")
