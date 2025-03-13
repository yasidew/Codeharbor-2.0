import json
import random
from sklearn.model_selection import train_test_split

# Step 1: Load the New Dataset
with open("refactoring_data.json", "r") as file:  # Updated dataset filename
    dataset = json.load(file)

# Step 2: Shuffle the Dataset for Randomization
random.shuffle(dataset)

# Step 3: Extract All Relevant Properties
inputs = [data["input"] for data in dataset]  # Raw Java code input
outputs = [data["output"] for data in dataset]  # Expected refactored output
types = [data["type"] for data in dataset]  # Design pattern type (e.g., Factory, Observer)
complexities = [data["complexity"] for data in dataset]  # Complexity level
languages = [data["language"] for data in dataset]  # Programming language
contexts = [data.get("context", "") for data in dataset]  # Context information
edge_cases = [data.get("edge_cases", []) for data in dataset]  # Edge cases list
dependencies = [data.get("dependencies", []) for data in dataset]  # Dependencies list
performance_notes = [data.get("performance_notes", "") for data in dataset]  # Performance-related notes
real_world_usage = [data.get("real_world_usage", "") for data in dataset]  # Real-world application
testing_notes = [data.get("testing_notes", "") for data in dataset]  # Testing information
comments = [data.get("comments", "") for data in dataset]  # Additional comments
sources = [data.get("source", "Unknown") for data in dataset]  # Source of inspiration

# Step 4: Split Dataset (Train 70%, Validation 15%, Test 15%)
X_train, X_temp, y_train, y_temp, t_train, t_temp, c_train, c_temp, e_train, e_temp, d_train, d_temp, p_train, p_temp, r_train, r_temp, test_train, test_temp, com_train, com_temp, s_train, s_temp = train_test_split(
    inputs, outputs, types, complexities, contexts, edge_cases, dependencies,
    performance_notes, real_world_usage, testing_notes, comments, sources,
    test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test, t_val, t_test, c_val, c_test, e_val, e_test, d_val, d_test, p_val, p_test, r_val, r_test, test_val, test_test, com_val, com_test, s_val, s_test = train_test_split(
    X_temp, y_temp, t_temp, c_temp, e_temp, d_temp, p_temp, r_temp, test_temp, com_temp, s_temp,
    test_size=0.5, random_state=42
)

# Step 5: Structure the Processed Data
data_splits = {
    "train": {
        "input": X_train,
        "output": y_train,
        "type": t_train,
        "complexity": c_train,
        "language": c_train,
        "context": c_train,
        "edge_cases": e_train,
        "dependencies": d_train,
        "performance_notes": p_train,
        "real_world_usage": r_train,
        "testing_notes": test_train,
        "comments": com_train,
        "source": s_train
    },
    "validation": {
        "input": X_val,
        "output": y_val,
        "type": t_val,
        "complexity": c_val,
        "language": c_val,
        "context": c_val,
        "edge_cases": e_val,
        "dependencies": d_val,
        "performance_notes": p_val,
        "real_world_usage": r_val,
        "testing_notes": test_val,
        "comments": com_val,
        "source": s_val
    },
    "test": {
        "input": X_test,
        "output": y_test,
        "type": t_test,
        "complexity": c_test,
        "language": c_test,
        "context": c_test,
        "edge_cases": e_test,
        "dependencies": d_test,
        "performance_notes": p_test,
        "real_world_usage": r_test,
        "testing_notes": test_test,
        "comments": com_test,
        "source": s_test
    }
}

# Step 6: Save the Processed Dataset
with open("processed_refactoring_data.json", "w") as out_file:
    json.dump(data_splits, out_file, indent=4)

print("✅ Dataset preprocessing complete. File saved as processed_refactoring_data.json")
