import json
import random
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
with open("dataset/datasets/strategy_pattern_data.json", "r") as file:
    dataset = json.load(file)

# Step 2: Shuffle the dataset for randomness
random.shuffle(dataset)

# Step 3: Extract all attributes
inputs = [data["input"] for data in dataset]
outputs = [data["output"] for data in dataset]
types = [data["type"] for data in dataset]
complexities = [data.get("complexity", "Unknown") for data in dataset]
languages = [data.get("language", "Unknown") for data in dataset]
contexts = [data.get("context", "Unknown") for data in dataset]
edge_cases = [data.get("edge_cases", []) for data in dataset]
dependencies = [data.get("dependencies", []) for data in dataset]
performance_notes = [data.get("performance_notes", "None") for data in dataset]
real_world_usages = [data.get("real_world_usage", "None") for data in dataset]
testing_notes = [data.get("testing_notes", "None") for data in dataset]
comments = [data.get("comments", "None") for data in dataset]
sources = [data.get("source", "None") for data in dataset]

# Step 4: Train-test-validation split
X_train, X_temp, y_train, y_temp, t_train, t_temp, c_train, c_temp, l_train, l_temp, ctx_train, ctx_temp, ec_train, ec_temp, dep_train, dep_temp, pn_train, pn_temp, rwu_train, rwu_temp, tn_train, tn_temp, cm_train, cm_temp, src_train, src_temp = train_test_split(
    inputs, outputs, types, complexities, languages, contexts, edge_cases,
    dependencies, performance_notes, real_world_usages, testing_notes,
    comments, sources, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test, t_val, t_test, c_val, c_test, l_val, l_test, ctx_val, ctx_test, ec_val, ec_test, dep_val, dep_test, pn_val, pn_test, rwu_val, rwu_test, tn_val, tn_test, cm_val, cm_test, src_val, src_test = train_test_split(
    X_temp, y_temp, t_temp, c_temp, l_temp, ctx_temp, ec_temp, dep_temp,
    pn_temp, rwu_temp, tn_temp, cm_temp, src_temp, test_size=0.5, random_state=42
)

# Step 5: Save the splits into separate files
data_splits = {
    "train": {
        "input": X_train,
        "output": y_train,
        "type": t_train,
        "complexity": c_train,
        "language": l_train,
        "context": ctx_train,
        "edge_cases": ec_train,
        "dependencies": dep_train,
        "performance_notes": pn_train,
        "real_world_usage": rwu_train,
        "testing_notes": tn_train,
        "comments": cm_train,
        "source": src_train,
    },
    "validation": {
        "input": X_val,
        "output": y_val,
        "type": t_val,
        "complexity": c_val,
        "language": l_val,
        "context": ctx_val,
        "edge_cases": ec_val,
        "dependencies": dep_val,
        "performance_notes": pn_val,
        "real_world_usage": rwu_val,
        "testing_notes": tn_val,
        "comments": cm_val,
        "source": src_val,
    },
    "test": {
        "input": X_test,
        "output": y_test,
        "type": t_test,
        "complexity": c_test,
        "language": l_test,
        "context": ctx_test,
        "edge_cases": ec_test,
        "dependencies": dep_test,
        "performance_notes": pn_test,
        "real_world_usage": rwu_test,
        "testing_notes": tn_test,
        "comments": cm_test,
        "source": src_test,
    },
}

# Step 6: Save the processed dataset
output_file = "dataset/strategy_factory_method_data.json"
with open(output_file, "w") as out_file:
    json.dump(data_splits, out_file, indent=4)

print(f"Dataset preprocessing complete. File saved as {output_file}")
