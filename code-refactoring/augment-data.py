import json

# Patterns for Singleton types with placeholders
patterns = [
    {
        "type": "ClusteredSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.locks.Lock;\nimport java.util.concurrent.locks.ReentrantLock;\n\npublic class {name} {{\n\n    private static volatile {name} instance;\n    private static final Lock lock = new ReentrantLock();\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            lock.lock();\n            try {{\n                if (instance == null) {{\n                    instance = new {name}();\n                    // Simulate cluster-wide initialization\n                    System.out.println(\"Cluster-wide Singleton instance created\");\n                }}\n            }} finally {{\n                lock.unlock();\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    }
]

# Generate examples for each pattern
examples = []
for pattern in patterns:
    for i in range(5):  # Generate 5 variations per pattern
        class_name = f"{pattern['type']}Example{i + 1}"
        examples.append({
            "type": pattern["type"],
            "input": pattern["input_template"].format(name=class_name),
            "output": pattern["output_template"].format(name=class_name)
        })

# Save the dataset to a JSON file
dataset_filename = "augmented_singleton_data.json"
with open(dataset_filename, "w") as json_file:
    json.dump(examples, json_file, indent=4)

print(f"Dataset generated and saved to {dataset_filename}.")

