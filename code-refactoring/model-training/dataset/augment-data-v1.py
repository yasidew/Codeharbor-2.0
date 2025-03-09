import json

# Patterns for design patterns with placeholders
patterns = [
    {
        "type": "Strategy",
        "input": "interface SortingStrategy {\n    void sort(int[] numbers);\n}\n\nclass BubbleSort implements SortingStrategy {\n    @Override\n    public void sort(int[] numbers) {\n        System.out.println(\"Sorting using Bubble Sort\");\n    }\n}\n\nclass QuickSort implements SortingStrategy {\n    @Override\n    public void sort(int[] numbers) {\n        System.out.println(\"Sorting using Quick Sort\");\n    }\n}",
        "output": "class Sorter {\n    private SortingStrategy sortingStrategy;\n\n    public void setSortingStrategy(SortingStrategy sortingStrategy) {\n        this.sortingStrategy = sortingStrategy;\n    }\n\n    public void executeSorting(int[] numbers) {\n        sortingStrategy.sort(numbers);\n    }\n}",
        "complexity": "Intermediate",
        "language": "Java",
        "context": "Switching between different sorting algorithms at runtime",
        "edge_cases": [
            "Handling large datasets",
            "Sorting already sorted data"
        ],
        "dependencies": [
            "None"
        ],
        "performance_notes": "Choosing the optimal algorithm improves efficiency",
        "real_world_usage": "Used in databases and search engines for sorting large datasets",
        "testing_notes": "Test with various input sizes and conditions",
        "comments": "Easily extendable to include new sorting algorithms",
        "source": "Inspired by sorting mechanisms in computer science"
    }
]

# Generate examples for each pattern
examples = []
unique_checker = set()  # To track unique (input, output) pairs

for pattern in patterns:
    for i in range(10):  # Generate 5 variations per pattern
        class_name = f"{pattern['type']}Example{i + 1}"
        generated_input = pattern["input_template"].format(name=class_name)
        generated_output = pattern["output_template"].format(name=class_name)

        # Check if the (input, output) pair is unique
        if (generated_input, generated_output) not in unique_checker:
            examples.append({
                "type": pattern["type"],
                "input": generated_input,
                "output": generated_output,
                "complexity": pattern["complexity"],
                "language": pattern["language"],
                "context": pattern["context"],
                "edge_cases": pattern["edge_cases"],
                "dependencies": pattern["dependencies"],
                "performance_notes": pattern["performance_notes"],
                "real_world_usage": pattern["real_world_usage"],
                "testing_notes": pattern["testing_notes"],
                "comments": pattern["comments"],
                "source": pattern["source"]
            })
            unique_checker.add((generated_input, generated_output))  # Mark as generated

# Save the dataset to a JSON file
dataset_filename = "augmented_strategy_method_data.json"
with open(dataset_filename, "w") as json_file:
    json.dump(examples, json_file, indent=4)

print(f"Dataset generated and saved to {dataset_filename}.")
