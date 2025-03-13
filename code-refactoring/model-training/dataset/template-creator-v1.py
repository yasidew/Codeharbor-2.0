import json

def convert_to_template_format(input_file, output_file):
    # Load the existing dataset
    with open(input_file, "r") as file:
        data = json.load(file)

    # Convert each entry to the new template format
    converted_data = []
    for entry in data:
        # Extract the class name from the input
        name = entry["input"].split("class")[1].split("{")[0].strip()

        # Replace the actual class name with '{name}', ensuring proper curly braces formatting
        output_template = (
            entry["output"]
            .replace(name, "{name}")
            .replace("{", "{{")
            .replace("}", "}}")
            .replace("{{name}}", "{name}")
        )

        # Handle additional metadata fields
        converted_entry = {
            "type": entry.get("type", "UnknownType"),  # Fallback for missing type
            "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
            "output_template": output_template,
            "complexity": entry.get("complexity", "Unknown"),  # Default to 'Unknown'
            "language": entry.get("language", "Java"),
            "context": entry.get("context", "Not Specified"),
            "edge_cases": entry.get("edge_cases", []),
            "dependencies": entry.get("dependencies", []),
            "performance_notes": entry.get("performance_notes", "None"),
            "real_world_usage": entry.get("real_world_usage", "None"),
            "testing_notes": entry.get("testing_notes", "None"),
            "comments": entry.get("comments", "No Comments"),
            "source": entry.get("source", "Unknown")
        }

        # Append the formatted entry to the new dataset
        converted_data.append(converted_entry)

    # Save the converted dataset to a new file
    with open(output_file, "w") as file:
        json.dump(converted_data, file, indent=4)

    print(f"Converted dataset saved to {output_file}")

# Define input and output file paths
input_file = "datasets/observer_pattern_data.json"
output_file = "converted_observer_pattern_data.json"

# Run the conversion
convert_to_template_format(input_file, output_file)
