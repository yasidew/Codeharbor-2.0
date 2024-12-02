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
        output_template = entry["output"].replace(name, "{name}").replace("{", "{{").replace("}", "}}").replace("{{name}}", "{name}")

        # Append the new formatted entry
        converted_data.append({
            "type": entry["type"],
            "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
            "output_template": output_template
        })

    # Save the converted dataset to a new file
    with open(output_file, "w") as file:
        json.dump(converted_data, file, indent=4)

    print(f"Converted dataset saved to {output_file}")

# Define input and output file paths
input_file = "singleton_data-temporary.json"
output_file = "converted_singleton_data.json"

# Run the conversion
convert_to_template_format(input_file, output_file)
