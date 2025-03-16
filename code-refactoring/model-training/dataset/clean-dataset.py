import json

# Load the dataset
input_file = "datasets/singleton_data.json"
output_file = "cleaned_singleton_data.json"

def load_dataset(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def save_cleaned_dataset(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

def find_and_remove_duplicates(data):
    seen = {}
    cleaned_data = []
    duplicate_records = []

    for idx, record in enumerate(data):
        unique_key = (record["input"], record["output"])  # Combine input and output to form a unique identifier
        if unique_key in seen:
            duplicate_records.append((seen[unique_key] + 1, idx + 1))  # Record line numbers for duplicates
        else:
            seen[unique_key] = idx  # Keep track of the first occurrence
            cleaned_data.append(record)  # Add unique record to cleaned data

    return cleaned_data, duplicate_records

def main():
    # Load the original dataset
    dataset = load_dataset(input_file)

    # Find duplicates and clean the dataset
    cleaned_dataset, duplicate_records = find_and_remove_duplicates(dataset)

    # Log duplicates to the console
    if duplicate_records:
        print("Duplicate Entries Found:")
        for original_line, duplicate_line in duplicate_records:
            print(f"Dataset starting in line {original_line} is duplicated with dataset starting in line {duplicate_line}")
    else:
        print("No duplicates found.")

    # Save the cleaned dataset to a new file
    save_cleaned_dataset(output_file, cleaned_dataset)
    print(f"Cleaned dataset saved to {output_file}")

if __name__ == "__main__":
    main()
