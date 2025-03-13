import json

# Metadata template for different Singleton types
metadata_template = {
    "CacheSingleton": {
        "complexity": "Intermediate",
        "language": "Java",
        "context": "Caching frequently accessed data with a size limit",
        "edge_cases": ["Eviction policy", "Thread-safety"],
        "dependencies": ["java.util.LinkedHashMap"],
        "performance_notes": "Designed to limit memory usage with least-recently-used eviction",
        "real_world_usage": "Used for caching in web applications or distributed systems",
        "testing_notes": "Test eviction logic and concurrent access handling",
        "comments": "Ensure thread-safety if accessed from multiple threads",
        "source": "Inspired by common caching mechanisms"
    },
    "LoggerSingleton": {
        "complexity": "Advanced",
        "language": "Java",
        "context": "Thread-safe logging in a multi-threaded application",
        "edge_cases": ["Thread-safety", "Lazy initialization"],
        "dependencies": ["java.io.FileWriter"],
        "performance_notes": "Avoids synchronization bottleneck with double-checked locking",
        "real_world_usage": "Used for logging events in a microservices architecture",
        "testing_notes": "Run multiple threads to simulate concurrent logging",
        "comments": "Ensure proper file handling to avoid resource leaks",
        "source": "Inspired by real-world logging frameworks"
    },
    # Add other types with their specific metadata
}

def add_metadata(input_file, output_file):
    # Load the existing dataset
    with open(input_file, "r") as file:
        data = json.load(file)

    # Add metadata to each entry
    enriched_data = []
    for entry in data:
        singleton_type = entry["type"]
        # Use the template for the type if available, else add generic metadata
        metadata = metadata_template.get(singleton_type, {
            "complexity": "Unknown",
            "language": "Java",
            "context": "No specific context provided",
            "edge_cases": [],
            "dependencies": [],
            "performance_notes": "No notes provided",
            "real_world_usage": "No usage details provided",
            "testing_notes": "No testing notes provided",
            "comments": "No comments available",
            "source": "Unknown source"
        })

        # Merge the existing entry with the metadata
        enriched_entry = {**entry, **metadata}
        enriched_data.append(enriched_entry)

    # Save the enriched dataset to a new file
    with open(output_file, "w") as file:
        json.dump(enriched_data, file, indent=4)

    print(f"Enriched dataset saved to {output_file}")

# Define input and output file paths
input_file = "datasets/singleton_data.json"  # Replace with your actual file name
output_file = "enriched_singleton_data.json"

# Run the enrichment process
add_metadata(input_file, output_file)
