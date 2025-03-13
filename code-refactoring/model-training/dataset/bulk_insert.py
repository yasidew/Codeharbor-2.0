import json
import psycopg2

# Database connection details
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "5432"
SCHEMA = "code_harbor"  # Ensure we are using the correct schema
DATASET_FILE = "datasets/observer_pattern_data.json"  # Your dataset file

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

# Ensure correct schema usage
cursor.execute(f"SET search_path TO {SCHEMA}, public;")

# Read the JSON dataset
with open(DATASET_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# SQL Insert Query
insert_query = """
INSERT INTO design_pattern_examples 
(type, input_code, output_code, complexity, language, context, edge_cases, dependencies, performance_notes, real_world_usage, testing_notes, comments, source) 
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s);
"""

# Track inserted records
total_records = len(dataset)
batch_size = 500  # Adjust batch size for efficiency
batch = []

for index, entry in enumerate(dataset):
    batch.append((
        entry["type"],
        entry["input"],
        entry["output"],
        entry["complexity"],
        entry["language"],
        entry["context"],
        json.dumps(entry["edge_cases"]),
        json.dumps(entry["dependencies"]),
        entry["performance_notes"],
        entry["real_world_usage"],
        entry["testing_notes"],
        entry["comments"],
        entry["source"]
    ))

    # Insert in batches
    if len(batch) >= batch_size or index == total_records - 1:
        cursor.executemany(insert_query, batch)
        conn.commit()
        batch.clear()  # Clear the batch after insertion
        print(f"Inserted {index + 1}/{total_records} records...")

# Close connection
cursor.close()
conn.close()

print(f"✅ Successfully inserted {total_records} records into PostgreSQL!")
