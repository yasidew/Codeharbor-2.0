import boto3
import os
import zipfile

def download_model_from_s3(bucket_name, s3_key, zip_path, extract_to):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "eu-north-1")
    )

    print(f"📥 Downloading {s3_key} from {bucket_name} to {zip_path}")
    s3.download_file(bucket_name, s3_key, zip_path)
    print("✅ Download complete.")

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"📦 Extracting to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1. Refactoring Model
    download_model_from_s3(
        bucket_name="codeharbor-models",
        s3_key="refactoring_model.zip",
        zip_path=os.path.join(project_root, "refactoring_model.zip"),
        extract_to=os.path.join(project_root, "refactoring_model")
    )

    # 2. Custom Seq2Seq Model
    download_model_from_s3(
        bucket_name="codeharbor-models",
        s3_key="custom_seq2seq_model.zip",
        zip_path=os.path.join(project_root, "custom_seq2seq_model.zip"),
        extract_to=os.path.join(project_root, "custom_seq2seq_model")
    )

    # 3. Java Seq2Seq Model
    download_model_from_s3(
        bucket_name="codeharbor-models",
        s3_key="java_seq2seq_model.zip",
        zip_path=os.path.join(project_root, "java_seq2seq_model.zip"),
        extract_to=os.path.join(project_root, "java_seq2seq_model")
    )
