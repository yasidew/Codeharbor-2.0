from datasets import load_dataset

def load_defect_detection_dataset(dataset_name="google/code_x_glue_cc_defect_detection"):
    """
    Load a defect detection dataset using the Hugging Face `mydatasets` library.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        DatasetDict: The loaded dataset.
    """
    try:
        dataset = load_dataset(dataset_name)
        print(dataset)
        return dataset
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return None
