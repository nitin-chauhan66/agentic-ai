from unstructured.staging.base import elements_from_dicts
import json
import os
from tqdm import tqdm

json_output_path = "parsed_data.json"

def load_elements_from_file(json_path: str):
    """Load parsed elements from a JSON file with progress bar."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return elements_from_dicts(data)

# Example usage
if __name__ == "__main__":
    if os.path.exists(json_output_path):
        print(f"[INFO] Parsed data already exists at '{json_output_path}'.")
        elements = load_elements_from_file(json_output_path)
        print(elements[10:15])
        print(len(elements))