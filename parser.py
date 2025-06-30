import os
import json
import requests
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_from_dicts
from unstructured.chunking.title import chunk_by_title
from dotenv import load_dotenv


load_dotenv()
# File paths
pdf_file_path = "sample.pdf"
json_output_path = "parsed_data.json"
download_url = "https://www.accenture.com/content/dam/accenture/final/capabilities/corporate-functions/marketing-and-communications/marketing---communications/document/Accenture-Fiscal-2023-Annual-Report.pdf"


def download_pdf(url: str, path: str):
    """Download the PDF from URL if it doesn't already exist."""
    if not os.path.exists(path):
        print(f"[INFO] Downloading PDF to '{path}'...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            print("[INFO] Download successful.")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download PDF: {e}")
            raise
    else:
        print(f"[INFO] PDF already exists at '{path}'.")


def save_elements_to_file(elements, json_path: str):
    """Serialize parsed elements to a JSON file with progress bar."""
    print("[INFO] Saving parsed elements to JSON...")
    serializable = [el.to_dict() for el in tqdm(elements, desc="Saving Elements")]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Parsed data saved to '{json_path}'.")


def load_elements_from_file(json_path: str):
    """Load parsed elements from a JSON file with progress bar."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        elements = elements_from_dicts(data)
    return elements


def parse_pdf_to_elements(pdf_path: str):
    """Parse the PDF using Unstructured into a list of elements."""
    print("[INFO] Parsing PDF...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_to_payload=True,
        extract_image_block_types=["Image", "Table", "Figure"],
        infer_table_structure=True,
    )
    print(f"[INFO] Parsed {len(elements)} elements.")
    return elements

# Parse PDF into text elements
def parse_pdf_to_text_elements(pdf_path: str):
    print("Parsing PDF...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        chunking_strategy="by_title",
        split_pdf_page=True,
        split_pdf_concurrency_level=15,
    )
    return elements

def get_parsed_text_elements():
    """
    Public interface:
    - Downloads the PDF if needed
    - Loads parsed data if available
    - Otherwise parses and saves the result
    """
    download_pdf(download_url, pdf_file_path)

    elements = parse_pdf_to_text_elements(pdf_file_path)
    # save_elements_to_file(elements, json_output_path)
    return elements

def get_parsed_elements():
    """
    Public interface:
    - Downloads the PDF if needed
    - Loads parsed data if available
    - Otherwise parses and saves the result
    """
    download_pdf(download_url, pdf_file_path)

    if os.path.exists(json_output_path):
        print(f"[INFO] Parsed data already exists at '{json_output_path}'.")
        return load_elements_from_file(json_output_path)

    elements = parse_pdf_to_elements(pdf_file_path)
    save_elements_to_file(elements, json_output_path)
    return elements


# Example usage
if __name__ == "__main__":
    elements = get_parsed_elements()
    raw_text_chunks = chunk_by_title(elements)
    print(raw_text_chunks[3].to_dict())
    print(f"[INFO] Total text chunk elements: {len(raw_text_chunks)}")
    print(f"[INFO] Total elements returned: {len(elements)}")
