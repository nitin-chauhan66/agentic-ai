from google import genai
from google.genai import types
import base64
from dotenv import load_dotenv
from unstructured.documents.elements import Table, Image, FigureCaption, CompositeElement
from parser import *

load_dotenv()
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

json_output_text_chunks_path = "text_chunks.json"
json_output_image_chunks_path = "image_chunks.json"
json_output_table_chunks_path = "table_chunks.json"

def process_image_chunks(chunks):
    processed_data = []
    for i, ele in tqdm(enumerate(chunks)):
      if i+1<len(chunks) and isinstance(chunks[i+1],FigureCaption):
        caption = chunks[i+1].text
      else:
        caption = "No Caption"
      if isinstance(ele,Image):
        image_data = {
            "caption":caption,
            "content":ele.text if ele.text else "" ,
            "page_number":ele.metadata.page_number,
            "image_base64":ele.metadata.image_base64,
            "content_type":"image/jpeg",
            "file_name":ele.metadata.filename
        }
        image_bytes = base64.b64decode(image_data["image_base64"])
        prompt = (
            f"Analyze the following image and provide a detailed description. "
            f"Caption: '{image_data['caption']}'. "
            f"Content: '{image_data['content']}'. "
            f"Please focus on the visual elements and context of the image, delivering a thorough and insightful description without any additional commentary."
        )
        response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                  types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                  ),
                  prompt
                ]
              )
        image_data["content"] = response.text
        processed_data.append(image_data)
    return processed_data

def process_table_chunks(chunks):
    processed_data = []
    for i, ele in tqdm(enumerate(chunks)):
      if isinstance(ele,Table):
        table_data = {
            "content":ele.text if ele.text else "",
            "table_text":ele.text if ele.text else "" ,
            "table_as_html":ele.metadata.text_as_html,
            "page_number":ele.metadata.page_number,
            "content_type":"table",
            "file_name":ele.metadata.filename
        }

        prompt = (
            f"Analyze the following table and provide a detailed description. "
            f"Table as HTML: '{table_data['table_as_html']}'. "
            f"Table Text: '{table_data['table_text']}'. "
            f"Please focus on the structure, content, and context of the table, delivering a thorough and insightful description without any additional commentary."
        )
        try:
          url = "http://localhost:11434/api/generate"
          data = {
            "model": "deepseek-r1:1.5b",
            "prompt": prompt,
            "max_tokens": 1000,
            "stream": False,
            "temperature": 0.2,
          }
          response = requests.post(url, json=data)
          response.raise_for_status()

          table_data["content"] = response.json().get(
              "response", "No response from model"
          )
        except Exception as e:
          encountered_errors.append(
          {
            "error": str(e),
            "error_message": "Error generating description with Ollama.",
          }
          )
        processed_data.append(table_data)
    return processed_data

def create_semantic_chunks(chunks):
    """
    Create semantic chunks from a PDF document based on title structure.

    Args:
        chunks: List of document elements from unstructured.partition_pdf

    Returns:
        List of semantic chunks
    """
    from unstructured.documents.elements import CompositeElement

    # Convert to more usable format
    processed_chunks = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, CompositeElement):
            chunk_data = {
                "content": chunk.text,
                "content_type": "text",
                "filename": (
                    chunk.metadata.filename if hasattr(chunk, "metadata") else ""
                ),
            }
            processed_chunks.append(chunk_data)

    print(f"Created {len(processed_chunks)} semantic chunks from document")
    return processed_chunks

def save_processed_chunks_to_file(processed_chunks, json_path: str):
    """Serialize parsed elements to a JSON file with progress bar."""
    print("[INFO] Saving processed chunks to JSON...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Parsed data saved to '{json_path}'.")



# Example usage
if __name__ == "__main__":
  elements = get_parsed_elements()
  raw_text_chunks = chunk_by_title(elements)

#   # processed_image_chunks = process_image_chunks(elements)
#   # save_processed_chunks_to_file(processed_image_chunks, json_output_image_chunks_path)

  processed_table_chunks = process_table_chunks(elements)
  save_processed_chunks_to_file(processed_table_chunks, json_output_table_chunks_path)


  # processed_text_chunks = create_semantic_chunks(raw_text_chunks)
  # save_processed_chunks_to_file(processed_text_chunks, json_output_text_chunks_path)
