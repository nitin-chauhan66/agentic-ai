json_output_text_chunks_path = "text_chunks.json"
json_output_image_chunks_path = "image_chunks.json"
json_output_table_chunks_path = "table_chunks.json"

def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index to create
    """
    # Delete the index if it exists (to ensure proper mapping)
    if client.indices.exists(index=index_name):
        print(
            f"Deleting existing index '{index_name}' to recreate with proper mappings..."
        )
        client.indices.delete(index=index_name)

    # Get dimension from a sample embedding
    from helper import get_embedding

    sample_embedding = get_embedding("Sample text for dimension detection")
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")

    # Define mappings with vector field for embeddings
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
                "base64_image": {"type": "binary", "doc_values": False, "index": False},
                "table_html": {"type": "text", "index": False},
                "metadata": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "caption": {"type": "text"},
                        "image_text": {"type": "text"},
                    }
                },
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",  # Use cosine similarity for embeddings
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def prepare_chunks_for_ingestion(chunks):
    """
    Prepare chunks for ingestion by adding embeddings and token counts.

    Args:
        chunks: List of chunks to prepare

    Returns:
        List of prepared chunks ready for ingestion
    """
    from helper import get_embedding
    from tqdm import tqdm

    prepared_chunks = []

    for i, chunk in tqdm(enumerate(chunks)):
        try:
            # Skip chunks without content
            if not chunk.get("content"):
                continue

            # Compute embedding
            embedding = get_embedding(chunk["content"])

            # Create document for ingestion
            ingestion_doc = {
                "content": chunk["content"],
                "content_type": chunk.get("content_type", "text"),
                "embedding": embedding,
                "metadata": {
                    "filename": chunk.get("filename", ""),
                    "caption": chunk.get("caption", ""),
                    "image_text": chunk.get("image_text", ""),
                },
            }

            # Add image-specific data if available
            if chunk.get("content_type") == "image" and "base64_image" in chunk:
                ingestion_doc["base64_image"] = chunk["base64_image"]

            # Add table-specific data if available
            if chunk.get("content_type") == "table" and "table_as_html" in chunk:
                ingestion_doc["table_html"] = chunk["table_as_html"]

            prepared_chunks.append(ingestion_doc)

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Prepared {i+1}/{len(chunks)} chunks")

        except Exception as e:
            print(f"Error preparing chunk: {str(e)}")

    print(f"Successfully prepared {len(prepared_chunks)} chunks for ingestion")
    return prepared_chunks

def ingest_chunks_into_opensearch(client, index_name, chunks):
    """
    Ingest prepared chunks into OpenSearch.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index
        chunks: Prepared chunks with embeddings and token counts

    Returns:
        Number of successfully ingested documents
    """
    # Track successful and failed operations
    successful = 0
    failed = 0

    # Use bulk API for better performance
    from opensearchpy import helpers

    # Prepare bulk operations
    operations = []
    for i, chunk in enumerate(chunks):
        operations.append({'_index': index_name, '_source': chunk})

        # Process in batches of 100
        if (i + 1) % 100 == 0 or i == len(chunks) - 1:
            try:
                success, failed_items = helpers.bulk(client, operations, stats_only=True)
                successful += success
                failed += len(operations) - success
                operations = []  # Reset for next batch
                print(f"Ingested {successful} chunks so far ({failed} failed)")
            except Exception as e:
                print(f"Bulk ingestion error: {str(e)}")
                failed += len(operations)
                operations = []  # Reset after error

    # Final bulk operation if any remaining
    if operations:
        try:
            success, failed_items = bulk(client, operations, stats_only=True)
            successful += success
            failed += len(operations) - success
        except Exception as e:
            print(f"Bulk ingestion error: {str(e)}")
            failed += len(operations)

    print(f"Ingestion complete: {successful} successful, {failed} failed")
    return successful


def ingest_all_content_into_opensearch(image_chunks = None, table_chunks = None, text_chunks = None, index_name = "localrag"):
    from helper import get_opensearch_client

    client = get_opensearch_client("localhost", 9200)

    create_index_if_not_exists(client, index_name)
    if image_chunks:
        ingest_chunks_into_opensearch(
            client = client, 
            chunks = image_chunks,
            index_name = index_name)
    if table_chunks:
        ingest_chunks_into_opensearch(
            client = client,
            chunks = table_chunks,
            index_name = index_name)
    if text_chunks:
        ingest_chunks_into_opensearch(client = client,
        index_name=index_name,
        chunks=text_chunks)

if __name__ == "__main__":
    from helper import *
    processed_image_chunks = load_chunks_from_cache_file(json_output_image_chunks_path)
    image_chunks = prepare_chunks_for_ingestion(processed_image_chunks)

    processed_table_chunks = load_chunks_from_cache_file(json_output_table_chunks_path)
    table_chunks = prepare_chunks_for_ingestion(processed_table_chunks)

    processed_text_chunks = load_chunks_from_cache_file(json_output_text_chunks_path)
    text_chunks = prepare_chunks_for_ingestion(processed_text_chunks)
    ingest_all_content_into_opensearch(
        text_chunks = text_chunks,
        image_chunks = image_chunks,
        table_chunks = table_chunks,
        index_name="localrag")




