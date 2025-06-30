from opensearchpy import OpenSearch
import requests
import json

def get_embedding(prompt,model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings/"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "model": model}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("embedding", [])
    else:
        raise Exception(
            f"Error fetching embedding: {response.status_code}, {response.text}"
        )


def get_opensearch_client(host, port):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if client.ping():
        print("Connected to OpenSearch!")
        info = client.info()
        print(f"Cluster name: {info['cluster_name']}")
        print(f"OpenSearch version: {info['version']['number']}")
    else:
        print("Connection failed!")
        raise ConnectionError("Failed to connect to OpenSearch.")
    return client

def load_chunks_from_cache_file(json_path: str):
    import os
    import json
    """Load processed chunks from a JSON file with progress bar."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    client = get_open_search_client("localhost", 9200)
    print(client.ping())
