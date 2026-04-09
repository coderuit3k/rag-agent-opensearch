import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

INDEX_NAME = "rag_index"

# Initialze OpenSearch client
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
    http_auth=(USERNAME, PASSWORD),
    http_compress=True,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

# Create index in OpenSearch
def create_index(index_name: str) -> None:
    """
    Create index in OpenSearch
    Args:
        index_name: The name of the index to create
    Returns:
        None
    """

    if client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists")
        return None

    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "title": {"type": "text"},
                "source": {"type": "keyword"},
                "category": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }

    client.indices.create(index=index_name, body=index_body)
    print(f"Index {index_name} creates successfully")

# Delete index
def delete_index(index_name: str) -> None:
    """
    Delete index in OpenSearch
    Args:
        index_name: The name of the index to delete
    Returns:
        None
    """

    if not client.indices.exists(index=index_name):
        print(f"Index {index_name} does not exist")
        return None

    client.indices.delete(index=index_name)
    print(f"Index {index_name} deleted successfully")