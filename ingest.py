import os
import re
import json
import requests
from requests.exceptions import RequestException, HTTPError
from dotenv import load_dotenv
from uuid import uuid4
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from service import create_index, delete_index

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

if not OPENAI_API_KEY or not OPENSEARCH_HOST or not USERNAME or not PASSWORD:
    raise ValueError("Missing environment variables")

# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=OPENAI_API_KEY
)

INDEX_NAME = "rag_index"

# Initialize OpenSearch client
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

# Get raw content and title from URL
def get_content_and_title_from_url(url: str) -> Tuple[str, str]:
    """
    Get content and title from url
    Args:
        url: The URL where to get content
    Returns:
        The raw content and title from URL
    """
    
    print(f"Beginning crawl from url: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Get text
            text = soup.get_text()

            # Get title
            title = soup.find("h1", class_="post-title").get_text(strip=True)

            return text.strip(), title.strip()

    except HTTPError as http_err:
        print(f"HTTP error occurred: {str(http_err)}")
        return "", ""
    except RequestException as e:
        print(f"Failed to fetch URL with error: {str(e)}")
        return "", ""
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return "", ""

# Clean raw content from URL
def clean_web_text(text: str) -> str:
    """
    Clean raw content from URL
    Args:
        text: The raw content from URL
    Returns:
        The cleaned content
    """

    # Remove continous whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace at the beginning and end of the text
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join([line for line in lines if line.strip()])

    # Remove the source of the image caption
    text = re.sub(r"\(Image source:.*?\)", "", text)

    # Remove any strange formatting symbols or unnecessary icons
    text = re.sub(r'[^\w\s\.,\?!:;\-\(\)\[\]"\'\/`#\*+=]', '', text)
    return text

# Extract main content from the cleaned text
def extract_main_content(text: str) -> str:
    """
    Extract main content from the cleaned text
    Args:
        text: The cleaned content
    Returns:
        The main content
    """

    # Remove all content before "Table of Contents"
    start_keyword = "Table of Contents"
    start_idx = text.find(start_keyword)

    if start_idx != -1:
        text = text[start_idx:]

    # Remove all content after "Citation#"
    end_keyword = "Citation#"
    end_idx = text.find(end_keyword)

    if end_idx != -1:
        text = text[:end_idx]

    return text.strip()

# Generate category for search in OpenSearch
def generate_category(text: str) -> Optional[str]:
    """
    Generate category for search in OpenSearch
    Args:
        text: The main content of URL
    Returns:
        The category of the main content
    """

    SYSTEM_PROMPT = f"""You are an expert in Machine Leaning and AI topics.

Your task is to classify a given content into ONE high-level category.

The allowed categories are:
- Reinforcement Learning
- Large Language Models
- Prompt Engineering
- AI Agents
- Computer Vision
- Natural Language Processing
- Optimizer
- Generative AI

Rules:
- You must choose ONLY ONE category from the list above.
- Do NOT create new categories.
- If undure, choose the closest match.
- Output MUST be a valid JSON object with this format:
{{"category": "<category_name>"}}

Do NOT include any explanation.
"""

    HUMAN_MESSGAE = f"""Classify the following content:

CONTENT:
{text}

Return the category.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_MESSGAE)
    ]

    # Get response
    response = llm.invoke(messages)

    if response is None:
        return None

    response = re.sub(r"```json|```", "", response.content).strip()

    # Parse response
    result = json.loads(response)

    return result["category"]

# Get final data to store
def get_final_data(urls: List[str]) -> List[Document]:
    """
    Get final data for RAG
    Args:
        urls: The lists of URLs to get data from
    Returns:
        The list of Document objects
    """

    docs = []
    for url in urls:
        text, title = get_content_and_title_from_url(url)
        cleaned_text = clean_web_text(text)
        main_content = extract_main_content(cleaned_text)

        # Generate category
        category = generate_category(main_content[:2000])

        doc = Document(
            page_content=main_content,
            metadata={
                "title": title,
                "source": url,
                "category": category
            }
        )

        docs.append(doc)

    return docs

# Split data into chunks
def split_data(docs: List[Document]) -> List[Document]:
    """
    Split data into chunks
    Args:
        docs: The list of Documen objects 
    Returns:
        The list of Document objects after splitting
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100
    )

    doc_splits = text_splitter.split_documents(docs)
    return doc_splits

# Store chunks
def store_chunks(chunks: List[Document]) -> None:
    """
    Store chunks into OpenSearch
    Args:
        chunks: The list of Document objects after splitting
    Returns:
        None
    """

    embed_docs = embeddings.embed_documents([doc.page_content for doc in chunks])

    # OpenSearch bulk API requires alternating metadata and document bodies
    docs = []
    for chunk, emb in zip(chunks, embed_docs):
        docs.append({
            "_index": INDEX_NAME,
            "_id": str(uuid4()),
            "_source": {
                "content": chunk.page_content,
                "title": chunk.metadata.get("title"),
                "source": chunk.metadata.get("source"),
                "category": chunk.metadata.get("category"),
                "embedding": emb
            }
        })

    # Bulk insert
    body = []
    for doc in docs:
        body.append({
            "index": {
                "_index": INDEX_NAME,
                "_id": doc["_id"]
            }
        })

        body.append(doc["_source"])

    client.bulk(body=body)
    client.indices.refresh(index=INDEX_NAME)

    print(f"Successfully stored {len(chunks)} into OpenSearch")

if __name__ == "__main__":
    urls = [
        "https://lilianweng.github.io/posts/2025-05-01-thinking/",
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
        "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
        "https://lilianweng.github.io/posts/2022-09-08-ntk/",
        "https://lilianweng.github.io/posts/2022-06-09-vlm/",
        "https://lilianweng.github.io/posts/2022-04-15-data-gen/",
        "https://lilianweng.github.io/posts/2022-02-20-active-learning/",
        "https://lilianweng.github.io/posts/2021-12-05-semi-supervised/",
        "https://lilianweng.github.io/posts/2021-09-25-train-large/",
        "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
        "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
        "https://lilianweng.github.io/posts/2021-03-21-lm-toxicity/",
        "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/"
    ]

    # Delete index
    delete_index(INDEX_NAME)

    # Create index
    create_index(INDEX_NAME)

    # Get final data
    docs = get_final_data(urls)

    # Split data
    chunks = split_data(docs)

    # Store chunks
    store_chunks(chunks)