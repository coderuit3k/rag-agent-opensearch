import os
import cohere
import json
import re
from typing import List, Annotated, Optional, Literal, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from ingest import generate_category

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

if not OPENAI_API_KEY or not COHERE_API_KEY or not OPENSEARCH_HOST or not TAVILY_API_KEY:
    raise ValueError("Missing environment variables")

# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=OPENAI_API_KEY
)

# Initialize TavilySearch
tavily_search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    search_depth="advanced",
    search_topic="general"
)

# Initialze Cohere
co = cohere.ClientV2(api_key=COHERE_API_KEY)

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

# Generate category for search results
def generate_category_search(text: str) -> Optional[str]:
    """
    Generate category for search results
    Args:
        text: The text content from search results
    Returns:
        The category of text
    """

    SYSTEM_PROMPT = f"""You are a content classfication system.

Your task is to assign ONE category to a given web content.

The allowed categories are:
- technology
- science
- business
- geography
- health
- education
- politics
- entertainment
- sports
- finance
- travel
- food
- lifestyle
- general

Rules:
- You must choose ONLY ONE category from the list above.
- Do NOT create new categories.
- If the content is unclear or mixed, choose the most dominant topic.
- If unsure, choose "general".
- Ouput MUST be a valid JSON in this format:
{{"category": "<category_name>"}}

Do NOT include any explanation.
"""

    HUMAN_MESSAGE = f"""Classify the following web content:

CONTENT:
{text}

Return the category.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_MESSAGE)
    ]

    # Get response
    response = llm.invoke(messages)

    if response is None:
        return None

    response = re.sub(r"```json|```", "", response.content).strip()

    # Parse response
    result = json.loads(response)

    return result["category"]

# Define State
class State(TypedDict):
    messages: Annotated[List, add_messages]
    contents: List[str]
    sources: List[str]
    titles: List[str]
    categories: List[str]
    scores: List[float]

# # Rewrite query for retrieval
# def rewrite_query_for_retrieval(state: State) -> Dict[str, List[HumanMessage]]:
#     """
#     Rewrite query to be more suitable and specific for retrieval
#     Args:
#         state: The current state of the graph
#     Returns:
#         A dict of rewritten query
#     """

#     query = state["messages"][-1].content

#     prompt = f"""You are a query rewriting expert for RAG system specialized in Machine Learning and AI topics.

# Your task is to rewrite the user's question to be more technical, specific, and aligned with academic writing syle similar to Lilian Weng's blog posts.
# Use precise ML/AI terminology. Make the query verbose enough to match relevant document chunks.

# Ouput ONLY the rewritten query, NO explanation.
# Original query: {query}
# Rewritten query:"""

#     repsonse = llm.invoke([{"role": "user", "content": prompt}])

#     rewritten_query = response.content.strip()

#     return {"messages": [HumanMessage(content=rewritten_query)]}

# # Rewrite query for search
# def rewrite_query_for_search(state: State) -> Dict[str, List[HumanMessage]]:
#     """
#     Rewrite query to be more suitable and specific for search
#     Args:
#         state: The current state of the graph
#     Returns:
#         A dict of rewritten query
#     """

#     query = state["messages"][-1].content

#     prompt = f"""You are a search query optimizer.

# Your task is to rewrite the user's question into a concise, keyword-focused query for a web search engine.
# Remove all filter words, keep obly the most important nouns, verbs, and technical terms.
# If the question implies recency, add the current year.

# Ouput ONLY the rewritten query, NO explanation.
# Original query: {query}
# Rewritten query:"""

#     response = llm.invoke([{"role": "user", "content": prompt}])

#     rewritten_query = response.content.strip()

#     return {"messages": [HumanMessage(content=rewritten_query)]}

# Define route
def route_question(state: State) -> Literal["search", "retrieve"]:
    """
    Use LLM to evaluate question and decide to use retrieve or search node
    Args:
        state: The current state
    Returns:
        "search" or "retrieve"
    """

    query = state["messages"][-1].content

    prompt = f"""You are a routing assistant.
Review the user's question. If the user is asking about Machine Learning, Reinforcement Learning, LLMs, Prompt Engineering, or AI Agents, it is highly likely related to Lilia Weng's blogs.
Reply exactly with "web_search" if it's general knowledge or current events.
Reply exactly with "vectorstore" if it's related to Lilian Weng's blogs.

Question:
{query}"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    decision = response.content.strip().lower()

    if "web_search" in decision:
        return "search"
    return "retrieve"

# Define retrieve node
def retrieve_node(state: State) -> Dict[str, List]:
    """
    Retrieve documents from OpenSearch
    Args:
        state: The current state
    Returns:
        A dict of contents, sources, titles and scores related to the query
    """

    query = state["messages"][-1].content
    query_embedding = embeddings.embed_query(query)

    # Create category of query to search more accurately
    category_of_query = generate_category(query)

    # Get all categories in OpenSearch
    response = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match_all": {}
            }
        }
    )

    categories = set()
    for hit in response["hits"]["hits"]:
        categories.add(hit["_source"]["category"])

    if category_of_query in categories:
        # Create body for hybrid search with category filter using script_score
        retrieve_body = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "match": {
                                        "content": query
                                    }
                                }
                            ],
                            "filter": [
                                {
                                    "term": {
                                        "category": category_of_query
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": """
                            double bm25_score = _score;
                            double vector_score = cosineSimilarity(params.query_vector, doc['embedding']);
                            return 0.3 * bm25_score + 0.7 * vector_score;
                        """,
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }
    else:
        # Create body for hybrid search using script_score
        retrieve_body = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "match": {
                                        "content": query
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": """
                            double bm25_score = _score;
                            double vector_score = cosineSimilarity(params.query_vector, doc['embedding']);
                            return 0.3 * bm25_score + 0.7 * vector_score;
                        """,
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

    # Search from OpenSearch
    try:
        response = client.search(index=INDEX_NAME, body=retrieve_body)

        print(f"Retrieve response: {response}")

        return {
            "contents": [hit["_source"]["content"] for hit in response["hits"]["hits"]],
            "sources": [hit["_source"]["source"] for hit in response["hits"]["hits"]],
            "titles": [hit["_source"]["title"] for hit in response["hits"]["hits"]],
            "categories": [hit["_source"]["category"] for hit in response["hits"]["hits"]],
            "scores": [hit["_score"] for hit in response["hits"]["hits"]]
        }
    except Exception as e:
        print(f"Retrieve failed: {str(e)}")
        return {
            "contents": [],
            "sources": [],
            "titles": [],
            "categories": [],
            "scores": []
        }

# Define search node
def search_node(state: State) -> Dict[str, List]:
    """
    Search the web for information related to a query.
    Args:
        state: The current state
    Returns:
        A dict of contents, sources, titles, categories and scores of answers from searching
    """

    query = state["messages"][-1].content

    try:
        search_response = tavily_search.invoke({"query": query})

        print(f"Search response: {search_response}")

        if search_response and search_response.get("results"):
            # Get contents
            contents = [res["content"] for res in search_response["results"]]

            # Get sources
            sources = [res["url"] for res in search_response["results"]]

            # Get titles
            titles = [res["title"] for res in search_response["results"]]

            # Get categories
            categories = [generate_category_search(res["content"]) for res in search_response["results"]]

            # Get scores
            scores = [res["score"] for res in search_response["results"]]

            return {
                "contents": contents,
                "sources": sources,
                "titles": titles,
                "categories": categories,
                "scores": scores
            }

    except Exception as e:
        print(f"Search failed: {str(e)}")
        return {
            "contents": [],
            "sources": [],
            "titles": [],
            "categories": [],
            "scores": []
        }

# Define rerank node
def rerank_node(state: State) -> Dict[str, List[str]]:
    """
    Rerank the results from OpenSearch using Cohere Rerank
    Args:
        state: The current state
    Returns:
        A dict of contents, sources, titles, categories from the reranked documents
    """

    query = state["messages"][-1].content

    # Get contents
    contents = state.get("contents", [])

    # Get sources
    sources = state.get("sources", [])

    # Get titles
    titles = state.get("titles", [])

    # Get categories
    categories = state.get("categories", [])

    # Get scores
    scores = state.get("scores", [])

    if not contents or not sources or not titles or not categories or not scores:
        return {
            "contents": [],
            "sources": [],
            "titles": [],
            "categories": [],
            "scores": []
        }

    try:
        # Get 3 results after reranking with the higheset relevance score
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=contents,
            top_n=3
        )

        print(f"Rerank response: {rerank_response}")

        # Get indices
        indices = [res.index for res in rerank_response.results]

        # Get relevance scores
        scores = [res.relevance_score for res in rerank_response.results]

        return {
            "contents": [contents[idx] for idx in indices],
            "sources": [sources[idx] for idx in indices],
            "titles": [titles[idx] for idx in indices],
            "categories": [categories[idx] for idx in indices],
            "scores": scores
        }

    except Exception as e:
        # If error, return 3 docs with highest score from OpenSearch
        print(f"Rerank ignored dur to error: {str(e)}")
        return {
            "contents": contents[:3],
            "sources": sources[:3],
            "titles": titles[:3],
            "categories": categories[:3],
            "scores": scores[:3]
        }

# Define generate node
def generate_node(state: State) -> Dict[str, List[str]]:
    """
    Generate an answer based on the retrieved documents
    Args:
        state: The current state
    Returns:
        A list of the generated documents
    """

    query = state["messages"][-1].content

    # Get contents
    contents = state.get("contents", [])

    # Get sources
    sources = state.get("sources", [])

    # Get titles
    titles = state.get("titles", [])

    # Get categories
    categories = state.get("categories", [])

    context = "\n\n".join([f"[Souce URL]: {source}\n[Title]: {title}\n[Category]: {category}\n[Content]: {content}" for content, source, title, category in zip(contents, sources, titles, categories)])

    SYSTEM_PROMPT = f"""You are a helpful AI assistant.
Use the following context to answer the user's question.
If the context is empty or doesn't contain the answer, just use your own knowledge or admit you don't know.
When answering, please cite the Source URLs provided in the context at the end of your answer.

Context:
{context}"""

    # Generate the final answer using system prompt context + user state context
    invoke_messages = [SystemMessage(content=SYSTEM_PROMPT), state["messages"][-1]]

    response = llm.invoke(invoke_messages)

    return {"messages": [response]}

# Define graph
graph = StateGraph(State)

# Add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("search", search_node)
graph.add_node("rerank", rerank_node)
graph.add_node("generate", generate_node)

# Add conditional edges
graph.add_conditional_edges(
    START,
    route_question,
    {
        "search": "search",
        "retrieve": "retrieve"
    }
)

# Add edges
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "generate")
graph.add_edge("search", "generate")
graph.add_edge("generate", END)


# Memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

def chat_loop():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # GỌI APP NHƯ BÌNH THƯỜNG
            result = app.invoke(
                {"messages": [("user", user_input)]}, 
                {"configurable": {"thread_id": "1"}}
            )
            print(f"\nAgent: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    chat_loop()