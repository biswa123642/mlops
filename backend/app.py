import os
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

logger = logging.getLogger("backend")


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# Environment variables
# ============================================================

FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "").strip()
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY", "").strip()
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "").strip()

AZURE_API_VERSION = os.getenv(
    "AZURE_API_VERSION",
    "2024-12-01-preview"
).strip()

AZURE_SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT",
    ""
).strip()

AZURE_SEARCH_API_KEY = os.getenv(
    "AZURE_SEARCH_API_KEY",
    ""
).strip()

AZURE_SEARCH_INDEX = os.getenv(
    "AZURE_SEARCH_INDEX",
    "rag-llmops"
).strip()


# ============================================================
# Validate configuration
# ============================================================

required_variables = {
    "FOUNDRY_ENDPOINT": FOUNDRY_ENDPOINT,
    "FOUNDRY_API_KEY": FOUNDRY_API_KEY,
    "CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_API_KEY": AZURE_SEARCH_API_KEY,
    "AZURE_SEARCH_INDEX": AZURE_SEARCH_INDEX,
}

missing_variables = [
    name for name, value in required_variables.items()
    if not value
]

if missing_variables:
    logger.warning(
        "Missing environment variables: %s",
        ", ".join(missing_variables)
    )


# ============================================================
# Azure AI Foundry client
# ============================================================

foundry_client = None

if FOUNDRY_ENDPOINT and FOUNDRY_API_KEY:
    foundry_client = AzureOpenAI(
        azure_endpoint=FOUNDRY_ENDPOINT,
        api_key=FOUNDRY_API_KEY,
        api_version=AZURE_API_VERSION,
    )


# ============================================================
# Azure AI Search client
# ============================================================

search_client = None

if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY:
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )


# ============================================================
# Configuration
# ============================================================

SEARCH_TOP_K = 5


# ============================================================
# Health check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "foundry_configured": bool(foundry_client),
        "search_configured": bool(search_client),
        "search_index": AZURE_SEARCH_INDEX,
    }), 200


# ============================================================
# Azure AI Search - Hybrid Search
# ============================================================

def search_documents(question):
    """
    Performs hybrid search against Azure AI Search.

    Keyword search:
        Searches the 'chunk' and other searchable fields.

    Vector search:
        Uses Azure AI Search's configured vectorizer to
        convert the question into a vector.

    Vector field:
        text_vector

    Index:
        rag-llmops
    """

    if search_client is None:
        raise RuntimeError(
            "Azure AI Search client is not configured."
        )

    logger.info("Searching Azure AI Search: %s", question)

    # Azure AI Search performs query vectorization using the
    # vectorizer configured on the index.
    vector_query = VectorizableTextQuery(
        text=question,
        k_nearest_neighbors=SEARCH_TOP_K,
        fields="text_vector",
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        top=SEARCH_TOP_K,
        select=[
            "chunk_id",
            "parent_id",
            "chunk",
            "title",
        ],
    )

    documents = []

    for result in results:
        chunk = result.get("chunk")

        if not chunk:
            continue

        documents.append({
            "chunk_id": result.get("chunk_id"),
            "parent_id": result.get("parent_id"),
            "title": result.get("title"),
            "chunk": chunk,
            "score": result.get("@search.score"),
        })

    logger.info(
        "Azure AI Search returned %d chunks",
        len(documents)
    )

    return documents


# ============================================================
# Build context
# ============================================================

def build_context(documents):
    """
    Converts Azure AI Search results into context for the
    Foundry chat model.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(documents, start=1):

        title = document.get("title") or "Unknown document"
        chunk = document.get("chunk") or ""

        context_parts.append(
            f"""
--- Document Chunk {index} ---
Title: {title}

Content:
{chunk}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# Chat with Foundry
# ============================================================

def generate_answer(question, context):
    """
    Sends the user's question and retrieved PDF context
    to the Azure AI Foundry chat deployment.
    """

    if foundry_client is None:
        raise RuntimeError(
            "Azure AI Foundry client is not configured."
        )

    if not context:
        return (
            "I could not find relevant information in the "
            "provided documents."
        )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the document context,
   clearly say that the information was not found in the
   provided documents.
4. Give a concise and accurate answer.
5. When possible, mention the document title that supports
   the answer.
"""

    user_prompt = f"""
Document context:

{context}

User question:
{question}
"""

    response = foundry_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ============================================================
# Chat API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.get_json(silent=True) or {}

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Question is required."
            }), 400

        logger.info("Received question: %s", question)

        # ----------------------------------------------------
        # 1. Search Azure AI Search
        # ----------------------------------------------------

        documents = search_documents(question)

        if not documents:
            return jsonify({
                "answer": (
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                "sources": [],
            }), 200

        # ----------------------------------------------------
        # 2. Build context
        # ----------------------------------------------------

        context = build_context(documents)

        # ----------------------------------------------------
        # 3. Ask Azure AI Foundry
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            context
        )

        # ----------------------------------------------------
        # 4. Return answer + sources
        # ----------------------------------------------------

        sources = []

        for document in documents:
            sources.append({
                "title": document.get("title"),
                "parent_id": document.get("parent_id"),
                "chunk_id": document.get("chunk_id"),
                "score": document.get("score"),
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
        }), 200

    except Exception as exc:

        logger.exception(
            "Error while processing /chat request"
        )

        return jsonify({
            "error": "Failed to process chat request.",
            "details": str(exc),
        }), 500


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":

    logger.info("Starting backend...")
    logger.info(
        "Azure Search index: %s",
        AZURE_SEARCH_INDEX
    )
    logger.info(
        "Chat deployment: %s",
        CHAT_DEPLOYMENT
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )