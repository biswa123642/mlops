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
# Flask application
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# Environment variables
# ============================================================

FOUNDRY_ENDPOINT = os.getenv(
    "FOUNDRY_ENDPOINT",
    ""
).strip()

FOUNDRY_API_KEY = os.getenv(
    "FOUNDRY_API_KEY",
    ""
).strip()

CHAT_DEPLOYMENT = os.getenv(
    "CHAT_DEPLOYMENT",
    ""
).strip()

AZURE_API_VERSION = os.getenv(
    "AZURE_API_VERSION",
    "2024-12-01-preview"
).strip()


# Azure AI Search

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
# Search configuration
# ============================================================

SEARCH_TOP_K = 5


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
    name
    for name, value in required_variables.items()
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

    logger.info("Azure AI Foundry client initialized")


# ============================================================
# Azure AI Search client
# ============================================================

search_client = None

if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY:

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(
            AZURE_SEARCH_API_KEY
        ),
    )

    logger.info(
        "Azure AI Search client initialized "
        "for index: %s",
        AZURE_SEARCH_INDEX
    )


# ============================================================
# Health endpoint
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
# Azure AI Search hybrid search
# ============================================================

def search_documents(question):

    if search_client is None:
        raise RuntimeError(
            "Azure AI Search client is not configured."
        )

    logger.info(
        "Searching Azure AI Search for: %s",
        question
    )

    # --------------------------------------------------------
    # Vector query
    #
    # Azure AI Search uses the vectorizer configured on the
    # Search index to convert the text question into a vector.
    #
    # Vector field:
    #     text_vector
    #
    # Vectorizer/profile are configured in Azure AI Search.
    # --------------------------------------------------------

    vector_query = VectorizableTextQuery(
        text=question,
        k_nearest_neighbors=SEARCH_TOP_K,
        fields="text_vector",
    )

    # --------------------------------------------------------
    # Hybrid search
    #
    # search_text  -> keyword/text search
    # vector_query -> vector search
    # --------------------------------------------------------

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
        "Azure AI Search returned %d relevant chunks",
        len(documents)
    )

    return documents


# ============================================================
# Build context for Foundry
# ============================================================

def build_context(documents):

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        title = (
            document.get("title")
            or "Unknown document"
        )

        chunk = (
            document.get("chunk")
            or ""
        )

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
# Generate answer using Azure AI Foundry
# ============================================================

def generate_answer(question, context):

    if foundry_client is None:
        raise RuntimeError(
            "Azure AI Foundry client is not configured."
        )

    if not context:

        return (
            "I could not find relevant information "
            "in the provided documents."
        )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:

1. Use the provided document context as the source of truth.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is not available in the provided
   documents, clearly say that the information was not
   found in the provided documents.
5. Give a clear and concise answer.
6. When possible, mention the document title that
   supports the answer.
"""

    user_prompt = f"""
Document context:

{context}

User question:

{question}
"""

    logger.info(
        "Sending retrieved context to Foundry chat model"
    )

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

    if not response.choices:

        raise RuntimeError(
            "Foundry returned no response choices."
        )

    answer = response.choices[0].message.content

    if not answer:

        raise RuntimeError(
            "Foundry returned an empty response."
        )

    return answer


# ============================================================
# Chat endpoint
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # ----------------------------------------------------
        # Read JSON request
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}

        # ----------------------------------------------------
        # Your existing frontend sends:
        #
        # {
        #     "message": "..."
        # }
        #
        # We also accept "question" for compatibility.
        # ----------------------------------------------------

        question = (
            data.get("message")
            or data.get("question")
            or ""
        ).strip()

        if not question:

            return jsonify({
                "reply": "Question is required.",
                "sources": [],
            }), 400

        logger.info(
            "Received chat question: %s",
            question
        )

        # ----------------------------------------------------
        # Step 1:
        # Search Azure AI Search
        # ----------------------------------------------------

        documents = search_documents(
            question
        )

        # ----------------------------------------------------
        # No relevant documents
        # ----------------------------------------------------

        if not documents:

            logger.info(
                "No relevant documents found."
            )

            return jsonify({
                "reply": (
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                "sources": [],
            }), 200

        # ----------------------------------------------------
        # Step 2:
        # Build RAG context
        # ----------------------------------------------------

        context = build_context(
            documents
        )

        # ----------------------------------------------------
        # Step 3:
        # Send context + question to Foundry
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            context
        )

        # ----------------------------------------------------
        # Step 4:
        # Return response to frontend
        #
        # Frontend expects:
        #
        # data.reply
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
            "reply": answer,
            "sources": sources,
        }), 200

    except Exception as exc:

        logger.exception(
            "Error processing /chat request"
        )

        return jsonify({
            "reply": (
                "Sorry, I was unable to process "
                "your request."
            ),
            "error": str(exc),
        }), 500


# ============================================================
# Application startup
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting backend application..."
    )

    logger.info(
        "Azure AI Search index: %s",
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
