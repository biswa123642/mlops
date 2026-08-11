import hashlib
import logging
import os
import tempfile
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from azure.storage.blob import BlobServiceClient

from openai import AzureOpenAI

from PyPDF2 import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger("vector-service")


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# Configuration
# ============================================================

# ------------------------------------------------------------
# Azure AI Foundry / Azure OpenAI
# ------------------------------------------------------------

AZURE_FOUNDRY_ENDPOINT = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT"
)

AZURE_FOUNDRY_API_KEY = os.environ.get(
    "AZURE_FOUNDRY_API_KEY"
)

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "text-embedding-ada-002",
)

AZURE_API_VERSION = os.environ.get(
    "AZURE_API_VERSION",
    "2024-12-01-preview",
)


# ------------------------------------------------------------
# Azure Blob Storage
# ------------------------------------------------------------

AZURE_STORAGE_ACCOUNT = os.environ.get(
    "AZURE_STORAGE_ACCOUNT"
)

AZURE_STORAGE_ACCESS_KEY = os.environ.get(
    "AZURE_STORAGE_ACCESS_KEY"
)

AZURE_STORAGE_CONTAINER = os.environ.get(
    "AZURE_STORAGE_CONTAINER",
    "llmops",
)

AZURE_STORAGE_PREFIX = os.environ.get(
    "AZURE_STORAGE_PREFIX",
    "documents/",
)


# ------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------

QDRANT_CLIENT_URL = os.environ.get(
    "QDRANT_CLIENT_URL",
    "http://qdrant:6333",
)

COLLECTION_NAME = os.environ.get(
    "QDRANT_COLLECTION_NAME",
    "margies_travel_embeddings",
)


# ------------------------------------------------------------
# Chunking
# ------------------------------------------------------------

CHUNK_SIZE = int(
    os.environ.get(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.environ.get(
        "CHUNK_OVERLAP",
        "150",
    )
)


# ============================================================
# Validate required configuration
# ============================================================

required_config = {
    "AZURE_FOUNDRY_ENDPOINT": AZURE_FOUNDRY_ENDPOINT,
    "AZURE_FOUNDRY_API_KEY": AZURE_FOUNDRY_API_KEY,
    "AZURE_STORAGE_ACCOUNT": AZURE_STORAGE_ACCOUNT,
    "AZURE_STORAGE_ACCESS_KEY": AZURE_STORAGE_ACCESS_KEY,
    "AZURE_STORAGE_CONTAINER": AZURE_STORAGE_CONTAINER,
    "QDRANT_CLIENT_URL": QDRANT_CLIENT_URL,
}

missing_config = [
    name
    for name, value in required_config.items()
    if not value
]

if missing_config:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing_config)
    )


# ============================================================
# Azure OpenAI / Foundry client
# ============================================================

openai_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_FOUNDRY_ENDPOINT,
    api_key=AZURE_FOUNDRY_API_KEY,
)


# ============================================================
# Azure Blob Storage client
# ============================================================

storage_account_url = (
    f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
)

blob_service_client = BlobServiceClient(
    account_url=storage_account_url,
    credential=AZURE_STORAGE_ACCESS_KEY,
)

container_client = (
    blob_service_client.get_container_client(
        AZURE_STORAGE_CONTAINER
    )
)


# ============================================================
# Qdrant client
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_CLIENT_URL
)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text.
    """

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# Text chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# Generate embedding
# ============================================================

def generate_embedding(
    text: str,
) -> List[float]:
    """
    Generate an embedding using Azure AI Foundry.
    """

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=text,
    )

    return response.data[0].embedding


# ============================================================
# Ensure Qdrant collection exists
# ============================================================

def ensure_collection(
    vector_size: int,
):
    """
    Create the Qdrant collection if it does not exist.
    """

    if qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        return

    logger.info(
        "Creating Qdrant collection '%s'",
        COLLECTION_NAME,
    )

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.DOT,
        ),
    )

    logger.info(
        "Qdrant collection created"
    )


# ============================================================
# Generate deterministic point ID
# ============================================================

def create_point_id(
    blob_name: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic ID for a PDF chunk.
    """

    value = (
        f"{blob_name}:{chunk_index}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# Check whether blob version is already indexed
# ============================================================

def blob_already_processed(
    blob_name: str,
    blob_etag: str,
) -> bool:

    try:

        if not qdrant_client.collection_exists(
            COLLECTION_NAME
        ):
            return False

        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_blob",
                        match=MatchValue(
                            value=blob_name
                        ),
                    ),
                    FieldCondition(
                        key="blob_etag",
                        match=MatchValue(
                            value=blob_etag
                        ),
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return len(points) > 0

    except Exception as exc:

        logger.warning(
            "Could not check existing vectors for '%s': %s",
            blob_name,
            exc,
        )

        return False


# ============================================================
# Delete old vectors for a blob
# ============================================================

def delete_existing_blob_points(
    blob_name: str,
):
    """
    Remove vectors belonging to a previous version
    of the same PDF.
    """

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        return

    logger.info(
        "Deleting previous vectors for '%s'",
        blob_name,
    )

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source_blob",
                    match=MatchValue(
                        value=blob_name
                    ),
                )
            ]
        ),
        wait=True,
    )


# ============================================================
# Extract text from PDF
# ============================================================

def extract_pdf_text(
    file_path: str,
) -> str:
    """
    Extract text from a PDF.
    """

    reader = PdfReader(
        file_path
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages
    ):

        try:

            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception as exc:

            logger.warning(
                "Could not extract page %s: %s",
                page_number,
                exc,
            )

    return normalize_text(
        "\n".join(pages)
    )


# ============================================================
# Process one PDF blob
# ============================================================

def process_blob(
    blob_client,
):
    """
    Download a PDF from Azure Blob Storage,
    extract text, create embeddings and store
    vectors in Qdrant.
    """

    blob_name = blob_client.blob_name

    properties = (
        blob_client.get_blob_properties()
    )

    blob_etag = properties.etag

    logger.info(
        "Processing blob: %s",
        blob_name,
    )

    # --------------------------------------------------------
    # Skip unchanged document
    # --------------------------------------------------------

    if blob_already_processed(
        blob_name,
        blob_etag,
    ):

        logger.info(
            "Blob already indexed and unchanged: %s",
            blob_name,
        )

        return {
            "blob": blob_name,
            "status": "skipped",
        }


    # --------------------------------------------------------
    # Download PDF to temporary filesystem
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=True,
    ) as temp_file:

        logger.info(
            "Downloading blob '%s'",
            blob_name,
        )

        download_stream = (
            blob_client.download_blob()
        )

        temp_file.write(
            download_stream.readall()
        )

        temp_file.flush()

        # ----------------------------------------------------
        # Extract PDF text
        # ----------------------------------------------------

        text = extract_pdf_text(
            temp_file.name
        )


    if not text.strip():

        logger.warning(
            "No text extracted from '%s'",
            blob_name,
        )

        return {
            "blob": blob_name,
            "status": "empty",
        }


    # --------------------------------------------------------
    # Chunk text
    # --------------------------------------------------------

    chunks = chunk_text(
        text
    )

    if not chunks:

        return {
            "blob": blob_name,
            "status": "empty",
        }


    logger.info(
        "Created %s chunks for '%s'",
        len(chunks),
        blob_name,
    )


    # --------------------------------------------------------
    # Delete previous version
    # --------------------------------------------------------

    if qdrant_client.collection_exists(
        COLLECTION_NAME
    ):

        delete_existing_blob_points(
            blob_name
        )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    points = []

    for chunk_index, chunk in enumerate(
        chunks
    ):

        logger.info(
            "Embedding chunk %s/%s from '%s'",
            chunk_index + 1,
            len(chunks),
            blob_name,
        )

        embedding = generate_embedding(
            chunk
        )


        # ----------------------------------------------------
        # Create Qdrant collection from first embedding
        # ----------------------------------------------------

        if not qdrant_client.collection_exists(
            COLLECTION_NAME
        ):

            ensure_collection(
                len(embedding)
            )


        # ----------------------------------------------------
        # Point ID
        # ----------------------------------------------------

        point_id = create_point_id(
            blob_name,
            chunk_index,
        )


        # ----------------------------------------------------
        # Qdrant point
        # ----------------------------------------------------

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source_blob": blob_name,
                    "blob_etag": blob_etag,
                    "chunk_index": chunk_index,
                },
            )
        )


    # --------------------------------------------------------
    # Store vectors
    # --------------------------------------------------------

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    logger.info(
        "Successfully indexed %s chunks from '%s'",
        len(points),
        blob_name,
    )

    return {
        "blob": blob_name,
        "status": "indexed",
        "chunks": len(points),
    }


# ============================================================
# Ingest all PDFs
# ============================================================

def ingest_documents():
    """
    Find PDFs in Azure Blob Storage and index them.
    """

    logger.info(
        "Starting document ingestion"
    )

    results = []

    blobs = container_client.list_blobs(
        name_starts_with=AZURE_STORAGE_PREFIX
    )

    for blob in blobs:

        blob_name = blob.name

        # ----------------------------------------------------
        # Only process PDFs
        # ----------------------------------------------------

        if not blob_name.lower().endswith(
            ".pdf"
        ):

            logger.info(
                "Skipping non-PDF: %s",
                blob_name,
            )

            continue


        try:

            blob_client = (
                container_client
                .get_blob_client(
                    blob_name
                )
            )

            result = process_blob(
                blob_client
            )

            results.append(
                result
            )

        except Exception as exc:

            logger.exception(
                "Failed to process '%s'",
                blob_name,
            )

            results.append(
                {
                    "blob": blob_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results


# ============================================================
# Health endpoint
# ============================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "healthy",
        }
    ), 200


# ============================================================
# Readiness endpoint
# ============================================================

@app.route(
    "/ready",
    methods=["GET"],
)
def ready():

    try:

        # Check Qdrant
        qdrant_client.get_collections()

        # Check Azure Blob Storage
        container_client.get_container_properties()

        return jsonify(
            {
                "status": "ready",
            }
        ), 200

    except Exception as exc:

        logger.error(
            "Readiness check failed: %s",
            exc,
        )

        return jsonify(
            {
                "status": "not_ready",
                "error": str(exc),
            }
        ), 503


# ============================================================
# Ingestion endpoint
# ============================================================

@app.route(
    "/ingest",
    methods=["POST"],
)
def ingest():

    try:

        results = ingest_documents()

        failed = [
            result
            for result in results
            if result.get("status") == "failed"
        ]

        if failed:

            return jsonify(
                {
                    "status": "completed_with_errors",
                    "results": results,
                }
            ), 200

        return jsonify(
            {
                "status": "completed",
                "results": results,
            }
        ), 200

    except Exception as exc:

        logger.exception(
            "Document ingestion failed"
        )

        return jsonify(
            {
                "status": "failed",
                "error": str(exc),
            }
        ), 500


# ============================================================
# Root endpoint
# ============================================================

@app.route(
    "/",
    methods=["GET"],
)
def root():

    return jsonify(
        {
            "service": "vector-service",
            "status": "running",
        }
    ), 200


# ============================================================
# Application startup
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting Vector Service"
    )

    logger.info(
        "Qdrant URL: %s",
        QDRANT_CLIENT_URL,
    )

    logger.info(
        "Blob container: %s/%s",
        AZURE_STORAGE_ACCOUNT,
        AZURE_STORAGE_CONTAINER,
    )

    logger.info(
        "Blob prefix: %s",
        AZURE_STORAGE_PREFIX,
    )

    logger.info(
        "Embedding deployment: %s",
        EMBEDDING_MODEL_NAME,
    )

    app.run(
        host="0.0.0.0",
        port=5173,
    )