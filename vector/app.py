import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List

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
# Azure OpenAI configuration
# ============================================================

AZURE_FOUNDRY_ENDPOINT = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT",
)

AZURE_FOUNDRY_API_KEY = os.environ.get(
    "AZURE_FOUNDRY_API_KEY",
)

# Your Azure embedding deployment name
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_EMBEDDING_DEPLOYMENT",
    "text-embedding-3-small",
)

AZURE_API_VERSION = os.environ.get(
    "AZURE_API_VERSION",
    "2024-12-01-preview",
)


# ============================================================
# Azure Blob Storage configuration
# ============================================================

AZURE_STORAGE_ACCOUNT = os.environ.get(
    "AZURE_STORAGE_ACCOUNT",
)

AZURE_STORAGE_ACCESS_KEY = os.environ.get(
    "AZURE_STORAGE_ACCESS_KEY",
)

AZURE_STORAGE_CONTAINER = os.environ.get(
    "AZURE_STORAGE_CONTAINER",
    "llmops",
)

AZURE_STORAGE_PREFIX = os.environ.get(
    "AZURE_STORAGE_PREFIX",
    "documents/",
)


# ============================================================
# Qdrant configuration
# ============================================================

# Self-hosted Qdrant installed with Helm.
# No QDRANT_API_KEY is used.
QDRANT_CLIENT_URL = os.environ.get(
    "QDRANT_CLIENT_URL",
    "http://qdrant:6333",
)

COLLECTION_NAME = os.environ.get(
    "QDRANT_COLLECTION_NAME",
    "margies_travel_embeddings",
)


# ============================================================
# Application configuration
# ============================================================

CHUNK_SIZE = int(
    os.environ.get(
        "CHUNK_SIZE",
        "1200",
    )
)

CHUNK_OVERLAP = int(
    os.environ.get(
        "CHUNK_OVERLAP",
        "150",
    )
)

EMBEDDING_BATCH_SIZE = int(
    os.environ.get(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)

PORT = int(
    os.environ.get(
        "PORT",
        "5173",
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
# Azure OpenAI client
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

# No QDRANT_API_KEY is required.
qdrant_client = QdrantClient(
    url=QDRANT_CLIENT_URL,
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
        line = " ".join(line.split())

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
    Split text into overlapping character-based chunks.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative"
        )

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
# Azure OpenAI embeddings
# ============================================================

def generate_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """
    Generate embeddings for a batch of text.
    """

    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT,
        input=texts,
    )

    ordered_results = sorted(
        response.data,
        key=lambda item: item.index,
    )

    return [
        item.embedding
        for item in ordered_results
    ]


def generate_embeddings_in_batches(
    texts: List[str],
) -> List[List[float]]:
    """
    Generate embeddings in batches.
    """

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):
        end = min(
            start + EMBEDDING_BATCH_SIZE,
            len(texts),
        )

        batch = texts[start:end]

        logger.info(
            "Generating embeddings for chunks %s-%s of %s",
            start + 1,
            end,
            len(texts),
        )

        embeddings = generate_embeddings(batch)

        all_embeddings.extend(embeddings)

    return all_embeddings


# ============================================================
# Qdrant collection
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
        "Creating Qdrant collection '%s' with size %s",
        COLLECTION_NAME,
        vector_size,
    )

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    logger.info(
        "Qdrant collection created",
    )


# ============================================================
# Deterministic Qdrant point ID
# ============================================================

def create_point_id(
    blob_name: str,
    blob_etag: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic UUID for a document chunk.
    """

    value = (
        f"{blob_name}:{blob_etag}:{chunk_index}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )
    )


# ============================================================
# Check existing blob version
# ============================================================

def blob_version_already_processed(
    blob_name: str,
    blob_etag: str,
) -> bool:
    """
    Check whether this exact blob version is indexed.
    """

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
                            value=blob_name,
                        ),
                    ),
                    FieldCondition(
                        key="blob_etag",
                        match=MatchValue(
                            value=blob_etag,
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
# Delete old blob versions
# ============================================================

def delete_old_blob_versions(
    blob_name: str,
    current_etag: str,
):
    """
    Delete old versions after the new version has been
    successfully inserted.
    """

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        return

    old_versions_filter = Filter(
        must=[
            FieldCondition(
                key="source_blob",
                match=MatchValue(
                    value=blob_name,
                ),
            ),
        ],
        must_not=[
            FieldCondition(
                key="blob_etag",
                match=MatchValue(
                    value=current_etag,
                ),
            ),
        ],
    )

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=old_versions_filter,
        wait=True,
    )

    logger.info(
        "Deleted old Qdrant versions for '%s'",
        blob_name,
    )


# ============================================================
# PDF text extraction
# ============================================================

def extract_pdf_text(
    file_path: str,
) -> str:
    """
    Extract embedded text from a PDF.

    Scanned PDFs require OCR.
    """

    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(
        reader.pages
    ):
        try:
            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

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
) -> Dict[str, Any]:
    """
    Download a PDF, extract text, generate embeddings,
    and store vectors in Qdrant.
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

    if blob_version_already_processed(
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

    chunks = chunk_text(text)

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

    embeddings = generate_embeddings_in_batches(
        chunks
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count"
        )

    if not embeddings:
        return {
            "blob": blob_name,
            "status": "empty",
        }

    vector_size = len(embeddings[0])

    ensure_collection(vector_size)

    points = []

    for chunk_index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings)
    ):
        point_id = create_point_id(
            blob_name,
            blob_etag,
            chunk_index,
        )

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

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    try:
        delete_old_blob_versions(
            blob_name,
            blob_etag,
        )

    except Exception as exc:
        logger.exception(
            "New version indexed, but old versions could "
            "not be deleted for '%s'",
            blob_name,
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
        "embedding_dimension": vector_size,
    }


# ============================================================
# Ingest all PDFs
# ============================================================

def ingest_documents() -> List[Dict[str, Any]]:
    """
    List PDF blobs and index them.
    """

    logger.info(
        "Starting document ingestion",
    )

    results = []

    blobs = container_client.list_blobs(
        name_starts_with=AZURE_STORAGE_PREFIX,
    )

    for blob in blobs:
        blob_name = blob.name

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
                container_client.get_blob_client(
                    blob_name,
                )
            )

            result = process_blob(
                blob_client,
            )

            results.append(result)

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
# Search endpoint
# ============================================================

@app.route(
    "/search",
    methods=["POST"],
)
def search():
    """
    Generate an embedding for a query and search Qdrant.

    This endpoint is called by the backend service.
    """

    try:
        data = request.get_json(
            silent=True,
        ) or {}

        query = str(
            data.get("query", ""),
        ).strip()

        try:
            limit = int(
                data.get("limit", 3),
            )
        except (TypeError, ValueError):
            limit = 3

        if not query:
            return jsonify(
                {
                    "error": "query is required",
                }
            ), 400

        if limit < 1:
            limit = 1

        if limit > 10:
            limit = 10

        if not qdrant_client.collection_exists(
            COLLECTION_NAME,
        ):
            return jsonify(
                {
                    "context": "",
                    "results": [],
                }
            ), 200

        query_embedding = generate_embeddings(
            [query],
        )[0]

        search_response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in search_response.points:
            payload = point.payload or {}

            results.append(
                {
                    "score": point.score,
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "source_blob": payload.get(
                        "source_blob",
                        "",
                    ),
                    "chunk_index": payload.get(
                        "chunk_index",
                        0,
                    ),
                }
            )

        context = "\n\n".join(
            item["text"]
            for item in results
            if item.get("text")
        )

        return jsonify(
            {
                "context": context,
                "results": results,
            }
        ), 200

    except Exception as exc:
        logger.exception(
            "Vector search failed",
        )

        return jsonify(
            {
                "error": "Vector search failed",
                "details": str(exc),
            }
        ), 500


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
        qdrant_client.get_collections()

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
    """
    This endpoint is called by your backend.

    No INGEST_TOKEN is required.
    """

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
            "Document ingestion failed",
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
# Local startup
# ============================================================

if __name__ == "__main__":
    logger.info(
        "Starting vector service",
    )

    logger.info(
        "Qdrant URL: %s",
        QDRANT_CLIENT_URL,
    )

    logger.info(
        "Qdrant collection: %s",
        COLLECTION_NAME,
    )

    logger.info(
        "Azure embedding deployment: %s",
        AZURE_EMBEDDING_DEPLOYMENT,
    )

    logger.info(
        "Blob storage: %s/%s",
        AZURE_STORAGE_ACCOUNT,
        AZURE_STORAGE_CONTAINER,
    )

    logger.info(
        "Blob prefix: %s",
        AZURE_STORAGE_PREFIX,
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
    )
