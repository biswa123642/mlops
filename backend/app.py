import logging
import os

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger("backend-service")


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# Configuration
# ============================================================

FOUNDRY_ENDPOINT = os.environ["FOUNDRY_ENDPOINT"]

FOUNDRY_API_KEY = os.environ["FOUNDRY_API_KEY"]

# This must be the chat model deployment name.
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

VECTOR_SERVICE_URL = os.getenv(
    "VECTOR_SERVICE_URL",
    "http://vector:5173",
)


# ============================================================
# Microsoft Foundry client
# ============================================================

foundry_base_url = (
    FOUNDRY_ENDPOINT.rstrip("/")
    + "/openai/v1/"
)

foundry_client = OpenAI(
    base_url=foundry_base_url,
    api_key=FOUNDRY_API_KEY,
)


# ============================================================
# Health check
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
# Chat endpoint
# ============================================================

@app.route(
    "/chat",
    methods=["POST"],
)
def chat():
    try:
        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:
            return jsonify(
                {
                    "error": "Request body is required.",
                }
            ), 400

        user_message = str(
            data.get("message", "")
        ).strip()

        if not user_message:
            return jsonify(
                {
                    "error": "message is required.",
                }
            ), 400

        limit = int(
            data.get("limit", 3)
        )

        if limit < 1 or limit > 10:
            limit = 3

        # ----------------------------------------------------
        # Search through vector service
        # ----------------------------------------------------

        vector_response = requests.post(
            f"{VECTOR_SERVICE_URL.rstrip('/')}/search",
            json={
                "query": user_message,
                "limit": limit,
            },
            timeout=60,
        )

        if vector_response.status_code != 200:
            logger.error(
                "Vector service returned %s: %s",
                vector_response.status_code,
                vector_response.text,
            )

            return jsonify(
                {
                    "error": "Vector service failed.",
                    "details": vector_response.text,
                }
            ), 502

        vector_data = vector_response.json()

        supporting_text = vector_data.get(
            "context",
            "",
        )

        results = vector_data.get(
            "results",
            [],
        )

        # ----------------------------------------------------
        # Generate answer using Foundry chat model
        # ----------------------------------------------------

        system_prompt = """
You are a helpful retrieval-augmented generation assistant.

Answer the user's question using only the supporting knowledge.

If the supporting knowledge does not contain enough
information to answer the question, say:
"I don't have enough information to answer that."

Do not invent facts.
Do not use information that is not present in the
supporting knowledge.
"""

        user_prompt = f"""
Supporting knowledge:

{supporting_text}

User question:

{user_message}
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

        if not response.choices:
            return jsonify(
                {
                    "error": "Chat model returned no choices.",
                }
            ), 502

        reply = (
            response.choices[0]
            .message.content
            or ""
        )

        return jsonify(
            {
                "model": response.model,
                "reply": reply,
                "context": supporting_text,
                "results": results,
            }
        ), 200

    except requests.exceptions.RequestException as exc:
        logger.exception(
            "Unable to connect to vector service"
        )

        return jsonify(
            {
                "error": "Unable to connect to vector service.",
                "details": str(exc),
            }
        ), 503

    except ValueError as exc:
        logger.exception(
            "Invalid request value"
        )

        return jsonify(
            {
                "error": str(exc),
            }
        ), 400

    except Exception as exc:
        logger.exception(
            "Chat error"
        )

        return jsonify(
            {
                "error": "Internal server error.",
                "details": str(exc),
            }
        ), 500


# ============================================================
# Application startup
# ============================================================

if __name__ == "__main__":
    logger.info(
        "Starting backend service"
    )

    logger.info(
        "Vector service URL: %s",
        VECTOR_SERVICE_URL,
    )

    logger.info(
        "Chat deployment: %s",
        CHAT_DEPLOYMENT,
    )

    app.run(
        host="0.0.0.0",
        port=5000,
    )
