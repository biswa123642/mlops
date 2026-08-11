import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from qdrant_client import QdrantClient


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

CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

VECTOR_SERVICE_URL = os.getenv(
    "VECTOR_SERVICE_URL",
    "http://vector:5173"
)


# ============================================================
# Microsoft Foundry client
# ============================================================

foundry_client = OpenAI(
    base_url=f"{FOUNDRY_ENDPOINT}/openai/v1/",
    api_key=FOUNDRY_API_KEY,
)


# ============================================================
# Health check
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    }), 200


# ============================================================
# Chat endpoint
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Request body is required."
            }), 400

        user_message = data.get(
            "message",
            ""
        ).strip()

        if not user_message:

            return jsonify({
                "error": "message is required."
            }), 400


        # ----------------------------------------------------
        # Generate embedding + search Qdrant
        # through Vector Service
        # ----------------------------------------------------

        # TODO:
        # This endpoint will be implemented in the vector
        # service.
        #
        # For now this is the expected API:
        #
        # POST http://vector:5173/search
        #
        # {
        #     "query": "user question",
        #     "limit": 3
        # }

        import requests

        vector_response = requests.post(

            f"{VECTOR_SERVICE_URL}/search",

            json={
                "query": user_message,
                "limit": 3
            },

            timeout=30
        )


        if vector_response.status_code != 200:

            return jsonify({

                "error": "Vector service failed.",

                "details": vector_response.text

            }), 502


        vector_data = vector_response.json()

        supporting_text = vector_data.get(
            "context",
            ""
        )


        # ----------------------------------------------------
        # Generate answer using Foundry chat model
        # ----------------------------------------------------

        system_prompt = """
You are a helpful RAG assistant.

Answer the user's question using the provided
supporting knowledge.

If the supporting knowledge does not contain
enough information to answer the question,
say that you don't have enough information.

Do not invent facts.
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
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.7
        )


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "model": response.model,

            "reply": response.choices[
                0
            ].message.content,

            "context": supporting_text

        }), 200


    except requests.exceptions.RequestException as e:

        return jsonify({

            "error": "Unable to connect to vector service.",

            "details": str(e)

        }), 503


    except Exception as e:

        print(
            f"Chat error: {e}"
        )

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# Application startup
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )