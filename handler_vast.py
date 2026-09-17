"""
Vast.ai HTTP Handler for Wan 2.2 Image-to-Video
=================================================
Flask app exposing the I2V pipeline as HTTP endpoints.
Dispatches to handler_svi (SVI mode) or handler_core based on HANDLER_MODE env var.

Endpoints:
  POST /generate/sync  — run the full I2V pipeline (synchronous)
  GET  /health         — health check for PyWorker readiness detection
"""

import os

from flask import Flask, request, jsonify

HANDLER_MODE = os.getenv("HANDLER_MODE", "svi")

if HANDLER_MODE == "svi":
    from handler_svi import process_request
else:
    from handler_core import process_request

app = Flask(__name__)


@app.route("/generate/sync", methods=["POST"])
def generate():
    payload = request.json
    # Accept either {input: {...}} wrapper or flat payload
    job_input = payload.get("input", payload)
    result = process_request(job_input)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=18000)
