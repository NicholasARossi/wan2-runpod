"""
RunPod Serverless Handler for Wan 2.2 Image-to-Video
=====================================================
Routes to the appropriate handler based on HANDLER_MODE env var:
  - "svi" → handler_svi.py (SVI v1 workflow with Power Lora Loader)
  - default → handler_core.py (original WanVideoWrapper workflow)
"""

import os
import runpod

HANDLER_MODE = os.getenv("HANDLER_MODE", "svi")

if HANDLER_MODE == "svi":
    from handler_svi import process_request
else:
    from handler_core import process_request

runpod.serverless.start({"handler": lambda job: process_request(job["input"])})
