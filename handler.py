"""
RunPod Serverless Handler for Wan 2.2 Image-to-Video
=====================================================
Routes to the appropriate handler based on HANDLER_MODE env var:
  - "svi" → handler_svi.py (SVI v1 workflow with Power Lora Loader)
  - default → handler_core.py (original WanVideoWrapper workflow)
"""

import os
import sys
import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("handler")

try:
    log.info("=== handler.py starting ===")
    log.info("Python %s", sys.version)
    log.info("CWD: %s", os.getcwd())
    log.info("ENV HANDLER_MODE=%s", os.getenv("HANDLER_MODE", "(unset, default svi)"))

    import runpod
    log.info("runpod SDK version: %s", getattr(runpod, "__version__", "unknown"))

    HANDLER_MODE = os.getenv("HANDLER_MODE", "svi")

    if HANDLER_MODE == "svi":
        log.info("Importing handler_svi...")
        from handler_svi import process_request
        log.info("handler_svi imported OK")
    else:
        log.info("Importing handler_core...")
        from handler_core import process_request
        log.info("handler_core imported OK")

    log.info("Starting runpod.serverless.start()...")
    runpod.serverless.start({"handler": lambda job: process_request(job["input"])})

except Exception:
    log.critical("FATAL: handler.py crashed during startup:\n%s", traceback.format_exc())
    sys.exit(1)
