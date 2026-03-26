"""
RunPod Serverless Handler for Wan 2.2 Image-to-Video
=====================================================
Thin wrapper around handler_core.process_request() for RunPod's
serverless infrastructure. All pipeline logic lives in handler_core.py.
"""

import runpod
from handler_core import process_request

runpod.serverless.start({"handler": lambda job: process_request(job["input"])})
