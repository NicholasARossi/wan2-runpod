#!/bin/bash

echo "=== wan2-runpod entrypoint ==="
echo "Starting ComfyUI in background..."
python3 /ComfyUI/main.py --listen --use-sage-attention 2>&1 | tee /tmp/comfyui.log &

# Start RunPod handler immediately — it has its own ComfyUI readiness check.
# RunPod expects the handler to register within seconds; waiting for ComfyUI
# here causes the worker to be killed before the handler starts.
echo "Starting RunPod handler (ComfyUI will be ready when first job arrives)..."
exec python3 handler.py
