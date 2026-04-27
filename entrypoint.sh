#!/bin/bash

echo "=== wan2-runpod entrypoint ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo "Disk free:"
df -h / 2>/dev/null | tail -1
echo "Python: $(python3 --version 2>&1)"

echo "Starting ComfyUI in background..."
python3 /ComfyUI/main.py --listen --use-sage-attention --disable-mmap 2>&1 | tee /tmp/comfyui.log &
COMFY_PID=$!
echo "ComfyUI PID: $COMFY_PID"

# Start RunPod handler immediately — it has its own ComfyUI readiness check.
# RunPod expects the handler to register within seconds; waiting for ComfyUI
# here causes the worker to be killed before the handler starts.
echo "Starting RunPod handler (ComfyUI will be ready when first job arrives)..."
python3 handler.py 2>&1
HANDLER_EXIT=$?
echo "!!! handler.py exited with code $HANDLER_EXIT !!!"
echo "=== Last 50 lines of ComfyUI log ==="
tail -50 /tmp/comfyui.log 2>/dev/null
echo "=== END ==="
exit $HANDLER_EXIT
