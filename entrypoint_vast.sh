#!/bin/bash
set -e

echo "=== wan2-vast entrypoint ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo "Disk free:"
df -h / 2>/dev/null | tail -1
echo "HANDLER_MODE: ${HANDLER_MODE:-core}"

echo "Starting ComfyUI in background..."
python3 /ComfyUI/main.py --listen --use-sage-attention &

echo "Waiting for ComfyUI to be ready..."
max_wait=120
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready! (${wait_count}s)"
        break
    fi
    if [ $((wait_count % 10)) -eq 0 ]; then
        echo "  Waiting... (${wait_count}/${max_wait}s)"
    fi
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "ERROR: ComfyUI failed to start within ${max_wait}s"
    exit 1
fi

echo "Starting Vast.ai handler..."
exec python3 /app/handler_vast.py 2>&1 | tee /var/log/handler_vast.log
