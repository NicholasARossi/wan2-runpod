#!/usr/bin/env python3
"""Submit the local GGUF workflow to ComfyUI for testing."""

import json
import sys
import time
import uuid
import urllib.request
import urllib.error
import websocket

SERVER = "127.0.0.1"
CLIENT_ID = str(uuid.uuid4())


def queue_prompt(prompt):
    url = f"http://{SERVER}:8188/prompt"
    data = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(url, data=data)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def main():
    workflow_path = "workflow/wan22_local_test.json"
    if len(sys.argv) > 1:
        workflow_path = sys.argv[1]

    with open(workflow_path) as f:
        workflow = json.load(f)

    print(f"Submitting workflow: {workflow_path}")
    print(f"Resolution: {workflow['235']['inputs']['value']}x{workflow['236']['inputs']['value']}")
    print(f"Frames: {workflow['541']['inputs']['num_frames']}")
    print(f"Steps: {workflow['569']['inputs']['value']}")
    print(f"HIGH model: {workflow['122']['inputs']['model']}")
    print(f"LOW model: {workflow['549']['inputs']['model']}")

    # Connect websocket first
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER}:8188/ws?clientId={CLIENT_ID}")
    print("WebSocket connected")

    # Submit
    try:
        resp = queue_prompt(workflow)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: ComfyUI rejected workflow (HTTP {e.code}):")
        print(body[:2000])
        ws.close()
        sys.exit(1)

    prompt_id = resp["prompt_id"]
    print(f"Job queued: {prompt_id}")

    start = time.time()
    while True:
        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            msg_type = msg.get("type", "")

            if msg_type == "progress":
                data = msg["data"]
                step = data.get("value", 0)
                total = data.get("max", 0)
                elapsed = int(time.time() - start)
                print(f"  Progress: {step}/{total} ({elapsed}s elapsed)")

            elif msg_type == "executing":
                data = msg["data"]
                node = data.get("node")
                if node is None and data.get("prompt_id") == prompt_id:
                    print(f"DONE! Total time: {int(time.time() - start)}s")
                    break
                elif node:
                    elapsed = int(time.time() - start)
                    print(f"  Executing node {node} ({elapsed}s)")

            elif msg_type == "execution_error":
                error_data = msg.get("data", {})
                print(f"ERROR on node {error_data.get('node_id', '?')}:")
                print(f"  {error_data.get('exception_message', 'unknown')}")
                print(f"  {error_data.get('exception_type', '')}")
                traceback = error_data.get("traceback", [])
                for line in traceback[-5:]:
                    print(f"  {line.strip()}")
                ws.close()
                sys.exit(1)

            elif msg_type == "execution_cached":
                cached = msg.get("data", {}).get("nodes", [])
                if cached:
                    print(f"  Cached nodes: {cached}")

    ws.close()

    # Get output
    url = f"http://{SERVER}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as resp:
        history = json.loads(resp.read())

    outputs = history[prompt_id]["outputs"]
    for node_id, output in outputs.items():
        if "gifs" in output:
            for vid in output["gifs"]:
                print(f"Video saved: {vid['fullpath']}")
                size_mb = vid.get("file_size", 0) / (1024 * 1024) if "file_size" in vid else 0
                if size_mb:
                    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
