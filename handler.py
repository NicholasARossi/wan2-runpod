"""
RunPod Serverless Handler for Wan 2.2 Image-to-Video
=====================================================
Forked from wlsdml1114/generate_video, rewritten with:
  - Proper parameter wiring (seed, cfg, steps all applied to workflow)
  - Input validation before ComfyUI submission
  - Sane defaults (no crashes on missing params)
  - Clear error messages (English, no raw tracebacks)

Workflow node map (wan22_i2v.json):
  244  LoadImage           — input image
  135  WanVideoTextEncode  — positive/negative prompt
  136  T5TextEncoder       — text encoder loader
  173  CLIPVisionLoader    — CLIP vision model
  171  ImageResizeKJv2     — resize input to target dims
  193  WanVideoClipVision  — encode image with CLIP
  122  WanVideoModelLoader — HIGH noise diffusion model
  549  WanVideoModelLoader — LOW noise diffusion model
  525  EnhancedBlockSwap   — VRAM optimization
  555  WanVideoSetBlockSwap— apply block swap to HIGH model
  279  WanVideoLoraSelectMulti — HIGH noise LoRAs (lora_0=lightx2v, lora_1-4=user)
  553  WanVideoLoraSelectMulti — LOW noise LoRAs  (lora_0=lightx2v, lora_1-4=user)
  556  WanVideoSetLoRAs    — apply HIGH LoRAs to model
  552  WanVideoSetLoRAs    — apply LOW LoRAs to model
  541  WanVideoI2VEncode   — encode start image + CLIP
  498  WanVideoContextOpts — FreeNoise context settings
  569  INTConstant         — steps
  570  CreateCFGSchedule   — CFG schedule (from cfg value)
  575  INTConstant         — high/low noise split step
  235  INTConstant         — width
  236  INTConstant         — height
  220  WanVideoSampler     — HIGH noise sampler (pass 1)
  540  WanVideoSampler     — LOW noise sampler (pass 2)
  612  WanVideoDecode      — decode latents to frames
  129  WanVideoVAELoader   — VAE model
  131  VHS_VideoCombine    — encode frames to mp4
"""

import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


# ── Utilities ──────────────────────────────────────────────────────────

def to_nearest_16(value: int) -> int:
    """Round to nearest multiple of 16, minimum 16."""
    adjusted = int(round(value / 16.0) * 16)
    return max(adjusted, 16)


def validate_input(job_input: dict) -> tuple[dict, str | None]:
    """Validate and normalize job input. Returns (normalized_input, error_or_none)."""
    errors = []

    # Image: at least one source required
    has_image = any(k in job_input for k in ("image_base64", "image_url", "image_path", "image"))
    if not has_image:
        errors.append("No image provided. Use image_base64, image_url, or image_path.")

    # Numeric bounds
    width = job_input.get("width", 480)
    height = job_input.get("height", 832)
    if not isinstance(width, (int, float)) or width <= 0:
        errors.append(f"Invalid width: {width}")
    if not isinstance(height, (int, float)) or height <= 0:
        errors.append(f"Invalid height: {height}")

    length = job_input.get("length", 81)
    if not isinstance(length, (int, float)) or length < 1:
        errors.append(f"Invalid length: {length}")

    steps = job_input.get("steps", 8)
    if not isinstance(steps, (int, float)) or steps < 1 or steps > 100:
        errors.append(f"Invalid steps: {steps} (must be 1-100)")

    cfg = job_input.get("cfg", 2.0)
    if not isinstance(cfg, (int, float)) or cfg < 0:
        errors.append(f"Invalid cfg: {cfg}")

    lora_pairs = job_input.get("lora_pairs", [])
    if not isinstance(lora_pairs, list):
        errors.append("lora_pairs must be a list")
    elif len(lora_pairs) > 4:
        errors.append(f"Max 4 LoRA pairs supported, got {len(lora_pairs)}")

    if errors:
        return job_input, "; ".join(errors)

    return job_input, None


def resolve_image(job_input: dict, task_id: str) -> tuple[str, str | None]:
    """Resolve image input to a local file path. Returns (path, error_or_none)."""
    temp_dir = task_id

    # Priority: image_base64 > image_url > image_path > image (auto-detect)
    if "image_base64" in job_input:
        return save_base64(job_input["image_base64"], temp_dir, "input_image.jpg")
    elif "image_url" in job_input:
        return download_url(job_input["image_url"], temp_dir, "input_image.jpg")
    elif "image_path" in job_input:
        path = job_input["image_path"]
        if not os.path.exists(path):
            return "", f"Image file not found: {path}"
        return path, None
    elif "image" in job_input:
        data = job_input["image"]
        if isinstance(data, str):
            if data.startswith("http://") or data.startswith("https://"):
                return download_url(data, temp_dir, "input_image.jpg")
            elif os.path.exists(data):
                return data, None
            else:
                return save_base64(data, temp_dir, "input_image.jpg")
        return "", "image parameter must be a string"

    return "", "No image provided"


def save_base64(b64_data: str, temp_dir: str, filename: str) -> tuple[str, str | None]:
    """Decode base64 data to file. Returns (path, error_or_none)."""
    try:
        decoded = base64.b64decode(b64_data)
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(temp_dir, filename))
        with open(path, "wb") as f:
            f.write(decoded)
        logger.info("Saved base64 image to %s (%d bytes)", path, len(decoded))
        return path, None
    except (binascii.Error, ValueError) as e:
        return "", f"Base64 decode failed: {e}"


def download_url(url: str, temp_dir: str, filename: str) -> tuple[str, str | None]:
    """Download file from URL. Returns (path, error_or_none)."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(temp_dir, filename))
        result = subprocess.run(
            ["wget", "-O", path, "--no-verbose", url],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return "", f"Download failed: {result.stderr.strip()}"
        logger.info("Downloaded image from %s to %s", url, path)
        return path, None
    except subprocess.TimeoutExpired:
        return "", "Image download timed out (60s)"
    except Exception as e:
        return "", f"Download error: {e}"


# ── ComfyUI Communication ─────────────────────────────────────────────

def queue_prompt(prompt: dict) -> dict:
    """Submit workflow to ComfyUI. Returns response dict."""
    url = f"http://{server_address}:8188/prompt"
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected workflow (HTTP {e.code}): {body}")


def get_history(prompt_id: str) -> dict:
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def get_videos(ws, prompt: dict) -> dict:
    """Submit prompt, wait for completion via websocket, return video data."""
    resp = queue_prompt(prompt)
    prompt_id = resp["prompt_id"]
    logger.info("Job queued, prompt_id=%s", prompt_id)

    while True:
        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            if msg["type"] == "executing":
                data = msg["data"]
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
            elif msg["type"] == "execution_error":
                error_data = msg.get("data", {})
                raise RuntimeError(
                    f"ComfyUI execution error on node {error_data.get('node_id', '?')}: "
                    f"{error_data.get('exception_message', 'unknown')}"
                )

    history = get_history(prompt_id)[prompt_id]
    output_videos = {}
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        if "gifs" in node_output:
            for video in node_output["gifs"]:
                with open(video["fullpath"], "rb") as f:
                    video_data = base64.b64encode(f.read()).decode("utf-8")
                output_videos.setdefault(node_id, []).append(video_data)

    return output_videos


def load_workflow(workflow_path: str) -> dict:
    if not os.path.isabs(workflow_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(current_dir, workflow_path)
    with open(workflow_path, "r") as f:
        return json.load(f)


# ── Main Handler ───────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})
    task_id = f"task_{uuid.uuid4()}"
    logger.info("Received job, task_id=%s", task_id)

    # 1. Validate input
    job_input, error = validate_input(job_input)
    if error:
        logger.error("Validation failed: %s", error)
        return {"error": error}

    # 2. Resolve image to local path
    image_path, error = resolve_image(job_input, task_id)
    if error:
        logger.error("Image resolution failed: %s", error)
        return {"error": error}

    # 3. Extract params with defaults
    prompt_text = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    width = to_nearest_16(int(job_input.get("width", 480)))
    height = to_nearest_16(int(job_input.get("height", 832)))
    length = int(job_input.get("length", 81))
    steps = int(job_input.get("steps", 8))
    seed = int(job_input.get("seed", random.randint(0, 2**53)))
    cfg = float(job_input.get("cfg", 2.0))
    context_overlap = int(job_input.get("context_overlap", 48))
    lora_pairs = job_input.get("lora_pairs", [])

    logger.info(
        "Params: %dx%d, %d frames, %d steps, seed=%d, cfg=%.1f, %d LoRAs",
        width, height, length, steps, seed, cfg, len(lora_pairs),
    )

    # 4. Load and configure workflow
    workflow = load_workflow("workflow/wan22_i2v.json")

    # Image
    workflow["244"]["inputs"]["image"] = image_path

    # Prompt
    workflow["135"]["inputs"]["positive_prompt"] = prompt_text
    workflow["135"]["inputs"]["negative_prompt"] = negative_prompt

    # Resolution
    workflow["235"]["inputs"]["value"] = width
    workflow["236"]["inputs"]["value"] = height

    # Frame count
    workflow["541"]["inputs"]["num_frames"] = length

    # Steps (shared by both samplers via node 569)
    workflow["569"]["inputs"]["value"] = steps

    # High/Low noise split point (60% of steps for high noise)
    split_step = max(1, int(steps * 0.5))
    workflow["575"]["inputs"]["value"] = split_step

    # Seed (both samplers)
    workflow["220"]["inputs"]["seed"] = seed
    workflow["540"]["inputs"]["seed"] = seed

    # CFG (applied via CFG schedule on high sampler, direct on low sampler)
    workflow["570"]["inputs"]["cfg_scale_start"] = cfg
    workflow["570"]["inputs"]["cfg_scale_end"] = cfg
    workflow["540"]["inputs"]["cfg"] = max(cfg * 0.5, 1.0)

    # Context options
    workflow["498"]["inputs"]["context_overlap"] = context_overlap
    workflow["498"]["inputs"]["context_frames"] = length

    # LoRA pairs: lora_0 = lightx2v (built-in), lora_1-4 = user LoRAs
    for i, lora_pair in enumerate(lora_pairs[:4]):
        slot = i + 1  # lora_0 is reserved for lightx2v
        high_name = lora_pair.get("high")
        low_name = lora_pair.get("low")
        high_weight = lora_pair.get("high_weight", 1.0)
        low_weight = lora_pair.get("low_weight", 1.0)

        if high_name:
            workflow["279"]["inputs"][f"lora_{slot}"] = high_name
            workflow["279"]["inputs"][f"strength_{slot}"] = high_weight
            logger.info("LoRA %d HIGH: %s (weight=%.2f)", slot, high_name, high_weight)

        if low_name:
            workflow["553"]["inputs"][f"lora_{slot}"] = low_name
            workflow["553"]["inputs"][f"strength_{slot}"] = low_weight
            logger.info("LoRA %d LOW: %s (weight=%.2f)", slot, low_name, low_weight)

    # 5. Connect to ComfyUI and run
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    http_url = f"http://{server_address}:8188/"

    # Wait for ComfyUI HTTP
    for attempt in range(180):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("ComfyUI HTTP ready (attempt %d)", attempt + 1)
            break
        except Exception:
            if attempt == 179:
                return {"error": "ComfyUI server not reachable after 3 minutes"}
            time.sleep(1)

    # WebSocket connection
    ws = websocket.WebSocket()
    for attempt in range(36):
        try:
            ws.connect(ws_url)
            logger.info("WebSocket connected (attempt %d)", attempt + 1)
            break
        except Exception as e:
            if attempt == 35:
                return {"error": f"WebSocket connection failed after 3 minutes: {e}"}
            time.sleep(5)

    try:
        videos = get_videos(ws, workflow)
    except RuntimeError as e:
        logger.error("Generation failed: %s", e)
        return {"error": str(e)}
    finally:
        ws.close()

    # 6. Return result
    for node_id in videos:
        if videos[node_id]:
            logger.info("Video generated successfully from node %s", node_id)
            return {"video": videos[node_id][0]}

    return {"error": "No video output produced. Check ComfyUI logs for details."}


runpod.serverless.start({"handler": handler})
