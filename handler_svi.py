"""
SVI v1 handler for Kenpechi SVI workflow on RunPod serverless.
==============================================================
Single-scene Wan 2.2 I2V using the SVI (Smooth Video Interpolation) sampler
with Power Lora Loader (rgthree) for per-scene LoRA toggling.

Workflow node map (svi_v1.json):
  1   LoadImage                — input image
  2   ImageResizeKJv2          — resize to internal dims (832x480)
  3   Safetensors_Models       — load HIGH + LOW diffusion models (component)
  4   LoraLoaderModelOnly      — Lightning distill HIGH (v1030)
  5   LoraLoaderModelOnly      — Lightning distill LOW (v1022)
  6   SVI_CLIP_VAE_Models      — SVI LoRAs + CLIP + VAE + CLIP Vision (component)
  7   PathchSageAttentionKJ    — sage attention HIGH
  8   PathchSageAttentionKJ    — sage attention LOW
  9   ModelPatchTorchSettings   — fp16 accumulation HIGH
  10  ModelPatchTorchSettings   — fp16 accumulation LOW
  11  ModelSamplingSD3          — shift=5 HIGH
  12  ModelSamplingSD3          — shift=5 LOW
  13  Power Lora Loader         — scene LoRAs HIGH
  14  Power Lora Loader         — scene LoRAs LOW
  15  CLIPTextEncode            — positive prompt
  16  CLIPTextEncode            — negative prompt
  17  1st_Section SVI sampler   — main sampler (component)
  18  VHS_VideoCombine          — encode to mp4

Key differences from handler_core.py:
  - SVI sampler instead of WanVideoSampler (7 steps, split at 3)
  - Power Lora Loader (rgthree) instead of WanVideoLoraSelectMulti
  - LoRA pairs use HIGH\\ and LOW\\ subdirectory paths
  - ModelSamplingSD3 shift=5, SageAttention, fp16 accumulation
  - No FreeNoise context options (SVI handles temporal coherence internally)
"""

import os
import random
import logging

from handler_core import (
    validate_input,
    resolve_image,
    load_workflow,
    get_videos,
    logger,
    server_address,
    client_id,
)

import urllib.request
import websocket
import time

DEFAULT_NEGATIVE_PROMPT = (
    "watermark, text, subtitles, letterbox, pillarbox, frame, border, "
    "split screen, noise, artifacts, blur, vignette"
)

# SVI defaults (from Kenpechi SVI v3.4 workflow)
SVI_DEFAULTS = {
    "width": 720,
    "height": 1072,
    "steps": 7,
    "split_step": 3,
    "shift": 5,
    "frame_rate": 16.0,
    "duration_seconds": 2.5,
    "structural_repulsion_boost": 1.0,
    "svi_motion_strength": 1.0,
}


def build_lora_slot(filename, strength, on=True):
    """Build a Power Lora Loader slot dict."""
    return {
        "on": on,
        "lora": filename,
        "strength": strength,
        "strengthTwo": None,
    }


def process_request(job_input: dict) -> dict:
    """Run the SVI I2V pipeline. Entry point for RunPod handler.

    API format:
        image_base64/image_url/image_path: input image
        prompt: positive prompt text
        negative_prompt: negative prompt (optional)
        width: output width (default 720)
        height: output height (default 1072)
        duration_seconds: video length in seconds (default 2.5)
        steps: total sampling steps (default 7)
        split_step: high/low noise split (default 3)
        seed: random seed (default random)
        lora_pairs: list of LoRA pair dicts:
            [{"high": "HIGH\\filename.safetensors", "low": "LOW\\filename.safetensors",
              "high_weight": 0.8, "low_weight": 0.8, "on": true}]
            Max 4 pairs per noise path.
    """
    import uuid
    task_id = f"task_{uuid.uuid4()}"
    logger.info("SVI request received, task_id=%s", task_id)

    # Health check
    if job_input.get("health_check"):
        logger.info("Health check passed")
        return {"status": "ok", "message": "svi-v1 handler is running"}

    # Validate
    job_input, error = validate_input(job_input)
    if error:
        logger.error("Validation failed: %s", error)
        return {"error": error}

    # Resolve image
    image_path, error = resolve_image(job_input, task_id)
    if error:
        logger.error("Image resolution failed: %s", error)
        return {"error": error}

    # Extract params
    prompt_text = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    width = int(job_input.get("width", SVI_DEFAULTS["width"]))
    height = int(job_input.get("height", SVI_DEFAULTS["height"]))
    duration = float(job_input.get("duration_seconds", SVI_DEFAULTS["duration_seconds"]))
    steps = int(job_input.get("steps", SVI_DEFAULTS["steps"]))
    split_step = int(job_input.get("split_step", SVI_DEFAULTS["split_step"]))
    seed = int(job_input.get("seed", random.randint(0, 2**53)))
    frame_rate = float(job_input.get("frame_rate", SVI_DEFAULTS["frame_rate"]))
    repulsion_boost = float(job_input.get("structural_repulsion_boost",
                                           SVI_DEFAULTS["structural_repulsion_boost"]))
    motion_strength = float(job_input.get("svi_motion_strength",
                                           SVI_DEFAULTS["svi_motion_strength"]))
    lora_pairs = job_input.get("lora_pairs", [])

    # Calculate frame count: duration * frame_rate + 1
    length = int(duration * frame_rate) + 1

    logger.info(
        "SVI params: %dx%d, %.1fs (%d frames), %d steps (split %d), seed=%d, %d LoRAs",
        width, height, duration, length, steps, split_step, seed, len(lora_pairs),
    )

    # Load workflow
    workflow = load_workflow("workflow/svi_v1.json")

    # Node 1: LoadImage
    workflow["1"]["inputs"]["image"] = image_path

    # Node 2: ImageResizeKJv2 (internal resize — keep defaults)

    # Node 15: Positive prompt
    workflow["15"]["inputs"]["text"] = prompt_text

    # Node 16: Negative prompt
    workflow["16"]["inputs"]["text"] = negative_prompt

    # Node 17: SVI Sampler
    workflow["17"]["inputs"]["width"] = width
    workflow["17"]["inputs"]["height"] = height
    workflow["17"]["inputs"]["length"] = length
    workflow["17"]["inputs"]["steps"] = steps
    workflow["17"]["inputs"]["end_at_step"] = split_step
    workflow["17"]["inputs"]["noise_seed"] = seed
    workflow["17"]["inputs"]["structural_repulsion_boost"] = repulsion_boost
    workflow["17"]["inputs"]["svi_motion_strength"] = motion_strength
    workflow["17"]["inputs"]["frame_rate"] = frame_rate

    # Node 18: Video output frame rate
    workflow["18"]["inputs"]["frame_rate"] = frame_rate

    # Configure LoRA pairs via Power Lora Loader nodes
    # Node 13 = HIGH, Node 14 = LOW
    # Each supports up to 4 slots (lora_01..lora_04)
    # Default slots from workflow are already populated; override with user LoRAs if provided
    if lora_pairs:
        # Build HIGH and LOW slot dicts
        high_slots = {}
        low_slots = {}
        for i, pair in enumerate(lora_pairs[:4]):
            slot_key = f"lora_{i+1:02d}"
            on = pair.get("on", True)

            if pair.get("high"):
                high_slots[slot_key] = build_lora_slot(
                    pair["high"],
                    pair.get("high_weight", 1.0),
                    on=on,
                )
            if pair.get("low"):
                low_slots[slot_key] = build_lora_slot(
                    pair["low"],
                    pair.get("low_weight", 1.0),
                    on=on,
                )

        # Clear existing default slots and apply user LoRAs
        for slot_key in ["lora_01", "lora_02", "lora_03", "lora_04"]:
            if slot_key in high_slots:
                workflow["13"]["inputs"][slot_key] = high_slots[slot_key]
                logger.info("HIGH %s: %s (w=%.2f, on=%s)",
                           slot_key, high_slots[slot_key]["lora"],
                           high_slots[slot_key]["strength"],
                           high_slots[slot_key]["on"])
            elif slot_key in workflow["13"]["inputs"]:
                # Turn off unused slots
                existing = workflow["13"]["inputs"][slot_key]
                if isinstance(existing, dict):
                    existing["on"] = False

            if slot_key in low_slots:
                workflow["14"]["inputs"][slot_key] = low_slots[slot_key]
                logger.info("LOW %s: %s (w=%.2f, on=%s)",
                           slot_key, low_slots[slot_key]["lora"],
                           low_slots[slot_key]["strength"],
                           low_slots[slot_key]["on"])
            elif slot_key in workflow["14"]["inputs"]:
                existing = workflow["14"]["inputs"][slot_key]
                if isinstance(existing, dict):
                    existing["on"] = False

    # Connect to ComfyUI and run
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

    # Return result
    for node_id in videos:
        if videos[node_id]:
            logger.info("Video generated from node %s", node_id)
            return {"video": videos[node_id][0]}

    return {"error": "No video output produced. Check ComfyUI logs for details."}
