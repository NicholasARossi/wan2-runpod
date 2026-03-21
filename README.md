[![Runpod](https://api.runpod.io/badge/NicholasARossi/wan2-runpod)](https://console.runpod.io/hub/NicholasARossi/wan2-runpod)

# wan2-runpod

Custom RunPod serverless endpoint for Wan 2.2 image-to-video generation with LoRA support.

Forked from [wlsdml1114/generate_video](https://github.com/wlsdml1114/generate_video) — rewritten handler with proper parameter wiring, input validation, and error handling.

## What This Does

- Runs Wan 2.2 14B I2V (image-to-video) as a RunPod serverless endpoint
- Accepts a base64-encoded image + text prompt, returns a base64-encoded video
- Supports up to 4 custom LoRA pairs (HIGH + LOW noise models)
- Uses ComfyUI as the inference backend inside the Docker container

## API

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "<base64_encoded_image>",
      "prompt": "titJob, paizuri, girlMove, gather She is being tittyfucked...",
      "width": 480,
      "height": 832,
      "length": 81,
      "steps": 8,
      "seed": 42,
      "cfg": 2.0,
      "lora_pairs": [
        {
          "high": "WAN-2.2-I2V-POV-Titfuck-Paizuri-HIGH-v1.0.safetensors",
          "low": "WAN-2.2-I2V-POV-Titfuck-Paizuri-LOW-v1.0.safetensors",
          "high_weight": 1.0,
          "low_weight": 1.0
        }
      ]
    }
  }'
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_base64` | string | required | Base64-encoded input image |
| `image_url` | string | - | Alternative: URL to download image from |
| `image_path` | string | - | Alternative: path on the server |
| `prompt` | string | "" | Motion/scene description + LoRA trigger words |
| `negative_prompt` | string | (default) | What to avoid |
| `width` | int | 480 | Output width (auto-rounded to 16x) |
| `height` | int | 832 | Output height (auto-rounded to 16x) |
| `length` | int | 81 | Number of frames (~5 sec at 16fps) |
| `steps` | int | 8 | Sampling steps (4-20, 8 recommended with lightx2v) |
| `seed` | int | random | Reproducibility seed |
| `cfg` | float | 2.0 | Classifier-free guidance scale |
| `context_overlap` | int | 48 | FreeNoise context overlap |
| `lora_pairs` | array | [] | Up to 4 LoRA pairs (see below) |

### LoRA Pairs Format

```json
{
  "high": "lora_high_noise.safetensors",
  "low": "lora_low_noise.safetensors",
  "high_weight": 1.0,
  "low_weight": 1.0
}
```

LoRA files must be on the RunPod network volume at `/runpod-volume/loras/`.

## Architecture

```
Docker Container (RunPod Serverless)
┌─────────────────────────────────────────────┐
│  entrypoint.sh                              │
│    ├─ Starts ComfyUI (background, port 8188)│
│    └─ Starts handler.py (foreground)        │
│                                             │
│  handler.py (RunPod serverless handler)     │
│    ├─ Receives job input via RunPod SDK     │
│    ├─ Validates params, loads workflow JSON  │
│    ├─ Wires params into workflow nodes       │
│    ├─ Submits workflow to ComfyUI /prompt    │
│    ├─ Waits for completion via WebSocket     │
│    └─ Returns base64 video                  │
│                                             │
│  ComfyUI (inference engine, port 8188)      │
│    ├─ Wan 2.2 14B fp8 models (baked in)     │
│    ├─ lightx2v 4-step acceleration LoRA     │
│    ├─ Custom nodes: WanVideoWrapper, etc.   │
│    └─ /runpod-volume/loras/ (user LoRAs)    │
└─────────────────────────────────────────────┘
```

## Differences from Upstream

| Issue | upstream (wlsdml1114) | This fork |
|-------|----------------------|-----------|
| seed/cfg/steps | Required or ignored depending on branch | Optional with sane defaults, always wired to workflow |
| Error handling | Raw tracebacks, Korean error messages | Descriptive English errors, input validation before ComfyUI |
| ksampler branch | Broken (confirmed by maintainer) | Not used — we use the WanVideoWrapper approach |
| LoRA handling | Fragile, untested | Validated, max 4 pairs, clear error if file not found |
| Parameter validation | None — crashes on bad input | Width/height rounded to 16x, bounds checking |

## Local Testing (Mac Studio)

For workflow validation without RunPod, see `workflow/wan22_local_test.json` which uses GGUF-quantized models compatible with Apple Silicon MPS backend. Slow (~30 min/video) but catches workflow JSON errors before deployment.

## License

MIT — with credit to [wlsdml1114/generate_video](https://github.com/wlsdml1114/generate_video) for the original ComfyUI serverless pattern.
