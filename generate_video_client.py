#!/usr/bin/env python3
"""
RunPod API client for wan2-runpod serverless endpoint.

Usage:
    python generate_video_client.py --image test.png --prompt "woman dancing"
    python generate_video_client.py --image test.png --prompt "titJob, paizuri..." --lora high.safetensors low.safetensors
"""

import argparse
import base64
import json
import os
import sys
import time
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class GenerateVideoClient:
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def health(self) -> dict:
        """Check endpoint health."""
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def submit(self, input_data: dict) -> str:
        """Submit a job, return job ID."""
        payload = {"input": input_data}
        resp = self.session.post(f"{self.base_url}/run", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"No job ID in response: {data}")
        log.info("Job submitted: %s", job_id)
        return job_id

    def poll(self, job_id: str, interval: int = 10, timeout: int = 1800) -> dict:
        """Poll for job completion."""
        start = time.time()
        while time.time() - start < timeout:
            resp = self.session.get(f"{self.base_url}/status/{job_id}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "COMPLETED":
                log.info("Job completed")
                return data
            elif status == "FAILED":
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")
            else:
                elapsed = int(time.time() - start)
                log.info("Status: %s (%ds elapsed)", status, elapsed)
                time.sleep(interval)

        raise TimeoutError(f"Job timed out after {timeout}s")

    def generate(
        self,
        image_path: str,
        prompt: str = "",
        negative_prompt: str | None = None,
        width: int = 480,
        height: int = 832,
        length: int = 81,
        steps: int = 8,
        seed: int | None = None,
        cfg: float = 2.0,
        context_overlap: int = 48,
        lora_pairs: list[dict] | None = None,
    ) -> dict:
        """Generate video from image. Returns job result with base64 video."""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        input_data = {
            "image_base64": image_b64,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "cfg": cfg,
            "context_overlap": context_overlap,
        }

        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        if seed is not None:
            input_data["seed"] = seed
        if lora_pairs:
            input_data["lora_pairs"] = lora_pairs

        job_id = self.submit(input_data)
        return self.poll(job_id)

    def save_video(self, result: dict, output_path: str) -> bool:
        """Save video from job result to file."""
        output = result.get("output", {})
        video_b64 = output.get("video")
        if not video_b64:
            log.error("No video data in result")
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        decoded = base64.b64decode(video_b64)
        with open(output_path, "wb") as f:
            f.write(decoded)

        size_mb = len(decoded) / (1024 * 1024)
        log.info("Saved: %s (%.1f MB)", output_path, size_mb)
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate video via wan2-runpod")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--prompt", "-p", default="", help="Motion/scene prompt")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--length", type=int, default=81, help="Frame count")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--lora", nargs=2, action="append", metavar=("HIGH", "LOW"),
                        help="LoRA pair: --lora high.safetensors low.safetensors")
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--endpoint", default=None,
                        help="RunPod endpoint ID (or RUNPOD_ENDPOINT_ID env)")
    parser.add_argument("--api-key", default=None,
                        help="RunPod API key (or RUNPOD_API_KEY env)")
    parser.add_argument("--health", action="store_true", help="Just check health")
    args = parser.parse_args()

    endpoint_id = args.endpoint or os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")

    if not endpoint_id or not api_key:
        print("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY env vars, or use --endpoint/--api-key")
        sys.exit(1)

    client = GenerateVideoClient(endpoint_id, api_key)

    if args.health:
        print(json.dumps(client.health(), indent=2))
        return

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    lora_pairs = []
    if args.lora:
        for high, low in args.lora:
            lora_pairs.append({
                "high": high, "low": low,
                "high_weight": args.lora_weight,
                "low_weight": args.lora_weight,
            })

    result = client.generate(
        image_path=args.image,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        lora_pairs=lora_pairs or None,
    )

    if client.save_video(result, args.output):
        print(f"Video saved to {args.output}")
    else:
        print("Failed to save video")
        sys.exit(1)


if __name__ == "__main__":
    main()
