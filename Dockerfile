# wan2-runpod: Wan 2.2 I2V RunPod Serverless Endpoint
# Uses same Blackwell-optimized base as LTX deployment (includes CUDA dev tools,
# PyTorch, and RunPod-compatible infra). The runtime-only nvidia base was causing
# worker crashes because ComfyUI custom nodes need CUDA compilation at startup.
FROM wlsdml1114/engui_genai-base_blackwell:1.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Extra system deps (base already has most)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RunPod SDK + websocket (base may have runpod but ensure latest)
RUN pip3 install --no-cache-dir --upgrade runpod websocket-client

# ComfyUI
WORKDIR /
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip3 install --no-cache-dir -r requirements.txt

# Custom nodes
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && pip3 install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && pip3 install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && pip3 install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && pip3 install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    cd rgthree-comfy && pip3 install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/wallen0322/ComfyUI-Wan22FMLF

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use && \
    cd ComfyUI-Easy-Use && pip3 install --no-cache-dir -r requirements.txt

# Download lightweight models only (~10GB). Base diffusion models (2x 14GB) are on
# the Network Volume at /runpod-volume/models/diffusion_models/ to keep image small.
RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/text_encoders \
    /ComfyUI/models/upscale_models /ComfyUI/models/clip_vision \
    /ComfyUI/models/vae /ComfyUI/models/loras/HIGH /ComfyUI/models/loras/LOW && \
    curl -fL -o /ComfyUI/models/clip_vision/clip_vision_h.safetensors \
        https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors & P1=$! ; \
    curl -fL -o /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors \
        https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors & P2=$! ; \
    curl -fL -o /ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth & P3=$! ; \
    curl -fL -o /ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors & P4=$! ; \
    curl -fL -o /ComfyUI/models/loras/HIGH/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors \
        https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors & P5=$! ; \
    curl -fL -o /ComfyUI/models/loras/LOW/Wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
        https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors & P6=$! ; \
    wait $P1 && wait $P2 && wait $P3 && wait $P4 && wait $P5 && wait $P6 && \
    echo "All model downloads complete"

# Verify critical model downloads
RUN test -s /ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    test -s /ComfyUI/models/clip_vision/clip_vision_h.safetensors && \
    test -s /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors && \
    echo "All baked model downloads verified"

# Copy our handler, workflow, and config
WORKDIR /app
COPY handler.py .
COPY handler_core.py .
COPY handler_svi.py .
COPY workflow/ workflow/
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
