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
    git clone https://github.com/wan22fmlf/ComfyUI-Wan22FMLF

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use && \
    cd ComfyUI-Easy-Use && pip3 install --no-cache-dir -r requirements.txt

# Download all models in parallel (baked into image to avoid cold-start downloads)
# ~40GB total — parallel downloads keep build under 30min limit
RUN mkdir -p /ComfyUI/models/text_encoders /ComfyUI/models/upscale_models \
    /ComfyUI/models/loras/HIGH /ComfyUI/models/loras/LOW && \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors \
        -O /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors \
        -O /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors & \
    wget -q https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors \
        -O /ComfyUI/models/loras/high_noise_model.safetensors & \
    wget -q https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors \
        -O /ComfyUI/models/loras/low_noise_model.safetensors & \
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
        -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors \
        -O /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
        -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors & \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
        -O /ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth & \
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        -O /ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors \
        -O /ComfyUI/models/loras/HIGH/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors & \
    wget -q https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
        -O /ComfyUI/models/loras/LOW/Wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors & \
    wait

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
