# syntax=docker/dockerfile:1.6
# =============================================================================
# diffusion-boundary (ddim) -- portable Blackwell image
# Everything baked in: micromamba env, torch 2.8.0+cu128, ddim code, and model
# weights (SD v1-4, CLIP-L, ViT classifier, DINO ViT-S/8, Inception-FID).
# Secrets (HF/W&B tokens) are NOT baked -- passed at runtime via env vars.
# Gated SD v1-4 is fetched at build via a BuildKit secret (id=hf_token), which
# never persists in a layer.
# =============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl bzip2 ca-certificates git openssh-server \
    && rm -rf /var/lib/apt/lists/*

# --- micromamba ---
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local bin/micromamba

# baked cache locations (image ENV wins over runtime .env; load_dotenv won't override)
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    HF_HOME=/opt/hf \
    HF_HUB_CACHE=/opt/hf/hub \
    HF_MODELS_CACHE=/opt/hf/models \
    HF_DATASETS_CACHE=/opt/hf/datasets \
    TORCH_HOME=/opt/torch

# --- code (ddim branch) ---
RUN git clone -b ddim --single-branch https://github.com/inesgomes/diffusion-boundary.git /opt/diffusion-boundary
WORKDIR /opt/diffusion-boundary

# --- python env + deps + torch cu128 (torch/vision/triton/nvidia pins stripped) ---
RUN micromamba create -y -n dfst -c conda-forge python=3.11 pip \
 && awk '/^      - /{sub(/^      - /,""); print}' environment.yml \
      | grep -vE '^(torch==|torchvision==|triton==|nvidia-)' > /tmp/req.txt \
 && micromamba run -n dfst pip install --no-cache-dir -r /tmp/req.txt \
 && micromamba run -n dfst pip install --no-cache-dir --force-reinstall \
      torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
 && micromamba run -n dfst pip install --no-cache-dir --no-deps \
      "numpy==1.26.4" "fsspec==2024.9.0" \
 && micromamba run -n dfst pip check \
 && micromamba clean -y -a

# --- single-GPU fix: experiment configs hardcode cuda:1 ---
RUN sed -i 's/device: cuda:1/device: cuda:0/' experiments/*.yml

# --- minimal 1-image ddim config (not in the repo) ---
COPY smoke-ddim-1.yml /opt/diffusion-boundary/experiments/smoke-ddim-1.yml

# --- bake model weights (SD v1-4 gated -> hf_token secret) ---
COPY warmup.py /tmp/warmup.py
# warmup must succeed (proves gated download worked); then defensively delete any
# HF token file, so even a future login() call can't bake the secret into a layer.
RUN --mount=type=secret,id=hf_token \
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" \
      micromamba run -n dfst python /tmp/warmup.py \
 && rm -f /opt/hf/token /opt/hf/stored_tokens \
          /root/.cache/huggingface/token /root/.huggingface/token

# --- duckdb for server-side dataset prep (own layer: keeps env+weights cached) ---
RUN micromamba run -n dfst pip install --no-cache-dir duckdb \
 && micromamba run -n dfst pip check

# --- runtime helpers ---
COPY start.sh /usr/local/bin/start.sh
COPY db-run.sh /usr/local/bin/db-run
RUN chmod +x /usr/local/bin/start.sh /usr/local/bin/db-run \
 && mkdir -p /workspace

EXPOSE 22
# start.sh installs the RunPod-injected PUBLIC_KEY and runs sshd in foreground
CMD ["/usr/local/bin/start.sh"]
