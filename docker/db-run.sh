#!/usr/bin/env bash
# Self-contained experiment runner. Works from ANY ssh shell: sshd sessions don't
# inherit the container's ENV, so we set the micromamba/HF cache paths explicitly and
# read the RunPod-injected secrets (HF_TOKEN, WANDB_API_KEY, FILESDIR, ...) straight
# from PID 1's environ.
#   db-run                            -> experiments/smoke-ddim-1.yml
#   db-run experiments/2-dogs.yml     -> that config
set -e
export MAMBA_ROOT_PREFIX=/opt/micromamba
export HF_HOME=/opt/hf HF_HUB_CACHE=/opt/hf/hub HF_MODELS_CACHE=/opt/hf/models \
       HF_DATASETS_CACHE=/opt/hf/datasets TORCH_HOME=/opt/torch
while IFS= read -r -d '' kv; do case "$kv" in
  HF_TOKEN=*|WANDB_API_KEY=*|ENTITY=*|FILESDIR=*|CUDA_VISIBLE_DEVICES=*) export "$kv";; esac
done < /proc/1/environ 2>/dev/null
mkdir -p "${FILESDIR:-/workspace/dfst_files}"
cd /opt/diffusion-boundary
CFG="${1:-experiments/smoke-ddim-1.yml}"
echo "=== db-run: $CFG @ $(date -u +%FT%TZ) (env dfst) ==="
micromamba run -n dfst python -u -m src --config "$CFG"
echo "RUN_DONE_CODE:$?"
