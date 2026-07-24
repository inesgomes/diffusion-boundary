#!/usr/bin/env bash
# RunPod start command: install the injected SSH public key and run sshd (foreground).
set -e
mkdir -p /run/sshd /root/.ssh
ssh-keygen -A 2>/dev/null || true

# RunPod injects account SSH keys via $PUBLIC_KEY
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true

# --- make the container ENV + RunPod-injected secrets reach SSH sessions ---
# sshd shells (interactive AND `ssh host cmd`) don't inherit PID 1's env. Publish it
# via sshd SetEnv (OpenSSH 7.8+), so plain SSH shells have micromamba/HF caches and the
# template secrets without any wrapper. db-run stays self-contained as a fallback.
{
  echo "SetEnv MAMBA_ROOT_PREFIX=/opt/micromamba HF_HOME=/opt/hf HF_HUB_CACHE=/opt/hf/hub HF_MODELS_CACHE=/opt/hf/models HF_DATASETS_CACHE=/opt/hf/datasets TORCH_HOME=/opt/torch"
  line="SetEnv"
  for v in HF_TOKEN WANDB_API_KEY ENTITY FILESDIR CUDA_VISIBLE_DEVICES; do
    [ -n "${!v:-}" ] && line="$line $v=${!v}"
  done
  [ "$line" != "SetEnv" ] && echo "$line"
} >> /etc/ssh/sshd_config

echo "diffusion-boundary image ready. env 'dfst', code at /opt/diffusion-boundary."
echo "run an experiment with:  db-run experiments/<config>.yml"

exec /usr/sbin/sshd -D -e
