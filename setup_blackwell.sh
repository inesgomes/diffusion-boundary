#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible setup for diffusion-boundary (ddim branch) on a Blackwell GPU
# (RTX 50-series / B-series, compute capability sm_120) on Linux.
#
# Proven on: WSL2 Ubuntu 24.04 + RTX 5070 Laptop, driver 596.49 (CUDA 13.2).
# The ONLY non-obvious part: environment.yml pins torch==2.4.1+cu121, which has
# NO Blackwell kernels. We install everything, then force torch 2.8.0+cu128 LAST
# (deps can silently downgrade torch back to cu124 — so the override must be last).
#
# Usage:  bash setup_blackwell.sh
# Run from the repo root (where environment.yml lives).
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME=dfst
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

echo "==> [0/6] system prerequisites (curl, bzip2)"
if ! command -v curl >/dev/null 2>&1 || ! command -v bzip2 >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq curl bzip2 ca-certificates
fi

echo "==> [1/6] micromamba"
if ! command -v micromamba >/dev/null 2>&1 && [ ! -x /usr/local/bin/micromamba ]; then
  cd /tmp
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  sudo install -m 0755 bin/micromamba /usr/local/bin/micromamba
fi
MM="$(command -v micromamba || echo /usr/local/bin/micromamba)"
$MM --version

echo "==> [2/6] create env '$ENV_NAME' (python 3.11)"
$MM create -y -n "$ENV_NAME" -c conda-forge python=3.11 pip

echo "==> [3/6] install project deps (torch/vision/triton/nvidia pins stripped)"
cd "$REPO_DIR"
awk '/^      - /{sub(/^      - /,""); print}' environment.yml \
  | grep -vE '^(torch==|torchvision==|triton==|nvidia-)' > /tmp/requirements-cu128.txt
$MM run -n "$ENV_NAME" pip install --no-cache-dir -r /tmp/requirements-cu128.txt

echo "==> [4/6] OVERRIDE torch with cu128 (Blackwell sm_120) -- MUST be last"
$MM run -n "$ENV_NAME" pip install --no-cache-dir --force-reinstall \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> [5/6] verify GPU (expect capability (12, 0) and matmul ok: True)"
$MM run -n "$ENV_NAME" python - <<'PY'
import torch, sys
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
cap = torch.cuda.get_device_capability(0)
print("device:", torch.cuda.get_device_name(0), "| capability:", cap)
x = torch.randn(4096, 4096, device="cuda")
ok = torch.isfinite((x @ x).sum()).item()
print("matmul ok:", ok)
if cap[0] < 9 or not ok:
    print("!! GPU verification FAILED — check driver / torch build"); sys.exit(1)
print("GPU OK")
PY

echo "==> [6/6] .env template (fill in your values, then re-run without overwriting)"
if [ ! -f "$REPO_DIR/.env" ]; then
  cat > "$REPO_DIR/.env" <<EOF
CUDA_VISIBLE_DEVICES=0
FILESDIR=$HOME/dfst_files
ENTITY=              # optional; blank -> your default W&B entity
HF_TOKEN=            # HF token (accept imagenet-1k + stable-diffusion-v1-4 licenses)
WANDB_API_KEY=       # from https://wandb.ai/authorize   (NOTE: var name is WANDB_API_KEY)
EOF
  mkdir -p "$HOME/dfst_files"
  grep -qx '.env' .gitignore 2>/dev/null || printf '\n.env\n' >> .gitignore
  echo "   wrote .env template — EDIT IT with your token/key before running."
else
  echo "   .env already exists — leaving it untouched."
fi

cat <<EOF

Done. To run the smoke test:
  export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
  micromamba run -n $ENV_NAME python -m src --config experiments/smoke-test.yml

Reminders:
  * The config path needs the .yml extension (code opens it verbatim).
  * WANDB var must be named WANDB_API_KEY (not WANDB_TOKEN).
  * VRAM: >=12-16GB for smoke/2-dogs, 24GB+ for 3-felines / 4-birds.
EOF
