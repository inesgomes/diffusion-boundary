# Portable Blackwell image for diffusion-boundary

A self-contained image so any RunPod Blackwell GPU (RTX 5090, B200, …) starts
ready in seconds — no per-pod environment setup, no region-locked volume.

## What's baked in
- Ubuntu 24.04 + micromamba env `dfst` (Python 3.11)
- **torch 2.8.0 + cu128** (Blackwell sm_100/sm_120 kernels)
- All project deps (diffusers, transformers, pymdma, wandb, captum, …)
- The `ddim` branch code at `/opt/diffusion-boundary`
- **Model weights**: SD v1-4, CLIP-L, ViT-base classifier, DINO ViT-S/8, Inception-FID
- Experiment configs patched `cuda:1 -> cuda:0` (single-GPU pods)

## What's NOT baked (supplied at runtime)
- **Secrets**: `HF_TOKEN`, `WANDB_API_KEY`, `ENTITY` — via RunPod env vars or a mounted `.env`
- **imagenet-1k** — streamed live at run time (gated + huge); cached under `FILESDIR`

## Build & push (one-time, local)
```bash
# 1. GitHub PAT (classic) with write:packages, then:
docker login ghcr.io -u <your-github-user>
# 2. build + push (reads HF token from ../.env as a build secret)
cd docker
GHCR_USER=<your-github-user> bash build_and_push.sh
```
Image is ~15–18 GB; the push time depends on your upload bandwidth (one-time).
After pushing, set the package to **Private** in GHCR settings.

## RunPod template
- **Container Image**: `ghcr.io/<your-github-user>/diffusion-boundary:cu128`
- **Registry credentials**: add your GHCR PAT (RunPod → Settings → Container Registry Auth) if the package is private
- **Docker command / start**: leave default (image `CMD` runs sshd)
- **Expose**: TCP port 22 (SSH)
- **Environment variables**:
  - `HF_TOKEN` = <your hf token>
  - `WANDB_API_KEY` = <your wandb key>
  - `ENTITY` = <your wandb entity>   (optional)
  - `FILESDIR` = `/workspace/dfst_files` (if you attach a volume) or `/root/dfst_files`
  - `CUDA_VISIBLE_DEVICES` = `0`
- **SSH key**: your account SSH public key (RunPod injects it as `PUBLIC_KEY`)

> The code calls `load_dotenv()`, so alternatively drop a `.env` at
> `/opt/diffusion-boundary/.env`. Image `ENV` (HF caches) is not overridden by it.

## Run
```bash
# inside the pod (SSH in), from anywhere:
db-run experiments/smoke-ddim-1.yml     # 1-image ddim sanity check
db-run experiments/2-dogs.yml           # ~1 h
db-run experiments/3-felines.yml        # multi-hour, full eval+viz
db-run experiments/4-birds.yml          # multi-hour, full eval+viz
```
Results log to W&B. `save-metrics-disk: True` (felines/birds) writes parquet to
`$FILESDIR/logs/<run-id>/` — put `FILESDIR` on a mounted volume or scp it off
before stopping the pod (container disk is ephemeral).
