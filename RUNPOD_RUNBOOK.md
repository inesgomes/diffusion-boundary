# RunPod Experiment Runbook (for an AI agent)

You are helping run **diffusion-boundary** experiments on RunPod GPU pods. The human
will typically just paste an **SSH string** (`ssh root@<ip> -p <port>`) and tell you which
experiment to run. This document is everything you need to do that reliably. Follow it
top to bottom; the "Gotchas" section encodes real failures — respect it.

## What the project does (1 paragraph)
Generates classifier-boundary images: Stable Diffusion v1-4 guided by a classifier
(ViT or ResNet-50) via `latentguidance`. Each run generates N images, evaluates them
(FID / KDN / precision / recall / coverage / density), logs metrics to **Weights &
Biases** and writes per-image metric tables (parquet) to a network volume. Runs are
driven by a YAML config; `alpha` (and scheduler / guidance-scale / guidance-freq) can
be **lists** → the run grid-searches them (one full run per combination).

## Key facts / environment
- **Image:** `ghcr.io/inesgomes/diffusion-boundary:cu128` — bakes the micromamba env
  `dfst` (torch 2.8.0+cu128), all model weights (SD v1-4, CLIP, ViT, DINO, Inception),
  `duckdb`, and the numpy/fsspec pins. Its `start.sh` publishes env to SSH sessions via
  sshd `SetEnv`, and it ships a self-contained `db-run` launcher.
- **RunPod template:** `diffusion-boundary-cu128` (embeds secrets `HF_TOKEN`,
  `WANDB_API_KEY`, `FILESDIR=/workspace/dfst_files`, `CUDA_VISIBLE_DEVICES=0`).
- **Network volume:** `jl3evn485h`, **region-locked to EU-RO-1** (a pod must be in
  EU-RO-1 to mount it). Mounted at `/workspace`; `FILESDIR=/workspace/dfst_files`.
- **W&B project:** `20260722_text_image_diffusion-gpu` (entity `phd_ines`).
- **SSH key:** `~/.ssh/runpod_ines` (the account key RunPod injects). Ports remap when a
  pod restarts — always use the latest SSH string the human gives you.
- **Code is dynamic:** the image bakes the `ddim` branch at `/opt/diffusion-boundary`,
  but you **scp this branch's `src/` over it every fresh pod** (see step 2). Batch>1
  needs the batching patch, which is only on this branch.

## Given an SSH string, do this

### Step 0 — connect
```bash
ssh -i ~/.ssh/runpod_ines -o StrictHostKeyChecking=no -p <port> root@<ip> 'echo ok'
```

### Step 1 — VERIFY THE POD (every time, before anything else)
```bash
ssh -i ~/.ssh/runpod_ines -p <port> root@<ip> '
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
  command -v db-run
  micromamba run -n dfst python -c "import numpy,duckdb;print(numpy.__version__,duckdb.__version__)"
  env | grep -c MAMBA_ROOT_PREFIX
  ls /workspace/dfst_files/dataset/ILSVRC/
  ps -eo args | grep "python -u -m src" | grep -v "grep\|bash -c" || echo IDLE'
```
Pass criteria:
- **GPU `memory.used` is ~0** (a few MiB). **If it is high while `nvidia-smi
  --query-compute-apps` shows no process, the GPU is occupied by a ghost allocation or a
  co-tenant — it WILL OOM. Do not run; ask for a different pod.** (This bit us: a pod
  reported 89 GB used with no visible process → every run OOM'd.)
- Card fits the batch size: **batch-8 guided runs peak ~51 GB → need a 96 GB card**
  (RTX PRO 6000). batch-1 ≈ 10 GB.
- `db-run` present; `numpy 1.26.4`; `duckdb` imports; `MAMBA_ROOT_PREFIX` count is 1
  (env is inherited → the new image).
- The dataset `ls` lists caches (volume attached). Process check says `IDLE`.

### Step 2 — deliver the code (fresh containers reset `/opt` to baked ddim)
```bash
scp -i ~/.ssh/runpod_ines -P <port> -r src root@<ip>:/opt/diffusion-boundary/
scp -i ~/.ssh/runpod_ines -P <port> -r experiments/2407 root@<ip>:/opt/diffusion-boundary/experiments/
# verify the batching patch landed (required for batch>1):
ssh ... 'grep -c "expand(prompt_emb.shape" /opt/diffusion-boundary/src/pipelines/latentguidance.py'  # -> 1
```

### Step 3 — make sure the reference dataset is cached (else it streams for HOURS)
The experiment computes the cache path as
`$FILESDIR/dataset/<name>_<split>_<N>_<idx1_idx2_...>` where `N = max(evaluation.num-images,
dataset.num-images)` and the indices are the imagenet class indices of `diffusion.args.classes`.
Already built on the volume:
- dogs → `imagenet-1k_train_2500_207_208`
- felines → `imagenet-1k_train_2500_288_290_293`
- birds → `imagenet-1k_train_2500_139_140_141_142`
- calibration set → `imagenet-1k_validation_1000_`

For **new classes**, prep the cache first with `dataset_prep/build_dataset_duckdb.py`
(DuckDB over HF parquet — fast, server-side filter). scp it + a config to the pod, then:
```bash
DRYRUN=1 <wrapper> experiments/<cfg>.yml   # validate class resolution first (exact, case-insensitive)
<wrapper> experiments/<cfg>.yml            # real build -> writes cache to $FILESDIR/dataset/...
```
The wrapper must export `MAMBA_ROOT_PREFIX`, `HF_HOME`, `REPO=/opt/diffusion-boundary`,
and pull `HF_TOKEN`/`FILESDIR` from `/proc/1/environ` (sshd shells: see `dataset_prep`).
The builder auto-budgets DuckDB memory to the container's cgroup cap and defaults to 16
threads — do NOT force high thread counts on a small-RAM container (it OOM-kills).

### Step 4 — run
Use the baked, self-contained `db-run`, detached, with a log:
```bash
ssh -i ~/.ssh/runpod_ines -p <port> root@<ip> \
  'setsid bash -c "db-run experiments/2407/<file>.yml > /opt/run.log 2>&1" </dev/null & echo launched'
```
Config notes: `device: cuda:0`; `batch-size: 8` for a 96 GB card; `alpha` as a list →
sweep; **one classifier per file** (classifier is not a grid dimension — to run ViT and
ResNet you need two files).

### Step 5 — monitor
```bash
ssh ... 'grep -a "Reference dataset path" /opt/run.log | tail -2   # must point at a CACHED dir (no streaming)
         grep -a "Generating images:" /opt/run.log | tail -1        # batches/313, s/batch
         grep -aiE "Traceback|out of memory|CUDA error" /opt/run.log | tail -3
         nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```
Expected: guided ~13 s/batch (~70 min/run for 2500 imgs @ 40 steps), `alpha=0` ~7.6
s/batch (~40 min, low VRAM — pure CFG, no guidance backward). Completion prints
`RUN_DONE_CODE:0`; metrics land at `$FILESDIR/logs/<wandb-run-id>/results_{synthetic,real}.parquet`.

### Step 6 — retrieve results (S3)
Metrics persist on the volume and sync to the RunPod S3 gateway:
```bash
AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
AWS_ACCESS_KEY_ID=<AWS_S3_USER> AWS_SECRET_ACCESS_KEY=<AWS_S3_KEY> \
aws s3 sync s3://jl3evn485h/dfst_files/logs ./results \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
```
(creds in `.env`; or use `rp-s3.ps1`). The `AWS_*_CHECKSUM_*=when_required` vars are
**required** — without them the RunPod gateway returns `412 Precondition Failed` on
download (AWS CLI ≥2.23 default checksums).

## Gotchas / lessons (respect these)
- **Verify the GPU is empty before every launch** (ghost VRAM / co-tenant → OOM with
  "N MiB free" while your process uses little). If occupied, get another pod; if this
  keeps happening, use Secure Cloud (dedicated), not Community.
- **Ports remap on pod restart.** Always use the latest SSH string.
- **Volume I/O (MooseFS) can hang** — processes wedge in **D state** (`kill -9` can't
  clear them until I/O returns), load average spikes into the dozens. Fix: stop/restart
  the pod. Metrics that hadn't saved are lost → just re-run that config.
- **Fresh container = baked ddim code** (no batching patch, no configs) → re-scp `src/`
  + configs every new pod. Confirm the patch with the grep in step 2.
- **Secrets** live in the RunPod template env and local `.env` — **never commit `.env`**.
  `db-run` and the prep wrapper read them from PID 1's environ (sshd shells don't inherit
  container ENV unless the new image's `SetEnv` is active).
- **ResNet-50 is batch-safe**: the classifier stays in `eval()`, so BatchNorm uses
  running stats (no cross-sample coupling); its label order matches imagenet
  (`id2label[207]=="golden retriever"`). Batched guidance gradients are per-sample →
  numerically equivalent to batch-1 modulo RNG (images won't be bit-identical to batch-1
  runs — fine for a fresh project, don't compare image-for-image).
- **Detached runs** survive SSH drops (`setsid`), but a **launcher SSH exiting 255** just
  means the connection dropped — check the pod's log / the volume, not the exit code.
- **Don't trust `free`/host RAM for DuckDB** — budget against the cgroup cap (the builder
  does this automatically).

## What's in this branch (the toolkit)
- `src/pipelines/latentguidance.py` — **batching fix** (expand uncond embedding to batch
  size so CFG works at batch>1).
- `dataset_prep/` — DuckDB reference-set builder (`build_dataset_duckdb.py`), class
  validators, streaming fallback.
- `docker/` — the self-sufficient image (Dockerfile, `start.sh` SetEnv, `db-run`,
  `build_and_push.sh`, `warmup.py`).
- `experiments/2407/` — the sweep configs (dogs/felines/birds × RN/ViT, alpha sweeps) +
  single-run rerun configs.
- `rp-s3.ps1` — S3 helper. `setup_blackwell.sh` — local (non-Docker) env setup.
