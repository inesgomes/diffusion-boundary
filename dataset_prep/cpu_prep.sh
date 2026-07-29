#!/usr/bin/env bash
# Run on a cheap CPU-only pod (image = same GHCR image, volume attached at /workspace,
# region = EU-RO-1). Streams + caches the reference dataset to the volume so the GPU
# run skips streaming entirely. No GPU used.
export MAMBA_ROOT_PREFIX=/opt/micromamba
MM=/usr/local/bin/micromamba
CFG="${1:-experiments/custom-adhoc.yml}"

echo "=== [1] fsspec/numpy patch (datasets streaming needs fsspec 2024.9.0) ==="
$MM run -n dfst pip install --no-cache-dir --no-deps "numpy==1.26.4" "fsspec==2024.9.0" 2>&1 | tail -2

echo "=== [2] env (HF token + FILESDIR from PID1, cache paths) ==="
export HF_HOME=/opt/hf HF_HUB_CACHE=/opt/hf/hub HF_DATASETS_CACHE=/opt/hf/datasets
while IFS= read -r -d '' kv; do case "$kv" in HF_TOKEN=*|FILESDIR=*) export "$kv";; esac; done < /proc/1/environ
export FILESDIR="${FILESDIR:-/workspace/dfst_files}"
mkdir -p "$FILESDIR"

echo "=== [3] verify config present ==="
if [ ! -f "/opt/diffusion-boundary/$CFG" ]; then echo "MISSING /opt/diffusion-boundary/$CFG (scp it first)"; exit 1; fi

echo "=== [4] prep (detached) -> /opt/prep.log ==="
setsid bash -c "$MM run -n dfst python -u /opt/prep_dataset.py $CFG > /opt/prep.log 2>&1" </dev/null &
echo "PREP_PID:$!"
sleep 4; echo "--- first log lines ---"; head -6 /opt/prep.log 2>/dev/null
