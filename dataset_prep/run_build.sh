#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX=/root/micromamba
export REPO=/mnt/c/Users/mbie0/Documents/dev/ines/diffusion-boundary
export FILESDIR=/mnt/c/Users/mbie0/AppData/Local/Temp/dfst_prep
mkdir -p "$FILESDIR"
CFG="${1:-experiments/custom-adhoc.yml}"
/usr/local/bin/micromamba run -n dfst python -u /root/build_dataset_duckdb.py "$CFG" > /root/build.log 2>&1
echo "RC=$?" >> /root/build.log
