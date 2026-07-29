"""Pre-build the reference dataset cache for an experiment config (CPU/network only).
Same code path the experiment uses, so the cache key matches exactly.

Env:
  REPO      repo root (default /opt/diffusion-boundary; set to local path when running locally)
  FILESDIR  where the cache is written -> $FILESDIR/dataset/<name>_<split>_<n>_<idx>/
  HF_TOKEN  needed for gated imagenet-1k

Usage:  python prep_dataset.py experiments/<config>.yml
"""
import os
import sys
import yaml

repo = os.environ.get("REPO", "/opt/diffusion-boundary")
sys.path.insert(0, repo)
os.chdir(repo)

from src.dataset.aux import get_tst_dataset

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/custom-adhoc.yml"
cfg = yaml.safe_load(open(cfg_path))

name = cfg["dataset"]["name"]
split = cfg["dataset"].get("split", "test")
n = max(int(cfg["evaluation"]["num-images"]), int(cfg["dataset"]["num-images"]))
classes = cfg["diffusion"]["args"]["classes"]

print(f"[prep] {name} split={split} n_samples={n} subset={classes}", flush=True)
print(f"[prep] FILESDIR={os.getenv('FILESDIR')}", flush=True)

_imgs, labels, _class_labels = get_tst_dataset(name, split, n, subset=classes)

print(f"[prep] cached {len(labels)} reference images -> {os.getenv('FILESDIR')}/dataset/", flush=True)
print("DATASET_PREP_DONE")
