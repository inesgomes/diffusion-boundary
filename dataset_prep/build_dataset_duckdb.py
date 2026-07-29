"""Fast reference-dataset builder via DuckDB over HF parquet.

Filters imagenet-1k by label SERVER-SIDE (reads only the cheap label column + the
matching image rows, not the 150GB of images), randomly samples within each target
class, and writes a datasets cache identical to what get_tst_dataset() produces --
so the experiment's load_from_disk hits it directly.

Env:  REPO, FILESDIR, HF_TOKEN
Usage: python build_dataset_duckdb.py experiments/<config>.yml
"""
import os
import sys
import time
import yaml

repo = os.environ.get("REPO", "/opt/diffusion-boundary")
sys.path.insert(0, repo)
os.chdir(repo)

from datasets import load_dataset, Dataset
import duckdb

cfg = yaml.safe_load(open(sys.argv[1]))
DATASET = cfg["dataset"]["name"]
SPLIT = cfg["dataset"].get("split", "test")
N = max(int(cfg["evaluation"]["num-images"]), int(cfg["dataset"]["num-images"]))
CLASSES = cfg["diffusion"]["args"]["classes"]
FILESDIR = os.environ["FILESDIR"]
tok = os.environ["HF_TOKEN"]

# canonical features (Image + ClassLabel with the 1000 names) -> match cache schema exactly
stream = load_dataset(DATASET, split=SPLIT, streaming=True)
features = stream.features
class_labels = features["label"].names

# EXACT match, CASE-INSENSITIVE: the full imagenet label string must equal the config
# class string ignoring case. No partial/substring/fuzzy matching. Hard error otherwise.
lower_map = {}
for i, n in enumerate(class_labels):
    lower_map.setdefault(n.lower(), (i, n))

subset_idx = []
for c in CLASSES:
    key = c.strip().lower()
    if key not in lower_map:
        raise SystemExit(f"[build] ERROR: class '{c}' has NO exact (case-insensitive) match in imagenet-1k labels -- aborting, not downloading.")
    i, canonical = lower_map[key]
    subset_idx.append(i)
    kind = "exact" if canonical == c else f"case-insensitive (canonical: '{canonical}')"
    print(f"[build] class '{c}'  ==>  idx {i}  [{kind}]", flush=True)

if len(set(subset_idx)) != len(subset_idx):
    raise SystemExit(f"[build] ERROR: duplicate class indices resolved {subset_idx} -- aborting.")

per = N // len(subset_idx)
print(f"[build] {DATASET} split={SPLIT} N={N} per_class={per} subset_idx={subset_idx}", flush=True)

if os.environ.get("DRYRUN"):
    print("DRYRUN: class resolution only, not downloading.")
    raise SystemExit(0)

con = duckdb.connect()
# IMPORTANT: budget against the CONTAINER cgroup limit, not `free`/host RAM. RunPod
# pods report the host's huge RAM via free(1) but cap the container much lower (e.g.
# ~58GB), and each httpfs thread buffers a large imagenet row-group image chunk, so a
# high thread count spikes RSS past memory_limit and gets OOM-killed. Keep threads
# modest and the budget well under the cgroup cap; temp_directory spills if exceeded.
os.makedirs("/root/duckdb_tmp", exist_ok=True)
# read the cgroup v2/v1 memory cap; budget ~40% of it, capped to a safe [6,24]GB range
def _cgroup_limit_gb():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                return int(v) / 1073741824
        except OSError:
            pass
    return None
_cap = _cgroup_limit_gb()
_mem = int(os.environ.get("DUCKDB_MEM_GB", max(6, min(24, int((_cap or 60) * 0.4)))))
con.execute("SET memory_limit='%dGB'" % _mem)
con.execute("SET max_memory='%dGB'" % _mem)
con.execute("SET temp_directory='/root/duckdb_tmp'")
# 16 is a safe/faster default: the buffer pool is hard-capped at memory_limit above, so
# 32 threads is what previously OOM'd (huge pool + many httpfs buffers); 8 was ~4x too
# slow (a 4-class fetch took ~65min). 16 stays under the cgroup cap and ~halves that.
con.execute("SET threads=%s" % os.environ.get("DUCKDB_THREADS", "16"))
con.execute("SET preserve_insertion_order=false")
print("[build] container cap=%s GB, duckdb memory_limit=%dGB, threads=%s" %
      (round(_cap,1) if _cap else "?", _mem, os.environ.get("DUCKDB_THREADS", "8")), flush=True)
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN '%s')" % tok)
glob = f"hf://datasets/{DATASET}/data/{SPLIT}-*.parquet"

# per-class server-side filter + random sample within the class, then shuffle the union
parts = [f"(SELECT image, label FROM '{glob}' WHERE label = {i} ORDER BY random() LIMIT {per})"
         for i in subset_idx]
query = "SELECT image, label FROM (" + " UNION ALL ".join(parts) + ") ORDER BY random()"

t = time.time()
rows = con.execute(query).fetchall()
print(f"[build] fetched {len(rows)} rows in {round(time.time()-t,1)}s", flush=True)

images, labels = [], []
for img, lab in rows:
    b = img["bytes"] if isinstance(img, dict) else img[0]
    images.append({"bytes": b, "path": None})
    labels.append(int(lab))

ds = Dataset.from_dict({"image": images, "label": labels}, features=features)

subset_name = "_".join(map(str, subset_idx))
path = f"{FILESDIR}/dataset/{DATASET}_{SPLIT}_{N}_{subset_name}"
os.makedirs(path, exist_ok=True)
ds.save_to_disk(path)

# sanity: reload like the experiment does
from datasets import load_from_disk
rl = load_from_disk(path)
print(f"[build] saved {len(rl)} imgs, cols={rl.column_names}, label0={rl['label'][0]} -> {path}", flush=True)
print("BUILD_DONE")
