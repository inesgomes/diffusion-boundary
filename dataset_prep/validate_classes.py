"""Validate that every class in every experiment YAML resolves to an imagenet-1k
label (exact + case-insensitive) and has >= the required number of images."""
import os, sys, glob, time, yaml

repo = os.environ.get("REPO", "/opt/diffusion-boundary")
sys.path.insert(0, repo); os.chdir(repo)

from datasets import load_dataset
import duckdb

stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
class_labels = stream.features["label"].names
lower_map = {}
for i, n in enumerate(class_labels):
    lower_map.setdefault(n.lower(), (i, n))

reqs = {}          # idx -> max per-class images required across experiments
details = []       # (yaml, cls, idx, how, per)
unmatched = []     # (yaml, cls)

for path in sorted(glob.glob("experiments/*.yml")):
    try:
        cfg = yaml.safe_load(open(path))
    except Exception:
        continue
    args = (cfg or {}).get("diffusion", {}).get("args", {})
    classes = args.get("classes")
    if not classes:
        continue
    N = max(int(cfg["evaluation"]["num-images"]), int(cfg["dataset"]["num-images"]))
    per = N // len(classes)
    for c in classes:
        if c in class_labels:
            idx, how = class_labels.index(c), "exact"
        elif c.lower() in lower_map:
            idx, how = lower_map[c.lower()][0], "case-insensitive"
        else:
            unmatched.append((os.path.basename(path), c)); continue
        reqs[idx] = max(reqs.get(idx, 0), per)
        details.append((os.path.basename(path), c, idx, how, per))

# one server-side count over all needed labels
tok = os.environ["HF_TOKEN"]
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN '%s')" % tok)
gp = "hf://datasets/ILSVRC/imagenet-1k/data/train-*.parquet"
idxs = sorted(reqs)
t = time.time()
counts = dict(con.execute(
    f"SELECT label, count(*) FROM '{gp}' WHERE label IN ({','.join(map(str, idxs))}) GROUP BY label"
).fetchall())
print(f"[counted {len(idxs)} distinct labels across all experiments in {round(time.time()-t,1)}s]\n")

print(f"{'experiment':18s} {'class':45s} {'idx':>4s} {'avail':>6s} {'need':>5s}  status")
print("-" * 92)
short = 0
for y, c, idx, how, per in details:
    avail = counts.get(idx, 0)
    ok = "OK" if avail >= per else "*** SHORT ***"
    if avail < per: short += 1
    tag = "" if how == "exact" else " [CASE-INSENSITIVE ONLY]"
    print(f"{y:18s} {c[:45]:45s} {idx:>4d} {avail:>6d} {per:>5d}  {ok}{tag}")

print()
if unmatched:
    print("=== UNMATCHED class names (get_tst_dataset would CRASH) ===")
    for y, c in unmatched:
        print(f"  {y}: '{c}'")
else:
    print("All class names matched an imagenet-1k label.")
print(f"\nSHORT count: {short}   UNMATCHED count: {len(unmatched)}")
print("VALIDATE_DONE")
