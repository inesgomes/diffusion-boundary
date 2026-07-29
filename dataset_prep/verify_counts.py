import os, time
import duckdb

tok = os.environ["HF_TOKEN"]
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN '%s')" % tok)
gp = "hf://datasets/ILSVRC/imagenet-1k/data/train-*.parquet"

t = time.time()
total = con.execute(f"SELECT count(*) FROM '{gp}'").fetchone()[0]
print(f"TOTAL train rows: {total:,}  (imagenet-1k train is 1,281,167)  [{round(time.time()-t,1)}s]", flush=True)

t = time.time()
mn, mx, av, nc = con.execute(
    f"SELECT min(c), max(c), round(avg(c),1), count(*) "
    f"FROM (SELECT label, count(*) c FROM '{gp}' GROUP BY label)"
).fetchone()
print(f"per-class: min={mn} max={mx} avg={av} n_classes={nc}  [{round(time.time()-t,1)}s]", flush=True)

low = con.execute(f"SELECT label, count(*) c FROM '{gp}' GROUP BY label ORDER BY c ASC LIMIT 5").fetchall()
print("5 smallest classes (label, count):", low, flush=True)

verdict = "GENUINE (1300 is the real cap; smaller classes exist)" if (mn is not None and mn < 1300 and total > 1_000_000) \
    else "SUSPICIOUS -- looks capped/artifact"
print("VERDICT:", verdict)
print("VERIFY_DONE")
