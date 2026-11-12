"""Download the official Allen GLIF neuron_config.json files for the cells in results/allen_glif_models_raw.json
(written by the API query) into data/glif/<specimen>_glif<level>.json, and a summary with the official explained
variance ratios.  usage: python3 experiments/glif_fetch.py [RAW.json SUMMARY.json]"""
import json, os, sys, urllib.request
RAW = sys.argv[1] if len(sys.argv) > 1 else "results/allen_glif_models_raw.json"
SUM = sys.argv[2] if len(sys.argv) > 2 else "results/allen_glif_models.json"
rows = json.load(open(RAW))
os.makedirs("data/glif", exist_ok=True)
summary = []
jobs = []
for r in rows:
    t = r["neuronal_model_template"]["name"]
    if not t[0].isdigit():
        continue
    lvl = int(t[0])
    wk = [w for w in r["well_known_files"] if w.get("path", "").endswith("neuron_config.json")]
    if not wk:
        continue
    out = f"data/glif/{r['specimen_id']}_glif{lvl}.json"
    if not os.path.exists(out):
        jobs.append((wk[0]["id"], out))
    ev = [x.get("explained_variance_ratio") for x in r.get("neuronal_model_runs", [])]
    summary.append(dict(specimen=r["specimen_id"], level=lvl, model_id=r["id"], ev=ev[0] if ev else None, config=out))
from concurrent.futures import ThreadPoolExecutor


def dl(job):
    fid, out = job
    tmp = out + ".part"
    urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", tmp)
    os.replace(tmp, out)


with ThreadPoolExecutor(16) as ex:
    list(ex.map(dl, jobs))
assert all(os.path.exists(m["config"]) for m in summary)
json.dump(summary, open(SUM, "w"), indent=1)
print(len(summary), "configs")
