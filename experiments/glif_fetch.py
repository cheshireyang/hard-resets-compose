"""Download the official Allen GLIF neuron_config.json files for the cells in results/allen_glif_models_raw.json
(written by the API query) into data/glif/<specimen>_glif<level>.json, and a summary with the official explained
variance ratios.  usage: python3 experiments/glif_fetch.py"""
import json, os, urllib.request
rows = json.load(open("results/allen_glif_models_raw.json"))
os.makedirs("data/glif", exist_ok=True)
summary = []
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
        urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{wk[0]['id']}", out)
    ev = [x.get("explained_variance_ratio") for x in r.get("neuronal_model_runs", [])]
    summary.append(dict(specimen=r["specimen_id"], level=lvl, model_id=r["id"], ev=ev[0] if ev else None, config=out))
json.dump(summary, open("results/allen_glif_models.json", "w"), indent=1)
print(len(summary), "configs")
