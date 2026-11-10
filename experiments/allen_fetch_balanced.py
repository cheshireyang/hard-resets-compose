"""Balanced fetch: up to K spiny and K aspiny/sparsely-spiny mouse cells that have at least two
'Noise 1' and two 'Noise 2' sweeps.  Parallel downloads.  Reuses allen_fetch.py helpers.
usage: python3 experiments/allen_fetch_balanced.py K"""
import json, os, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import urllib.request
from allen_fetch import query, nwb_file_id, extract, OUT

K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rows = query("model::ApiCellTypesSpecimenDetail,rma::criteria,[donor__species$eq'Mus musculus'],"
             "rma::options[num_rows$eq3000]")
rows = [r for r in rows if r.get("ef__avg_firing_rate") and 2 < r["ef__avg_firing_rate"] < 80]
random.Random(0).shuffle(rows)
groups = {"spiny": [r for r in rows if r.get("tag__dendrite_type") == "spiny"],
          "aspiny": [r for r in rows if r.get("tag__dendrite_type") in ("aspiny", "sparsely spiny")]}
print({k: len(v) for k, v in groups.items()}, flush=True)


def get(r):
    sid = r["specimen__id"]
    if os.path.exists(f"{OUT}/{sid}.npz"):
        z = np.load(f"{OUT}/{sid}.npz")
        names = [str(z[k]) for k in z.files if k.endswith("__name")]
        return sid, names.count("Noise 1") >= 2 and names.count("Noise 2") >= 2
    fid = nwb_file_id(sid)
    if fid is None:
        return sid, False
    path = f"{OUT}/{sid}.nwb"
    urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", path)
    try:
        sw = extract(path)
    finally:
        os.remove(path)
    names = [v["stim_name"] for v in sw.values()]
    ok = names.count("Noise 1") >= 2 and names.count("Noise 2") >= 2
    if ok:
        np.savez(f"{OUT}/{sid}.npz", meta=json.dumps(dict(specimen=sid, dendrite=r.get("tag__dendrite_type"),
                                                          region=r.get("structure__acronym"), line=r.get("line_name"))),
                 **{f"{k}__i_pa": v["i_pa"] for k, v in sw.items()},
                 **{f"{k}__spikes": v["spike_times"] for k, v in sw.items()},
                 **{f"{k}__rate": np.array(v["rate"]) for k, v in sw.items()},
                 **{f"{k}__name": np.array(v["stim_name"]) for k, v in sw.items()})
    return sid, ok


for gname, cand in groups.items():
    good, i = 0, 0
    with ThreadPoolExecutor(8) as ex:
        while good < K and i < len(cand):
            batch = cand[i:i + 8]; i += 8
            for fut in as_completed([ex.submit(get, r) for r in batch]):
                try:
                    sid, ok = fut.result()
                except Exception as e:  # network or file errors: skip the cell
                    print("skip", repr(e)[:80], flush=True); continue
                good += ok
    print(gname, "usable:", good, flush=True)
