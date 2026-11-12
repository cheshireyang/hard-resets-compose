"""Fetch a fixed list of Allen cells (the ones used in the paper) by specimen id; same extraction as
allen_fetch_balanced.py.  usage: python3 experiments/allen_fetch_ids.py results/allen_cells_78.json"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, urllib.request
from allen_fetch import query, nwb_file_id, extract, OUT

ids = json.load(open(sys.argv[1]))
meta = {r["specimen__id"]: r for r in query("model::ApiCellTypesSpecimenDetail,rma::criteria,[donor__species$eq'Mus musculus'],"
                                             "rma::options[num_rows$eq3000]")}


def get(sid):
    if os.path.exists(f"{OUT}/{sid}.npz"):
        return sid, "cached"
    r = meta.get(sid, {})
    fid = nwb_file_id(sid)
    path = f"{OUT}/{sid}.nwb"
    urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", path)
    try:
        sw = extract(path)
    finally:
        os.remove(path)
    np.savez(f"{OUT}/{sid}.npz", meta=json.dumps(dict(specimen=sid, dendrite=r.get("tag__dendrite_type"),
                                                      region=r.get("structure__acronym"), line=r.get("line_name"))),
             **{f"{k}__i_pa": v["i_pa"] for k, v in sw.items()},
             **{f"{k}__spikes": v["spike_times"] for k, v in sw.items()},
             **{f"{k}__rate": np.array(v["rate"]) for k, v in sw.items()},
             **{f"{k}__name": np.array(v["stim_name"]) for k, v in sw.items()})
    return sid, "ok"


with ThreadPoolExecutor(6) as ex:
    for fut in as_completed([ex.submit(get, s) for s in ids]):
        try:
            print(*fut.result(), flush=True)
        except Exception as e:
            print("fail", repr(e)[:100], flush=True)
print("have", len([f for f in os.listdir(OUT) if f.endswith(".npz")]), "cells")
