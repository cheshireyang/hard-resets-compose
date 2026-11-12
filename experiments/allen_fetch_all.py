"""Fetch every mouse cell that has official GLIF models (results/allen_glif_models_raw_all.json): extract the Noise 1 /
Noise 2 sweeps, detect spikes at the native sampling rate, and store the stimulus resampled to 20 kHz (exact
per-interval means; both the 0.2 ms bins of our fits and the 5e-5 s step of the GLIF models are unions of 20 kHz
intervals).  Cells with fewer than two sweeps of either stimulus are recorded as skipped.
usage: python3 experiments/allen_fetch_all.py NTHREADS"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, urllib.request
from allen_fetch import query, nwb_file_id, extract, OUT

RATE = 20000.0
rows = json.load(open("results/allen_glif_models_raw_all.json"))
mouse = sorted({r["specimen_id"] for r in rows if ((r.get("specimen") or {}).get("donor") or {}).get("organism_id") == 2})
meta = {r["specimen__id"]: r for r in query("model::ApiCellTypesSpecimenDetail,rma::criteria,[donor__species$eq'Mus musculus'],"
                                             "rma::options[num_rows$eq5000]")}
skip_path = "results/allen_fetch_all_skipped.json"
skipped = set(json.load(open(skip_path))) if os.path.exists(skip_path) else set()


def resample(i_pa, rate):
    tq = np.arange(len(i_pa) + 1) / rate
    Q = np.concatenate([[0.0], np.cumsum(i_pa.astype(np.float64)) / rate])
    n = int(np.floor(tq[-1] * RATE))
    return (np.diff(np.interp(np.arange(n + 1) / RATE, tq, Q)) * RATE).astype(np.float32)


def get(sid):
    if os.path.exists(f"{OUT}/{sid}.npz") or sid in skipped:
        return sid, "cached"
    fid = nwb_file_id(sid)
    if fid is None:
        return sid, "no-nwb"
    path = f"{OUT}/{sid}.nwb"
    urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", path)
    try:
        sw = extract(path)
    finally:
        os.remove(path)
    names = [v["stim_name"] for v in sw.values()]
    if names.count("Noise 1") < 2 or names.count("Noise 2") < 2:
        return sid, "too-few-sweeps"
    r = meta.get(sid, {})
    np.savez(f"{OUT}/{sid}.npz", meta=json.dumps(dict(specimen=sid, dendrite=r.get("tag__dendrite_type"), region=r.get("structure__acronym"),
                                                      line=r.get("line_name"), layer=r.get("structure__layer"))),
             **{f"{k}__i_pa": resample(v["i_pa"], v["rate"]) for k, v in sw.items()},
             **{f"{k}__spikes": v["spike_times"] for k, v in sw.items()},
             **{f"{k}__rate": np.array(RATE) for k, v in sw.items()},
             **{f"{k}__name": np.array(v["stim_name"]) for k, v in sw.items()})
    return sid, "ok"


if __name__ == "__main__":
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(len(mouse), "mouse specimens with GLIF models", flush=True)
    done = 0
    with ThreadPoolExecutor(nt) as ex:
        futs = {ex.submit(get, s): s for s in mouse}
        for fut in as_completed(futs):
            try:
                sid, st = fut.result()
            except Exception as e:
                sid, st = futs[fut], "error " + repr(e)[:80]
            if st in ("too-few-sweeps", "no-nwb"):
                skipped.add(sid)
            done += 1
            if done % 25 == 0 or st.startswith("error"):
                print(done, sid, st, flush=True)
                json.dump(sorted(skipped), open(skip_path, "w"))
    json.dump(sorted(skipped), open(skip_path, "w"))
    print("have", len([f for f in os.listdir(OUT) if f.endswith(".npz")]), "npz; skipped", len(skipped))
