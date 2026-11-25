"""Robust parallel fetcher for a prioritized list of Allen Cell Types specimen IDs.
Fetches NWB v1 files with retries and atomic writes, extracts Noise 1/2 sweeps into data/allen/<sid>.npz.
usage: python3 experiments/allen_fetch_target.py [LIST_JSON] [THREADS]"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from allen_fetch import query, nwb_file_id, extract

OUT = os.environ.get("ALLEN_OUT", "data/allen")
os.makedirs(OUT, exist_ok=True)


def download_file(url, dst, retries=4):
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                expected = resp.headers.get("Content-Length")
                expected = int(expected) if expected else None
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
            if expected is not None and got < expected:
                raise IOError(f"truncated {got}/{expected}")
            os.replace(tmp, dst)
            return
        except Exception as e:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    list_path = sys.argv[1] if len(sys.argv) > 1 else "results/allen_cells_78.json"
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ids = json.load(open(list_path))
    meta = {r["specimen__id"]: r for r in query(
        "model::ApiCellTypesSpecimenDetail,rma::criteria,[donor__species$eq'Mus musculus'],rma::options[num_rows$eq3000]"
    )}

    def fetch_one(sid):
        out_npz = f"{OUT}/{sid}.npz"
        if os.path.exists(out_npz):
            return sid, "cached"
        r = meta.get(sid, {})
        fid = None
        for attempt in range(3):
            try:
                fid = nwb_file_id(sid)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if fid is None:
            return sid, "no_nwb"
        nwb = f"{OUT}/{sid}.nwb"
        try:
            download_file(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", nwb)
            sw = extract(nwb)
        finally:
            if os.path.exists(nwb):
                try: os.remove(nwb)
                except OSError: pass
        tmp_npz = f"{OUT}/{sid}.tmp.npz"
        np.savez(
            tmp_npz,
            meta=json.dumps(dict(
                specimen=sid, dendrite=r.get("tag__dendrite_type"),
                region=r.get("structure__acronym"), line=r.get("line_name"),
            )),
            **{f"{k}__i_pa": v["i_pa"] for k, v in sw.items()},
            **{f"{k}__spikes": v["spike_times"] for k, v in sw.items()},
            **{f"{k}__rate": np.array(v["rate"]) for k, v in sw.items()},
            **{f"{k}__name": np.array(v["stim_name"]) for k, v in sw.items()},
        )
        os.replace(tmp_npz, out_npz)
        return sid, f"ok({len(sw)})"

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(n_threads) as ex:
        futs = [ex.submit(fetch_one, s) for s in ids]
        for fut in as_completed(futs):
            done += 1
            try:
                sid, st = fut.result()
                print(f"[{done}/{len(ids)} {time.time()-t0:.0f}s] {sid}: {st}", flush=True)
            except Exception as e:
                print(f"[{done}/{len(ids)} {time.time()-t0:.0f}s] FAIL: {repr(e)[:120]}", flush=True)
    have = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
    print(f"DONE: {have} npz files in {OUT} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
