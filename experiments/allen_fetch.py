"""Fetch Allen Cell Types electrophysiology (NWB v1 / HDF5) for a few mouse cells and extract the
'Noise 1' / 'Noise 2' current-clamp sweeps (stimulus current, recorded spike times).
Writes data/allen/<specimen_id>.npz with arrays per sweep.
usage: python3 experiments/allen_fetch.py N_CELLS"""
import io, json, os, sys, urllib.request, urllib.parse
import h5py
import numpy as np

API = "https://api.brain-map.org/api/v2/data/query.json?criteria="
OUT = os.environ.get("ALLEN_OUT", "data/allen")
os.makedirs(OUT, exist_ok=True)


def query(crit):
    url = API + urllib.parse.quote(crit, safe=":,[]$='()")
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    assert d["success"], d
    return d["msg"]


def nwb_file_id(specimen_id):
    rows = query(f"model::Specimen,rma::criteria,[id$eq{specimen_id}],"
                 "rma::include,ephys_result(well_known_files(well_known_file_type[name$eq'NWBDownload']))")
    for r in rows:
        for wkf in (r.get("ephys_result") or {}).get("well_known_files", []):
            return wkf["id"]
    return None


def spikes_from_voltage(v, rate, thr_mv=-20.0, refractory_ms=1.0):
    above = v > thr_mv
    on = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    keep, last = [], -1e9
    for i in on:
        if (i - last) / rate * 1000 >= refractory_ms:
            keep.append(i); last = i
    return np.array(keep) / rate  # seconds


def extract(path):
    f = h5py.File(path, "r")
    out = {}
    for name in f["stimulus/presentation"].keys():
        st = f["stimulus/presentation"][name]
        sname = st["aibs_stimulus_name"][()]
        sname = sname.decode() if isinstance(sname, bytes) else str(sname)
        if not sname.startswith("Noise"):
            continue
        acq = f["acquisition/timeseries"][name]
        rate = float(acq["starting_time"].attrs["rate"])
        i_pa = st["data"][()] * float(st["data"].attrs.get("conversion", 1.0)) * 1e12   # A -> pA
        v_mv = acq["data"][()] * float(acq["data"].attrs.get("conversion", 1.0)) * 1e3  # V -> mV
        n = min(len(i_pa), len(v_mv))
        out[name] = dict(stim_name=sname, rate=rate, i_pa=i_pa[:n].astype(np.float32),
                         spike_times=spikes_from_voltage(v_mv[:n], rate))
    return out


if __name__ == "__main__":
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rows = query("model::ApiCellTypesSpecimenDetail,rma::criteria,[donor__species$eq'Mus musculus'],"
                 "rma::options[num_rows$eq400]")
    # prefer cells with a moderate firing rate
    rows = [r for r in rows if r.get("ef__avg_firing_rate") and 5 < r["ef__avg_firing_rate"] < 60]
    got = 0
    for r in rows:
        sid = r["specimen__id"]
        if os.path.exists(f"{OUT}/{sid}.npz"):
            got += 1
            if got >= n_cells: break
            continue
        fid = nwb_file_id(sid)
        if fid is None:
            continue
        path = f"{OUT}/{sid}.nwb"
        urllib.request.urlretrieve(f"https://api.brain-map.org/api/v2/well_known_file_download/{fid}", path)
        sw = extract(path)
        if len(sw) < 2:
            os.remove(path); continue
        np.savez(f"{OUT}/{sid}.npz", meta=json.dumps(dict(specimen=sid, dendrite=r.get("tag__dendrite_type"),
                                                          region=r.get("structure__acronym"), line=r.get("line_name"))),
                 **{f"{k}__i_pa": v["i_pa"] for k, v in sw.items()},
                 **{f"{k}__spikes": v["spike_times"] for k, v in sw.items()},
                 **{f"{k}__rate": np.array(v["rate"]) for k, v in sw.items()},
                 **{f"{k}__name": np.array(v["stim_name"]) for k, v in sw.items()})
        os.remove(path)
        print(sid, r.get("tag__dendrite_type"), r.get("structure__acronym"),
              {k: (v["stim_name"], len(v["i_pa"]) / v["rate"], len(v["spike_times"])) for k, v in sw.items()}, flush=True)
        got += 1
        if got >= n_cells:
            break
    print("cells:", got)
