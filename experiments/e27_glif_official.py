"""E27: spike-time F1 of the official Allen GLIF models (levels 1-5, Teeter et al. 2018) on the same held-out
'Noise 2' sweeps, with the same metric as E15 (F1, +-4 ms, 0.2 ms bins, same truncation).  Simulated with allensdk's
GlifNeuron at the configuration's own time step (stimulus block-averaged to it).  Runs under the allensdk venv:
    /root/venv_allen311/bin/python experiments/e27_glif_official.py [NPROC]"""
import json, os, sys, glob
from multiprocessing import Pool
import numpy as np
from allensdk.model.glif.glif_neuron import GlifNeuron

DT_MS, WIN_MS = 0.2, 4.0
DATA = os.environ.get("E27_DATA", "data/allen")
MODELS = os.environ.get("E27_MODELS", "results/allen_glif_models.json")
OUTJ = os.environ.get("E27_OUT", "results/e27_glif_official.json")
SPK = os.environ.get("E27_SPK", "results/e27_spikes")


def binned(sp_s, n):
    s = np.zeros(n, dtype=np.float32)
    idx = (np.asarray(sp_s) * 1000.0 / DT_MS).astype(np.int64)
    s[idx[(idx >= 0) & (idx < n)]] = 1.0                 # spikes outside the evaluation window are dropped
    return s


def pool(a, w):
    out = np.zeros_like(a)
    for k in range(-w, w + 1):
        if k >= 0: out[k:] = np.maximum(out[k:], a[:len(a) - k])
        else: out[:k] = np.maximum(out[:k], a[-k:])
    return out


def f1(model, true, w):
    """model, true: lists of binned trains (one per sweep); same aggregation as E15 (sums over sweeps)."""
    rec_n = sum((pool(m, w) * t).sum() for m, t in zip(model, true)); rec_d = max(sum(t.sum() for t in true), 1)
    pre_n = sum((pool(t, w) * m).sum() for m, t in zip(model, true)); pre_d = max(sum(m.sum() for m in model), 1)
    rec, pre = rec_n / rec_d, pre_n / pre_d
    return float(2 * rec * pre / (rec + pre + 1e-12))


def load_sweeps(path, name):
    z = np.load(path, allow_pickle=False)
    out = []
    for s in sorted({k.split("__")[0] for k in z.files if "__" in k}):
        if str(z[f"{s}__name"]) == name:
            out.append((z[f"{s}__i_pa"].astype(np.float64), float(z[f"{s}__rate"]), z[f"{s}__spikes"]))
    return out


def simulate(cfg, i_pa, rate):
    nrn = GlifNeuron.from_dict(cfg)
    dt = nrn.dt
    # exact mean current over every model step (the sampling rate need not be a multiple of 1/dt)
    tq = np.arange(len(i_pa) + 1) / rate
    Q = np.concatenate([[0.0], np.cumsum(i_pa) / rate])
    n = int(np.floor(tq[-1] / dt))
    stim = np.diff(np.interp(np.arange(n + 1) * dt, tq, Q)) / dt * 1e-12   # pA -> A
    out = nrn.run(stim)
    return np.asarray(out["interpolated_spike_times"], dtype=np.float64)


def run(job):
    sid, cfgs = job
    test = load_sweeps(f"{DATA}/{sid}.npz", "Noise 2")
    if len(test) < 2:
        return None
    # E15 truncation: common length of the binned sweeps, rounded down to a multiple of 256 bins
    nb = min(int(len(i) // int(round(r * DT_MS / 1000.0))) for i, r, _ in test)
    nb = (nb // 256) * 256
    true = [binned(sp, nb) for _, _, sp in test]
    w = int(WIN_MS / DT_MS)
    res = dict(specimen=sid, n_test=len(test), bins=nb, levels={})
    for lvl, path in sorted(cfgs.items()):
        cfg = json.load(open(path))
        try:
            sp_model = [simulate(cfg, i, r) for i, r, _ in test]
            model = [binned(sp, nb) for sp in sp_model]
            res["levels"][lvl] = dict(f1=f1(model, true, w), rate_model=float(np.mean([m.mean() for m in model])) * 1000 / DT_MS)
            os.makedirs(SPK, exist_ok=True)
            np.savez(f"{SPK}/{sid}_glif{lvl}.npz", *sp_model)
        except Exception as e:
            res["levels"][lvl] = dict(error=repr(e)[:200])
    res["rate_true"] = float(np.mean([t.mean() for t in true])) * 1000 / DT_MS
    return res


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    only = sys.argv[2:]
    models = json.load(open(MODELS))
    by = {}
    for m in models:
        by.setdefault(m["specimen"], {})[m["level"]] = m["config"]
    jobs = [(sid, c) for sid, c in sorted(by.items()) if os.path.exists(f"{DATA}/{sid}.npz") and (not only or str(sid) in only)]
    rows = []
    with Pool(nproc) as p:
        for r in p.imap_unordered(run, jobs):
            if r is None: continue
            rows.append(r)
            print(r["specimen"], {k: round(v.get("f1", -1), 3) for k, v in r["levels"].items()}, f"rate {r['rate_true']:.1f} Hz", flush=True)
    rows.sort(key=lambda r: r["specimen"])
    json.dump(rows, open(OUTJ if not only else "results/e27_glif_official_test.json", "w"), indent=1)
    print(len(rows), "cells")
