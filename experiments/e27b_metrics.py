"""E27b: robustness of the official-GLIF vs exact-gradient-fit comparison to the metric.  For every cell with official
GLIF models: our fitted hard- and soft-reset LIF (E15b/c parameters) and our ALIF (E28) are re-simulated on the Noise 2
sweeps; the official GLIF spike times come from E27.  Metrics on the same truncated window:
  F1 at +-4 ms and +-10 ms (E15 aggregation), and an explained-variance ratio of Gaussian-smoothed spike trains
  (sigma = 10 ms): EV(model) / EV(repeat), EV(a, b) = 1 - var(a - b) / var(b), averaged over test repeats, with the
  repeat-to-repeat EV as normaliser (the form of the GLIF paper's metric; not claimed to be identical to it).
usage: python3 experiments/e27b_metrics.py"""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
_argv = sys.argv[:]
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])
sys.argv = _argv
from lifscan import triton_alif as A

SIG_MS = 10.0
DATA = os.environ.get("E27_DATA", "data/allen")
SPK = os.environ.get("E27_SPK", "results/e27_spikes")
BASE = {}
for f in (glob.glob(os.environ["E28_BASE_GLOB"]) if "E28_BASE_GLOB" in os.environ else glob.glob("results/e15b_allen_w*.json") + glob.glob("results/e15c_allen_w*.json")):
    for r in json.load(open(f)): BASE[r["cell"]["specimen"]] = r
ALIF = {}
for f in glob.glob(os.environ.get("E28_GLOB", "results/e28_allen_alif_w*.json")):
    for r in json.load(open(f)):
        if not r.get("failed"): ALIF[r["cell"]["specimen"]] = r
OFF = {r["specimen"]: r for r in json.load(open(os.environ.get("E27_OUT", "results/e27_glif_official.json")))}


def smooth(S):                                   # S: [T, n] binned trains -> Gaussian-smoothed (FFT)
    sig = SIG_MS / DT_MS; L = int(4 * sig)
    k = torch.exp(-0.5 * (torch.arange(-L, L + 1, device=S.device, dtype=S.dtype) / sig) ** 2); k = k / k.sum()
    n = S.shape[0] + 2 * L + 1
    out = torch.fft.irfft(torch.fft.rfft(S, n=n, dim=0) * torch.fft.rfft(k, n=n)[:, None], n=n, dim=0)
    return out[L:L + S.shape[0]]


def ev(a, b):
    return float(1 - torch.var(a - b) / torch.var(b))


def ev_ratio(Sm, St):
    """Sm, St: [T, n] model and recorded trains for the n test repeats."""
    sm, st_ = smooth(Sm), smooth(St)
    n = St.shape[1]
    model = np.mean([ev(sm[:, i], st_[:, i]) for i in range(n)])
    rep = np.mean([ev(st_[:, j], st_[:, i]) for i in range(n) for j in range(n) if i != j])
    return float(model / rep), float(rep)


def trains_from_times(times, T):
    S = torch.zeros(T, len(times), device=dev)
    for j, sp in enumerate(times):
        idx = (np.asarray(sp) * 1000.0 / DT_MS).astype(np.int64)
        S[idx[(idx >= 0) & (idx < T)], j] = 1.0
    return S


rows = []
for sid in sorted(OFF):
    if sid not in BASE: continue
    meta, sw = load_cell(f"{DATA}/{sid}.npz")
    Xte, Ste = stack(sw["Noise 2"])
    T = Xte.shape[0]
    w4, w10 = int(4.0 / DT_MS), int(10.0 / DT_MS)
    preds = {}
    b = BASE[sid]["models"]
    with torch.no_grad():
        for kind in ("hard", "soft"):
            m = b[kind]
            p = dict(zb=torch.tensor(math.log(m["beta"] / (1 - m["beta"])), device=dev), gain=torch.tensor(m["gain"], device=dev),
                     bias=torch.tensor(m["bias"], device=dev), ur=torch.tensor(m["ur"] if m["ur"] is not None else 0.0, device=dev))
            preds[kind] = simulate(kind, Xte, p, mode="serial")
        if sid in ALIF:
            a = ALIF[sid]; z = torch.zeros(Xte.shape[1], device=dev)
            I = (a["gain"] * Xte + a["bias"]).contiguous()
            preds["alif"] = A.alif(I, z, z, a["beta"], a["rho"], a["q"], 1.0, a["ur"], fwd="serial")[0]
    for lvl in OFF[sid]["levels"]:
        z = np.load(f"{SPK}/{sid}_glif{lvl}.npz")
        preds[f"glif{lvl}"] = trains_from_times([z[k] for k in z.files], T)
    row = dict(specimen=sid, dendrite=meta["dendrite"], metrics={})
    for k, Sm in preds.items():
        evr, rep = ev_ratio(Sm, Ste)
        row["metrics"][k] = dict(f1_4=f1(Sm, Ste, w4), f1_10=f1(Sm, Ste, w10), ev_ratio=evr)
    row["ev_repeat"] = rep
    rows.append(row)
    print(sid, {k: round(v["ev_ratio"], 3) for k, v in row["metrics"].items()}, flush=True)
json.dump(rows, open(os.environ.get("E27B_OUT", "results/e27b_metrics.json"), "w"), indent=1)
