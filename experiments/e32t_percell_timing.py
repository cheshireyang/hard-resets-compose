"""E32t: per-cell iteration time (one fit at a time) on the GPU used for E32, so that batched and per-cell numbers come
from the same device.  Hard LIF (E15 protocol: serial vs auto) and ALIF (E28 protocol: serial vs scan) on the first
NCELLS database cells; 30 iterations, median of iterations 5..29.
usage: NCELLS=32 python3 experiments/e32t_percell_timing.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
os.environ.setdefault("E28_BASE_GLOB", "results/e15all_allen_w*.json")
os.environ.setdefault("E28_GLIF", "results/allen_glif_models_all.json")
os.environ.setdefault("E28_DATA", "data/allen_all")
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e28_allen_alif.py")).read().rsplit("rows = []", 1)[0])
sys.argv = _argv
fit_alif = fit                                   # E28's fit(Xtr, Str, b, q0, rho0, mode, iters)
_e15 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")
_ns = {"__file__": _e15}
exec(open(_e15).read().split("rows = []")[0], _ns)
fit_hard = _ns["fit"]
NCELLS = int(os.environ.get("NCELLS", "32"))
rows = []
for sid in sorted(s for s in BASE if s in GLIF and os.path.exists(f"{DATA}/{s}.npz"))[:NCELLS]:
    meta, sw = load_cell(f"{DATA}/{sid}.npz")
    if len(sw["Noise 1"]) < 2 or len(sw["Noise 2"]) < 2: continue
    Xtr, Str = stack(sw["Noise 1"])
    r = dict(cell=sid, N=Xtr.shape[1], T=Xtr.shape[0])
    for mode in ("serial", "auto"):
        _, _, tt = fit_hard("hard", Xtr, Str, 7, mode=mode, iters=30); r[f"hard_ms_{mode}"] = 1000 * float(np.median(tt[5:]))
    for mode in ("serial", "scan"):
        _, _, tt = fit_alif(Xtr, Str, BASE[sid], 0.05, 0.99, mode=mode, iters=30); r[f"alif_ms_{mode}"] = 1000 * float(np.median(tt[5:]))
    rows.append(r)
    print(r, flush=True)
    json.dump(rows, open("results/e32t_percell_timing.json", "w"), indent=1)
