"""E28t: clean per-iteration timing of the ALIF fit (E28), serial kernels vs scans, on every cell that E28 fitted,
from its best initialisation; nothing else runs on the machine.  30 iterations, median of iterations 5..29."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["E28_ITERS"] = "0"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e28_allen_alif.py")).read().split("\nrows = []\ncells = sorted")[0]
sys.argv = [sys.argv[0]]
exec(src)
import numpy as np
done = [r for r in json.load(open("results/e28_allen_alif_w0.json")) if not r.get("failed")]
rows = []
for r in done:
    sid = r["cell"]["specimen"]
    meta, sw = load_cell(f"data/allen/{sid}.npz")
    Xtr, Str = stack(sw["Noise 1"])
    q0, rho0 = r["init"]
    out = dict(specimen=sid, T=Xtr.shape[0], N=Xtr.shape[1])
    for mode in ("serial", "scan"):
        _, _, tt = fit(Xtr, Str, BASE[sid], q0, rho0, mode=mode, iters=30)
        out[f"ms_{mode}"] = 1000 * float(np.median(tt[5:]))
    rows.append(out)
    print(sid, out["N"], f"serial {out['ms_serial']:.2f} scan {out['ms_scan']:.2f} x{out['ms_serial'] / out['ms_scan']:.1f}", flush=True)
json.dump(rows, open("results/e28t_alif_timing.json", "w"), indent=1)
